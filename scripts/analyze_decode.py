"""Decode parity computation via partition analysis (d4, d8, d16).

For a target degree-d parity (over a chosen set of bit indices), this script
enumerates a family of set partitions of those indices and, for each partition,
estimates how much of the degree-d Fourier component of the residual block's
output can be explained by the product of sub-degree Walsh directions.

Algorithm for partition {B_1, ..., B_k} at block `block_idx`:
  v_i = E_x[r(x) * chi_{B_i}(x)]        # Walsh direction for each part
  predicted_scalar = sum_a prod_i v_i[a]  # multilinear inner product (= sum of element-wise products)
  true_vector = E_x[h_block(x) * chi_I(x)]   # full-degree Walsh direction
  cosine = predicted_scalar / (prod_i ||v_i||)

The table is sorted by |predicted_scalar| and saved as a CSV.  A bar chart
shows the contribution of each partition.

Partition families used:
  d4  → all 15 Bell(4) set partitions
  d8  → "tree" family (26 balanced binary-tree partitions)
  d16 → "tree" family (677 partitions)

Usage:
    python scripts/analyze_decode.py --run-dir runs/my_exp/N_2048
    python scripts/analyze_decode.py --run-dir runs/my_exp/N_2048 \\
        --degree 4 --block-idx 3 --indices 0 1 2 3
    python scripts/analyze_decode.py --run-dir runs/my_exp/N_2048 --degree 8
    python scripts/analyze_decode.py --run-dir runs/my_exp/N_2048 --degree 16
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from parity_net.checkpoint import load_checkpoint
from parity_net.data import load_dataset, target_names, tree_parity_specs
from parity_net.train import max_target_degree_for_model, resolve_device, resolve_dtype


# ── Partition generators ───────────────────────────────────────────────────────

def all_set_partitions(items: list) -> Iterator[list[tuple]]:
    """Yield all set partitions of `items` (Bell number count)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in all_set_partitions(rest):
        # Add first to each existing block
        for i in range(len(partition)):
            new_partition = [list(b) for b in partition]
            new_partition[i] = sorted(new_partition[i] + [first])
            yield [tuple(sorted(b)) for b in new_partition]
        # Or start a new singleton block
        yield [tuple([first])] + [tuple(b) for b in partition]


def tree_partitions(items: list, max_count: int = 10_000) -> list[list[tuple]]:
    """Binary-tree partitions: recursively split each block in half.

    Produces at most max_count partitions (stops adding new ones after that).
    """
    result: list[list[tuple]] = [[tuple(items)]]
    seen: set[frozenset] = {frozenset([frozenset(items)])}
    queue: list[list[tuple]] = [[tuple(items)]]

    while queue and len(result) < max_count:
        partition = queue.pop(0)
        for block_idx, block in enumerate(partition):
            block = list(block)
            if len(block) < 2:
                continue
            mid = len(block) // 2
            left, right = tuple(sorted(block[:mid])), tuple(sorted(block[mid:]))
            new_partition = [b for j, b in enumerate(partition) if j != block_idx] + [left, right]
            new_partition = sorted(new_partition)
            key = frozenset(frozenset(b) for b in new_partition)
            if key not in seen:
                seen.add(key)
                result.append(new_partition)
                if len(result) < max_count:
                    queue.append(new_partition)

    return result


def partition_label(partition: list[tuple]) -> str:
    return " | ".join("*".join(f"x{i}" for i in sorted(block)) for block in sorted(partition))


# ── Walsh direction estimation ─────────────────────────────────────────────────

@torch.no_grad()
def block_residual_update(block, h: torch.Tensor) -> torch.Tensor:
    update = block.activation(block.linear(h))
    if block.post_activation_linear is not None:
        update = block.post_activation_linear(update)
    return update


@torch.no_grad()
def compute_walsh_direction(
    model, x: torch.Tensor, block_idx: int, indices: tuple[int, ...], batch_size: int
) -> torch.Tensor:
    """E_x[r(x) * prod_{i in indices} x_i]  →  (N,) vector."""
    N = model.config.N
    direction = torch.zeros(N, device=x.device, dtype=x.dtype)
    for start in range(0, x.shape[0], batch_size):
        x_b = x[start : start + batch_size]
        h = model.embedding(x_b)
        for i, block in enumerate(model.blocks):
            if i == block_idx:
                r = block_residual_update(block, h)
                break
            h = block(h)
        idx = torch.tensor(indices, device=x.device, dtype=torch.long)
        chi = x_b[:, idx].prod(dim=1)
        direction += (r * chi.unsqueeze(1)).sum(dim=0)
    return direction / x.shape[0]


@torch.no_grad()
def compute_walsh_direction_post_residual(
    model, x: torch.Tensor, block_idx: int, indices: tuple[int, ...], batch_size: int
) -> torch.Tensor:
    """Same but measures the full block output (h + r(h)) instead of r(h)."""
    N = model.config.N
    direction = torch.zeros(N, device=x.device, dtype=x.dtype)
    for start in range(0, x.shape[0], batch_size):
        x_b = x[start : start + batch_size]
        h = model.embedding(x_b)
        for i, block in enumerate(model.blocks):
            if i == block_idx:
                h_out = block(h)
                break
            h = block(h)
        idx = torch.tensor(indices, device=x.device, dtype=torch.long)
        chi = x_b[:, idx].prod(dim=1)
        direction += (h_out * chi.unsqueeze(1)).sum(dim=0)
    return direction / x.shape[0]


# ── Main analysis ──────────────────────────────────────────────────────────────

def decode(
    model,
    x: torch.Tensor,
    block_idx: int,
    indices: list[int],
    partitions: list[list[tuple]],
    batch_size: int,
) -> pd.DataFrame:
    """Build the partition decoding table."""
    # Cache Walsh directions for every subset that appears in any partition
    all_subsets: set[tuple] = set()
    for partition in partitions:
        for block in partition:
            all_subsets.add(tuple(sorted(block)))
    full_key = tuple(sorted(indices))
    all_subsets.add(full_key)

    print(f"  Computing Walsh directions for {len(all_subsets)} subsets …")
    directions: dict[tuple, torch.Tensor] = {}
    for subset in sorted(all_subsets, key=len):
        directions[subset] = compute_walsh_direction(model, x, block_idx, subset, batch_size)

    true_vector = directions[full_key]
    true_norm = true_vector.norm().item()
    print(f"  True d{len(indices)} mode norm (from r): {true_norm:.5f}")

    rows = []
    for partition in partitions:
        parts = [tuple(sorted(b)) for b in partition]
        vecs = [directions[p] for p in parts]
        # Multilinear inner product: element-wise product of all vectors, then sum
        product = vecs[0].clone()
        for v in vecs[1:]:
            product = product * v
        predicted_scalar = product.sum().item()
        part_norms = [v.norm().item() for v in vecs]
        denom = 1.0
        for n in part_norms:
            denom *= max(n, 1e-12)
        rows.append({
            "partition": partition_label(partition),
            "num_blocks": len(partition),
            "predicted_scalar": predicted_scalar,
            "predicted_abs": abs(predicted_scalar),
            "true_norm": true_norm,
            "cosine": predicted_scalar / denom,
            "part_norms": str([round(n, 5) for n in part_norms]),
        })

    df = pd.DataFrame(rows).sort_values("predicted_abs", ascending=False).reset_index(drop=True)
    return df


def run(
    run_dir: Path,
    degree: int,
    block_idx: int | None,
    indices: list[int] | None,
    num_samples: int | None,
    batch_size: int,
    max_partitions: int,
) -> None:
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    model, payload, _ = load_checkpoint(ckpt_path, device)
    config = payload["config"]
    training = config["training"]
    task = config.get("task") or config["model"]
    dtype = resolve_dtype(training["dtype"])
    model = model.to(device=device, dtype=dtype)

    relevant_dim = int(task["relevant_dim"])
    L = model.config.L

    # Defaults for indices and block_idx
    if indices is None:
        # Use the first contiguous block of `degree` relevant bits
        indices = list(range(degree))
    if block_idx is None:
        # Put the decode at the last block that reads out this degree (log2(degree)-1)
        import math
        block_idx = min(int(math.log2(degree)) - 1, L - 1)

    print(f"\nDecode d{degree}  |  block_idx={block_idx}  |  indices={indices}")

    test_data = load_dataset(run_dir / "test_data.pt", device, dtype)
    x = test_data.x
    if num_samples is not None and num_samples < x.shape[0]:
        x = x[:num_samples]

    # Choose partition family
    if degree <= 4:
        partitions = list(all_set_partitions(indices))
        print(f"  Using all Bell({degree})={len(partitions)} set partitions")
    else:
        partitions = tree_partitions(indices, max_count=max_partitions)
        print(f"  Using tree family: {len(partitions)} partitions")

    df = decode(model, x, block_idx, indices, partitions, batch_size)

    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    analysis_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    csv_path = analysis_dir / f"decode_d{degree}_block{block_idx}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTop-10 partitions by |predicted_scalar|:")
    print(df.head(10)[["partition", "num_blocks", "predicted_abs", "cosine"]].to_string(index=False))
    print(f"Full table: {csv_path}")

    # Bar chart
    top_k = min(30, len(df))
    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.4), 4))
    colors = ["steelblue" if v >= 0 else "tomato" for v in df["predicted_scalar"].head(top_k)]
    ax.bar(range(top_k), df["predicted_abs"].head(top_k).values, color=colors)
    ax.axhline(y=df["true_norm"].iloc[0], color="k", linestyle="--", label="true mode norm")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(df["partition"].head(top_k).tolist(), rotation=90, fontsize=6)
    ax.set_ylabel("|predicted scalar|")
    ax.set_title(f"d{degree} partition decoding  –  block {block_idx}  –  {run_dir.name}")
    ax.legend()
    fig.tight_layout()
    plot_path = plots_dir / f"decode_d{degree}_block{block_idx}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--degree", type=int, default=4,
                        help="Target parity degree: 4, 8, or 16 (default: 4)")
    parser.add_argument("--block-idx", type=int, default=None,
                        help="Which block's residual update to decode (default: log2(degree)-1)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Bit indices for the parity monomial (default: 0..degree-1)")
    parser.add_argument("--num-samples", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-partitions", type=int, default=1000,
                        help="Max partitions for tree family (d8/d16)")
    args = parser.parse_args()
    run(
        Path(args.run_dir),
        args.degree,
        args.block_idx,
        args.indices,
        args.num_samples,
        args.batch_size,
        args.max_partitions,
    )


if __name__ == "__main__":
    main()
