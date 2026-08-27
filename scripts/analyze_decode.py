"""Partition-based decoding of parity modes, one table per block.

For a target degree-d parity (over bit indices 0..d-1 by default), at every
residual block:

  1. original_mode  = E_x[r(x) * chi_I(x)]        (block residual update)
  2. For each set partition of I into parts {B_1,...,B_k}:
       h_reduced = E_x[h(x)] + sum_k E_x[h(x)*chi_{B_k}(x)] * chi_{B_k}(x)
       modified_mode = E_x[r(h_reduced) * chi_I(x)]
  3. Report: norm(original_mode), norm(modified_mode), cosine(original, modified)

Partition families:
  d4  → all 15 Bell(4) set partitions
  d8  → "tree" family (~26 binary-tree partitions)
  d16 → "tree" family (~677 partitions, slow)

Output per degree: one combined PNG (table per block) + per-block CSVs.

Usage:
    python scripts/analyze_decode.py --run-dir runs/my_exp/N_2048 --degree 4
    python scripts/analyze_decode.py --run-dir runs/my_exp/N_2048 --degree 8
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
import torch.nn.functional as F

from parity_net.checkpoint import load_checkpoint
from parity_net.data import load_dataset
from parity_net.train import resolve_device, resolve_dtype


# ── Partition generators ───────────────────────────────────────────────────────

def all_set_partitions(items: list) -> list[list[tuple]]:
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    result = []
    for partition in all_set_partitions(rest):
        for i in range(len(partition)):
            new_p = [list(b) for b in partition]
            new_p[i] = sorted(new_p[i] + [first])
            result.append([tuple(sorted(b)) for b in new_p])
        result.append([tuple([first])] + [tuple(b) for b in partition])
    return result


def tree_partitions(items: list, max_count: int = 10_000) -> list[list[tuple]]:
    result: list[list[tuple]] = [[tuple(items)]]
    seen: set[frozenset] = {frozenset([frozenset(items)])}
    queue: list[list[tuple]] = [[tuple(items)]]
    while queue and len(result) < max_count:
        partition = queue.pop(0)
        for bi, block in enumerate(partition):
            block = list(block)
            if len(block) < 2:
                continue
            mid = len(block) // 2
            left, right = tuple(sorted(block[:mid])), tuple(sorted(block[mid:]))
            new_p = sorted([b for j, b in enumerate(partition) if j != bi] + [left, right])
            key = frozenset(frozenset(b) for b in new_p)
            if key not in seen:
                seen.add(key)
                result.append(new_p)
                if len(result) < max_count:
                    queue.append(new_p)
    return result


def partition_label(partition: list[tuple]) -> str:
    return " | ".join("*".join(f"x{i+1}" for i in sorted(b)) for b in sorted(partition))


# ── Block helpers ──────────────────────────────────────────────────────────────

@torch.no_grad()
def block_residual_update(block, h: torch.Tensor) -> torch.Tensor:
    update = block.activation(block.linear(h))
    if block.post_activation_linear is not None:
        update = block.post_activation_linear(update)
    return update


@torch.no_grad()
def input_to_block(model, x_batch: torch.Tensor, block_idx: int) -> torch.Tensor:
    h = model.embedding(x_batch)
    for i, block in enumerate(model.blocks):
        if i == block_idx:
            break
        h = block(h)
    return h


# ── Core decode at one block ───────────────────────────────────────────────────

@torch.no_grad()
def decode_block(
    model,
    x: torch.Tensor,
    block_idx: int,
    indices: list[int],
    partitions: list[list[tuple]],
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    """
    Returns (df, original_norm) where df has columns:
      partition, norm_modified, cosine
    sorted by cosine descending.
    """
    # Collect all unique subsets that appear across partitions
    all_subsets: set[tuple] = set()
    for partition in partitions:
        for b in partition:
            all_subsets.add(tuple(sorted(b)))
    full_key = tuple(sorted(indices))

    n = model.config.N
    device = x.device
    dtype = x.dtype

    # One pass: accumulate block-input directions and original mode
    constant_acc = torch.zeros(n, device=device, dtype=torch.float64)
    subset_accs = {s: torch.zeros(n, device=device, dtype=torch.float64) for s in all_subsets}
    full_mode_acc = torch.zeros(n, device=device, dtype=torch.float64)
    total = 0

    for start in range(0, x.shape[0], batch_size):
        x_b = x[start:start + batch_size]
        h = input_to_block(model, x_b, block_idx)
        r = block_residual_update(model.blocks[block_idx], h)

        h64 = h.to(dtype=torch.float64)
        r64 = r.to(dtype=torch.float64)

        idx_I = torch.tensor(list(full_key), device=device, dtype=torch.long)
        chi_I = x_b[:, idx_I].prod(dim=1).to(dtype=torch.float64)
        full_mode_acc += (chi_I.unsqueeze(1) * r64).sum(dim=0)
        constant_acc += h64.sum(dim=0)

        for s in all_subsets:
            idx_s = torch.tensor(list(s), device=device, dtype=torch.long)
            chi_s = x_b[:, idx_s].prod(dim=1).to(dtype=torch.float64)
            subset_accs[s] += (chi_s.unsqueeze(1) * h64).sum(dim=0)

        total += x_b.shape[0]

    input_constant = (constant_acc / total).to(dtype=dtype)
    input_dirs = {s: (v / total).to(dtype=dtype) for s, v in subset_accs.items()}
    full_mode = (full_mode_acc / total).cpu()
    original_norm = full_mode.norm().item()

    # One pass per partition: reconstruct h_reduced and compute modified mode
    rows = []
    for partition in partitions:
        parts = [tuple(sorted(b)) for b in partition]
        mod_acc = torch.zeros(n, device=device, dtype=torch.float64)
        mod_total = 0

        for start in range(0, x.shape[0], batch_size):
            x_b = x[start:start + batch_size]
            bs = x_b.shape[0]

            h_reduced = input_constant.unsqueeze(0).expand(bs, -1).clone()
            for p in parts:
                idx_p = torch.tensor(list(p), device=device, dtype=torch.long)
                chi_p = x_b[:, idx_p].prod(dim=1)
                h_reduced = h_reduced + chi_p.unsqueeze(1) * input_dirs[p].unsqueeze(0)

            r_red = block_residual_update(model.blocks[block_idx], h_reduced)

            idx_I = torch.tensor(list(full_key), device=device, dtype=torch.long)
            chi_I = x_b[:, idx_I].prod(dim=1).to(dtype=torch.float64)
            mod_acc += (chi_I.unsqueeze(1) * r_red.to(dtype=torch.float64)).sum(dim=0)
            mod_total += bs

        modified_mode = (mod_acc / mod_total).cpu()
        modified_norm = modified_mode.norm().item()
        cosine = F.cosine_similarity(
            modified_mode.unsqueeze(0), full_mode.unsqueeze(0), dim=1, eps=1e-12
        ).item()

        part_dir_norms = [
            torch.linalg.vector_norm(input_dirs[p]).item() for p in parts
        ]
        score = modified_norm * cosine / max(original_norm, 1e-12)

        rows.append({
            "partition": partition_label(partition),
            "norm_modified": round(modified_norm, 6),
            "cosine": round(cosine, 6),
            "score": round(score, 6),
            "input_direction_norms": [round(n, 5) for n in part_dir_norms],
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df, original_norm


# ── Figure: table per block ────────────────────────────────────────────────────

_HEADER_COLOR = "#2C5F8A"
_ROW_COLORS = ["#FFFFFF", "#EDF3FA"]


def render_table_figure(
    block_dfs: list[tuple[pd.DataFrame, float]],
    degree: int,
    run_name: str,
    out_path: Path,
    max_rows: int = 30,
) -> None:
    n_blocks = len(block_dfs)
    col_labels = ["partition", "norm_modified", "cosine"]

    # One subplot per block, sized by the largest table shown
    display_rows = min(max_rows, max(len(df) for df, _ in block_dfs))
    row_h = 0.28   # inches per data row
    header_h = 1.0  # inches for axes title + header row
    panel_h = display_rows * row_h + header_h
    fig, axes = plt.subplots(n_blocks, 1, figsize=(11, panel_h * n_blocks + 0.4))
    if n_blocks == 1:
        axes = [axes]

    for block_idx, (df, orig_norm) in enumerate(block_dfs):
        ax = axes[block_idx]
        ax.axis("off")

        display_df = df[col_labels].head(max_rows)
        cell_text = [
            [row["partition"], f"{row['norm_modified']:.5f}", f"{row['cosine']:.5f}"]
            for _, row in display_df.iterrows()
        ]

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)

        # Column widths: partition gets most of the space
        col_widths = [0.60, 0.20, 0.20]
        for (row_i, col_i), cell in tbl.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_width(col_widths[col_i])
            if row_i == 0:
                cell.set_facecolor(_HEADER_COLOR)
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor(_ROW_COLORS[(row_i - 1) % 2])
            # Left-align partition column
            if col_i == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.03

        shown = len(display_df)
        total = len(df)
        note = f"  (showing top {shown} of {total})" if total > shown else ""
        ax.set_title(
            f"Block {block_idx}  —  d{degree} parity  —  norm_original = {orig_norm:.5f}{note}",
            fontsize=10, pad=6, loc="left",
        )

    fig.suptitle(f"Partition decoding d{degree}  –  {run_name}", fontsize=12, y=1.002)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def run(
    run_dir: Path,
    degree: int,
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
    dtype = resolve_dtype(training["dtype"])
    model = model.to(device=device, dtype=dtype).eval()

    if indices is None:
        indices = list(range(degree))

    test_data = load_dataset(run_dir / "test_data.pt", device, dtype)
    x = test_data.x
    if num_samples is not None and num_samples < x.shape[0]:
        x = x[:num_samples]

    if degree <= 4:
        partitions = all_set_partitions(indices)
        print(f"d{degree}: {len(partitions)} partitions (all Bell({degree}))")
    else:
        partitions = tree_partitions(indices, max_count=max_partitions)
        print(f"d{degree}: {len(partitions)} partitions (tree family, max={max_partitions})")

    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    analysis_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    n_blocks = len(model.blocks)
    block_results: list[tuple[pd.DataFrame, float]] = []

    for block_idx in range(n_blocks):
        print(f"\n  Block {block_idx} / {n_blocks - 1} …")
        df, orig_norm = decode_block(model, x, block_idx, indices, partitions, batch_size)
        csv_path = analysis_dir / f"decode_d{degree}_block{block_idx}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    norm_original = {orig_norm:.5f}  |  top cosine = {df['cosine'].iloc[0]:.4f}")
        block_results.append((df, orig_norm))

    plot_path = plots_dir / f"decode_d{degree}_blocks.png"
    render_table_figure(block_results, degree, run_dir.name, plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Bit indices for the parity (default: 0..degree-1)")
    parser.add_argument("--num-samples", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-partitions", type=int, default=1000,
                        help="Max partitions for tree family (d8/d16)")
    args = parser.parse_args()
    run(
        Path(args.run_dir),
        args.degree,
        args.indices,
        args.num_samples,
        args.batch_size,
        args.max_partitions,
    )


if __name__ == "__main__":
    main()
