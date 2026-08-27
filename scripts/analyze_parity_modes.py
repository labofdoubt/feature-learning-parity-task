"""Parity-mode Gram matrices and cross-layer alignment.

Two analyses in one script:

1. **Parity-mode Gram matrices per block** (Section 6 of the notebooks):
   For each residual block, estimates the "Walsh direction" of each parity
   target: v_{spec} = E_x[r(x) * chi_spec(x)] where r(x) is the block's
   pre-residual update and chi_spec is the parity monomial.  Plots the
   pairwise Gram (dot-product) matrix of these direction vectors, one heatmap
   per block.

2. **Cross-layer alignment** (Section 8):
   Picks a single parity monomial (--align-indices) and estimates its Walsh
   direction vector in every block.  Plots the pairwise cosine matrix across
   blocks, showing how consistently the same direction is used across depth.

Usage:
    python scripts/analyze_parity_modes.py --run-dir runs/my_exp/N_2048
    python scripts/analyze_parity_modes.py --run-dir runs/my_exp/N_2048 \\
        --degrees 2 4 8 16 --align-indices 0 1 2 3 4 5 6 7 \\
        --num-samples 50000
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from parity_net.checkpoint import load_checkpoint
from parity_net.data import TargetSpec, load_dataset, target_names, tree_parity_specs
from parity_net.train import max_target_degree_for_model, resolve_device, resolve_dtype


@torch.no_grad()
def block_residual_update(block, h: torch.Tensor) -> torch.Tensor:
    update = block.activation(block.linear(h))
    if block.post_activation_linear is not None:
        update = block.post_activation_linear(update)
    return update


@torch.no_grad()
def compute_walsh_directions(
    model,
    x: torch.Tensor,
    block_idx: int,
    specs: list[TargetSpec],
    batch_size: int,
    use_post_residual: bool = False,
) -> torch.Tensor:
    """Returns (len(specs), N) tensor: E_x[r_block(x) * chi_spec(x)]."""
    N = model.config.N
    directions = torch.zeros(len(specs), N, device=x.device, dtype=x.dtype)
    n_total = x.shape[0]

    model.eval()
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        x_batch = x[start:stop]

        h = model.embedding(x_batch)
        for i, block in enumerate(model.blocks):
            if i == block_idx:
                if use_post_residual:
                    r = block(h)  # h + update
                else:
                    r = block_residual_update(block, h)
                break
            h = block(h)

        for spec_idx, spec in enumerate(specs):
            idx = torch.tensor(spec.indices, device=x.device, dtype=torch.long)
            chi = x_batch[:, idx].prod(dim=1)  # (batch,)  ±1 parity monomial
            directions[spec_idx] += (r * chi.unsqueeze(1)).sum(dim=0)

    return directions / n_total


def plot_gram_heatmap(matrix: torch.Tensor, labels: list[str], title: str, ax) -> None:
    mat = matrix.cpu().float().numpy()
    vmax = max(abs(mat.min()), abs(mat.max()), 1e-6)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def run(
    run_dir: Path,
    degrees: list[int],
    align_indices: list[int],
    num_samples: int | None,
    batch_size: int,
    normalize_gram: bool,
) -> None:
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    model, payload, _ = load_checkpoint(ckpt_path, device)
    config = payload["config"]
    training = config["training"]
    task = config.get("task") or config["model"]
    dtype = resolve_dtype(training["dtype"])
    model = model.to(device=device, dtype=dtype)

    target_names_ = target_names(
        int(task["relevant_dim"]),
        list(task.get("exclude_targets", [])),
        max_target_degree_for_model(model.config),
    )
    relevant_dim = int(task["relevant_dim"])

    # Build the list of parity specs we'll analyse
    all_specs = tree_parity_specs(relevant_dim, task.get("exclude_targets", []),
                                  max_target_degree_for_model(model.config))
    if degrees:
        all_specs = [s for s in all_specs if s.degree in degrees]
    spec_labels = [s.name for s in all_specs]

    test_data = load_dataset(run_dir / "test_data.pt", device, dtype)
    x = test_data.x
    if num_samples is not None and num_samples < x.shape[0]:
        x = x[:num_samples]

    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    analysis_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    n_blocks = len(model.blocks)

    # ── 1. Parity-mode Gram matrices per block ────────────────────────────────
    fig_gram, axes_gram = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5), squeeze=False)

    for block_idx in range(n_blocks):
        directions = compute_walsh_directions(model, x, block_idx, all_specs, batch_size)
        # Gram matrix: raw dot products or cosine
        if normalize_gram:
            norms = directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
            normed = directions / norms
            gram = normed @ normed.T
            kind = "cosine"
        else:
            gram = directions @ directions.T
            kind = "raw"

        pd.DataFrame(gram.cpu().numpy(), index=spec_labels, columns=spec_labels).to_csv(
            analysis_dir / f"parity_mode_gram_block{block_idx}_{kind}.csv"
        )

        ax = axes_gram[0][block_idx]
        plot_gram_heatmap(gram, spec_labels, f"Block {block_idx}  ({kind} Gram)", ax)

    fig_gram.suptitle(f"Parity-mode Gram matrices  –  {run_dir.name}")
    fig_gram.tight_layout()
    gram_path = plots_dir / f"parity_mode_gram_{kind}.png"
    fig_gram.savefig(gram_path, dpi=150)
    plt.close(fig_gram)
    print(f"Gram plot: {gram_path}")

    # ── 2. Cross-layer alignment for a chosen monomial ────────────────────────
    align_spec = TargetSpec(
        name="custom_" + "_".join(map(str, align_indices)),
        degree=len(align_indices),
        indices=tuple(align_indices),
    )

    block_directions = []
    for block_idx in range(n_blocks):
        d = compute_walsh_directions(model, x, block_idx, [align_spec], batch_size)
        block_directions.append(d[0])  # (N,)

    stack = torch.stack(block_directions)  # (n_blocks, N)
    norms = stack.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine_mat = (stack / norms) @ (stack / norms).T  # (n_blocks, n_blocks)

    block_labels = [f"block_{i}" for i in range(n_blocks)]
    pd.DataFrame(cosine_mat.cpu().numpy(), index=block_labels, columns=block_labels).to_csv(
        analysis_dir / f"parity_cross_block_cosines_{align_spec.name}.csv"
    )

    fig_align, ax_align = plt.subplots(figsize=(5, 4))
    plot_gram_heatmap(cosine_mat, block_labels,
                      f"Cross-layer cosines  chi_{{{''.join(map(str, align_indices))}}}", ax_align)
    # Annotate cells
    for i in range(cosine_mat.shape[0]):
        for j in range(cosine_mat.shape[1]):
            ax_align.text(j, i, f"{cosine_mat[i, j].item():.2f}", ha="center", va="center", fontsize=8)

    fig_align.suptitle(f"Alignment  –  {run_dir.name}  –  indices {align_indices}")
    fig_align.tight_layout()
    align_path = plots_dir / f"parity_cross_block_alignment_{align_spec.name}.png"
    fig_align.savefig(align_path, dpi=150)
    plt.close(fig_align)
    print(f"Alignment plot: {align_path}")

    # Print norms per block for the chosen monomial
    print(f"\nWalsh direction norms for chi_{{{align_indices}}} across blocks:")
    for i, (norm, d) in enumerate(zip(norms.squeeze(1).tolist(), block_directions)):
        print(f"  block {i}: ||v|| = {norm:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Which parity degrees to include in the Gram analysis")
    parser.add_argument("--align-indices", type=int, nargs="+", default=list(range(16)),
                        help="Bit indices of the monomial for the cross-layer alignment")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of test samples to use (default: all)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--normalize-gram", action="store_true",
                        help="Plot cosine Gram instead of raw dot-product Gram")
    args = parser.parse_args()
    run(
        Path(args.run_dir),
        args.degrees,
        args.align_indices,
        args.num_samples,
        args.batch_size,
        args.normalize_gram,
    )


if __name__ == "__main__":
    main()
