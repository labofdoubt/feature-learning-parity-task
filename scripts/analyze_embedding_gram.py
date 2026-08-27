"""Embedding Gram-matrix heatmap.

Computes E^T E where E is the N×input_dim embedding weight matrix, then plots
two heatmaps: the raw Gram (input_dim × input_dim) and the cosine version
(each entry divided by the product of column norms).

Usage:
    python scripts/analyze_embedding_gram.py --run-dir runs/my_exp/N_2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from parity_net.checkpoint import load_checkpoint
from parity_net.train import resolve_device, resolve_dtype


def run(run_dir: Path) -> None:
    device = resolve_device("cpu")  # gram matrix is small, CPU is fine
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    model, payload, _ = load_checkpoint(ckpt_path, device)
    config = payload["config"]
    dtype = resolve_dtype(config["training"]["dtype"])
    model = model.to(device=device, dtype=dtype)

    W = model.embedding.weight.detach().double()  # (N, input_dim)
    gram = W.T @ W  # (input_dim, input_dim)
    col_norms = gram.diagonal().sqrt().clamp_min(1e-12)
    cosine_gram = gram / col_norms.unsqueeze(0) / col_norms.unsqueeze(1)

    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    analysis_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    pd.DataFrame(gram.numpy()).to_csv(analysis_dir / "embedding_gram.csv")
    pd.DataFrame(cosine_gram.numpy()).to_csv(analysis_dir / "embedding_cosine_gram.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in zip(
        axes,
        [gram, cosine_gram],
        ["Embedding Gram  (E^T E)", "Embedding Cosine Gram"],
    ):
        im = ax.imshow(matrix.numpy(), aspect="auto", cmap="RdBu_r",
                       vmin=-matrix.abs().max().item(), vmax=matrix.abs().max().item())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("input dim")
        ax.set_ylabel("input dim")

    fig.suptitle(f"Embedding Gram  –  {run_dir.name}")
    fig.tight_layout()
    plot_path = plots_dir / "embedding_gram.png"
    fig.savefig(plot_path, dpi=200)
    fig.savefig(plot_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    d = gram.shape[0]
    off_diag = gram[~torch.eye(d, dtype=torch.bool)].float()
    print(f"Gram diagonal: mean={gram.diagonal().mean():.4f}  std={gram.diagonal().std():.4f}")
    print(f"Gram off-diag: mean={off_diag.mean():.4f}  std={off_diag.std():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run(Path(args.run_dir))


if __name__ == "__main__":
    main()
