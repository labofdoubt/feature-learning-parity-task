"""Plot train/test MSE curves from metrics.csv.

Produces:
  plots/curves_by_degree.png  – per-degree test MSE over time
  plots/curves_total.png      – train batch vs test total MSE

Usage:
    python scripts/analyze_curves.py --run-dir runs/my_exp/N_2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def run(run_dir: Path) -> None:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in {run_dir}")

    df = pd.read_csv(csv_path)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    step = df["step"]

    # ── Per-degree test MSE ───────────────────────────────────────────────────
    degree_cols = [c for c in df.columns if c.startswith("test_mse_d")]
    if degree_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in sorted(degree_cols):
            label = col.replace("test_mse_", "")
            ax.semilogy(step, df[col], label=label)
        ax.set_xlabel("step")
        ax.set_ylabel("test MSE")
        ax.set_title(f"Test MSE by degree  –  {run_dir.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "curves_by_degree.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Total train vs test MSE ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    if "train_mse" in df.columns:
        ax.semilogy(step, df["train_mse"], label="train (batch)", alpha=0.6)
    if "test_mse" in df.columns:
        ax.semilogy(step, df["test_mse"], label="test")
    if "train_set_mse" in df.columns:
        ax.semilogy(step, df["train_set_mse"], label="train (pool)", linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title(f"Total MSE  –  {run_dir.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "curves_total.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run(Path(args.run_dir))


if __name__ == "__main__":
    main()
