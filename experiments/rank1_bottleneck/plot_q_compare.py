"""Compare two Q sweeps (e.g. original inputs vs. non-singleton modes replaced by noise):
panel 1 final test MSE of both vs Q, panel 2 full loss (MSE + (wd/2) sum ||theta||^2) of both.

Reads the CSVs written by plot_q_sweep.py --full-loss.
Usage: python experiments/rank1_bottleneck/plot_q_compare.py --a orig.csv --b noise.csv \
         --label-a "x1..x4 + x1x2, x3x4" --label-b "x1..x4 + two noise bits" --out fig.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--xlabel", default="Q_1 / Q_2 (singleton / second-row embedding norm at init)")
    args = ap.parse_args()
    a = pd.read_csv(args.a).sort_values("Q"); b = pd.read_csv(args.b).sort_values("Q")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]
    ax.semilogy(a.Q, a.test_mse, "o-", color="C3", label=args.label_a)
    ax.semilogy(b.Q, b.test_mse, "o-", color="C0", label=args.label_b)
    ax.set_ylabel("final test MSE"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax = axes[1]
    ax.semilogy(a.Q, a.full_loss, "o-", color="C3", label=args.label_a)
    ax.semilogy(b.Q, b.full_loss, "o-", color="C0", label=args.label_b)
    ax.set_ylabel("full loss = MSE + (wd/2) sum ||theta||^2"); ax.set_xlabel(args.xlabel); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(args.title or f"{args.label_a} vs {args.label_b}", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=140); print(args.out)
    m = a[["Q", "test_mse", "full_loss"]].merge(b[["Q", "test_mse", "full_loss"]], on="Q", suffixes=("_a", "_b"))
    print(m.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
