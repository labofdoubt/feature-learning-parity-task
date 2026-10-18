"""Two-row figure: d4 (top) and d32 (bottom) test MSE curves for a set of runs
(e.g. two weight decays), plus faint train curves.

Usage: python scripts/plot_d4_d32_curves.py --runs-root runs --runs NAME1 NAME2 --labels "wd 1e-3" "wd 1e-4" --title ... --out file.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for run, label in zip(args.runs, args.labels):
        f = Path(args.runs_root) / run / "metrics.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f); m = m[m["test_mse"].notna()]
        for ax, deg in zip(axes, ("d4", "d32")):
            line, = ax.plot(m["step"], m[f"test_mse_{deg}"], lw=1.3,
                            label=f"{label}: test {m.iloc[-1][f'test_mse_{deg}']:.3g} @ {int(m.iloc[-1]['step']) // 1000}k")
            # train per-degree losses are not in metrics.csv; show total train MSE faintly on the d32 panel only
    for ax, deg in zip(axes, ("d4", "d32")):
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5); ax.grid(alpha=0.3); ax.set_ylabel(f"test MSE {deg}"); ax.legend(fontsize=9)
    axes[1].set_xlabel("step")
    fig.suptitle(args.title, fontsize=11); fig.tight_layout(); fig.savefig(args.out, dpi=130); print(args.out)


if __name__ == "__main__":
    main()
