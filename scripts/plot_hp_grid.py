"""3x3 grid of validation curves on the (wd, lr) plane for one HP family.

Usage: python scripts/plot_hp_grid.py --runs-root runs --prefix hp_d32_var1N --out file.png
Expects runs named <prefix>_lr<lr>_wd<wd>; rows = wd (descending), cols = lr (descending).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LRS = ["1e-3", "3e-4", "1e-4"]
WDS = ["1e-2", "1e-3", "1e-4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for i, wd in enumerate(WDS):
        for j, lr in enumerate(LRS):
            ax = axes[i][j]
            f = Path(args.runs_root) / f"{args.prefix}_lr{lr}_wd{wd}" / "metrics.csv"
            if not f.exists():
                ax.set_title(f"lr {lr}, wd {wd}: missing", fontsize=9); continue
            m = pd.read_csv(f); m = m[m["test_mse"].notna()]
            ax.plot(m["step"], m["test_mse"], label="test", lw=1.2)
            ax.plot(m["step"], m["train_mse"], label="train", lw=0.8, alpha=0.7)
            last = m.iloc[-1]
            ax.set_title(f"lr {lr}, wd {wd}  →  test {last['test_mse']:.3g} @ {int(last['step']) // 1000}k", fontsize=9)
            ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5); ax.grid(alpha=0.3)
            if i == 2: ax.set_xlabel("step")
            if j == 0: ax.set_ylabel("MSE")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{args.prefix}: validation (test) and train MSE on the (wd, lr) grid", fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=130); print(args.out)


if __name__ == "__main__":
    main()
