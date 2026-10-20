"""Phase diagram (Q4/Q1, lambda) of the FINAL MSE for the d8 frozen-accessibility x WD
experiment (exp 7), the companion of the f44 phase diagram in merge_d8_wd.py: same cells,
same target-ratio binning and seed averaging, colored by the mean final MSE of the d8 output
over the full 2^8 cube (log scale), failures (|f8| < 0.5) crossed, boundary 5/32 in red.

Usage: python analysis/accessibility/plot_d8_mse_phase.py \
         --results analysis_output/accessibility_d8_wd/results_merged.csv --out reports/d8_mse_phase_diagram.pdf
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RC = 5 / 32
TLIST = [0.05, 0.08, 0.10, 0.12, 0.135, 0.145, 0.152, 0.156, 0.160, 0.168, 0.18, 0.20, 0.25, 0.30]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="analysis_output/accessibility_d8_wd/results_merged.csv")
ap.add_argument("--out", default="reports/d8_mse_phase_diagram.pdf")
ap.add_argument("--vmin", type=float, default=1e-15)
A = ap.parse_args()

cells = defaultdict(list)
with open(A.results) as f:
    for r in csv.DictReader(f):
        r41, lam = float(r["r41"]), float(r["lam"])
        tbin = min(TLIST, key=lambda t: abs(t - r41))
        cells[(tbin, lam)].append((r41, float(r["mse"]), float(r["f8"])))
r41 = np.array([np.mean([c[0] for c in v]) for v in cells.values()])
mse = np.array([np.mean([c[1] for c in v]) for v in cells.values()])
f8 = np.array([np.mean([c[2] for c in v]) for v in cells.values()])
lam = np.array([k[1] for k in cells])
lam_y = np.where(lam > 0, lam, 1e-6 / 3)

fig, ax = plt.subplots(figsize=(7.6, 4.8))
sc = ax.scatter(r41, lam_y, c=np.clip(mse, A.vmin, 1.0), s=210, marker="s", cmap="viridis_r",
                norm=LogNorm(vmin=A.vmin, vmax=1.0), edgecolors="k", linewidths=0.4)
fail = np.abs(f8) < 0.5
ax.scatter(r41[fail], lam_y[fail], marker="x", s=90, c="k", label="failed (|f8|<0.5)")
ax.axvline(RC, color="r", ls="--", lw=1.2, label=r"$Q_4/Q_1=5/32$")
ax.set_yscale("log")
ax.set_xlabel("measured $Q_4/Q_1$")
ax.set_ylabel(r"stage-2 weight decay $\lambda$  (bottom row = 0)")
ax.set_title("exp 7 (d8, frozen): mean final MSE on the full cube, 3 seeds", fontsize=10)
fig.colorbar(sc, label=f"final MSE (clipped below at {A.vmin:g})")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
out = Path(A.out)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
print(out)
for l in sorted(set(lam)):
    sel = lam == l
    order = np.argsort(r41[sel])
    print(f"lam={l:<7g}", " ".join(f"{v:.2g}" for v in mse[sel][order]))
