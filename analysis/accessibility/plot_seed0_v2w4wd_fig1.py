"""Plot the preliminary seed-0 analogue of Fig. 1 in the V2-decay report."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_output" / "accessibility_alpha_v2w4wd"
ALPHAS = [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0]
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
RC = 1 / np.sqrt(8)


files = [OUT / f"results_v2w4_g{i}.csv" for i in range(8)]
missing = [str(path) for path in files if not path.exists()]
assert not missing, f"Missing worker results: {missing}"

rows = []
for path in files:
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
seed0 = [row for row in rows if int(row["seed"]) == 0]
assert len(seed0) == 64, f"Expected 64 seed-0 cells, found {len(seed0)}"
keys = {(float(row["alpha"]), float(row["lam"])) for row in seed0}
assert len(keys) == 64, "Duplicate seed-0 cells"
assert sorted({float(row["alpha"]) for row in seed0}) == ALPHAS
assert sorted({float(row["lam"]) for row in seed0}) == LAMBDAS

seed0.sort(key=lambda row: (float(row["lam"]), float(row["alpha"])))


def lam_y(value: float) -> float:
    return value if value > 0 else 1e-5 / 3


fig = plt.figure(figsize=(14.4, 4.6))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.30)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
cax = fig.add_subplot(grid[0, 2])
for ax, xcol, xlabel in (
    (axes[0], "R0", r"$R_0$ (measured, at d4 onset)"),
    (axes[1], "R_final", r"$R_{final}$"),
):
    sc = ax.scatter(
        [float(row[xcol]) for row in seed0],
        [lam_y(float(row["lam"])) for row in seed0],
        c=[float(row["frac22_w"]) for row in seed0],
        s=300,
        marker="s",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        edgecolors="k",
        linewidths=0.4,
    )
    fail = [row for row in seed0 if abs(float(row["f4"])) < 0.5]
    ax.scatter(
        [float(row[xcol]) for row in fail],
        [lam_y(float(row["lam"])) for row in fail],
        marker="x",
        s=110,
        c="k",
        label=r"failed ($|\hat f_4|<0.5$)",
    )
    ax.axvline(RC, color="r", ls="--", lw=1.2, label=r"$R_c=1/\sqrt{8}$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"weight decay $\lambda$  (bottom row = 0)")
    ax.set_title("seed 0 preliminary: 2+2 fraction", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

fig.colorbar(sc, cax=cax, label="fraction of 2+2→4 computation")
fig.suptitle("Matched decay on $V_2$ and the d4 readout row", fontsize=12)
fig.subplots_adjust(left=0.07, right=0.94, bottom=0.14, top=0.84)
fig.savefig(OUT / "fig1_seed0_preliminary.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_seed0_preliminary.png", dpi=180, bbox_inches="tight")
print(f"validated {len(seed0)} seed-0 cells and wrote {OUT / 'fig1_seed0_preliminary.pdf'}")
