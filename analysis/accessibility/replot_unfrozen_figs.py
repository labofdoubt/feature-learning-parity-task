"""Regenerate unfrozen-experiment figures from saved CSVs (fixed \\sqrt{8} label)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/root/parity/analysis_output/accessibility_unfrozen")
RC = 1 / np.sqrt(8)
df = pd.read_csv(OUT / "results.csv")
tr = pd.read_csv(OUT / "trajectories.csv")
lam_y = lambda l: l if l > 0 else 1e-5 / 3

def phase_plot(xcol, fname, xlabel):
    m = df.groupby(["pre_step", "lam"]).agg(
        x=(xcol, "mean"), f=("frac22_w", "mean"), f4=("f4", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(m.x, [lam_y(l) for l in m.lam], c=m.f, s=340, marker="s",
                    cmap="coolwarm", vmin=0, vmax=1, edgecolors="k", linewidths=0.4)
    fail = m[m.f4.abs() < 0.5]
    ax.scatter(fail.x, [lam_y(l) for l in fail.lam], marker="x", s=120, c="k",
               label="failed (|f4|<0.5)")
    ax.axvline(RC, color="r", ls="--", lw=1.2, label=r"$R_c=1/\sqrt{8}$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"global weight decay $\lambda$  (bottom row = 0)")
    ax.set_title(f"unfrozen d2+d4: mean 2+2 fraction vs ({xlabel}, λ), 3 seeds", fontsize=10)
    fig.colorbar(sc, label="fraction of 2+2→4 computation")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=170, bbox_inches="tight")

phase_plot("R0", "phase_R0", "$R_0$ (at d4 onset)")
phase_plot("R_final", "phase_Rfinal", "$R_{final}$")

pres = sorted(tr[tr.seed == 0].pre_step.unique())
pre_sel = [pres[0], pres[len(pres) // 2], pres[-1]]
lam_sel = [0.0, 3e-3, 3e-2]
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for i, pre in enumerate(pre_sel):
    for j, lam in enumerate(lam_sel):
        ax = axes[i, j]
        g = tr[(tr.seed == 0) & (tr.pre_step == pre) & (tr.lam == lam)].sort_values("step")
        ax.plot(g.step, g.frac22_w, color="tab:blue", label="f 2+2")
        ax.plot(g.step, g.frac1111_w, color="tab:orange", label="f 1⁴")
        ax.plot(g.step, g.e4.abs(), color="tab:green", ls=":", label="|e4|")
        ax2 = ax.twinx()
        ax2.plot(g.step, g.R, color="k", lw=1, alpha=0.6)
        ax2.axhline(RC, color="r", ls="--", lw=0.7)
        ax2.set_ylabel("R(t)", fontsize=7)
        ax.set_title(f"pre={pre} (R0={g.R0.iloc[0]:.2f}), λ={lam:g}", fontsize=8)
        if i == 0 and j == 0:
            ax.legend(fontsize=6)
fig.suptitle("phase-2 dynamics (seed 0): circuit fractions, |e4|, and R(t)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "trajectories_panels.pdf", bbox_inches="tight")
fig.savefig(OUT / "trajectories_panels.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.6, 4.6))
cm = plt.cm.plasma
for i, pre in enumerate(pre_sel):
    for j, lam in enumerate(lam_sel):
        g = tr[(tr.seed == 0) & (tr.pre_step == pre) & (tr.lam == lam)].sort_values("step")
        c = cm((i * 3 + j) / 8)
        ax.plot(g.R, g.frac22_w, "-", color=c, lw=1.2, label=f"pre={pre}, λ={lam:g}")
        ax.plot(g.R.iloc[:1], g.frac22_w.iloc[:1], "o", color=c, ms=6)
        ax.plot(g.R.iloc[-1:], g.frac22_w.iloc[-1:], "s", color=c, ms=6)
ax.axvline(RC, color="r", ls="--", label=r"$R_c$")
ax.set_xlabel("R(t)"); ax.set_ylabel("f 2+2(t)")
ax.set_title("phase plane: R(t) vs f22(t)  (o start, ■ end)", fontsize=9)
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "phase_plane.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase_plane.png", dpi=170, bbox_inches="tight")
print("figures done")

import subprocess
r = subprocess.run(["rclone", "copy", str(OUT),
                    "gdrive:parity/analysis_output/accessibility_unfrozen/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
