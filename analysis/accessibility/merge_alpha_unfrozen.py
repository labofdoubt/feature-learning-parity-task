"""Merge experiment-4 slices (texas + texas2 + texas3) and draw the figures.

Figures (per the request): (1) (R0, lambda) phase diagram in the wd-report
style; (2) f22 vs R0 cuts at fixed lambda; (3) (R_final, lambda) phase diagram
in the unfrozen-report style; (4) phase-plane R(t) vs f22(t); plus (5) drift
R_final vs R0 per lambda -- the direct answer to whether the drift direction
depends on the initial value.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="accessibility_alpha_unfrozen")
ap.add_argument("--tags", default="texas,texas2,texas3")
ap.add_argument("--title", default="exp 4 (alpha at converged-d2 onset, unfrozen)")
A = ap.parse_args()
OUT = Path("/root/parity/analysis_output") / A.dir
RC = 1 / np.sqrt(8)
TAGS = A.tags.split(",")

df = pd.concat([pd.read_csv(OUT / f"results_{t}.csv") for t in TAGS], ignore_index=True)
tr = pd.concat([pd.read_csv(OUT / f"trajectories_{t}.csv") for t in TAGS], ignore_index=True)
assert not df.duplicated(["seed", "alpha", "lam"]).any(), "overlapping slices!"
df.to_csv(OUT / "results_merged.csv", index=False)
print("cells:", len(df), "| alphas:", sorted(df.alpha.unique()),
      "| seeds:", sorted(df.seed.unique()))

lam_y = lambda l: l if l > 0 else 1e-5 / 3
m = df.groupby(["alpha", "lam"]).agg(R0=("R0", "mean"), Rf=("R_final", "mean"),
                                     f=("frac22_w", "mean"), f4=("f4", "mean")).reset_index()

# 1. phase diagram vs R0
for xcol, fname, xlabel in (("R0", "phase_R0", "$R_0$ (measured, at d4 onset)"),
                            ("Rf", "phase_Rfinal", "$R_{final}$")):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(m[xcol], [lam_y(l) for l in m.lam], c=m.f, s=300, marker="s",
                    cmap="coolwarm", vmin=0, vmax=1, edgecolors="k", linewidths=0.4)
    fail = m[m.f4.abs() < 0.5]
    ax.scatter(fail[xcol], [lam_y(l) for l in fail.lam], marker="x", s=110, c="k",
               label="failed (|f4|<0.5)")
    ax.axvline(RC, color="r", ls="--", lw=1.2, label=r"$R_c=1/\sqrt{8}$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"phase-2 weight decay $\lambda$  (bottom row = 0)")
    ax.set_title(f"{A.title}: mean 2+2 fraction", fontsize=10)
    fig.colorbar(sc, label="fraction of 2+2→4 computation")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=170, bbox_inches="tight")

# 2. cuts at fixed lambda
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for lam in [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    g = m[m.lam == lam].sort_values("R0")
    ax.plot(g.R0, g.f, "o-", label=f"λ={lam:g}")
ax.axvline(RC, color="r", ls="--")
ax.set_xlabel("$R_0$ (measured, at d4 onset)")
ax.set_ylabel("fraction of 2+2→4 computation")
ax.set_title("unfrozen: does initial accessibility still select the circuit?", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "cuts_fixed_lambda.pdf", bbox_inches="tight")
fig.savefig(OUT / "cuts_fixed_lambda.png", dpi=170, bbox_inches="tight")

# 3. drift: R_final vs R0, per lambda
fig, ax = plt.subplots(figsize=(6.6, 4.6))
cm = plt.cm.viridis
lams = sorted(m.lam.unique())
for i, lam in enumerate(lams):
    g = m[m.lam == lam].sort_values("R0")
    ax.plot(g.R0, g.Rf, "o-", color=cm(i / (len(lams) - 1)), label=f"λ={lam:g}")
lim = max(m.R0.max(), m.Rf.max()) * 1.05
ax.plot([0, lim], [0, lim], "k:", lw=1, label="no drift")
ax.axvline(RC, color="r", ls="--", lw=0.9)
ax.axhline(RC, color="r", ls="--", lw=0.9)
ax.set_xlabel("$R_0$")
ax.set_ylabel("$R_{final}$")
ax.set_title("representation drift: where does R end up, given where it started?",
             fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "drift.pdf", bbox_inches="tight")
fig.savefig(OUT / "drift.png", dpi=170, bbox_inches="tight")

# 4. phase plane R(t) vs f22(t): all alphas, three lambdas, seed 0
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
alphas = sorted(tr.alpha.unique())
cma = plt.cm.plasma
for j, lam in enumerate([0.0, 3e-3, 3e-2]):
    ax = axes[j]
    for i, a in enumerate(alphas):
        g = tr[(tr.seed == 0) & (tr.alpha == a) & (tr.lam == lam)].sort_values("step")
        if not len(g):
            continue
        c = cma(i / (len(alphas) - 1))
        ax.plot(g.R, g.frac22_w, "-", color=c, lw=1.1, label=f"α={a:g}")
        ax.plot(g.R.iloc[:1], g.frac22_w.iloc[:1], "o", color=c, ms=5)
        ax.plot(g.R.iloc[-1:], g.frac22_w.iloc[-1:], "s", color=c, ms=5)
    ax.axvline(RC, color="r", ls="--")
    ax.set_xlabel("R(t)")
    ax.set_title(f"λ={lam:g}", fontsize=10)
    if j == 0:
        ax.set_ylabel("f 2+2(t)")
        ax.legend(fontsize=6, ncol=2)
fig.suptitle("phase plane (seed 0): trajectories from α-set initial conditions "
             "(o start, ■ end)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "phase_plane.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase_plane.png", dpi=170, bbox_inches="tight")
print("figures saved")

import subprocess
r = subprocess.run(["rclone", "copy", str(OUT),
                    f"gdrive:parity/analysis_output/{A.dir}/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
