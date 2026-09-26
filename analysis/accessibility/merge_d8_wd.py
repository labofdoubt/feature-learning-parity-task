"""Merge exp-7 (d8 frozen accessibility x WD) slices and draw the figures.

Figures: (1) phase diagram (Q4/Q1, lambda) colored by f44_w, failures marked,
boundary 5/32; (2) f44 vs ratio cuts at fixed lambda; (3) composition vs
lambda at representative ratios (1^8 -> 4+4 -> dead ladders); (4) suppression
|f8| vs lambda per ratio; (5) control: Q4/Q2 and Q4/Q1 vs alpha from pilots.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="d8_s0a,d8_s0b,d8_s1a,d8_s1b,d8_s2a,d8_s2b")
A = ap.parse_args()
OUT = Path("/root/parity/analysis_output/accessibility_d8_wd")
RC = 5 / 32
TAGS = A.tags.split(",")

df = pd.concat([pd.read_csv(OUT / f"results_{t}.csv") for t in TAGS], ignore_index=True)
df = df.drop_duplicates(["seed", "alpha", "lam"])
df.to_csv(OUT / "results_merged.csv", index=False)
# bin conditions by nominal target ratio: group alphas per seed in order
df["tgt"] = df.groupby("seed")["r41"].transform(lambda s: s.round(3))
print("cells:", len(df), "| ratio range:", round(df.r41.min(), 3), "-", round(df.r41.max(), 3),
      "| r42 range:", round(df.r42.min(), 3), "-", round(df.r42.max(), 3))

lam_y = lambda l: l if l > 0 else 1e-6 / 3
# average across seeds at matched target index: round r41 to nearest of the target list
TLIST = [0.05, 0.08, 0.10, 0.12, 0.135, 0.145, 0.152, 0.156, 0.160, 0.168, 0.18, 0.20, 0.25, 0.30]
df["tbin"] = df.r41.apply(lambda r: min(TLIST, key=lambda t: abs(t - r)))
m = df.groupby(["tbin", "lam"]).agg(r41=("r41", "mean"), f44=("f44_w", "mean"),
                                    f18=("f18_w", "mean"), f24=("f24_w", "mean"),
                                    f8=("f8", "mean"), sq=("sqnorm", "mean")).reset_index()

# 1. phase diagram
fig, ax = plt.subplots(figsize=(7.6, 4.8))
sc = ax.scatter(m.r41, [lam_y(l) for l in m.lam], c=m.f44, s=210, marker="s",
                cmap="coolwarm", vmin=0, vmax=1, edgecolors="k", linewidths=0.4)
fail = m[m.f8.abs() < 0.5]
ax.scatter(fail.r41, [lam_y(l) for l in fail.lam], marker="x", s=90, c="k",
           label="failed (|f8|<0.5)")
ax.axvline(RC, color="r", ls="--", lw=1.2, label=r"$Q_4/Q_1=5/32$")
ax.set_yscale("log")
ax.set_xlabel("measured $Q_4/Q_1$")
ax.set_ylabel(r"stage-2 weight decay $\lambda$  (bottom row = 0)")
ax.set_title("exp 7 (d8, frozen): mean 4+4 fraction, 3 seeds", fontsize=10)
fig.colorbar(sc, label="fraction of 4+4→8 computation")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "phase_diagram.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase_diagram.png", dpi=170, bbox_inches="tight")

# 2. cuts at fixed lambda
fig, ax = plt.subplots(figsize=(6.6, 4.3))
for lam in [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    g = m[m.lam == lam].sort_values("r41")
    ax.plot(g.r41, g.f44, "o-", label=f"λ={lam:g}")
ax.axvline(RC, color="r", ls="--")
ax.set_xlabel("measured $Q_4/Q_1$")
ax.set_ylabel("fraction of 4+4→8 computation")
ax.set_title("does the d8 transition sit at 5/32, and does λ sharpen it?", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "cuts_fixed_lambda.pdf", bbox_inches="tight")
fig.savefig(OUT / "cuts_fixed_lambda.png", dpi=170, bbox_inches="tight")

# 3. composition vs lambda at representative ratios
reps = [0.10, 0.145, 0.168, 0.25]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax, tb in zip(axes, reps):
    g = m[m.tbin == tb].sort_values("lam")
    xs = [lam_y(l) for l in g.lam]
    ax.plot(xs, g.f18, "o-", color="tab:blue", label="$1^8$")
    ax.plot(xs, g.f44, "o-", color="tab:red", label="$4{+}4$")
    ax.plot(xs, g.f24, "o-", color="tab:green", label="$2^4$")
    ax.plot(xs, g.f8.abs(), "k:", label="$|\\hat f_8|$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$  ($3{\cdot}10^{-7}$ = 0)")
    ax.set_title(f"$Q_4/Q_1 \\approx {g.r41.mean():.3f}$", fontsize=10)
axes[0].set_ylabel("fraction / coefficient")
axes[0].legend(fontsize=8)
fig.suptitle("circuit composition vs weight decay: 1⁸ → 4+4 → no d8", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "composition_vs_lambda.pdf", bbox_inches="tight")
fig.savefig(OUT / "composition_vs_lambda.png", dpi=170, bbox_inches="tight")

# 4. suppression
fig, ax = plt.subplots(figsize=(6.6, 4.3))
cm = plt.cm.viridis
tbins = sorted(m.tbin.unique())
for i, tb in enumerate(tbins):
    g = m[m.tbin == tb].sort_values("lam")
    ax.plot([lam_y(l) for l in g.lam], g.f8.abs(), "o-",
            color=cm(i / max(len(tbins) - 1, 1)), label=f"{g.r41.mean():.3f}", ms=3)
ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$  ($3{\cdot}10^{-7}$ = 0)")
ax.set_ylabel(r"$|\hat f_8|$")
ax.set_title("suppression of d8 by weight decay (color: $Q_4/Q_1$)", fontsize=10)
ax.legend(fontsize=6, ncol=2, title="$Q_4/Q_1$", title_fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "suppression.pdf", bbox_inches="tight")
fig.savefig(OUT / "suppression.png", dpi=170, bbox_inches="tight")

# 5. control: pilot curves Q4/Q1 and Q4/Q2 vs alpha
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for tag in TAGS:
    p = OUT / f"pilot_{tag}.json"
    if not p.exists():
        continue
    pj = json.loads(p.read_text())
    for k, v in pj.items():
        al = [q["alpha"] for q in v["pilot"]]
        axes[0].plot(al, [q["r41"] for q in v["pilot"]], "-", alpha=0.6)
        axes[1].plot(al, [q["r42"] for q in v["pilot"]], "-", alpha=0.6)
axes[0].axhline(RC, color="r", ls="--", label="5/32")
axes[0].set_xscale("log"); axes[0].set_xlabel(r"$\alpha$"); axes[0].set_ylabel("$Q_4/Q_1$")
axes[0].legend(fontsize=8); axes[0].set_title("pilot map: accessibility vs α", fontsize=10)
axes[1].axhline(1.0, color="k", ls=":", label="ideal $Q_4/Q_2=1$")
axes[1].axhline(1 / np.sqrt(8), color="r", ls="--", label=r"$1/\sqrt{8}$ (2⁴ viability)")
axes[1].set_xscale("log"); axes[1].set_xlabel(r"$\alpha$"); axes[1].set_ylabel("$Q_4/Q_2$")
axes[1].legend(fontsize=8); axes[1].set_title("2⁴-route control: $Q_4/Q_2$ stays near 1", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "q42_control.pdf", bbox_inches="tight")
fig.savefig(OUT / "q42_control.png", dpi=170, bbox_inches="tight")
print("figures saved")

import subprocess
r = subprocess.run(["rclone", "copy", str(OUT),
                    "gdrive:parity/analysis_output/accessibility_d8_wd/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
