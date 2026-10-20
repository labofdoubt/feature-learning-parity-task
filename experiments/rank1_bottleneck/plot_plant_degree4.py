"""Summary figures for the planted degree-4 circuit experiment (plant_degree4.py).

Reads runs/<tag>_<arm>_s<seed>/analysis/{intervention.json,trajectory.csv,final.json} and writes
  <out>_summary.png  : final degree-4 share and final objective vs rho (coherent seeds, target-orthogonal
                       controls, orig/prep references, predicted rho_crit and limiting objectives)
  <out>_traj.png     : trajectories (share, objective, planted/rest target coefficient, template
                       projections) for representative arms
  <out>_summary.csv  : one row per arm.

Usage: python experiments/rank1_bottleneck/plot_plant_degree4.py --runs-root runs --tag plantd4_Q8 --out reports/rank1_bottleneck/plantd4_Q8
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RHO_CRIT = 0.28310
L_SINGLE, L_D4 = 0.20196, 0.03764


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--tag", default="plantd4_Q8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-arms", nargs="*", default=None, help="arms to show in the trajectory figure")
    args = ap.parse_args()
    rows = []
    trajs = {}
    for d in sorted(Path(args.runs_root).glob(f"{args.tag}_*_s*")):
        mt = re.match(rf"{re.escape(args.tag)}_(.+)_s(\d+)$", d.name)
        fin = d / "analysis" / "final.json"
        if not mt or not fin.exists():
            continue
        arm, seed = mt.group(1), int(mt.group(2))
        meta = json.loads((d / "analysis" / "intervention.json").read_text())
        f = json.loads(fin.read_text())
        kind = "orig" if arm == "orig" else "prep" if arm == "prep" else "coherent" if arm.startswith("rho") else "orthogonal"
        rho = 0.0 if arm in ("orig", "prep") else float(arm[3:] if kind == "coherent" else arm[4:])
        tr = pd.read_csv(d / "analysis" / "trajectory.csv")
        trajs[(arm, seed)] = tr
        # classify: rescue if final share > 0.9 and L near degree-4; decay if share < 0.05; else mixed/unresolved
        s, L = f["final"]["share_d4"], f["final"]["L"]
        outcome = "rescue" if s > 0.9 else "decay" if s < 0.05 else "mixed/unresolved"
        rows.append(dict(arm=arm, kind=kind, rho=rho, seed=seed, steps=meta["steps"], share_d4_final=s, L_final=L,
                         mse_final=f["final"]["mse"], a_final=f["final"]["a"],
                         a_planted_final=f["final"].get("a_planted"), a_rest_final=f["final"].get("a_rest"),
                         abl_only_d4_mse=f["ablation"]["only_degree4"]["mse"], abl_only_single_mse=f["ablation"]["only_singletons"]["mse"],
                         share_d4_max=float(tr.share_d4.max()), outcome=outcome))
    df = pd.DataFrame(rows).sort_values(["kind", "rho", "seed"])
    df.to_csv(f"{args.out}_summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---- summary figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for kind, mk, col in (("coherent", "o", "C0"), ("orthogonal", "s", "C1")):
        sub = df[df.kind == kind]
        if len(sub):
            axes[0].plot(sub.rho, sub.share_d4_final, mk, color=col, ms=7, label=f"{kind} seed, final")
            axes[0].plot(sub.rho, sub.share_d4_max, mk, color=col, ms=7, mfc="none", alpha=0.6, label=f"{kind} seed, max over run")
            axes[1].plot(sub.rho, sub.L_final, mk, color=col, ms=7, label=f"{kind} seed")
    for name, ls in (("orig", ":"), ("prep", "--")):
        sub = df[df.kind == name]
        if len(sub):
            axes[0].axhline(sub.share_d4_final.iloc[0], color="gray", ls=ls, label=f"{name} control")
            axes[1].axhline(sub.L_final.iloc[0], color="gray", ls=ls, label=f"{name} control")
    axes[1].axhline(L_SINGLE, color="C3", ls="-.", lw=1, label=f"predicted singleton objective {L_SINGLE}")
    axes[1].axhline(L_D4, color="C2", ls="-.", lw=1, label=f"predicted degree-4 objective {L_D4}")
    for ax in axes:
        ax.axvline(RHO_CRIT, color="r", ls="--", lw=1, label=f"predicted rho_crit = {RHO_CRIT}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("degree-4 pre-activation share"); axes[0].set_ylim(-0.03, 1.03); axes[0].legend(fontsize=7, loc="center left")
    axes[1].set_ylabel("final objective L = MSE + P"); axes[1].set_xlabel("seed strength rho"); axes[1].legend(fontsize=7, loc="center left")
    fig.suptitle(f"{args.tag}: planted degree-4 circuit in the trained singleton solution, {int(df.steps.max())} post-intervention steps", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{args.out}_summary.png", dpi=140)

    # ---- trajectories
    arms = args.traj_arms or [a for a in ["prep", "rho0.23", "rho0.28", "rho0.30", "rho0.34", "rho0.40", "orth0.34"] if any(k[0] == a for k in trajs)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (arm, seed), tr in sorted(trajs.items()):
        if arm not in arms:
            continue
        lab = f"{arm} s{seed}"
        x = tr.post_step.clip(lower=1)
        axes[0, 0].semilogx(x, tr.share_d4, label=lab)
        axes[0, 1].semilogx(x, tr.L, label=lab)
        if "a_planted" in tr:
            axes[1, 0].semilogx(x, tr.a_planted, label=f"{lab} planted")
            axes[1, 0].semilogx(x, tr.a_rest, ":", label=f"{lab} rest")
            axes[1, 1].semilogx(x, tr.b_par, label=f"{lab} b"); axes[1, 1].semilogx(x, tr.w_par, "--", label=f"{lab} w"); axes[1, 1].semilogx(x, tr.r_par, ":", label=f"{lab} r")
    axes[0, 0].set_ylabel("degree-4 share"); axes[0, 1].set_ylabel("objective L"); axes[0, 1].axhline(L_SINGLE, color="C3", ls="-.", lw=1); axes[0, 1].axhline(L_D4, color="C2", ls="-.", lw=1)
    axes[1, 0].set_ylabel("target coefficient: planted set / rest"); axes[1, 1].set_ylabel("template projections b, w, r")
    for ax in axes.flat:
        ax.grid(alpha=0.3); ax.set_xlabel("post-intervention step (log)")
    axes[0, 0].legend(fontsize=7); axes[1, 1].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{args.tag}: trajectories", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{args.out}_traj.png", dpi=140)
    print(f"{args.out}_summary.png", f"{args.out}_traj.png")


if __name__ == "__main__":
    main()
