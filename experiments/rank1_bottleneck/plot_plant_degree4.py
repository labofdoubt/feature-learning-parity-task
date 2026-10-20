"""Summary figures for the planted degree-4 circuit experiment (plant_degree4.py).

Reads runs/<tag>_<arm>_s<seed>/analysis/{intervention.json,trajectory.csv,final.json} and writes
  <out>_summary.png  : final degree-4 share and final objective vs seed strength rho
  <out>_traj.png     : trajectories of a few representative arms (below / at / above the barrier,
                       and one target-orthogonal control)
  <out>_summary.csv  : one row per arm.

Usage: python experiments/rank1_bottleneck/plot_plant_degree4.py --runs-root runs --tag plantd4_Q8 \
           --out reports/rank1_bottleneck/plantd4_Q8 [--traj-arms rho0.28 rho0.29 rho0.30 orth0.40] [--extra-tag plantd4_Q8L]
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
OUTCOME_COLOR = {"rescue": "#2ca02c", "decay": "#d62728", "mixed": "#ff7f0e"}
ARM_COLORS = ["#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e", "#8c564b", "#17becf"]


def classify(share: float) -> str:
    return "rescue" if share > 0.9 else "decay" if share < 0.05 else "mixed"


def load_arms(runs_root: Path, tags: list[str]):
    rows, trajs = [], {}
    for tag in tags:
        for d in sorted(runs_root.glob(f"{tag}_*_s*")):
            mt = re.match(rf"{re.escape(tag)}_(.+)_s(\d+)$", d.name)
            fin = d / "analysis" / "final.json"
            if not mt or not fin.exists():
                continue
            arm, seed = mt.group(1), int(mt.group(2))
            meta = json.loads((d / "analysis" / "intervention.json").read_text())
            f = json.loads(fin.read_text())
            kind = "orig" if arm == "orig" else "prep" if arm == "prep" else "coherent" if arm.startswith("rho") else "orthogonal"
            rho = 0.0 if kind in ("orig", "prep") else float(arm[3:] if kind == "coherent" else arm[4:])
            tr = pd.read_csv(d / "analysis" / "trajectory.csv")
            trajs[(tag, arm, seed)] = tr
            s, L = f["final"]["share_d4"], f["final"]["L"]
            rows.append(dict(tag=tag, arm=arm, kind=kind, rho=rho, seed=seed, steps=int(meta["steps"]),
                             share_d4_final=s, L_final=L, mse_final=f["final"]["mse"], a_final=f["final"]["a"],
                             a_planted_final=f["final"].get("a_planted"), a_rest_final=f["final"].get("a_rest"),
                             ablation_only_degree4_mse=f["ablation"]["only_degree4"]["mse"],
                             ablation_only_singletons_mse=f["ablation"]["only_singletons"]["mse"],
                             outcome=classify(s)))
    return pd.DataFrame(rows), trajs


def summary_figure(df: pd.DataFrame, out: str, tag: str) -> None:
    main = df[df.tag == tag]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)
    seeds = sorted(main.seed.unique())
    base_seed = seeds[0]
    for ax, col, ylab in ((axes[0], "share_d4_final", "final degree-4 pre-activation share"),
                          (axes[1], "L_final", "final objective  L = MSE + (λ/2)Σ‖θ‖²")):
        for kind, marker, lab in (("coherent", "o", "coherent seed"), ("orthogonal", "s", "target-orthogonal control")):
            sub = main[main.kind == kind]
            for _, r in sub.iterrows():
                primary = r.seed == base_seed
                ax.plot(r.rho + (0 if primary else 0.004 * (seeds.index(r.seed))), r[col], marker,
                        ms=9 if primary else 5, mfc=OUTCOME_COLOR[r.outcome], mec="k", mew=0.6, alpha=1 if primary else 0.8, zorder=3)
        ax.axvline(RHO_CRIT, color="k", ls="--", lw=1, zorder=1)
        ax.grid(alpha=0.25); ax.set_ylabel(ylab)
    axes[0].set_ylim(-0.04, 1.04)
    axes[1].axhline(L_SINGLE, color="#d62728", ls=":", lw=1.2, zorder=1)
    axes[1].axhline(L_D4, color="#2ca02c", ls=":", lw=1.2, zorder=1)
    axes[1].set_xlabel("seed strength  ρ")
    # one legend, built by hand
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc="w", mec="k", label="coherent seed (fill = outcome)"),
               Line2D([], [], marker="s", ls="", ms=9, mfc="w", mec="k", label="target-orthogonal control"),
               Line2D([], [], marker="o", ls="", ms=5, mfc="w", mec="k", label=f"repeat with another minibatch seed (small, offset)"),
               Line2D([], [], marker="o", ls="", ms=9, mfc=OUTCOME_COLOR["rescue"], mec="k", label="rescue: degree-4 solution"),
               Line2D([], [], marker="o", ls="", ms=9, mfc=OUTCOME_COLOR["mixed"], mec="k", label="mixed / unresolved"),
               Line2D([], [], marker="o", ls="", ms=9, mfc=OUTCOME_COLOR["decay"], mec="k", label="decay: singleton solution"),
               Line2D([], [], color="k", ls="--", lw=1, label=f"predicted barrier ρ_crit = {RHO_CRIT}"),
               Line2D([], [], color="#d62728", ls=":", lw=1.2, label=f"predicted singleton objective {L_SINGLE}"),
               Line2D([], [], color="#2ca02c", ls=":", lw=1.2, label=f"predicted degree-4 objective {L_D4}")]
    axes[1].legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(0.01, 0.52), framealpha=0.95)
    steps = int(main.steps.max())
    fig.suptitle(f"Planted degree-4 circuit in the trained singleton solution: outcome after {steps//1000}k post-intervention steps", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}_summary.png", dpi=150)


def trajectory_figure(trajs: dict, df: pd.DataFrame, out: str, arms: list[str]) -> None:
    """One line per selected arm (primary seed), same color in every panel, one legend."""
    sel = []
    for a in arms:
        cands = [k for k in trajs if k[1] == a]
        if not cands:
            continue
        # prefer the longest run, then the lowest seed
        k = sorted(cands, key=lambda k: (-len(trajs[k]), k[2]))[0]
        sel.append(k)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for i, k in enumerate(sel):
        tr = trajs[k]; col = ARM_COLORS[i % len(ARM_COLORS)]
        arm = k[1]
        lab = ("coherent ρ = " + arm[3:]) if arm.startswith("rho") else ("target-orthogonal ρ = " + arm[4:]) if arm.startswith("orth") else arm
        x = tr.post_step / 1000
        axes[0].plot(x, tr.share_d4, color=col, lw=2, label=lab)
        axes[1].plot(x, tr.L, color=col, lw=2)
        if "a_planted" in tr:
            axes[2].plot(x, tr.a_planted, color=col, lw=2)
            axes[2].plot(x, tr.a_rest, color=col, lw=1.2, ls="--")
    axes[0].set_ylabel("degree-4 pre-activation share"); axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("objective  L"); axes[1].axhline(L_SINGLE, color="#d62728", ls=":", lw=1.2); axes[1].axhline(L_D4, color="#2ca02c", ls=":", lw=1.2)
    axes[1].text(0.02, L_SINGLE + 0.004, "singleton", color="#d62728", fontsize=8, transform=axes[1].get_yaxis_transform())
    axes[1].text(0.02, L_D4 + 0.004, "degree-4", color="#2ca02c", fontsize=8, transform=axes[1].get_yaxis_transform())
    axes[2].set_ylabel("target coefficient"); axes[2].set_ylim(-0.03, 1.03)
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[Line2D([], [], color="gray", lw=2, label="planted units (solid)"),
                            Line2D([], [], color="gray", lw=1.2, ls="--", label="all other units (dashed)")], fontsize=8, loc="center right")
    for ax in axes:
        ax.set_xlabel("post-intervention step  (×10³)"); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("Trajectories below, near and above the barrier", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}_traj.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--tag", default="plantd4_Q8")
    ap.add_argument("--extra-tag", nargs="*", default=[], help="additional tags (e.g. longer continuations) to load for trajectories/table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-arms", nargs="*", default=["rho0.28", "rho0.29", "rho0.30", "orth0.40"])
    args = ap.parse_args()
    df, trajs = load_arms(Path(args.runs_root), [args.tag] + list(args.extra_tag))
    df = df.sort_values(["tag", "kind", "rho", "seed"])
    df.to_csv(f"{args.out}_summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    summary_figure(df, args.out, args.tag)
    trajectory_figure(trajs, df, args.out, args.traj_arms)
    print(f"{args.out}_summary.png", f"{args.out}_traj.png")


if __name__ == "__main__":
    main()
