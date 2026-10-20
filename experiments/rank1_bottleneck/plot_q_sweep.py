"""Degree-2 share of the bottleneck code vs the singleton/pair embedding norm ratio Q_1/Q_2.

Reads runs named <prefix>Q<q> (metrics.csv) and plots, against Q: the final degree-2 share
of the code power, its value at a few intermediate steps, and the final test MSE.

Usage: python experiments/rank1_bottleneck/plot_q_sweep.py --runs-root runs --prefix rank1_d4_sp_K4_wd1e-3_ --out fig.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def full_loss(run_dir: Path) -> float:
    """MSE on all 2^d inputs + (wd/2) * sum ||theta||^2 over the parameters the forward pass uses,
    from checkpoints/final.pt (the objective SGD with weight_decay descends)."""
    import itertools
    import torch
    import torch.nn.functional as F
    from .config import Rank1Config
    from .model import Rank1BottleneckNet
    ckpt = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
    cfg = Rank1Config(**ckpt["config"])
    model = Rank1BottleneckNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    x = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=cfg.d)))
    if cfg.noise_modes:
        x = x.repeat(4096, 1)
    with torch.no_grad():
        mse = float(F.mse_loss(model(x), x.prod(1)))
        used = ([] if cfg.freeze_embedding else [model.embedding]) + [model.linear.weight]
        if cfg.use_bottleneck:
            used += [model.W_enc, model.W_dec]
        sq = sum(float((p ** 2).sum()) for p in used)
        sq_ro = float((model.readout.weight ** 2).sum())
    return mse + 0.5 * (cfg.weight_decay * sq + cfg.readout_wd() * sq_ro)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, nargs="*", default=[500, 1000])
    ap.add_argument("--metric", default="bottleneck/fraction_degree2",
                    help="column to plot as the degree-2 share (e.g. preact/fraction_degree2)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--suffix", default="", help="run-name suffix after the Q value, e.g. _50k")
    ap.add_argument("--full-loss", action="store_true", help="also plot MSE + (wd/2) sum ||theta||^2 from final.pt")
    ap.add_argument("--xlabel", default="Q_1 / Q_2 (singleton / pair embedding norm at init)")
    args = ap.parse_args()
    rows = []
    for d in sorted(Path(args.runs_root).glob(f"{args.prefix}Q*{args.suffix}")):
        mt = re.search(r"Q([0-9.]+)" + re.escape(args.suffix) + "$", d.name)
        f = d / "metrics.csv"
        if not mt or not f.exists():
            continue
        m = pd.read_csv(f); last = m.iloc[-1]
        row = dict(Q=float(mt.group(1)), share2_final=last[args.metric],
                   test_mse=last["test_mse"], step=int(last["step"]))
        for s in args.steps:
            sub = m[m.step == s]
            row[f"share2_{s}"] = float(sub[args.metric].iloc[0]) if len(sub) else float("nan")
        if args.full_loss:
            row["full_loss"] = full_loss(d)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Q")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]
    ax.plot(df.Q, df.share2_final, "o-", label=f"final (step {int(df.step.iloc[0])})")
    for s in args.steps:
        ax.plot(df.Q, df[f"share2_{s}"], ".--", alpha=0.6, label=f"step {s}")
    ax.set_ylabel(f"share ({args.metric})"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax = axes[1]
    ax.semilogy(df.Q, df.test_mse, "o-", color="C3", label="final test MSE")
    if args.full_loss:
        ax.semilogy(df.Q, df.full_loss, "o-", color="k", label="full loss = MSE + (wd/2) sum ||theta||^2")
        ax.legend(fontsize=9)
    ax.set_ylabel("final value" if args.full_loss else "final test MSE"); ax.set_xlabel(args.xlabel); ax.grid(alpha=0.3)
    fig.suptitle(args.title or f"{args.prefix}: d4 target from x1..x4 + x1x2, x3x4", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=140); print(args.out)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
