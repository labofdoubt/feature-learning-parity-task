"""Full (regularized) objective of the final checkpoints of a Q sweep.

For every run <prefix>Q<q><suffix>: exact test MSE on all 2^d inputs, the weight-decay term
(wd / 2) * sum ||theta||^2 over the parameters that the forward pass uses (embedding, linear,
readout, and the bottleneck matrices when the bottleneck is on), and their sum, which is the
objective SGD with `weight_decay` descends. Writes a CSV and a figure.

Usage: python -m experiments.rank1_bottleneck.full_loss --prefix rank1_d8_s44_nobn_wd9e-2_ --suffix _50k --out fig.png
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from .config import Rank1Config
from .model import Rank1BottleneckNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--xlabel", default="Q_1 / Q_4 (singleton / degree-4 embedding norm at init)")
    args = ap.parse_args()
    rows = []
    for d in sorted(Path(args.runs_root).glob(f"{args.prefix}Q*{args.suffix}")):
        mt = re.search(r"Q([0-9.]+)" + re.escape(args.suffix) + "$", d.name)
        f = d / "checkpoints" / "final.pt"
        if not mt or not f.exists():
            continue
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        cfg = Rank1Config(**ckpt["config"])
        model = Rank1BottleneckNet(cfg)
        model.load_state_dict(ckpt["model_state"])
        x = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=cfg.d)))
        with torch.no_grad():
            mse = float(F.mse_loss(model(x), x.prod(1)))
            used = [model.embedding, model.linear.weight, model.readout.weight]
            if cfg.use_bottleneck:
                used += [model.W_enc, model.W_dec]
            sq = {n: float((p ** 2).sum()) for n, p in zip(["embedding", "linear", "readout", "W_enc", "W_dec"], used)}
        penalty = 0.5 * sum((cfg.readout_wd() if k == "readout" else cfg.weight_decay) * v for k, v in sq.items())
        rows.append(dict(Q=float(mt.group(1)), test_mse=mse, wd_penalty=penalty, full_loss=mse + penalty,
                         **{f"sumsq_{k}": v for k, v in sq.items()}))
    df = pd.DataFrame(rows).sort_values("Q")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(df.Q, df.test_mse, "o-", color="C3", label="test MSE")
    ax.semilogy(df.Q, df.full_loss, "o-", color="k", label="full loss = MSE + (wd/2) sum ||theta||^2")
    ax.set_xlabel(args.xlabel); ax.set_ylabel("final value"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(args.title or f"{args.prefix}{args.suffix}: full objective at the final checkpoint", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=140); print(args.out)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
