"""Output Walsh spectrum contributed by groups of hidden units of a trained rank-K run.

Units are grouped by the shape of their exact pre-activation decomposition (which singleton
sum dominates and the relative sign of the two sums); for each group the contribution
sum_j r_j relu(z_j(x)) is expanded exactly in all 2^d Walsh modes.

Usage: python -m experiments.rank1_bottleneck.group_spectrum --run runs/<name>
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .config import Rank1Config
from .model import Rank1BottleneckNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final.pt")
    ap.add_argument("--min-power", type=float, default=1e-9, help="ignore units with weaker output power")
    args = ap.parse_args()
    ckpt = torch.load(Path(args.run) / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Rank1Config(**ckpt["config"])
    model = Rank1BottleneckNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    d = cfg.d
    x = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=d)))
    supports = [s for k in range(d + 1) for s in itertools.combinations(range(d), k)]
    chi = torch.stack([x[:, list(s)].prod(1) if s else torch.ones(len(x)) for s in supports], 1)   # (2^d, 2^d)
    names = ["1"] + ["".join(f"x{i+1}" for i in s) for s in supports[1:]]
    with torch.no_grad():
        _, parts = model(x, return_parts=True)
        a = F.relu(model.linear(parts["bottleneck_out"]))
        if cfg.residual_after_bottleneck:
            a = parts["bottleneck_out"] + a
        out = a * model.readout.weight[0]                       # (2^d, N) per-unit output contributions
        C = model.preact_coefficients()                         # (N, M)
    sing = {i: m for m, s in enumerate(model.supports) if len(s) == 1 for i in s}
    diff12 = (C[:, sing[0]] - C[:, sing[1]]) / 2; sum12 = (C[:, sing[0]] + C[:, sing[1]]) / 2
    diff34 = (C[:, sing[2]] - C[:, sing[3]]) / 2; sum34 = (C[:, sing[2]] + C[:, sing[3]]) / 2
    s1 = torch.where(diff12.abs() > sum12.abs(), diff12, sum12)   # dominant combination of x1, x2
    s2 = torch.where(diff34.abs() > sum34.abs(), diff34, sum34)
    power = (out ** 2).mean(0)
    label = []
    for j in range(cfg.d_hid):
        if power[j] < args.min_power:
            label.append("silent"); continue
        lead = "x1-x2" if diff12[j].abs() > sum12[j].abs() else "x1+x2"
        lead2 = "x3-x4" if diff34[j].abs() > sum34[j].abs() else "x3+x4"
        dom = "A" if s1[j].abs() >= s2[j].abs() else "B"
        sign = "+" if s1[j] * s2[j] > 0 else "-"
        label.append(f"{dom}:{lead}/{lead2}:{sign}")
    label = pd.Series(label)
    rows = {}
    for g in sorted(label.unique()):
        idx = torch.tensor((label == g).values)
        contrib = out[:, idx].sum(1)
        rows[f"{g} (n={int(idx.sum())})"] = (chi.t() @ contrib / len(x)).numpy()
    rows["total"] = (chi.t() @ out.sum(1) / len(x)).numpy()
    df = pd.DataFrame(rows, index=names)
    pd.set_option("display.width", 200)
    print(df.to_string(float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
