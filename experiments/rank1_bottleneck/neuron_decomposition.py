"""Per-neuron decomposition for a trained rank-K bottleneck run.

For every hidden unit j: the exact Walsh coefficients of its pre-activation z_j(x) on the
input modes (x_1, ..., x_d, pairs, ...) and the amount of the target mode (the full product
x_1...x_d) it contributes to the output, r_j * <relu(z_j), chi_target> computed exactly over
all 2^d inputs. Rows are sorted by target power (the square of that contribution).

Usage: python -m experiments.rank1_bottleneck.neuron_decomposition --run runs/<name> --out file.csv
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .config import Rank1Config
from .model import Rank1BottleneckNet, features


def mode_name(support: tuple[int, ...]) -> str:
    return "".join(f"x{i + 1}" for i in support)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt = torch.load(Path(args.run) / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Rank1Config(**ckpt["config"])
    model = Rank1BottleneckNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    d = cfg.d
    x = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=d)))        # all 2^d inputs
    target = x.prod(dim=1)                                                   # chi_{1..d}
    with torch.no_grad():
        _, parts = model(x, return_parts=True)
        z = model.linear(parts["bottleneck_out"])                            # (2^d, N) pre-activations
        a = F.relu(z)
        if cfg.residual_after_bottleneck:
            a = parts["bottleneck_out"] + a
        r = model.readout.weight[0].detach()                                 # (N,)
        coef_target = (a * target[:, None]).mean(0)                          # exact Walsh coefficient per unit
        contrib = r * coef_target                                            # unit's share of the output target coefficient
        C = model.preact_coefficients()                                      # (N, M) exact input decomposition
        # sanity: the exact coefficients reproduce z on the full input set
        assert torch.allclose(features(x, model.supports) @ C.t(), z, atol=1e-4)
    df = pd.DataFrame({"neuron": range(cfg.d_hid), "target_power": (contrib ** 2).numpy(),
                       "target_contrib": contrib.numpy()})
    for m, s in enumerate(model.supports):
        df[mode_name(s)] = C[:, m].numpy()
    df["readout"] = r.numpy()
    df = df.sort_values("target_power", ascending=False).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"output target coefficient {contrib.sum():.4f} (sum over units); top units:")
    print(df.head(15).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
