"""Overlap between the d4 readout rows and the d2 content of a block's nonlinearity output.

For block b (default 1, where the d4 layerwise readout sits), takes the
post-nonlinearity, pre-residual activation vector a(x) = phi(V_b h_b(x)) and its
Walsh coefficients on the eight d2 target modes, v_p = E[a(x) chi_{d2_p}(x)] (a
vector over hidden units; read from the neuron-Walsh artifact of that step), and
plots the 4 x 8 matrix  w_k . v_p  of dot products with the d4 readout rows w_k
taken from the checkpoint, on a log10 |.| color scale with the sign marked.

Usage: python scripts/analyze_readout_d2_overlap.py --run-dir runs/<name> --step 10000 [--block 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parity_net.checkpoint import load_checkpoint


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--block", type=int, default=1)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    model, _, _ = load_checkpoint(run_dir / "checkpoints" / f"step_{args.step:08d}.pt", torch.device("cpu"))
    art = torch.load(run_dir / "analysis" / "neuron_walsh" / f"step_{args.step:08d}.pt",
                     map_location="cpu", weights_only=False)
    names = [m["name"] for m in art["modes"]]
    loc = next(l for l in art["locations"] if l["block"] == args.block)
    C = loc["coefficients"].double()                                   # [M, N] post-nonlinearity coefficients
    d2 = [f"d2_{p}" for p in range(8)]
    V = torch.stack([C[names.index(n)] for n in d2], dim=1)            # N x 8
    if model.config.use_layerwise_readouts:
        key = str(int(np.log2(4)) - 1)                                   # the d4 readout block
        W = model.layerwise_readouts[key].weight.double()               # 4 x N
        rows = [f"d4_{k}" for k in range(W.shape[0])]
    else:
        W = model.readout.weight.double()
        rows = model.target_names
    D = (W @ V).numpy()                                                 # rows x 8
    out = run_dir / "analysis"; out.mkdir(exist_ok=True)
    np.savetxt(out / f"readout_d2_overlap_block{args.block}_step{args.step}.csv", D, delimiter=",",
               header=",".join(d2), comments="")
    mag = np.log10(np.abs(D) + 1e-300)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(mag, cmap="viridis", aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:+.2e}", ha="center", va="center", fontsize=8,
                    color="white" if mag[i, j] < mag.max() - 0.6 else "black")
    ax.set_xticks(range(8)); ax.set_xticklabels(d2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xlabel(f"d2 mode of block-{args.block} nonlinearity output  v_p = E[phi(V h) chi_p]")
    ax.set_ylabel("d4 readout row w_k")
    ax.set_title(f"{run_dir.name} @ step {args.step}: w_k · v_p  (color = log10 |dot|, text = signed value)")
    fig.colorbar(im, ax=ax, label="log10 |w_k · v_p|")
    fig.tight_layout()
    path = out / f"readout_d2_overlap_block{args.block}_step{args.step}.png"
    fig.savefig(path, dpi=150)
    print(path)
    print("rows:", rows); print(np.array2string(D, formatter={"float_kind": lambda v: f"{v:+.2e}"}))


if __name__ == "__main__":
    main()
