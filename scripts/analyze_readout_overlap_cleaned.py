"""Overlap of the d4 readout rows with the d4 modes that block 1's nonlinearity
PRODUCES ITSELF, after removing all d4 content from its input.

On the uniform eval set: h_in = bottleneck_0(h_1) is block 1's input; its d4
components v_k = E[h_in chi_{d4_k}] are subtracted exactly,
    h_in' = h_in - sum_k v_k chi_{d4_k}(x),
then a = phi(V_1 h_in') is decomposed, u_k = E[a chi_{d4_k}], and the matrix
w_k . u_l (readout rows x produced modes) is plotted on a log10|.| color scale
with signed values. The uncleaned matrix is printed for comparison.

Usage: python scripts/analyze_readout_overlap_cleaned.py --run-dir runs/<name> --step 10000
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

from analysis.neuron_walsh.compute import walsh_matrix
from analysis.neuron_walsh.modes import build_mode_registry
from parity_net.checkpoint import load_checkpoint
from parity_net.data import load_dataset
from parity_net.train import resolve_device


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device("auto")
    model, _, _ = load_checkpoint(run_dir / "checkpoints" / f"step_{args.step:08d}.pt", device)
    model.eval()
    dtype = next(model.parameters()).dtype
    x = load_dataset(run_dir / "eval_uniform.pt", device, dtype).x
    n = x.shape[0]
    modes = [m for m in build_mode_registry(model.config.relevant_dim) if m["degree"] == 4]
    names = [m["name"] for m in modes]
    chi = walsh_matrix(x, modes).to(dtype)                               # n x 4
    b0, b1 = model.blocks[0], model.blocks[1]

    def block1_input(xb):
        return b0.bottleneck(b0(model.embedding(xb), apply_bottleneck=False))

    # pass 1: d4 content of block 1's input
    V = torch.zeros(len(modes), model.config.N, dtype=torch.float64, device=device)
    for s in range(0, n, args.batch_size):
        V += chi[s:s + args.batch_size].double().t() @ block1_input(x[s:s + args.batch_size]).double()
    V /= n
    # pass 2: nonlinearity output with and without that content
    U_clean = torch.zeros_like(V); U_raw = torch.zeros_like(V)
    for s in range(0, n, args.batch_size):
        xb, cb = x[s:s + args.batch_size], chi[s:s + args.batch_size]
        h_in = block1_input(xb)
        h_clean = h_in - cb @ V.to(dtype)
        U_raw += cb.double().t() @ b1.activation(b1.linear(h_in)).double()
        U_clean += cb.double().t() @ b1.activation(b1.linear(h_clean)).double()
    U_raw /= n; U_clean /= n
    W = model.layerwise_readouts["1"].weight.double()                     # 4 x N (d4 readout rows)
    D_clean = (W @ U_clean.t()).cpu().numpy(); D_raw = (W @ U_raw.t()).cpu().numpy()
    print("removed d4 content of block-1 input, ||v_k||:", [f"{float(V[k].norm()):.2e}" for k in range(4)])
    print("d4 power of nonlinearity output, raw   :", [f"{float((U_raw[k]**2).sum()):.2e}" for k in range(4)])
    print("d4 power of nonlinearity output, clean :", [f"{float((U_clean[k]**2).sum()):.2e}" for k in range(4)])
    print("w_k . u_l (clean), rows = readout d4_k, cols = produced d4_l:")
    print(np.array2string(D_clean, formatter={"float_kind": lambda v: f"{v:+.2e}"}))
    print("w_k . u_l (raw):"); print(np.array2string(D_raw, formatter={"float_kind": lambda v: f"{v:+.2e}"}))
    out = run_dir / "analysis"; out.mkdir(exist_ok=True)
    tag = f"readout_d4_overlap_cleaned_block1_step{args.step}"
    np.savetxt(out / f"{tag}.csv", D_clean, delimiter=",", header=",".join(names), comments="")
    mag = np.log10(np.abs(D_clean) + 1e-300)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mag, cmap="viridis")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{D_clean[i, j]:+.2e}", ha="center", va="center", fontsize=9,
                    color="white" if mag[i, j] < mag.max() - 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_yticks(range(4)); ax.set_yticklabels(names)
    ax.set_xlabel("d4 mode produced by block-1 nonlinearity (input d4 content removed)")
    ax.set_ylabel("d4 readout row w_k")
    ax.set_title(f"{run_dir.name} @ step {args.step}: w_k · u_l\n(color = log10 |dot|, text = signed value)")
    fig.colorbar(im, ax=ax, label="log10 |w_k · u_l|")
    fig.tight_layout(); fig.savefig(out / f"{tag}.png", dpi=150); print(out / f"{tag}.png")


if __name__ == "__main__":
    main()
