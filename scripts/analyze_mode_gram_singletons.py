"""Gram matrix of the tree target modes plus the degree-1 singletons in the
residual stream after each residual addition (one heatmap per block).

v_S^{(b)} = E_x[h_b(x) chi_S(x)] on the uniform eval set; plots the cosine
Gram  v_S.v_T / (|v_S||v_T|)  per block (raw Gram saved as CSV alongside).

Usage:
  python scripts/analyze_mode_gram_singletons.py --run-dir runs/<name> [--checkpoint final]
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
import pandas as pd
import torch

from analysis.neuron_walsh.compute import walsh_matrix
from analysis.neuron_walsh.modes import build_mode_registry
from parity_net.checkpoint import load_checkpoint
from parity_net.data import load_dataset
from parity_net.train import resolve_device, resolve_dtype


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="final")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--cosine", action="store_true", help="plot the cosine-normalized Gram instead of raw dot products")
    ap.add_argument("--log", action="store_true",
                    help="symmetric log color scale (sign kept; linear below 1e-4 of the block max)")
    ap.add_argument("--decades", action="store_true",
                    help="color = floor(log10 |G|), one color per order of magnitude; "
                         "negative entries are marked with a minus sign")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device("auto")
    ckpt = run_dir / "checkpoints" / ("final.pt" if args.checkpoint == "final" else f"step_{int(args.checkpoint):08d}.pt")
    model, payload, _ = load_checkpoint(ckpt, device)
    dtype = resolve_dtype(payload["config"]["training"]["dtype"])
    model = model.to(dtype=dtype).eval()
    step = int(payload.get("step", 0))
    data_path = run_dir / "eval_uniform.pt"
    if not data_path.exists():
        data_path = run_dir / "test_data.pt"
    x = load_dataset(data_path, device, dtype).x
    modes = build_mode_registry(model.config.relevant_dim, include_singletons=True)
    names = [m["name"] for m in modes]
    L = len(model.blocks)
    V = [torch.zeros(len(modes), model.config.N, dtype=torch.float64, device=device) for _ in range(L)]
    for s in range(0, x.shape[0], args.batch_size):
        xb = x[s:s + args.batch_size]
        Xw = walsh_matrix(xb, modes)
        _, acts = model(xb, return_activations=True)
        for b in range(L):
            V[b] += Xw.t() @ acts[b + 1].double()
    out = run_dir / "analysis"; out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 15))
    for b, ax in enumerate(axes.flat[:L]):
        v = V[b] / x.shape[0]
        gram = (v @ v.t()).cpu().numpy()
        pd.DataFrame(gram, index=names, columns=names).to_csv(out / f"mode_gram_singletons_block{b}.csv")
        if args.cosine:
            norms = np.sqrt(np.diag(gram))
            shown, kind = gram / (np.outer(norms, norms) + 1e-30), "cosine Gram"
            im = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
        else:
            lim = float(np.abs(gram).max())
            shown, kind = gram, "raw Gram (dot products)"
            if args.decades:
                from matplotlib.colors import BoundaryNorm
                kind = "log10|G| by decade (− marks negative)"
                mag = np.log10(np.abs(gram) + 1e-300)
                top = int(np.ceil(mag.max()))
                lo = max(top - 8, int(np.floor(mag[np.isfinite(mag)].min())))
                bounds = np.arange(lo, top + 1)
                cmap = plt.get_cmap("viridis", len(bounds) - 1)
                im = ax.imshow(np.clip(mag, lo, top), cmap=cmap,
                               norm=BoundaryNorm(bounds, cmap.N))
                for i in range(gram.shape[0]):
                    for j in range(gram.shape[1]):
                        if gram[i, j] < 0:
                            ax.text(j, i, "−", ha="center", va="center", fontsize=6, color="white")
            elif args.log:
                from matplotlib.colors import SymLogNorm
                kind += ", symlog"
                im = ax.imshow(shown, cmap="RdBu_r",
                               norm=SymLogNorm(linthresh=lim * 1e-4, vmin=-lim, vmax=lim))
            else:
                im = ax.imshow(shown, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"after residual addition {b + 1} (block {b})  {kind}")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        if args.decades:
            cb.set_ticks(bounds); cb.set_ticklabels([f"1e{int(b)}" for b in bounds])
    fig.suptitle(f"{run_dir.name} @ step {step}: Gram of mode directions v_S = E[h χ_S], tree modes + singletons")
    fig.tight_layout()
    fname = ("mode_gram_singletons_cosine.png" if args.cosine
             else "mode_gram_singletons_decades.png" if args.decades
             else "mode_gram_singletons_log.png" if args.log else "mode_gram_singletons.png")
    fig.savefig(out / fname, dpi=130)
    print(out / fname)


if __name__ == "__main__":
    main()
