"""Static version of the Streamlit Overview heatmap ("highest-degree mode produced").

For each (step, block) panel: neuron-by-mode power of the post-nonlinearity
activations on the tree modes (degrees 2,4,8,16), neurons grouped by the
highest-degree mode they produce above max(floor, mult * var_i / n), groups in
hierarchy order, sorted within a group by that mode's power; color = log10 power.

Usage: python scripts/plot_overview_static.py --run-dir runs/<name> --steps 20000 40000 --blocks 0 1 [--out file.png]
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

from analysis.neuron_walsh import metrics as M


def panel(ax, art, block, floor, mult, log_floor):
    modes = [m for m in art["modes"] if m["degree"] > 1]
    idx = np.array([m["index"] for m in modes]); names = [m["name"] for m in modes]
    loc = next(l for l in art["locations"] if l["block"] == block)
    C = loc["coefficients"].numpy().astype(np.float64)[idx]; var = loc["activation_variance"].numpy().astype(np.float64)
    P = C ** 2; N = P.shape[1]
    thr = np.maximum(floor, mult * var / max(1, art["num_samples"]))
    g = M.highest_degree_mode(C, np.array([m["degree"] for m in modes]), thr)
    order = np.lexsort((-P[g, np.arange(N)], g))
    Z = np.log10(np.maximum(P[:, order], log_floor))
    im = ax.imshow(Z, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("neuron (sorted)")
    counts = np.bincount(g, minlength=len(names))
    ax.set_title(f"block_{block}/nonlinearity_0 @ step {art['step']}  "
                 f"(d4 groups: {', '.join(f'{names[i]}={counts[i]}' for i in range(len(names)) if names[i].startswith('d4'))})",
                 fontsize=8)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--floor", type=float, default=1e-6)
    ap.add_argument("--mult", type=float, default=10.0)
    ap.add_argument("--log-floor", type=float, default=1e-8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    fig, axes = plt.subplots(len(args.blocks), len(args.steps),
                             figsize=(7 * len(args.steps), 3.2 * len(args.blocks)), squeeze=False)
    for j, step in enumerate(args.steps):
        art = torch.load(run_dir / "analysis" / "neuron_walsh" / f"step_{step:08d}.pt",
                         map_location="cpu", weights_only=False)
        for i, block in enumerate(args.blocks):
            im = panel(axes[i][j], art, block, args.floor, args.mult, args.log_floor)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, label="log10 power")
    fig.suptitle(f"{run_dir.name}: highest-degree mode produced (floor {args.floor:g}, mult {args.mult:g})", fontsize=10)
    out = Path(args.out) if args.out else run_dir / "analysis" / "overview_static.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(out)


if __name__ == "__main__":
    main()
