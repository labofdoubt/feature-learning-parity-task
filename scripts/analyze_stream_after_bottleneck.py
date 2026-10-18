"""Overview-style heatmap of the residual stream right after a bottleneck.

Decomposes every coordinate of h_in = bottleneck_b(h_{b+1}^+) (the stream the
next block receives) into the tracked modes of the given degrees on the uniform
eval set, C[S, i] = E[h_in_i chi_S], and renders it like the Overview heatmap:
coordinates grouped by the highest-degree mode they carry above the floor,
sorted by that mode's power, color = log10 power.

Usage: python scripts/analyze_stream_after_bottleneck.py --run-dir runs/<name> --steps 20000 40000 [--bottleneck 0] [--degrees 2 4]
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
import torch

from analysis.neuron_walsh.compute import walsh_matrix
from analysis.neuron_walsh.modes import build_mode_registry
from parity_net.checkpoint import load_checkpoint
from parity_net.data import load_dataset
from parity_net.train import resolve_device
sys.path.insert(0, str(_ROOT / "scripts"))
from plot_overview_static import panel  # noqa: E402


@torch.no_grad()
def stream_coefficients(model, x, modes, bottleneck_idx, batch_size):
    N = model.config.N
    C = torch.zeros(len(modes), N, dtype=torch.float64, device=x.device)
    S1 = torch.zeros(N, dtype=torch.float64, device=x.device); S2 = torch.zeros_like(S1)
    for s in range(0, x.shape[0], batch_size):
        xb = x[s:s + batch_size]
        h = model.embedding(xb)
        for l, block in enumerate(model.blocks):
            h = block(h)                               # bottleneck applied inside
            if l == bottleneck_idx:
                break
        Xw = walsh_matrix(xb, modes)
        C += Xw.t() @ h.double(); S1 += h.double().sum(0); S2 += (h.double() ** 2).sum(0)
    n = x.shape[0]
    mean = S1 / n
    return C / n, (S2 / n - mean ** 2).clamp_min(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--bottleneck", type=int, default=0)
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--floor", type=float, default=1e-8)
    ap.add_argument("--mult", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()
    run_dir = Path(args.run_dir); device = resolve_device("auto")
    fig, axes = plt.subplots(1, len(args.steps), figsize=(7 * len(args.steps), 4.2), squeeze=False)
    for j, step in enumerate(args.steps):
        model, _, _ = load_checkpoint(run_dir / "checkpoints" / f"step_{step:08d}.pt", device); model.eval()
        dtype = next(model.parameters()).dtype
        x = load_dataset(run_dir / "eval_uniform.pt", device, dtype).x
        reg = build_mode_registry(model.config.relevant_dim, include_singletons=True)
        modes = [m for m in reg if m["degree"] in args.degrees]
        C, var = stream_coefficients(model, x, modes, args.bottleneck, args.batch_size)
        # re-index the registry so the panel sees only these modes (all degree > 1 kept)
        local = [dict(m, index=i) for i, m in enumerate(modes)]
        art = dict(step=step, num_samples=x.shape[0], modes=local,
                   locations=[dict(block=args.bottleneck, coefficients=C.float().cpu(),
                                   activation_variance=var.float().cpu())])
        panel(axes[0][j], art, args.bottleneck, args.floor, args.mult, args.floor)
        axes[0][j].set_title(f"stream after bottleneck {args.bottleneck} @ step {step}", fontsize=9)
        axes[0][j].set_xlabel("residual coordinate (sorted)")
    fig.suptitle(f"{run_dir.name}\nresidual stream after bottleneck {args.bottleneck}, modes of degree {args.degrees}, "
                 f"grouped by highest-degree mode carried (floor {args.floor:g}, mult {args.mult:g}); color = log10 power",
                 fontsize=9, y=1.08)
    out = run_dir / "analysis" / f"stream_after_bottleneck{args.bottleneck}_d{'_'.join(map(str, args.degrees))}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print(out)


if __name__ == "__main__":
    main()
