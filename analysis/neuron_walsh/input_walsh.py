"""Exact Walsh decomposition of every neuron's INPUT (pre-activation).

For block b the input to the nonlinearity is z^{(b)}(x) = V_b h_b(x), where h_b
is the residual stream entering the block (post-bottleneck when the model has
one). The uniform eval set enumerates every configuration of the
`relevant_dim` relevant bits, so instead of projecting on a hand-picked mode
set the FULL Walsh spectrum over the relevant bits is computed exactly with a
fast Walsh-Hadamard transform: for N neurons that is one (2^k x N) butterfly
pass per bit, i.e. O(k 2^k N) — for k=16, N=2048 well under a second on a GPU.
Dependence on the irrelevant bits is averaged out over `repeats` random draws
and reported separately as `irrelevant_power`.

Walsh index convention: coefficient index s is the bitmask of the support,
bit j <-> relevant input coordinate j (0-based), so s=0 is the constant term.

Only a sparse summary is stored per checkpoint (the full [2^k, N] spectrum is
never written):
    <run_dir>/analysis/neuron_walsh/input_walsh/step_<step>.pt
with, per block: top-K coefficients by magnitude (`top_idx`, `top_coef`),
total power E[z^2] (`power`), constant term (`mean`), a robust noise scale
(`noise_sigma` = 1.4826 * median |c_s| over s != 0), power by support size
(`degree_power` [N, k+1]), the coefficients on the tracked tree/singleton
modes (`tree_coef` [M, N], registry order) and `irrelevant_power`.

Usage:
  python -m analysis.neuron_walsh.input_walsh --run <run_name> \\
    [--runs-root /root/parity/runs] [--checkpoints final|all|10000,20000,...] \\
    [--repeats 2] [--topk 64]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from parity_net.train import resolve_device

from analysis.backward_walsh.checkpoint_utils import list_checkpoints, load_model_at_checkpoint
from .compute import walsh_matrix
from .modes import build_mode_registry

EPS = 1e-12
MAX_RELEVANT_DIM = 20


def relevant_configs(k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """All 2^k relevant-bit configurations, row i has bit j = (i >> j) & 1 -> {-1,+1}."""
    keys = torch.arange(2 ** k, device=device, dtype=torch.long)
    bits = torch.arange(k, device=device, dtype=torch.long)
    return ((keys.unsqueeze(1) >> bits) & 1).to(dtype).mul_(2).sub_(1)


def fwht_(z: torch.Tensor) -> torch.Tensor:
    """In-place fast Walsh-Hadamard transform along dim 0 (length 2^k), natural
    (Sylvester) order: out[s] = sum_i z[i] * (-1)^{popcount(i & s)}."""
    n, cols = z.shape
    h = 1
    while h < n:
        v = z.view(-1, 2, h, cols)
        a, b = v[:, 0], v[:, 1]
        s, d = a + b, a - b
        v[:, 0], v[:, 1] = s, d
        h *= 2
    return z


def _popcount(idx: torch.Tensor, k: int) -> torch.Tensor:
    bits = torch.arange(k, device=idx.device, dtype=torch.long)
    return ((idx.unsqueeze(1) >> bits) & 1).sum(dim=1)


@torch.no_grad()
def preactivations(model, x: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """z^{(b)}(x) for every block, each [n, N] (float32, on x.device)."""
    L = len(model.blocks)
    out = [[] for _ in range(L)]
    for start in range(0, x.shape[0], batch_size):
        h = model.embedding(x[start:start + batch_size])
        for l, block in enumerate(model.blocks):
            out[l].append(block.linear(h).float())
            h = block(h)  # advance the residual stream (bottleneck included)
    return [torch.cat(chunks, dim=0) for chunks in out]


@torch.no_grad()
def walsh_summary(z_rep: torch.Tensor, k: int, topk: int, modes: list[dict],
                  neuron_chunk: int | None = None) -> dict:
    """z_rep: [repeats, 2^k, N] pre-activations ordered by config index.
    Returns the sparse per-neuron summary described in the module docstring."""
    repeats, n_cfg, N = z_rep.shape
    device = z_rep.device
    z_mean = z_rep.mean(dim=0)                                       # [2^k, N]
    irrelevant_power = (z_rep.var(dim=0, unbiased=repeats > 1).mean(dim=0)
                        if repeats > 1 else torch.zeros(N, device=device))
    # chi_S(x) = prod_{j in S} x_j = (-1)^{|S|} (-1)^{popcount(i & s)} under the
    # bit convention of relevant_configs, so the Sylvester transform picks up a
    # sign (-1)^{|S|} per support size.
    all_idx = torch.arange(n_cfg, device=device)
    deg = _popcount(all_idx, k)                                      # [2^k]
    sign = (1.0 - 2.0 * (deg % 2)).to(z_mean.dtype)                  # (-1)^{|S|}
    tree_idx = torch.tensor([sum(1 << j for j in m["support"]) for m in modes],
                            device=device, dtype=torch.long)
    if neuron_chunk is None:
        # keep the working spectrum around 512 MB
        neuron_chunk = max(1, min(N, (512 * 2 ** 20) // (4 * n_cfg)))

    top_idx = torch.empty(N, topk, dtype=torch.int32)
    top_coef = torch.empty(N, topk, dtype=torch.float32)
    power = torch.empty(N); mean = torch.empty(N); noise = torch.empty(N)
    degree_power = torch.empty(N, k + 1)
    tree_coef = torch.empty(len(modes), N)
    for c0 in range(0, N, neuron_chunk):
        c1 = min(N, c0 + neuron_chunk)
        spec = z_mean[:, c0:c1].clone()
        fwht_(spec)
        spec.mul_(sign.unsqueeze(1)).div_(n_cfg)                     # c_S [2^k, chunk]
        sq = spec.square()
        power[c0:c1] = sq.sum(dim=0).cpu()
        mean[c0:c1] = spec[0].cpu()
        degree_power[c0:c1] = torch.zeros(k + 1, c1 - c0, device=device).index_add_(
            0, deg, sq).t().cpu()
        tree_coef[:, c0:c1] = spec[tree_idx].cpu()
        mag = spec.abs()
        noise[c0:c1] = (1.4826 * mag[1:].median(dim=0).values).cpu()
        v, i = mag.topk(topk, dim=0)                                  # [topk, chunk]
        top_idx[c0:c1] = i.t().to(torch.int32).cpu()
        top_coef[c0:c1] = torch.gather(spec, 0, i).t().cpu()
        del spec, sq, mag
    return dict(
        top_idx=top_idx, top_coef=top_coef, power=power, mean=mean,
        noise_sigma=noise, degree_power=degree_power, tree_coef=tree_coef,
        irrelevant_power=irrelevant_power.float().cpu(),
    )


@torch.no_grad()
def run_input_walsh_checkpoint(model, step: int, run: str, out_dir: Path, *,
                               repeats: int = 2, topk: int = 64,
                               batch_size: int = 4096, seed: int = 0,
                               modes: list[dict] | None = None,
                               with_sanity: bool = False) -> Path:
    cfg = model.config
    k, d = cfg.relevant_dim, cfg.input_dim
    if k > MAX_RELEVANT_DIM:
        raise ValueError(f"relevant_dim={k} > {MAX_RELEVANT_DIM}: the exact 2^k "
                         "spectrum is too large; use the tracked-mode compute instead")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if modes is None:
        modes = build_mode_registry(k, include_singletons=True)
    n_cfg = 2 ** k
    rel = relevant_configs(k, device, dtype)                          # [2^k, k]
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    z_by_block = None
    for r in range(repeats):
        irrel = (torch.randint(0, 2, (n_cfg, d - k), device=device, generator=gen)
                 .to(dtype).mul_(2).sub_(1)) if d > k else rel[:, :0]
        x = torch.cat([rel, irrel], dim=1)
        zs = preactivations(model, x, batch_size)                     # L x [2^k, N]
        if z_by_block is None:
            z_by_block = [[] for _ in zs]
        for l, z in enumerate(zs):
            z_by_block[l].append(z)
        if with_sanity and r == 0:
            # exact spectrum vs direct projection on the tracked modes
            Xw = walsh_matrix(x, modes)                               # [2^k, M] f64
            worst = 0.0
            for l, z in enumerate(zs):
                direct = (Xw.t() @ z.double()) / n_cfg                # [M, N]
                s = walsh_summary(z.unsqueeze(0), k, topk=1, modes=modes)
                err = (s["tree_coef"].double().to(device) - direct).abs().max()
                worst = max(worst, float(err / (direct.abs().max() + EPS)))
            print(f"  sanity: fwht vs direct tracked-mode projection, max rel err {worst:.2e}")
    blocks = []
    for l, chunks in enumerate(z_by_block):
        z_rep = torch.stack(chunks, dim=0)                            # [R, 2^k, N]
        summary = walsh_summary(z_rep, k, topk, modes)
        summary.update(block=l, name=f"block_{l}/input_0")
        blocks.append(summary)
        del z_rep
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:08d}.pt"
    torch.save(dict(
        run=run, step=step, relevant_dim=k, input_dim=d, repeats=repeats,
        topk=topk, modes=modes, blocks=blocks,
        note="exact Walsh spectrum of pre-activations V_b h_b over the relevant bits; "
             "irrelevant bits averaged over `repeats` draws",
    ), path)
    print(f"  saved {path.name}  ({len(blocks)} blocks, top-{topk} of {n_cfg} modes per neuron)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-root", default="/root/parity/runs")
    parser.add_argument("--checkpoints", default="all",
                        help="'final', 'all', or comma-separated steps")
    parser.add_argument("--repeats", type=int, default=2,
                        help="independent irrelevant-bit draws averaged per config")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.runs_root) / args.run
    out_dir = run_dir / "analysis" / "neuron_walsh" / "input_walsh"
    ckpts = list_checkpoints(run_dir / "checkpoints")
    final = run_dir / "checkpoints" / "final.pt"
    if args.checkpoints == "final":
        selected = []
    elif args.checkpoints == "all":
        selected = ckpts
    else:
        wanted = {int(s) for s in args.checkpoints.split(",")}
        selected = [(s, p) for s, p in ckpts if s in wanted]
    use_final = args.checkpoints in ("final", "all") and final.exists()
    print(f"Checkpoints: {len(selected)}" + (" + final" if use_final else ""))

    first = True
    for n, (step, ckpt_path) in enumerate(selected):
        print(f"[{n + 1}/{len(selected)}] step={step}")
        model, _, _ = load_model_at_checkpoint(ckpt_path, device)
        run_input_walsh_checkpoint(model, step, args.run, out_dir, repeats=args.repeats,
                                   topk=args.topk, batch_size=args.batch_size,
                                   with_sanity=first)
        first = False
        del model
    if use_final:
        model, payload, _ = load_model_at_checkpoint(final, device)
        step = int(payload.get("step", 0))
        print(f"[final] step={step}")
        run_input_walsh_checkpoint(model, step, args.run, out_dir, repeats=args.repeats,
                                   topk=args.topk, batch_size=args.batch_size,
                                   with_sanity=first)
    print(f"Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
