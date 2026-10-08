"""Offline neuron-level Walsh coefficient computation.

For every residual block, captures the activation IMMEDIATELY AFTER the
nonlinearity and BEFORE residual addition (and before any optional
post_activation_linear): a^{(b)}(x) = phi(V_b h_b(x)), named
`block_{b}/nonlinearity_0` (this architecture has one nonlinearity per block).

Coefficients are computed on the fixed UNIFORM evaluation dataset
(eval_uniform.pt, falling back to test_data.pt — both uniform), never on the
training distribution:

    C[S, i] = E_uniform[a_i(x) * chi_S(x)]        (batched  X_walsh^T H / n)

Per checkpoint one artifact is saved to
    <run_dir>/analysis/neuron_walsh/step_<step>.pt
holding, for every location: coefficients [M, N], normalized correlations
[M, N] (a_iS / (sigma_i + eps) — the Pearson correlation since Var(chi)=1),
activation mean/variance [N], the independent layer-level Walsh power [M]
(vector accumulation E[a * chi_S], for the sum_i a_iS^2 == H(S) sanity check),
plus the mode registry and step metadata. Full activations are never stored.

Usage:
  python -m analysis.neuron_walsh.compute --run <run_name> \\
    [--runs-root /root/parity/runs] [--checkpoints final|all|10000,20000,...] \\
    [--num-samples 65536] [--batch-size 2048]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from parity_net.data import load_dataset
from parity_net.train import resolve_device

from analysis.backward_walsh.checkpoint_utils import list_checkpoints, load_model_at_checkpoint
from .modes import build_mode_registry

EPS = 1e-12


def walsh_matrix(x: torch.Tensor, modes: list[dict]) -> torch.Tensor:
    """X_walsh [n, M] in {-1, +1} (float64) for the given batch."""
    cols = []
    for m in modes:
        idx = torch.tensor(m["support"], device=x.device, dtype=torch.long)
        cols.append(x[:, idx].prod(dim=1))
    return torch.stack(cols, dim=1).double()


@torch.no_grad()
def compute_locations(model, x: torch.Tensor, modes: list[dict], batch_size: int):
    """Batched accumulation over the uniform set. Returns one dict per block:
    {name, block, nonlinearity, coeffs f64[M,N], mean f64[N], var f64[N],
     layer_coeff_vec f64[M,N] — kept only transiently for layer_power}."""
    model.eval()
    L = len(model.blocks)
    N = model.config.N
    M = len(modes)
    n_total = x.shape[0]
    dev = x.device

    C = [torch.zeros(M, N, dtype=torch.float64, device=dev) for _ in range(L)]
    S1 = [torch.zeros(N, dtype=torch.float64, device=dev) for _ in range(L)]
    S2 = [torch.zeros(N, dtype=torch.float64, device=dev) for _ in range(L)]

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        xb = x[start:stop]
        Xw = walsh_matrix(xb, modes)              # (b, M), cached per batch
        h = model.embedding(xb)
        for l, block in enumerate(model.blocks):
            a = block.activation(block.linear(h)).double()   # post-phi, pre-residual
            C[l] += Xw.t() @ a                    # (M, N) accumulation
            S1[l] += a.sum(0)
            S2[l] += (a * a).sum(0)
            h = block(h)                          # advance the residual stream

    out = []
    for l in range(L):
        coeffs = C[l] / n_total
        mean = S1[l] / n_total
        var = (S2[l] / n_total - mean * mean).clamp_min(0.0)
        out.append(dict(
            name=f"block_{l}/nonlinearity_0", block=l, nonlinearity=0,
            coeffs=coeffs, mean=mean, var=var,
        ))
    return out


def sanity_checks(model, x, modes, locations, batch_size) -> dict:
    """(1) batched C vs naive per-neuron sums on a subset; (2) sum_i a_iS^2 vs
    an independent vector-level accumulation (different code path in
    analysis.backward_walsh.run_4xd4.pass_update_walsh, which for models
    without post_activation_linear captures the same tensor); (3) analytic
    normalized correlation vs direct Pearson. Returns max relative errors."""
    report = {}

    # (1) naive vs batched on a 4096-sample subset, 3 neurons x 2 modes
    xs = x[:4096]
    locs_small = compute_locations(model, xs, modes, batch_size)
    l, mode_ids, neuron_ids = 0, [0, len(modes) - 1], [0, 1, model.config.N - 1]
    with torch.no_grad():
        h = model.embedding(xs)
        a = model.blocks[0].activation(model.blocks[0].linear(h)).double()
    worst = 0.0
    for mi in mode_ids:
        idx = torch.tensor(modes[mi]["support"], device=x.device, dtype=torch.long)
        chi = xs[:, idx].prod(dim=1).double()
        for ni in neuron_ids:
            direct = float((a[:, ni] * chi).mean())
            batched = float(locs_small[l]["coeffs"][mi, ni])
            worst = max(worst, abs(direct - batched) / (abs(direct) + EPS))
    report["naive_vs_batched_max_rel_err"] = worst

    # (3) Pearson on the same subset for one (mode, neuron)
    mi, ni = mode_ids[-1], neuron_ids[-1]
    idx = torch.tensor(modes[mi]["support"], device=x.device, dtype=torch.long)
    chi = xs[:, idx].prod(dim=1).double()
    an = a[:, ni]
    pearson = float(((an - an.mean()) * (chi - chi.mean())).mean()
                    / (an.std(unbiased=False) * chi.std(unbiased=False) + EPS))
    analytic = float(locs_small[l]["coeffs"][mi, ni]
                     / (locs_small[l]["var"][ni].sqrt() + EPS))
    report["pearson_vs_analytic_abs_err"] = abs(pearson - analytic)

    # (2) layer-level recovery via the independent existing implementation
    try:
        from analysis.backward_walsh.modes import ModeInfo
        from analysis.backward_walsh.run_4xd4 import pass_update_walsh
        mode_infos = [ModeInfo(m["support"], m["name"], "tree") for m in modes]
        vec = pass_update_walsh(model, x, mode_infos, batch_size)  # [L][M] (N,)
        worst = 0.0
        for loc in locations:
            l = loc["block"]
            for mi in range(len(modes)):
                h_layer = float(vec[l][mi].square().sum())
                h_sum = float(loc["coeffs"][mi].square().sum())
                worst = max(worst, abs(h_layer - h_sum) / (abs(h_layer) + EPS))
        report["layer_power_max_rel_err"] = worst
    except Exception as e:  # post_activation_linear models differ by design
        report["layer_power_max_rel_err"] = f"skipped: {e}"
    return report


def run_checkpoint(model, x, modes, batch_size, step, run, out_dir,
                   with_sanity=False):
    locations = compute_locations(model, x, modes, batch_size)
    if with_sanity:
        report = sanity_checks(model, x, modes, locations, batch_size)
        print(f"  sanity: {report}")
    else:
        report = {}

    payload_locs = []
    for loc in locations:
        coeffs = loc["coeffs"]
        var = loc["var"]
        corr = coeffs / (var.sqrt() + EPS).unsqueeze(0)
        payload_locs.append(dict(
            name=loc["name"], block=loc["block"], nonlinearity=loc["nonlinearity"],
            coefficients=coeffs.float().cpu(),
            correlations=corr.float().cpu(),
            activation_mean=loc["mean"].float().cpu(),
            activation_variance=loc["var"].float().cpu(),
            layer_power=coeffs.square().sum(dim=1).cpu(),  # f64 [M]
        ))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:08d}.pt"
    torch.save(dict(
        run=run, step=step, num_samples=x.shape[0],
        modes=modes, locations=payload_locs, sanity=report,
        note="post-nonlinearity pre-residual activations; uniform data",
    ), path)
    print(f"  saved {path.name}  "
          f"({len(payload_locs)} locations, C shape "
          f"[{len(modes)}, {payload_locs[0]['coefficients'].shape[1]}])")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-root", default="/root/parity/runs")
    parser.add_argument("--checkpoints", default="all",
                        help="'all', 'final', or comma-separated steps")
    parser.add_argument("--num-samples", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.runs_root) / args.run
    out_dir = run_dir / "analysis" / "neuron_walsh"

    data_path = run_dir / "eval_uniform.pt"
    if not data_path.exists():
        data_path = run_dir / "test_data.pt"
    print(f"Data: {data_path.name} (uniform)")
    ds = load_dataset(data_path, device, torch.float32)
    x = ds.x
    if args.num_samples < x.shape[0]:
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(42))
        x = x[perm[: args.num_samples]]
    print(f"Samples: {x.shape[0]}")

    ckpts = list_checkpoints(run_dir / "checkpoints")
    final = run_dir / "checkpoints" / "final.pt"
    if args.checkpoints == "final":
        selected = []
    elif args.checkpoints == "all":
        selected = ckpts
    else:
        wanted = {int(s) for s in args.checkpoints.split(",")}
        selected = [(s, p) for s, p in ckpts if s in wanted]
    print(f"Checkpoints: {len(selected)}"
          + (" + final" if args.checkpoints in ("final", "all") and final.exists() else ""))

    modes = None
    first = True
    for k, (step, ckpt_path) in enumerate(selected):
        print(f"[{k + 1}/{len(selected)}] step={step}")
        model, payload, dtype = load_model_at_checkpoint(ckpt_path, device)
        if modes is None:
            modes = build_mode_registry(model.config.relevant_dim, include_singletons=True)
            print(f"Tracked modes: {len(modes)} tree modes "
                  f"(degrees {sorted(set(m['degree'] for m in modes))})")
        run_checkpoint(model, x.to(dtype=dtype), modes, args.batch_size,
                       step, args.run, out_dir, with_sanity=first)
        first = False
        del model

    if args.checkpoints in ("final", "all") and final.exists():
        model, payload, dtype = load_model_at_checkpoint(final, device)
        if modes is None:
            modes = build_mode_registry(model.config.relevant_dim, include_singletons=True)
        step = int(payload.get("step", 0))
        print(f"[final] step={step}")
        run_checkpoint(model, x.to(dtype=dtype), modes, args.batch_size,
                       step, args.run, out_dir, with_sanity=first)
        del model

    print(f"Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
