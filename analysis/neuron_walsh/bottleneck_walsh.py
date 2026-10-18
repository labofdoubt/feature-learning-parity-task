"""Walsh analysis of the TopK bottlenecks: which modes each code neuron carries
and which modes pass through the bottleneck.

For every block b with a bottleneck, the bottleneck input is the residual
stream right after the residual addition, h_b^+(x). Three tensors are analysed
on the uniform eval set, each [n, d_bottleneck]:
    pre  = W_enc h_b^+ (+ bias)       every code neuron BEFORE TopK
    post = TopK(pre)                   the sparse code that passes through
    out  = W_dec post (+ bias)         the residual stream handed to the next
                                       block (one row per residual coordinate)
and for each the Walsh coefficients on the tracked modes (tree targets plus
the degree-1 singletons), C[S, j] = E[code_j(x) chi_S(x)], the per-neuron
mean/variance, and the layer-level Walsh power of the decoder output
W_dec post (what the next block actually receives). `select_freq[j]` is the
fraction of inputs on which code neuron j is among the K kept entries.

Artifacts: <run_dir>/analysis/neuron_walsh/bottleneck_walsh/step_<step>.pt

Usage:
  python -m analysis.neuron_walsh.bottleneck_walsh --run <run_name> \\
    [--runs-root /root/parity/runs] [--checkpoints final|all|10000,...] \\
    [--num-samples 65536] [--batch-size 2048]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from parity_net.data import load_dataset
from parity_net.train import resolve_device

from analysis.backward_walsh.checkpoint_utils import list_checkpoints, load_model_at_checkpoint
from .compute import walsh_matrix
from .modes import build_mode_registry

EPS = 1e-12


@torch.no_grad()
def compute_bottlenecks(model, x: torch.Tensor, modes: list[dict], batch_size: int) -> list[dict]:
    model.eval()
    M = len(modes)
    n_total = x.shape[0]
    dev = x.device
    bns = [(l, b.bottleneck) for l, b in enumerate(model.blocks) if getattr(b, "bottleneck", None) is not None]
    if not bns:
        raise ValueError("model has no bottlenecks")
    acc = {}
    for l, bn in bns:
        d = bn.dim
        acc[l] = dict(
            C_pre=torch.zeros(M, d, dtype=torch.float64, device=dev),
            C_post=torch.zeros(M, d, dtype=torch.float64, device=dev),
            S1_pre=torch.zeros(d, dtype=torch.float64, device=dev),
            S2_pre=torch.zeros(d, dtype=torch.float64, device=dev),
            S1_post=torch.zeros(d, dtype=torch.float64, device=dev),
            S2_post=torch.zeros(d, dtype=torch.float64, device=dev),
            sel=torch.zeros(d, dtype=torch.float64, device=dev),
            V_out=torch.zeros(M, model.config.N, dtype=torch.float64, device=dev),
            S1_out=torch.zeros(model.config.N, dtype=torch.float64, device=dev),
            S2_out=torch.zeros(model.config.N, dtype=torch.float64, device=dev),
        )
    for start in range(0, n_total, batch_size):
        xb = x[start:start + batch_size]
        Xw = walsh_matrix(xb, modes)                          # [b, M] f64
        h = model.embedding(xb)
        for l, block in enumerate(model.blocks):
            bn = getattr(block, "bottleneck", None)
            if bn is None:
                h = block(h)
                continue
            h_pre = block(h, apply_bottleneck=False)          # bottleneck input
            pre = bn.encoder(h_pre)
            post = bn.encode(h_pre)                           # sparse code
            out = bn.decoder(post)
            a = acc[l]
            pre64, post64 = pre.double(), post.double()
            a["C_pre"] += Xw.t() @ pre64
            a["C_post"] += Xw.t() @ post64
            a["S1_pre"] += pre64.sum(0); a["S2_pre"] += (pre64 * pre64).sum(0)
            a["S1_post"] += post64.sum(0); a["S2_post"] += (post64 * post64).sum(0)
            a["sel"] += (post != 0).double().sum(0)
            out64 = out.double()
            a["V_out"] += Xw.t() @ out64
            a["S1_out"] += out64.sum(0); a["S2_out"] += (out64 * out64).sum(0)
            h = out
    results = []
    for l, bn in bns:
        a = acc[l]
        def stats(S1, S2):
            mean = S1 / n_total
            var = (S2 / n_total - mean * mean).clamp_min(0.0)
            return mean, var
        mean_pre, var_pre = stats(a["S1_pre"], a["S2_pre"])
        mean_post, var_post = stats(a["S1_post"], a["S2_post"])
        mean_out, var_out = stats(a["S1_out"], a["S2_out"])
        results.append(dict(
            block=l, name=f"block_{l}/bottleneck", dim=bn.dim, k=bn.k, abs_topk=bn.abs_topk,
            C_pre=(a["C_pre"] / n_total).float().cpu(),
            C_post=(a["C_post"] / n_total).float().cpu(),
            mean_pre=mean_pre.float().cpu(), var_pre=var_pre.float().cpu(),
            mean_post=mean_post.float().cpu(), var_post=var_post.float().cpu(),
            select_freq=(a["sel"] / n_total).float().cpu(),
            # decoder output per residual coordinate [M, N], and its layer power [M]
            C_out=(a["V_out"] / n_total).float().cpu(),
            mean_out=mean_out.float().cpu(), var_out=var_out.float().cpu(),
            output_power=(a["V_out"] / n_total).square().sum(dim=1).cpu(),
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-root", default="/root/parity/runs")
    parser.add_argument("--checkpoints", default="all")
    parser.add_argument("--num-samples", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.runs_root) / args.run
    out_dir = run_dir / "analysis" / "neuron_walsh" / "bottleneck_walsh"
    data_path = run_dir / "eval_uniform.pt"
    if not data_path.exists():
        data_path = run_dir / "test_data.pt"
    ds = load_dataset(data_path, device, torch.float32)
    x = ds.x
    if args.num_samples < x.shape[0]:
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(42))
        x = x[perm[: args.num_samples]]
    print(f"Data: {data_path.name}, samples: {x.shape[0]}")

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

    todo = list(selected) + ([(None, final)] if use_final else [])
    modes = None
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (step, path) in enumerate(todo):
        model, payload, dtype = load_model_at_checkpoint(path, device)
        if step is None:
            step = int(payload.get("step", 0))
        if modes is None:
            modes = build_mode_registry(model.config.relevant_dim, include_singletons=True)
        print(f"[{i + 1}/{len(todo)}] step={step}")
        res = compute_bottlenecks(model, x.to(dtype=dtype), modes, args.batch_size)
        torch.save(dict(run=args.run, step=step, num_samples=x.shape[0], modes=modes,
                        bottlenecks=res,
                        note="bottleneck encoder outputs before (pre) and after (post) TopK; uniform data"),
                   out_dir / f"step_{step:08d}.pt")
        print(f"  saved step_{step:08d}.pt  ({len(res)} bottlenecks, dim {res[0]['dim']}, "
              f"K {[r['k'] for r in res]})")
        del model
    print(f"Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
