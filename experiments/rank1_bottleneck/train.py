"""Online training of the rank-1 bottleneck net on the product-of-d-bits target.

Usage: python -m experiments.rank1_bottleneck.train --config <yaml> [--resume-from ckpt]

Writes to cfg.output_dir: config.yaml, metrics.csv, tb_logs/, checkpoints/step_*.pt + final.pt,
test_data.pt (the enumerated / sampled test set). TensorBoard also logs, every validate_every
steps, the exact bottleneck Walsh coefficients (w_enc . v_m) for each mode, their power split
into "linear" (singletons) and "product", the cosines of w_enc with v_prod and with sum_i v_i,
and the weight norms.
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from parity_net.train import resolve_device

from .config import Rank1Config, load_config, save_config
from .model import Rank1BottleneckNet


def make_test_set(cfg: Rank1Config, device: torch.device) -> torch.Tensor:
    if cfg.d <= 16:
        keys = torch.arange(2 ** cfg.d, device=device)
        bits = torch.arange(cfg.d, device=device)
        return (((keys.unsqueeze(1) >> bits) & 1).float() * 2 - 1)      # all 2^d inputs
    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed + 1)
    return torch.randint(0, 2, (cfg.test_samples, cfg.d), device=device, generator=gen).float() * 2 - 1


@torch.no_grad()
def evaluate(model: Rank1BottleneckNet, x: torch.Tensor, batch: int = 8192) -> dict[str, float]:
    model.eval()
    preds = torch.cat([model(x[s:s + batch]) for s in range(0, x.shape[0], batch)])
    y = x.prod(dim=1)
    model.train()
    return dict(test_mse=float(F.mse_loss(preds, y)), test_acc=float((torch.sign(preds) == y).float().mean()))


@torch.no_grad()
def bottleneck_stats(model: Rank1BottleneckNet) -> dict[str, float]:
    cfg = model.cfg
    names = cfg.mode_names(); supports = model.supports
    C = model.code_coefficients()                       # (K, M)
    p = (C ** 2).sum(0)                                  # power per mode through the bottleneck
    total = float(p.sum()) + 1e-30
    out = {f"bottleneck/power_{n}": float(p[i]) for i, n in enumerate(names)}
    degrees = sorted({len(m) for m in supports})
    for deg in degrees:
        idx = [i for i, m in enumerate(supports) if len(m) == deg]
        out[f"bottleneck/power_degree{deg}"] = float(p[idx].sum())
        out[f"bottleneck/fraction_degree{deg}"] = float(p[idx].sum()) / total
    # legacy names used by the K = 1 analyses
    singles = [i for i, m in enumerate(supports) if len(m) == 1]
    out["bottleneck/power_linear"] = float(p[singles].sum())
    pi = model.product_idx
    out["bottleneck/coef_product"] = float(C[0, pi]) if pi is not None else 0.0
    out["bottleneck/power_product"] = float(p[pi]) if pi is not None else 0.0
    out["bottleneck/product_fraction"] = out["bottleneck/power_product"] / total
    for i in singles:
        out[f"bottleneck/coef_{names[i]}"] = float(C[0, i])
    # the same decomposition one step later: what reaches the nonlinearity (summed over units)
    pp = model.preact_mode_power(); ptot = float(pp.sum()) + 1e-30
    for i, n in enumerate(names):
        out[f"preact/power_{n}"] = float(pp[i])
    for deg in degrees:
        idx = [i for i, m in enumerate(supports) if len(m) == deg]
        out[f"preact/power_degree{deg}"] = float(pp[idx].sum())
        out[f"preact/fraction_degree{deg}"] = float(pp[idx].sum()) / ptot
    # per-unit dominance: fraction of units whose largest pre-activation coefficient is a degree-2 mode
    Cp = model.preact_coefficients() ** 2
    deg_of = torch.tensor([len(m) for m in supports], device=Cp.device)
    out["preact/units_dominated_by_degree2"] = float((deg_of[Cp.argmax(1)] == 2).float().mean())
    out["bottleneck/cos_enc_product"] = model.enc_cosine_to("product")
    out["bottleneck/cos_enc_linear_sum"] = model.enc_cosine_to("linear_sum")
    out["norms/W_enc"] = float(model.W_enc.norm()); out["norms/W_dec"] = float(model.W_dec.norm())
    out["norms/w_enc"] = float(model.w_enc.norm()); out["norms/w_dec"] = float(model.w_dec.norm())
    out["norms/embedding_product_row"] = float(model.embedding[pi].norm()) if pi is not None else 0.0
    out["norms/embedding_singleton_rows_mean"] = float(model.embedding[singles].norm(dim=1).mean())
    out["norms/linear"] = float(model.linear.weight.norm()); out["norms/readout"] = float(model.readout.weight.norm())
    return out


def train(cfg: Rank1Config, resume_from: str | Path | None = None) -> Path:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    model = Rank1BottleneckNet(cfg).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    x_test = make_test_set(cfg, device)
    torch.save(dict(x=x_test.cpu()), out_dir / "test_data.pt")
    start = 0
    history: list[dict] = []
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=device)
        model.load_state_dict(payload["model_state"]); opt.load_state_dict(payload["optimizer_state"])
        start = int(payload["step"])
        if (out_dir / "metrics.csv").exists():
            history = pd.read_csv(out_dir / "metrics.csv").to_dict("records")
    writer = SummaryWriter(str(out_dir / "tb_logs"))
    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed + 7 + start)

    def checkpoint(step: int, name: str | None = None) -> None:
        torch.save(dict(step=step, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                        config=cfg.__dict__), out_dir / "checkpoints" / (name or f"step_{step:08d}.pt"))

    def log_row(step: int, train_mse: float) -> None:
        row = dict(step=step, train_mse=train_mse, **evaluate(model, x_test), **bottleneck_stats(model))
        history.append(row)
        for k, v in row.items():
            if k != "step":
                writer.add_scalar(k if "/" in k else f"loss/{k}", v, step)
        pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)

    t0 = time.perf_counter()
    for step in tqdm(range(start + 1, cfg.num_steps + 1), initial=start, total=cfg.num_steps, desc="rank1", mininterval=5):
        x = torch.randint(0, 2, (cfg.batch_size, cfg.d), device=device, generator=gen).float() * 2 - 1
        loss = F.mse_loss(model(x), x.prod(dim=1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % cfg.validate_every == 0 or step == cfg.num_steps:
            log_row(step, float(loss))
        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            checkpoint(step)
    checkpoint(cfg.num_steps, "final.pt")
    writer.close()
    print(f"done: {cfg.num_steps} steps in {time.perf_counter() - t0:.0f}s; final test mse {history[-1]['test_mse']:.4g}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()
    train(load_config(args.config), resume_from=args.resume_from)


if __name__ == "__main__":
    main()
