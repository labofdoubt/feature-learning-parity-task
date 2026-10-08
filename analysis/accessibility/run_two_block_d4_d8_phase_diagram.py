"""Train the fully unfrozen two-block model on the d4-parent / d8-target decay grid.

Task (reports/two_block_theory/two_block_d4_d8_layerwise_decay_phase_theory.tex): the exact
8-bit cube, two supervised degree-4 parents chi_L = x0x1x2x3 and chi_R = x4x5x6x7, and the
degree-8 target chi_8 = chi_L * chi_R; there are no degree-2 targets.  Network, objective and
protocols are those of run_two_block_phase_diagram.py: a bias-free two-block residual ReLU
network with explicit penalties

    lambda_E = lambda_2 = lambda_p = lambda_8 = lambda_0,   lambda_1 = ratio * lambda_0,

``joint`` training from the common random initialization, and ``parent_warm`` training that
first fits E, V1 and the parent readout to the parents read from h1.  Training is
deterministic SGD (no Langevin noise).  The fixed-representation gate of the theory is
R4 = Q4 / Q1^(1) > 5/32.

Width parametrizations (hidden V1, V2: Var 1/N and rate lr throughout; E: Var 1/d):
  --readout-init fan_in   readouts Var 1/N,    readout rate lr,      embedding rate lr
  --readout-init mup      readouts Var 1/N^2,  readout rate lr/N,    embedding rate lr
                          (the muP-like readout of the d2/d4 runs)
  --readout-init base --base-width N0
                          readouts Var N0/N^2, readout rate lr N0/N, embedding rate lr N/N0:
                          a width-N0 network with Var(W)=1/N0 and rate lr in every layer,
                          carried to width N (muP for SGD relative to the base width N0)
--readout-lr / --embedding-lr override the derived rates.

Output columns mirror the d2/d4 trainer with degree-specific names (A4L, A4R, A8, Q4, mse4L,
mse4R, mse8, f8_*); R is R4 and C_abs_share is the 4+4->8 share of the d8 output.  In the
model, ``Wp`` holds the two parent readouts and ``w4`` the degree-8 target readout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]   # run as a script by the launcher
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.accessibility.run_two_block_phase_diagram import (  # noqa: E402
    EPS,
    Decays,
    TwoBlockPhaseNet,
    append_rows,
    atomic_torch_save,
    build_sgd,
    clone_state,
    decay_coefficients,
    load_state,
    load_torch,
    mean_and_cv,
    parse_csv,
    plateau_reached,
    resolve_device,
    safe_cosine,
    state_is_finite,
    total_gradient_norm,
    walsh_vector,
)

TASK = "d4d8"
N_BITS = 8
SINGLE_MASKS = tuple(1 << bit for bit in range(N_BITS))
PARENT_MASKS = (0b00001111, 0b11110000)      # chi_L = x0x1x2x3, chi_R = x4x5x6x7
TARGET_MASK = 0b11111111                      # chi_8 = chi_L * chi_R
RC8 = 5.0 / 32.0                              # 4+4->8 beats the direct gate when R4 > 5/32


@dataclass(frozen=True)
class Cube8:
    x: torch.Tensor      # (256, 8), bit b of the row index -> x_b = +1 (0) or -1 (1)
    chi: torch.Tensor    # (256, 256) Walsh characters; column m is the subset with bitmask m
    y4: torch.Tensor     # (256, 2) parent targets chi_L, chi_R
    y8: torch.Tensor     # (256, 1) target chi_8


def build_cube8(device: torch.device, dtype: torch.dtype) -> Cube8:
    index = torch.arange(2 ** N_BITS, device=device)
    bits = torch.arange(N_BITS, device=device)
    in_subset = ((index[:, None] >> bits[None, :]) & 1).to(dtype)    # (2^8, 8)
    x = 1.0 - 2.0 * in_subset                                        # same convention as build_cube
    chi = torch.where(in_subset[None, :, :] > 0, x[:, None, :],
                      torch.ones_like(x[:, None, :])).prod(dim=2)    # (point, mode)
    return Cube8(x=x, chi=chi, y4=chi[:, list(PARENT_MASKS)], y8=chi[:, [TARGET_MASK]])


def loss_terms8(model: TwoBlockPhaseNet, data: Cube8, decays: Decays, gamma4: float,
                gamma8: float) -> dict[str, torch.Tensor]:
    pred4, pred8, _ = model(data.x)
    mse4_each = ((pred4 - data.y4) ** 2).mean(dim=0)
    mse8 = ((pred8 - data.y8) ** 2).mean()
    loss4 = 0.5 * gamma4 * mse4_each.sum()
    loss8 = 0.5 * gamma8 * mse8
    reg_e = 0.5 * decays.embedding * model.E.square().sum()
    reg_1 = 0.5 * decays.block1 * model.V1.square().sum()
    reg_2 = 0.5 * decays.block2 * model.V2.square().sum()
    reg_p = 0.5 * decays.parent_readout * model.Wp.square().sum()
    reg_8 = 0.5 * decays.target_readout * model.w4.square().sum()
    regularizer = reg_e + reg_1 + reg_2 + reg_p + reg_8
    return {
        "objective": loss4 + loss8 + regularizer,
        "data_loss": loss4 + loss8,
        "loss4": loss4,
        "loss8": loss8,
        "mse4L": mse4_each[0],
        "mse4R": mse4_each[1],
        "mse8": mse8,
        "regularizer": regularizer,
        "reg_E": reg_e,
        "reg_V1": reg_1,
        "reg_V2": reg_2,
        "reg_W4": reg_p,
        "reg_w8": reg_8,
    }


def phase_label8(metrics: dict[str, float], active_threshold: float = 0.5) -> str:
    """Diagnostic label; continuous route measurements remain authoritative."""
    parents_active = min(metrics["A4L"], metrics["A4R"]) >= active_threshold
    if metrics["A8"] < active_threshold:
        return "parents_only" if parents_active else "suppressed"
    if metrics["block2_abs_share"] < 0.05:
        return "H_D"
    return "H_C" if metrics["block2_parent_purity"] >= 0.5 else "H_D"


@torch.no_grad()
def measure8(model: TwoBlockPhaseNet, data: Cube8, decays: Decays, gamma4: float,
             gamma8: float) -> dict[str, float | str]:
    pred4, pred8, parts = model(data.x)
    terms = loss_terms8(model, data, decays, gamma4, gamma8)

    h0_single = [walsh_vector(parts["h0"], data.chi[:, m]) for m in SINGLE_MASKS]
    h1_single = [walsh_vector(parts["h1"], data.chi[:, m]) for m in SINGLE_MASKS]
    b1_single = [walsh_vector(parts["b1"], data.chi[:, m]) for m in SINGLE_MASKS]
    h1_parent = [walsh_vector(parts["h1"], data.chi[:, m]) for m in PARENT_MASKS]
    q1_0, q1_0_cv = mean_and_cv(float(q.norm()) for q in h0_single)
    q1_1, q1_1_cv = mean_and_cv(float(q.norm()) for q in h1_single)
    q4, q4_cv = mean_and_cv(float(q.norm()) for q in h1_parent)
    rho_values = [safe_cosine(q0, qb) for q0, qb in zip(h0_single, b1_single)]
    finite_rho = [value for value in rho_values if np.isfinite(value)]

    target = data.chi[:, TARGET_MASK]
    q8_h1 = walsh_vector(parts["h1"], target)
    q8_b2 = walsh_vector(parts["b2"], target)
    c1_neuron = model.w4 * q8_h1
    c2_neuron = model.w4 * q8_b2

    # Block-2 pre-activation spectrum over all 255 non-constant modes (column m-1 = mask m).
    z2_power = (parts["z2"].T @ data.chi[:, 1:] / data.x.shape[0]).square()
    total_power = z2_power.sum(dim=1)
    parent_score = z2_power[:, [m - 1 for m in PARENT_MASKS]].sum(dim=1) / (total_power + EPS)
    single_score = z2_power[:, [m - 1 for m in SINGLE_MASKS]].sum(dim=1) / (total_power + EPS)
    other_score = torch.clamp(1.0 - parent_score - single_score, min=0.0)

    abs_c1, abs_c2 = c1_neuron.abs(), c2_neuron.abs()
    d1_abs = float(abs_c1.sum())
    c_abs = float((abs_c2 * parent_score).sum())
    d2_abs = float((abs_c2 * single_score).sum())
    other_abs = float((abs_c2 * other_score).sum())
    total_abs = d1_abs + c_abs + d2_abs + other_abs
    block2_abs = c_abs + d2_abs + other_abs
    parent_purity = c_abs / block2_abs if block2_abs > 1e-20 else 0.0
    single_purity = d2_abs / block2_abs if block2_abs > 1e-20 else 0.0

    a4l = float((pred4[:, 0] * data.y4[:, 0]).mean())
    a4r = float((pred4[:, 1] * data.y4[:, 1]).mean())
    a8 = float((pred8[:, 0] * data.y8[:, 0]).mean())
    f8_h1, f8_block2 = float(c1_neuron.sum()), float(c2_neuron.sum())
    accessibility = q4 / max(q1_1, 1e-30)
    metrics: dict[str, float | str] = {
        "Q1_0": q1_0,
        "Q1_1": q1_1,
        "Q4": q4,
        "R": accessibility,
        "Q1_0_cv": q1_0_cv,
        "Q1_1_cv": q1_1_cv,
        "Q4_cv": q4_cv,
        "singleton_overlap_rho": float(np.mean(finite_rho)) if finite_rho else float("nan"),
        "Q8_h1": float(q8_h1.norm()),
        "Q8_block2": float(q8_b2.norm()),
        "d8_feature_alignment": safe_cosine(q8_h1, q8_b2),
        "A4L": a4l,
        "A4R": a4r,
        "A4_mean": 0.5 * (a4l + a4r),
        "A8": a8,
        "d8_accuracy": float((torch.sign(pred8) == data.y8).to(torch.float64).mean()),
        "f8_h1_signed": f8_h1,
        "f8_block2_signed": f8_block2,
        "f8_C_signed": float((c2_neuron * parent_score).sum()),
        "f8_D2_signed": float((c2_neuron * single_score).sum()),
        "f8_other_signed": float((c2_neuron * other_score).sum()),
        "route_reconstruction_error": a8 - f8_h1 - f8_block2,
        "D1_abs_share": d1_abs / max(total_abs, 1e-30),
        "C_abs_share": c_abs / max(total_abs, 1e-30),
        "D2_abs_share": d2_abs / max(total_abs, 1e-30),
        "other_abs_share": other_abs / max(total_abs, 1e-30),
        "block2_abs_share": block2_abs / max(total_abs, 1e-30),
        "block2_parent_purity": parent_purity,
        "block2_single_purity": single_purity,
        "theory_C_from_R": float(accessibility > RC8),
        "theory_margin_R": accessibility - RC8,
        "E_norm": float(model.E.norm()),
        "V1_norm": float(model.V1.norm()),
        "V2_norm": float(model.V2.norm()),
        "W4_norm": float(model.Wp.norm()),
        "w8_norm": float(model.w4.norm()),
    }
    metrics.update({name: float(value) for name, value in terms.items()})
    metrics["phase"] = phase_label8(metrics)  # type: ignore[arg-type]
    metrics["gate_agrees_with_R"] = (
        float((metrics["phase"] == "H_C") == bool(metrics["theory_C_from_R"]))
        if metrics["phase"] in ("H_C", "H_D") else float("nan")
    )
    return metrics


def pretrain_parents8(model: TwoBlockPhaseNet, data: Cube8, gamma4: float, hidden_lr: float,
                      readout_lr: float, momentum: float, max_steps: int, min_steps: int,
                      plateau_window: int, plateau_tol: float,
                      embedding_lr: float | None = None):
    """Fit the two degree-4 parents read from h1 (E, V1 and the parent readout only)."""
    optimizer = build_sgd([model.V1], [model.Wp], hidden_lr, readout_lr, momentum,
                          embedding=([model.E], hidden_lr if embedding_lr is None
                                     else embedding_lr))
    history: list[float] = []
    converged = False
    start = time.perf_counter()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        h0 = data.x @ model.E.T
        h1 = h0 + torch.relu(h0 @ model.V1.T)
        loss = 0.5 * gamma4 * ((h1 @ model.Wp.T - data.y4) ** 2).mean(dim=0).sum()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite parent-pretraining loss at step {step}")
        loss.backward()
        optimizer.step()
        history.append(loss_value)
        if step >= min_steps and plateau_reached(history, plateau_window, plateau_tol):
            converged = True
            break
    diagnostics = measure8(model, data, decay_coefficients(0.0, 1.0), gamma4, gamma8=0.0)
    return clone_state(model), {
        "steps": step,
        "converged": float(converged),
        "loss": history[-1],
        "R": float(diagnostics["R"]),
        "Q1_1": float(diagnostics["Q1_1"]),
        "Q4": float(diagnostics["Q4"]),
        "A4_mean": float(diagnostics["A4_mean"]),
        "seconds": time.perf_counter() - start,
    }


def train_cell8(model: TwoBlockPhaseNet, data: Cube8, decays: Decays, gamma4: float,
                gamma8: float, hidden_lr: float, readout_lr: float, momentum: float,
                max_steps: int, min_steps: int, plateau_window: int, plateau_tol: float,
                snapshot_every: int, embedding_lr: float | None = None):
    optimizer = build_sgd([model.V1, model.V2], [model.Wp, model.w4], hidden_lr, readout_lr,
                          momentum, embedding=([model.E], hidden_lr if embedding_lr is None
                                               else embedding_lr))
    history: list[float] = []
    trajectory: list[dict[str, float | str]] = []
    initial = measure8(model, data, decays, gamma4, gamma8)
    trajectory.append({"step": 0, **initial})
    start = time.perf_counter()
    converged = False
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms8(model, data, decays, gamma4, gamma8)
        terms["objective"].backward()
        optimizer.step()
        objective = float(terms["objective"].detach())
        if not math.isfinite(objective):
            raise FloatingPointError(f"non-finite objective at step {step}")
        history.append(objective)
        if step % snapshot_every == 0:
            trajectory.append({"step": step, **measure8(model, data, decays, gamma4, gamma8)})
        if step >= min_steps and plateau_reached(history, plateau_window, plateau_tol):
            converged = True
            break

    final = measure8(model, data, decays, gamma4, gamma8)
    if trajectory[-1]["step"] != step:
        trajectory.append({"step": step, **final})
    model.zero_grad(set_to_none=True)
    loss_terms8(model, data, decays, gamma4, gamma8)["objective"].backward()
    final.update({
        "steps": step,
        "converged": float(converged),
        "gradient_norm": total_gradient_norm(model),
        "seconds": time.perf_counter() - start,
        "initial_R": float(initial["R"]),
        "initial_Q1_1": float(initial["Q1_1"]),
        "initial_Q4": float(initial["Q4"]),
        "initial_A8": float(initial["A8"]),
    })
    return final, trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", default="0.1,0.3,1,2,5,10,30,100")
    parser.add_argument("--lambda0s", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--protocols", default="joint,parent_warm",
                        help="comma list drawn from joint,parent_warm")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir",
                        default="/root/parity/analysis_output/two_block_d4_d8_phase_diagram")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--gamma4", type=float, default=2.0, help="weight of each parent loss")
    parser.add_argument("--gamma8", type=float, default=2.0, help="weight of the d8 loss")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="SGD learning rate for the embedding and hidden matrices")
    parser.add_argument("--readout-lr", type=float, default=None,
                        help="SGD learning rate for the readouts (default: lr for fan_in, "
                             "lr/N for mup, lr*N0/N for base)")
    parser.add_argument("--readout-init", choices=("mup", "fan_in", "base"), default="fan_in",
                        help="readout init variance: fan_in = 1/N (default), mup = 1/N^2, "
                             "base = N0/N^2 with N0 = --base-width")
    parser.add_argument("--base-width", type=int, default=None,
                        help="base width N0 of --readout-init base")
    parser.add_argument("--embedding-lr", type=float, default=None,
                        help="SGD learning rate for E (default: lr, or lr*N/N0 for base)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--pretrain-momentum", type=float, default=None,
                        help="momentum of parent pretraining (default: --momentum)")
    parser.add_argument("--max-steps", type=int, default=250_000)
    parser.add_argument("--min-steps", type=int, default=250_000,
                        help="plateau stopping is only checked from here on (= max: none)")
    parser.add_argument("--plateau-window", type=int, default=5_000)
    parser.add_argument("--plateau-tol", type=float, default=1e-4)
    parser.add_argument("--snapshot-every", type=int, default=1_000)
    parser.add_argument("--pretrain-max-steps", type=int, default=50_000)
    parser.add_argument("--pretrain-min-steps", type=int, default=50_000)
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="save final states for every completed cell")
    parser.add_argument("--checkpoint-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ratios = parse_csv(args.ratios, float)
    lambda0s = parse_csv(args.lambda0s, float)
    seeds = parse_csv(args.seeds, int)
    protocols = parse_csv(args.protocols, str)
    unknown = sorted(set(protocols) - {"joint", "parent_warm"})
    if unknown:
        raise ValueError(f"unknown protocols: {unknown}")
    if args.min_steps > args.max_steps or args.pretrain_min_steps > args.pretrain_max_steps:
        raise ValueError("min-steps cannot exceed max-steps")
    if args.snapshot_every <= 0 or args.plateau_window <= 0:
        raise ValueError("snapshot-every and plateau-window must be positive")
    if (args.readout_init == "base") != (args.base_width is not None):
        raise ValueError("--base-width is required by, and only used with, --readout-init base")
    if args.readout_lr is not None:
        readout_lr = args.readout_lr
    elif args.readout_init == "fan_in":
        readout_lr = args.lr
    elif args.readout_init == "mup":
        readout_lr = args.lr / args.width
    else:
        readout_lr = args.lr * args.base_width / args.width
    if args.embedding_lr is not None:
        embedding_lr = args.embedding_lr
    elif args.readout_init == "base":
        embedding_lr = args.lr * args.width / args.base_width
    else:
        embedding_lr = args.lr
    if args.lr <= 0 or readout_lr <= 0 or embedding_lr <= 0:
        raise ValueError("learning rates must be positive")
    pretrain_momentum = args.momentum if args.pretrain_momentum is None else args.pretrain_momentum

    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    out = Path(args.output_dir)
    results_path = out / f"results_{args.tag}.csv"
    trajectories_path = out / f"trajectories_{args.tag}.csv"
    manifest_path = out / f"manifest_{args.tag}.json"
    pretrain_dir = out / "pretrain"
    checkpoints_dir = out / "checkpoints"
    cell_count = len(ratios) * len(lambda0s) * len(seeds) * len(protocols)
    print(f"task={TASK} device={device} dtype={args.dtype} width={args.width} cells={cell_count} "
          f"hidden_lr={args.lr:g} readout_lr={readout_lr:g} embedding_lr={embedding_lr:g} "
          f"readout_init={args.readout_init} base_width={args.base_width} "
          f"momentum={args.momentum:g} steps={args.min_steps}-{args.max_steps} "
          f"ratios={ratios} lambda0s={lambda0s} protocols={protocols}", flush=True)
    if args.dry_run:
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out.mkdir(parents=True, exist_ok=True)
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task": TASK,
        "tag": args.tag,
        "axis_definition": {"lambda_E": "lambda0", "lambda_1": "ratio * lambda0",
                            "lambda_2": "lambda0", "lambda_p": "lambda0", "lambda_8": "lambda0"},
        "ratios": ratios,
        "lambda0s": lambda0s,
        "seeds": seeds,
        "protocols": protocols,
        "arguments": vars(args),
        "derived_readout_lr": readout_lr,
        "derived_embedding_lr": embedding_lr,
        "critical_accessibility": RC8,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        old.setdefault("derived_embedding_lr", old["arguments"]["lr"])   # older manifests
        keys = ("task", "ratios", "lambda0s", "seeds", "protocols", "derived_readout_lr",
                "derived_embedding_lr")
        if any(old.get(key) != manifest.get(key) for key in keys):
            raise RuntimeError(f"{manifest_path} exists with a different grid")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    completed: set[str] = set()
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if existing.cell_id.duplicated().any():
            raise RuntimeError(f"duplicate cells already present in {results_path}")
        completed = set(existing.cell_id.astype(str))
        print(f"resume: {len(completed)} completed cells", flush=True)

    data = build_cube8(device, dtype)
    pretrain_info_path = out / f"pretrain_{args.tag}.json"
    pretrain_info: dict[str, dict] = (json.loads(pretrain_info_path.read_text())
                                      if pretrain_info_path.exists() else {})
    completed_this_run = 0
    for seed in seeds:
        base = TwoBlockPhaseNet(args.width, seed, device, dtype, readout_init=args.readout_init,
                                input_dim=N_BITS, base_width=args.base_width)
        protocol_states = {"joint": clone_state(base)}
        if "parent_warm" in protocols:
            warm_path = pretrain_dir / f"parent_warm_{args.tag}_N{args.width}_seed{seed}.pt"
            expected = {"task": TASK, "hidden_lr": args.lr, "readout_lr": readout_lr,
                        "momentum": pretrain_momentum, "readout_init": args.readout_init,
                        "embedding_lr": embedding_lr, "base_width": args.base_width}
            if warm_path.exists():
                payload = load_torch(warm_path, device)
                # warm starts saved before embedding_lr/base_width existed used lr and no N0
                payload.setdefault("embedding_lr", payload.get("hidden_lr"))
                payload.setdefault("base_width", None)
                mismatched = [key for key, value in expected.items() if payload.get(key) != value]
                if mismatched:
                    raise RuntimeError(f"{warm_path} was made with different {mismatched}")
                warm_state = {name: value.to(device=device, dtype=dtype)
                              for name, value in payload["state_dict"].items()}
                if not state_is_finite(warm_state):
                    raise RuntimeError(f"{warm_path} contains non-finite parameters")
                warm_diagnostics = payload["diagnostics"]
                print(f"[seed {seed}] loaded {warm_path.name}", flush=True)
            else:
                warm_state, warm_diagnostics = pretrain_parents8(
                    base, data, args.gamma4, args.lr, readout_lr, pretrain_momentum,
                    args.pretrain_max_steps, args.pretrain_min_steps,
                    args.plateau_window, args.plateau_tol, embedding_lr=embedding_lr)
                atomic_torch_save({
                    "state_dict": {name: value.cpu() for name, value in warm_state.items()},
                    "diagnostics": warm_diagnostics, "width": args.width, "seed": seed,
                    **expected,
                }, warm_path)
                print(f"[seed {seed}] saved {warm_path.name}: {warm_diagnostics}", flush=True)
            protocol_states["parent_warm"] = warm_state
            pretrain_info[str(seed)] = warm_diagnostics
            pretrain_info_path.write_text(json.dumps(pretrain_info, indent=2, sort_keys=True))

        for protocol in protocols:
            for ratio in ratios:
                for lambda0 in lambda0s:
                    cell_id = f"{protocol}_s{seed}_l{lambda0:.8g}_r{ratio:.8g}"
                    if cell_id in completed:
                        print(f"skip {cell_id}", flush=True)
                        continue
                    load_state(base, protocol_states[protocol])
                    decays = decay_coefficients(lambda0, ratio)
                    final, trajectory = train_cell8(
                        base, data, decays, args.gamma4, args.gamma8, args.lr, readout_lr,
                        args.momentum, args.max_steps, args.min_steps, args.plateau_window,
                        args.plateau_tol, args.snapshot_every, embedding_lr=embedding_lr)
                    common = {
                        "cell_id": cell_id, "task": TASK, "tag": args.tag,
                        "protocol": protocol, "seed": seed, "width": args.width,
                        "lambda0": lambda0, "ratio": ratio,
                        "lambda_E": decays.embedding, "lambda_1": decays.block1,
                        "lambda_2": decays.block2, "lambda_p": decays.parent_readout,
                        "lambda_8": decays.target_readout,
                        "gamma4": args.gamma4, "gamma8": args.gamma8,
                        "hidden_lr": args.lr, "readout_lr": readout_lr,
                        "embedding_lr": embedding_lr, "momentum": args.momentum,
                        "readout_init": args.readout_init, "base_width": args.base_width,
                    }
                    append_rows(results_path, [{**common, **final}])
                    append_rows(trajectories_path, [{**common, **row} for row in trajectory])
                    if args.save_checkpoints:
                        checkpoint_dtype = (torch.float64 if args.checkpoint_dtype == "float64"
                                            else torch.float32)
                        atomic_torch_save({
                            "state_dict": {
                                name: value.detach().to(device="cpu", dtype=checkpoint_dtype)
                                for name, value in base.state_dict().items()
                            },
                            "metadata": common,
                        }, checkpoints_dir / f"{cell_id}.pt")
                    completed.add(cell_id)
                    completed_this_run += 1
                    print(f"[{completed_this_run}/{cell_count}] {cell_id} phase={final['phase']} "
                          f"obj={float(final['objective']):.5g} A8={float(final['A8']):.3f} "
                          f"A4={float(final['A4_mean']):.3f} R4={float(final['R']):.3f} "
                          f"Cshare={float(final['C_abs_share']):.3f} "
                          f"grad={float(final['gradient_norm']):.2e} steps={int(final['steps'])}",
                          flush=True)
    print(f"complete: wrote {results_path} and {trajectories_path}", flush=True)


if __name__ == "__main__":
    main()
