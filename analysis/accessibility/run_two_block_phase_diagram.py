"""Train the fully unfrozen two-block model on the theoretical decay grid.

This experiment implements the objective in
``reports/two_block_layerwise_decay_phase_theory.tex`` exactly:

    lambda_E = lambda_p = lambda_4 = lambda_2 = lambda_0,
    lambda_1 = decay_ratio * lambda_0.

Both degree-2 parent losses and the degree-4 loss are active during the main
training phase, and every parameter is trainable.  The regularizer is written
explicitly rather than through optimizer ``weight_decay`` so that the two d2
readout rows and the d4 readout row can have independently auditable costs.

Two initialization protocols are supported. ``joint`` trains all targets from
the common random initialization. ``parent_warm`` first trains E, V1, and the
d2 readout against h1, then unfreezes the entire model and optimizes the same
final objective as ``joint``.  Agreement tests equilibrium selection; a
disagreement measures dynamical lock-in.

The script records continuous observables rather than relying only on a hard
phase label: Q1, Q2, R=Q2/Q1, target amplitudes, signed d4 contributions from
h1 and the block-2 residual, block-2 preactivation purity, circuit shares,
individual regularization costs, and the final gradient norm.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


SINGLES = ((0,), (1,), (2,), (3,))
PARENTS = ((0, 1), (2, 3))
QUAD = (0, 1, 2, 3)
NONEMPTY_SUBSETS = tuple(
    subset
    for degree in range(1, 5)
    for subset in itertools.combinations(range(4), degree)
)
RC = 1.0 / math.sqrt(8.0)
EPS = 1e-30


@dataclass(frozen=True)
class Decays:
    embedding: float
    block1: float
    block2: float
    parent_readout: float
    target_readout: float


@dataclass(frozen=True)
class CubeData:
    x: torch.Tensor
    y2: torch.Tensor
    y4: torch.Tensor
    chi: dict[tuple[int, ...], torch.Tensor]


def parse_csv(text: str, cast=float) -> list:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a non-empty comma-separated list")
    return values


def decay_coefficients(lambda0: float, ratio: float) -> Decays:
    """Map the two plotted controls to the five coefficients in the theory."""
    if lambda0 < 0 or ratio <= 0:
        raise ValueError("lambda0 must be non-negative and ratio must be positive")
    return Decays(
        embedding=lambda0,
        block1=ratio * lambda0,
        block2=lambda0,
        parent_readout=lambda0,
        target_readout=lambda0,
    )


def build_cube(device: torch.device, dtype: torch.dtype) -> CubeData:
    x = torch.tensor(
        [
            [1.0 if (mask >> bit) & 1 == 0 else -1.0 for bit in range(4)]
            for mask in range(16)
        ],
        device=device,
        dtype=dtype,
    )
    chi = {support: x[:, list(support)].prod(dim=1) for support in NONEMPTY_SUBSETS}
    y2 = torch.stack([chi[support] for support in PARENTS], dim=1)
    y4 = chi[QUAD].unsqueeze(1)
    return CubeData(x=x, y2=y2, y4=y4, chi=chi)


class TwoBlockPhaseNet(nn.Module):
    """Bias-free, pure-ReLU, two-block residual network used by the theory."""

    def __init__(
        self,
        width: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least 2")
        self.width = width
        generator = torch.Generator(device=device).manual_seed(seed)

        # Repository muP convention: Var(E)=1/d, Var(V)=1/N, Var(W)=1/N^2.
        self.E = nn.Parameter(
            torch.randn(width, 4, generator=generator, device=device, dtype=dtype) / 2.0
        )
        self.V1 = nn.Parameter(
            torch.randn(width, width, generator=generator, device=device, dtype=dtype)
            / math.sqrt(width)
        )
        self.V2 = nn.Parameter(
            torch.randn(width, width, generator=generator, device=device, dtype=dtype)
            / math.sqrt(width)
        )
        self.Wp = nn.Parameter(
            torch.randn(2, width, generator=generator, device=device, dtype=dtype) / width
        )
        self.w4 = nn.Parameter(
            torch.randn(width, generator=generator, device=device, dtype=dtype) / width
        )

    def parts(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h0 = x @ self.E.T
        z1 = h0 @ self.V1.T
        b1 = torch.relu(z1)
        h1 = h0 + b1
        z2 = h1 @ self.V2.T
        b2 = torch.relu(z2)
        h2 = h1 + b2
        return {"h0": h0, "z1": z1, "b1": b1, "h1": h1,
                "z2": z2, "b2": b2, "h2": h2}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        parts = self.parts(x)
        y2 = parts["h2"] @ self.Wp.T
        y4 = (parts["h2"] @ self.w4).unsqueeze(1)
        return y2, y4, parts


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({name: value.detach().clone() for name, value in state.items()})


def loss_terms(
    model: TwoBlockPhaseNet,
    data: CubeData,
    decays: Decays,
    gamma2: float,
    gamma4: float,
) -> dict[str, torch.Tensor]:
    pred2, pred4, _ = model(data.x)
    mse2_each = ((pred2 - data.y2) ** 2).mean(dim=0)
    mse4 = ((pred4 - data.y4) ** 2).mean()
    loss2 = 0.5 * gamma2 * mse2_each.sum()
    loss4 = 0.5 * gamma4 * mse4
    reg_e = 0.5 * decays.embedding * model.E.square().sum()
    reg_1 = 0.5 * decays.block1 * model.V1.square().sum()
    reg_2 = 0.5 * decays.block2 * model.V2.square().sum()
    reg_p = 0.5 * decays.parent_readout * model.Wp.square().sum()
    reg_4 = 0.5 * decays.target_readout * model.w4.square().sum()
    regularizer = reg_e + reg_1 + reg_2 + reg_p + reg_4
    return {
        "objective": loss2 + loss4 + regularizer,
        "data_loss": loss2 + loss4,
        "loss2": loss2,
        "loss4": loss4,
        "mse2a": mse2_each[0],
        "mse2b": mse2_each[1],
        "mse4": mse4,
        "regularizer": regularizer,
        "reg_E": reg_e,
        "reg_V1": reg_1,
        "reg_V2": reg_2,
        "reg_Wp": reg_p,
        "reg_w4": reg_4,
    }


def walsh_vector(values: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return (values * chi.unsqueeze(1)).mean(dim=0)


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = float(left.norm() * right.norm())
    if denom < 1e-20:
        return float("nan")
    return float(torch.dot(left, right) / denom)


def mean_and_cv(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    return mean, float(array.std() / max(abs(mean), 1e-30))


def phase_label(metrics: dict[str, float], active_threshold: float = 0.5) -> str:
    """A diagnostic label; continuous route measurements remain authoritative."""
    parents_active = min(metrics["A2a"], metrics["A2b"]) >= active_threshold
    target_active = metrics["A4"] >= active_threshold
    if not target_active:
        return "parents_only" if parents_active else "suppressed"
    if metrics["block2_abs_share"] < 0.05:
        return "H_D"
    return "H_C" if metrics["block2_pair_purity"] >= 0.5 else "H_D"


@torch.no_grad()
def measure(
    model: TwoBlockPhaseNet,
    data: CubeData,
    decays: Decays,
    gamma2: float,
    gamma4: float,
) -> dict[str, float | str]:
    pred2, pred4, parts = model(data.x)
    terms = loss_terms(model, data, decays, gamma2, gamma4)

    h0_single = [walsh_vector(parts["h0"], data.chi[s]) for s in SINGLES]
    h1_single = [walsh_vector(parts["h1"], data.chi[s]) for s in SINGLES]
    b1_single = [walsh_vector(parts["b1"], data.chi[s]) for s in SINGLES]
    h1_parent = [walsh_vector(parts["h1"], data.chi[s]) for s in PARENTS]
    q1_0, q1_0_cv = mean_and_cv(float(q.norm()) for q in h0_single)
    q1_1, q1_1_cv = mean_and_cv(float(q.norm()) for q in h1_single)
    q2, q2_cv = mean_and_cv(float(q.norm()) for q in h1_parent)
    rho_values = [safe_cosine(q0, qb) for q0, qb in zip(h0_single, b1_single)]
    finite_rho = [value for value in rho_values if np.isfinite(value)]

    target = data.chi[QUAD]
    q4_h1 = walsh_vector(parts["h1"], target)
    q4_b2 = walsh_vector(parts["b2"], target)
    c1_neuron = model.w4 * q4_h1
    c2_neuron = model.w4 * q4_b2
    f4_h1 = float(c1_neuron.sum())
    f4_block2 = float(c2_neuron.sum())

    z2_coeff = {support: walsh_vector(parts["z2"], data.chi[support])
                for support in NONEMPTY_SUBSETS}
    single_power = sum(z2_coeff[s].square() for s in SINGLES)
    parent_power = sum(z2_coeff[s].square() for s in PARENTS)
    total_power = sum(z2_coeff[s].square() for s in NONEMPTY_SUBSETS)
    pair_score = parent_power / (total_power + EPS)
    single_score = single_power / (total_power + EPS)
    other_score = torch.clamp(1.0 - pair_score - single_score, min=0.0)

    abs_c1 = c1_neuron.abs()
    abs_c2 = c2_neuron.abs()
    d1_abs = float(abs_c1.sum())
    c_abs = float((abs_c2 * pair_score).sum())
    d2_abs = float((abs_c2 * single_score).sum())
    other_abs = float((abs_c2 * other_score).sum())
    total_abs = d1_abs + c_abs + d2_abs + other_abs
    block2_abs = c_abs + d2_abs + other_abs
    if block2_abs > 1e-20:
        block2_pair_purity = c_abs / block2_abs
        block2_single_purity = d2_abs / block2_abs
    else:
        block2_pair_purity = 0.0
        block2_single_purity = 0.0

    f4_c_signed = float((c2_neuron * pair_score).sum())
    f4_d2_signed = float((c2_neuron * single_score).sum())
    f4_other_signed = float((c2_neuron * other_score).sum())
    a2a = float((pred2[:, 0] * data.y2[:, 0]).mean())
    a2b = float((pred2[:, 1] * data.y2[:, 1]).mean())
    a4 = float((pred4[:, 0] * data.y4[:, 0]).mean())

    metrics: dict[str, float | str] = {
        "Q1_0": q1_0,
        "Q1_1": q1_1,
        "Q2": q2,
        "R": q2 / max(q1_1, 1e-30),
        "Q1_0_cv": q1_0_cv,
        "Q1_1_cv": q1_1_cv,
        "Q2_cv": q2_cv,
        "singleton_overlap_rho": float(np.mean(finite_rho)) if finite_rho else float("nan"),
        "Q4_h1": float(q4_h1.norm()),
        "Q4_block2": float(q4_b2.norm()),
        "d4_feature_alignment": safe_cosine(q4_h1, q4_b2),
        "A2a": a2a,
        "A2b": a2b,
        "A2_mean": 0.5 * (a2a + a2b),
        "A4": a4,
        "d4_accuracy": float((torch.sign(pred4) == data.y4).to(torch.float64).mean()),
        "f4_h1_signed": f4_h1,
        "f4_block2_signed": f4_block2,
        "f4_C_signed": f4_c_signed,
        "f4_D2_signed": f4_d2_signed,
        "f4_other_signed": f4_other_signed,
        "route_reconstruction_error": a4 - f4_h1 - f4_block2,
        "D1_abs_share": d1_abs / max(total_abs, 1e-30),
        "C_abs_share": c_abs / max(total_abs, 1e-30),
        "D2_abs_share": d2_abs / max(total_abs, 1e-30),
        "other_abs_share": other_abs / max(total_abs, 1e-30),
        "block2_abs_share": block2_abs / max(total_abs, 1e-30),
        "block2_pair_purity": block2_pair_purity,
        "block2_single_purity": block2_single_purity,
        "theory_C_from_R": float(q2 / max(q1_1, 1e-30) > RC),
        "theory_margin_R": q2 / max(q1_1, 1e-30) - RC,
        "E_norm": float(model.E.norm()),
        "V1_norm": float(model.V1.norm()),
        "V2_norm": float(model.V2.norm()),
        "Wp_norm": float(model.Wp.norm()),
        "w4_norm": float(model.w4.norm()),
    }
    metrics.update({name: float(value) for name, value in terms.items()})
    metrics["phase"] = phase_label(metrics)  # type: ignore[arg-type]
    metrics["gate_agrees_with_R"] = (
        float((metrics["phase"] == "H_C") == bool(metrics["theory_C_from_R"]))
        if metrics["phase"] in ("H_C", "H_D")
        else float("nan")
    )
    return metrics


def total_gradient_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()
                         if p.grad is not None))


def plateau_reached(history: list[float], window: int, tolerance: float) -> bool:
    if len(history) <= window:
        return False
    old = history[-window - 1]
    new = history[-1]
    return abs(old - new) / max(abs(old), 1e-12) < tolerance


def build_sgd(
    hidden_parameters: Iterable[nn.Parameter],
    readout_parameters: Iterable[nn.Parameter],
    hidden_lr: float,
    readout_lr: float,
    momentum: float,
) -> torch.optim.SGD:
    """Build muP-compatible SGD groups without optimizer-level weight decay."""
    return torch.optim.SGD(
        [
            {"params": list(hidden_parameters), "lr": hidden_lr},
            {"params": list(readout_parameters), "lr": readout_lr},
        ],
        momentum=momentum,
    )


def state_is_finite(state: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in state.values())


def pretrain_parent_state(
    model: TwoBlockPhaseNet,
    data: CubeData,
    gamma2: float,
    hidden_lr: float,
    readout_lr: float,
    momentum: float,
    max_steps: int,
    min_steps: int,
    plateau_window: int,
    plateau_tol: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Place d2 parents in h1 without changing the final experiment objective."""
    optimizer = build_sgd(
        [model.E, model.V1], [model.Wp], hidden_lr, readout_lr, momentum
    )
    history: list[float] = []
    converged = False
    start = time.perf_counter()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        h0 = data.x @ model.E.T
        h1 = h0 + torch.relu(h0 @ model.V1.T)
        pred2 = h1 @ model.Wp.T
        loss = 0.5 * gamma2 * ((pred2 - data.y2) ** 2).mean(dim=0).sum()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite parent-pretraining loss at step {step}")
        loss.backward()
        optimizer.step()
        history.append(loss_value)
        if step >= min_steps and plateau_reached(history, plateau_window, plateau_tol):
            converged = True
            break
    zero_decays = decay_coefficients(0.0, 1.0)
    diagnostics = measure(model, data, zero_decays, gamma2, gamma4=0.0)
    return clone_state(model), {
        "steps": step,
        "converged": float(converged),
        "loss": history[-1],
        "R": float(diagnostics["R"]),
        "Q1_1": float(diagnostics["Q1_1"]),
        "Q2": float(diagnostics["Q2"]),
        "seconds": time.perf_counter() - start,
    }


def train_cell(
    model: TwoBlockPhaseNet,
    data: CubeData,
    decays: Decays,
    gamma2: float,
    gamma4: float,
    hidden_lr: float,
    readout_lr: float,
    momentum: float,
    max_steps: int,
    min_steps: int,
    plateau_window: int,
    plateau_tol: float,
    snapshot_every: int,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    optimizer = build_sgd(
        [model.E, model.V1, model.V2],
        [model.Wp, model.w4],
        hidden_lr,
        readout_lr,
        momentum,
    )
    history: list[float] = []
    trajectory: list[dict[str, float | str]] = []
    initial = measure(model, data, decays, gamma2, gamma4)
    trajectory.append({"step": 0, **initial})
    start = time.perf_counter()
    converged = False

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(model, data, decays, gamma2, gamma4)
        terms["objective"].backward()
        optimizer.step()
        objective = float(terms["objective"].detach())
        if not math.isfinite(objective):
            raise FloatingPointError(f"non-finite objective at step {step}")
        history.append(objective)

        if step % snapshot_every == 0:
            trajectory.append({"step": step, **measure(model, data, decays, gamma2, gamma4)})
        if step >= min_steps and plateau_reached(history, plateau_window, plateau_tol):
            converged = True
            break

    final = measure(model, data, decays, gamma2, gamma4)
    if trajectory[-1]["step"] != step:
        trajectory.append({"step": step, **final})

    model.zero_grad(set_to_none=True)
    final_terms = loss_terms(model, data, decays, gamma2, gamma4)
    final_terms["objective"].backward()
    final.update({
        "steps": step,
        "converged": float(converged),
        "gradient_norm": total_gradient_norm(model),
        "seconds": time.perf_counter() - start,
        "initial_R": float(initial["R"]),
        "initial_Q1_1": float(initial["Q1_1"]),
        "initial_Q2": float(initial["Q2"]),
        "initial_A4": float(initial["A4"]),
    })
    return final, trajectory


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", default="0.05,0.1,0.25,0.5,1,2,5,20")
    parser.add_argument("--lambda0s", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--protocols", default="joint,parent_warm",
                        help="comma list drawn from joint,parent_warm")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir",
                        default="/root/parity/analysis_output/two_block_phase_diagram")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--gamma2", type=float, default=2.0)
    parser.add_argument("--gamma4", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="SGD learning rate for embedding and hidden matrices")
    parser.add_argument(
        "--readout-lr",
        type=float,
        default=None,
        help="SGD learning rate for Wp and w4 (default: lr/width, the muP scaling)",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--min-steps", type=int, default=15_000)
    parser.add_argument("--plateau-window", type=int, default=5_000)
    parser.add_argument("--plateau-tol", type=float, default=1e-4)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--pretrain-max-steps", type=int, default=50_000)
    parser.add_argument("--pretrain-min-steps", type=int, default=10_000)
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="save final states for every completed cell")
    parser.add_argument("--checkpoint-dtype", choices=("float32", "float64"),
                        default="float32",
                        help="precision of saved final states (float64 keeps them exact)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ratios = parse_csv(args.ratios, float)
    lambda0s = parse_csv(args.lambda0s, float)
    seeds = parse_csv(args.seeds, int)
    protocols = parse_csv(args.protocols, str)
    unknown = sorted(set(protocols) - {"joint", "parent_warm"})
    if unknown:
        raise ValueError(f"unknown protocols: {unknown}")
    if args.min_steps > args.max_steps:
        raise ValueError("min-steps cannot exceed max-steps")
    if args.pretrain_min_steps > args.pretrain_max_steps:
        raise ValueError("pretrain-min-steps cannot exceed pretrain-max-steps")
    if args.snapshot_every <= 0 or args.plateau_window <= 0:
        raise ValueError("snapshot-every and plateau-window must be positive")
    if args.lr <= 0 or (args.readout_lr is not None and args.readout_lr <= 0):
        raise ValueError("learning rates must be positive")

    readout_lr = args.lr / args.width if args.readout_lr is None else args.readout_lr

    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    out = Path(args.output_dir)
    results_path = out / f"results_{args.tag}.csv"
    trajectories_path = out / f"trajectories_{args.tag}.csv"
    manifest_path = out / f"manifest_{args.tag}.json"
    pretrain_dir = out / "pretrain"
    checkpoints_dir = out / "checkpoints"
    cell_count = len(ratios) * len(lambda0s) * len(seeds) * len(protocols)

    print(
        f"device={device} dtype={args.dtype} width={args.width} cells={cell_count} "
        f"hidden_lr={args.lr:g} readout_lr={readout_lr:g} "
        f"ratios={ratios} lambda0s={lambda0s} protocols={protocols}",
        flush=True,
    )
    if args.dry_run:
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out.mkdir(parents=True, exist_ok=True)
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tag": args.tag,
        "axis_definition": {
            "lambda_E": "lambda0",
            "lambda_1": "ratio * lambda0",
            "lambda_2": "lambda0",
            "lambda_p": "lambda0",
            "lambda_4": "lambda0",
        },
        "ratios": ratios,
        "lambda0s": lambda0s,
        "seeds": seeds,
        "protocols": protocols,
        "arguments": vars(args),
        "derived_readout_lr": readout_lr,
        "critical_accessibility": RC,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        keys = ("ratios", "lambda0s", "seeds", "protocols", "derived_readout_lr")
        if any(old.get(key) != manifest[key] for key in keys):
            raise RuntimeError(f"{manifest_path} exists with a different grid")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    completed: set[str] = set()
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if "cell_id" not in existing:
            raise RuntimeError(f"cannot resume {results_path}: missing cell_id")
        if existing.cell_id.duplicated().any():
            raise RuntimeError(f"duplicate cells already present in {results_path}")
        completed = set(existing.cell_id.astype(str))
        print(f"resume: {len(completed)} completed cells", flush=True)

    data = build_cube(device, dtype)
    pretrain_info: dict[str, dict] = {}
    pretrain_info_path = out / f"pretrain_{args.tag}.json"
    if pretrain_info_path.exists():
        pretrain_info = json.loads(pretrain_info_path.read_text())

    completed_this_run = 0
    for seed in seeds:
        base = TwoBlockPhaseNet(args.width, seed, device, dtype)
        joint_state = clone_state(base)
        protocol_states = {"joint": joint_state}

        if "parent_warm" in protocols:
            warm_path = pretrain_dir / f"parent_warm_{args.tag}_N{args.width}_seed{seed}.pt"
            if warm_path.exists():
                payload = load_torch(warm_path, device)
                warm_state = {name: value.to(device=device, dtype=dtype)
                              for name, value in payload["state_dict"].items()}
                warm_diagnostics = payload["diagnostics"]
                if not state_is_finite(warm_state):
                    raise RuntimeError(
                        f"{warm_path} contains non-finite parameters; move it aside and rerun"
                    )
                if payload.get("hidden_lr") != args.lr or payload.get("readout_lr") != readout_lr:
                    raise RuntimeError(
                        f"{warm_path} was created with different optimizer learning rates"
                    )
                print(f"[seed {seed}] loaded {warm_path.name}", flush=True)
            else:
                load_state(base, joint_state)
                warm_state, warm_diagnostics = pretrain_parent_state(
                    base,
                    data,
                    args.gamma2,
                    args.lr,
                    readout_lr,
                    args.momentum,
                    args.pretrain_max_steps,
                    args.pretrain_min_steps,
                    args.plateau_window,
                    args.plateau_tol,
                )
                atomic_torch_save(
                    {
                        "state_dict": {name: value.cpu() for name, value in warm_state.items()},
                        "diagnostics": warm_diagnostics,
                        "width": args.width,
                        "seed": seed,
                        "hidden_lr": args.lr,
                        "readout_lr": readout_lr,
                    },
                    warm_path,
                )
                print(f"[seed {seed}] saved {warm_path.name}", flush=True)
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
                    final, trajectory = train_cell(
                        base,
                        data,
                        decays,
                        args.gamma2,
                        args.gamma4,
                        args.lr,
                        readout_lr,
                        args.momentum,
                        args.max_steps,
                        args.min_steps,
                        args.plateau_window,
                        args.plateau_tol,
                        args.snapshot_every,
                    )
                    common = {
                        "cell_id": cell_id,
                        "tag": args.tag,
                        "protocol": protocol,
                        "seed": seed,
                        "width": args.width,
                        "lambda0": lambda0,
                        "ratio": ratio,
                        "lambda_E": decays.embedding,
                        "lambda_1": decays.block1,
                        "lambda_2": decays.block2,
                        "lambda_p": decays.parent_readout,
                        "lambda_4": decays.target_readout,
                        "gamma2": args.gamma2,
                        "gamma4": args.gamma4,
                        "hidden_lr": args.lr,
                        "readout_lr": readout_lr,
                    }
                    append_rows(results_path, [{**common, **final}])
                    append_rows(trajectories_path, [{**common, **row} for row in trajectory])
                    if args.save_checkpoints:
                        checkpoint = checkpoints_dir / f"{cell_id}.pt"
                        checkpoint_dtype = (torch.float64 if args.checkpoint_dtype == "float64"
                                            else torch.float32)
                        atomic_torch_save(
                            {
                                "state_dict": {
                                    name: value.detach().to(device="cpu", dtype=checkpoint_dtype)
                                    for name, value in base.state_dict().items()
                                },
                                "metadata": common,
                            },
                            checkpoint,
                        )
                    completed.add(cell_id)
                    completed_this_run += 1
                    print(
                        f"[{completed_this_run}/{cell_count}] {cell_id} "
                        f"phase={final['phase']} obj={float(final['objective']):.5g} "
                        f"A4={float(final['A4']):.3f} R={float(final['R']):.3f} "
                        f"Cshare={float(final['C_abs_share']):.3f} "
                        f"grad={float(final['gradient_norm']):.2e} steps={int(final['steps'])}",
                        flush=True,
                    )

    print(f"complete: wrote {results_path} and {trajectories_path}", flush=True)


if __name__ == "__main__":
    main()
