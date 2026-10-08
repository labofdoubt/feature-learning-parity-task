"""Stationary theory versus training on a two-block phase diagram (d2/d4 or d4/d8 task).

Reads one grid written with ``--save-checkpoints`` by ``run_two_block_phase_diagram.py``
(d2 parents, d4 target; deterministic or Langevin-annealed) or by
``run_two_block_d4_d8_phase_diagram.py`` (d4 parents, d8 target) -- the task is taken from
the grid's manifest -- (``results_*.csv`` plus ``checkpoints/<cell_id>.pt``; both protocols
may share the directory) and writes the figures below.  Names refer to the d2/d4 task; for
d4/d8 read 4+4->8 for 2+2->4, d8 for d4 and d4 parents for d2 targets, with R = Q4/Q1
gated at 5/32 instead of 1/sqrt(8):

  fig1_theory_vs_annealed.pdf   rows: theory / parent-warm / joint; columns: phase,
      f_C (readout-weighted total compositional share, as in the deterministic report),
      accessibility R, and phi_out = share of block 2's readout-weighted d4 amplitude
      carried by 2+2->4 neurons (theory: 1 in H_C, 0 otherwise)
  fig2_block_intrinsic_shares.pdf   the readout-free versions built from squared
      neuron d4 amplitudes: f_C^sq and phi_sq
  fig3_final_mse.pdf   final MSE of the d2 targets (mean of both parents) and of d4
  figures.pdf          the three figures on consecutive pages
  cell_metrics.csv, comparison_metrics.json

Per block-2 neuron n with d4 Walsh amplitude a_n (post-ReLU) and readout weight w4_n:
  phi_out = sum_{n in 2+2->4} |w4_n a_n| / sum_{n alive} |w4_n a_n|
  phi_sq  = sum_{n in 2+2->4} a_n^2      / sum_{n alive} a_n^2
  f_C^sq  = sum_{block 2} a_n^2 * pair_score_n / (sum_{block 1} a^2 + sum_{block 2} a^2),
the last mirroring f_C (whose pair_score is the parent-mode share of Var z_n) with
squared amplitudes in place of |w4 * a|.  2+2->4 neurons and dead units follow
two_block_neuron_types.classify at activity threshold tau (default 1e-6).  Cells
where no neuron produces a target mode above tau (collapsed networks) have no
defined share and are drawn grey.

Usage:
  python -m analysis.accessibility.plot_two_block_langevin_phase \
      --grid-dir runs/two_block_phase_langevin_T1e-7_seed0/analysis/phase_grid \
      --out-dir runs/two_block_phase_langevin_T1e-7_seed0/analysis/figures \
      --reference analysis_output/two_block_phase_diagram/results_merged.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from analysis.accessibility.merge_two_block_phase_diagram import (
    PHASE_CODE,
    PHASE_COLORS,
    PHASES,
    RC,
    log_edges,
    solve_theory_cell,
)
from analysis.accessibility.plot_two_block_d4_d8_layerwise_phase import solve_cell as solve_d4_d8
from analysis.accessibility.run_two_block_d4_d8_phase_diagram import (
    PARENT_MASKS,
    RC8,
    TARGET_MASK,
    build_cube8,
)
from analysis.accessibility.run_two_block_phase_diagram import PARENTS, QUAD
from analysis.accessibility.two_block_neuron_types import (
    BLUE_RAMP,
    EPS,
    MODE,
    load_model,
    phase_boundary_segments,
    walsh_coefficients,
)

PROTOCOL_ROWS = (("parent_warm", "Parent-warm"), ("joint", "Joint"))


def read_arguments(grid_dir: Path) -> dict:
    """Manifest record of one grid: training arguments plus the task name."""
    manifests = sorted(grid_dir.glob("manifest_*.json"))
    manifest = json.loads(manifests[0].read_text()) if manifests else {}
    return {**manifest.get("arguments", {}), "task": manifest.get("task", "d2d4")}


def run_description(grid_dir: Path, seed: int) -> tuple[str, bool]:
    """('Langevin-annealed training (T0=..., momentum ..., Nk steps, seed s)', noisy).

    Read from the grid's manifest so deterministic and annealed grids are labelled alike.
    """
    arguments = read_arguments(grid_dir)
    temperature = float(arguments.get("temperature_initial") or 0.0)
    momentum = float(arguments.get("momentum", 0.9))
    steps = int(arguments.get("max_steps", 50_000))
    steps_text = f"{steps / 1000:g}k" if steps >= 1000 else str(steps)
    readout = ""   # the d2/d4 default (muP readout, lr 0.01, readout lr lr/N) is not spelled out
    task = arguments["task"]
    readout_init = arguments.get("readout_init", "mup")
    rate = arguments.get("readout_lr")
    if task == "d4d8" or readout_init != "mup" or rate is not None:
        lr = float(arguments.get("lr", 1e-3 if task == "d4d8" else 0.01))
        embedding = ""
        if readout_init == "base":
            variance, default_rate = "N_0/N^2", "lr$\\cdot N_0/N$"
            embedding = (f", embedding lr lr$\\cdot N/N_0$, $N_0={arguments['base_width']}$"
                         if arguments.get("embedding_lr") is None else "")
        elif readout_init == "fan_in":
            variance, default_rate = "1/N", (f"{lr:g}" if task == "d4d8" else "lr/N")
        else:
            variance, default_rate = "1/N^2", "lr/N"
        if arguments.get("embedding_lr") is not None:
            embedding = f", embedding lr {float(arguments['embedding_lr']):g}"
        rate_text = f"{float(rate):g}" if rate is not None else default_rate
        readout = (f", hidden lr {lr:g}, "
                   f"readout init var ${variance}$, readout lr {rate_text}{embedding}")
    if temperature > 0:
        return (f"Langevin-annealed training ($T_0={tex_power(temperature)}$, momentum "
                f"{momentum:g}, {steps_text} steps{readout}, seed {seed})"), True
    return (f"deterministic training (momentum {momentum:g}, {steps_text} steps{readout}, "
            f"seed {seed})"), False


def route_shares(z2: np.ndarray, a1: np.ndarray, a2: np.ndarray, w_target: np.ndarray,
                 parent_cols: list[int], target_col: int, tau: float = 1e-6,
                 share_min: float = 0.9, balance_min: float = 0.8) -> dict[str, float]:
    """phi_out, phi_sq and f_C^sq from per-neuron Walsh coefficients (NaN if nothing is produced).

    Arrays are (neurons, modes) with column 0 the constant mode.  A block-2 neuron is a
    compositional unit when >= share_min of Var(z) lies on the two parent modes with
    min/max |coef| >= balance_min and it produces the target mode with |a| >= tau; a neuron
    is alive when some parent or target mode exceeds tau.
    """
    power = z2 ** 2
    parent_share = power[:, parent_cols].sum(axis=1) / np.maximum(power[:, 1:].sum(axis=1), EPS)
    magnitude = np.abs(z2[:, parent_cols])
    balance = magnitude.min(axis=1) / np.maximum(magnitude.max(axis=1), EPS)
    modes = list(parent_cols) + [target_col]
    alive1 = np.abs(a1[:, modes]).max(axis=1) >= tau
    alive2 = np.abs(a2[:, modes]).max(axis=1) >= tau
    q1, q2 = a1[:, target_col], a2[:, target_col]
    compose = (parent_share >= share_min) & (balance >= balance_min) & (np.abs(q2) >= tau)
    readout = np.abs(w_target * q2)
    power2 = q2 ** 2

    def share(values: np.ndarray) -> float:
        total = values[alive2].sum()
        return float(values[compose].sum() / total) if alive2.any() and total > 0 else np.nan

    total_power = (q1 ** 2).sum() + power2.sum()
    f_c_sq = (float((power2 * parent_share).sum() / total_power)
              if (alive1.any() or alive2.any()) and total_power > 0 else np.nan)
    return {
        "phi_out": share(readout),
        "phi_sq": share(power2),
        "f_C_sq": f_c_sq,
        "n_compose_block2": int(compose.sum()),
        "n_alive_block2": int(alive2.sum()),
        "target_power_block1": float((q1 ** 2).sum()),
        "target_power_block2": float(power2.sum()),
    }


def neuron_route_shares(model, tau: float = 1e-6) -> dict[str, float]:
    """d2/d4 task: 2+2->4 units read x0x1 and x2x3 and produce x0x1x2x3."""
    coefficients = walsh_coefficients(model)
    _, a1 = coefficients[1]
    z2, a2 = coefficients[2]
    return route_shares(z2, a1, a2, model.w4.detach().cpu().numpy(),
                        [MODE[parent] for parent in PARENTS], MODE[QUAD], tau)


@torch.no_grad()
def neuron_route_shares_d4_d8(model, tau: float = 1e-6) -> dict[str, float]:
    """d4/d8 task: 4+4->8 units read chi_L and chi_R and produce chi_8 (256 Walsh modes)."""
    device = next(model.parameters()).device
    data = build_cube8(device, torch.float64)
    _, _, parts = model(data.x)

    def coefficients(values: torch.Tensor) -> np.ndarray:   # column m = subset with bitmask m
        return (values.T @ data.chi / data.x.shape[0]).cpu().numpy()

    return route_shares(coefficients(parts["z2"]), coefficients(parts["b1"]),
                        coefficients(parts["b2"]), model.w4.detach().cpu().numpy(),
                        list(PARENT_MASKS), TARGET_MASK, tau)


def theory_d4_d8(lambda0: float, ratio: float) -> dict[str, float | str]:
    code, share, accessibility, amplitude8, amplitude4 = solve_d4_d8(lambda0, ratio)
    return {"theory_phase": PHASES[code], "theory_C_share": share, "theory_R": accessibility,
            "theory_A4": amplitude8, "theory_A2": amplitude4}


@dataclass(frozen=True)
class TaskSpec:
    rc: float                    # critical accessibility of the local gate
    rc_label: str                # mathtext for rc
    compose: str                 # mathtext name of the compositional unit
    parent: str                  # degree label of the supervised parents
    target: str                  # degree label of the target
    parent_mse: tuple[str, str]  # result columns of the two parent MSEs
    target_mse: str
    theory: Callable[[float, float], dict]
    shares: Callable


TASKS = {
    "d2d4": TaskSpec(RC, r"1/\sqrt{8}", r"2{+}2{\to}4", "d2", "d4", ("mse2a", "mse2b"),
                     "mse4", solve_theory_cell, neuron_route_shares),
    "d4d8": TaskSpec(RC8, r"5/32", r"4{+}4{\to}8", "d4", "d8", ("mse4L", "mse4R"),
                     "mse8", theory_d4_d8, neuron_route_shares_d4_d8),
}


def load_results(grid_dir: Path) -> pd.DataFrame:
    paths = sorted(grid_dir.glob("results_*.csv"))
    if not paths:
        raise SystemExit(f"no results_*.csv under {grid_dir}")
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["protocol", "seed", "lambda0", "ratio"]
    if results.duplicated(key).any():
        raise RuntimeError(f"duplicate cells in {grid_dir}")
    return results


def theory_table(lambda0s: np.ndarray, ratios: np.ndarray, task: TaskSpec = TASKS["d2d4"]
                 ) -> pd.DataFrame:
    rows = [{"lambda0": lam, "ratio": ratio, **task.theory(lam, ratio)}
            for ratio in ratios for lam in lambda0s]
    theory = pd.DataFrame(rows)
    theory["theory_phi"] = (theory.theory_phase == "H_C").astype(float)
    return theory


def protocol_metrics(frame: pd.DataFrame, rc: float = RC) -> dict[str, float | int]:
    """Agreement with theory for one protocol (one row per grid cell)."""
    active = frame[frame.phase.isin(["H_C", "H_D"])]
    return {
        "cells": int(len(frame)),
        "phase_agreement": float((frame.phase == frame.theory_phase).mean()),
        "gate_agreement_active": (float(((active.phase == "H_C") == (active.R > rc)).mean())
                                  if len(active) else float("nan")),
        "C_share_mae": float((frame.C_abs_share - frame.theory_C_share).abs().mean()),
        "H_C_cells": int((frame.phase == "H_C").sum()),
        "H_D_cells": int((frame.phase == "H_D").sum()),
        "suppressed_cells": int((frame.phase == "suppressed").sum()),
        "theory_H_C_cells": int((frame.theory_phase == "H_C").sum()),
    }


def selection_metrics(selected: pd.DataFrame, rc: float = RC) -> dict[str, float | int]:
    """protocol_metrics of a lower-objective selection plus how often each protocol won."""
    out = {**protocol_metrics(selected, rc),
           "won_by_parent_warm": int((selected.selected_from == "parent_warm").sum()),
           "won_by_joint": int((selected.selected_from == "joint").sum())}
    if "phi_out" in selected:
        active = selected[selected.phase.isin(["H_C", "H_D"]) & selected.phi_out.notna()]
        out["phi_out_gt_half_matches_theory_H_C_active"] = float(
            ((active.phi_out > 0.5) == (active.theory_phase == "H_C")).mean())
    return out


def grid_values(frame: pd.DataFrame, column: str, lambda0s, ratios) -> np.ndarray:
    return (frame.pivot(index="ratio", columns="lambda0", values=column)
            .reindex(index=ratios, columns=lambda0s).to_numpy(dtype=float))


def tex_power(value: float) -> str:
    """1e-07 -> '10^{-7}', 3e-08 -> '3\\times10^{-8}' (for mathtext labels)."""
    mantissa, exponent = f"{value:.0e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return power if mantissa == "1" else f"{mantissa}\\times{power}"


def short_number(value: float) -> str:
    if not np.isfinite(value):
        return "–"
    if value == 0:
        return "0"
    if abs(value) >= 1e-2:
        return f"{value:.2f}"
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def annotate(ax, values: np.ndarray, x_centres, y_centres, cmap, norm, formatter=short_number,
             fontsize: float = 5.4) -> None:
    for i, y in enumerate(y_centres):
        for j, x in enumerate(x_centres):
            value = values[i, j]
            if np.isfinite(value):
                red, green, blue, _ = cmap(norm(value))
                dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.45
                colour = "white" if dark else "#0b0b0b"
            else:
                colour = "#52514e"
            ax.text(x, y, formatter(value), ha="center", va="center", fontsize=fontsize,
                    color=colour)


def theory_outline(ax, theory: pd.DataFrame, lambda0s, ratios, x_edges, y_edges) -> None:
    """Dashed outline of the theory's H_C region (edges between H_C and any other phase)."""
    frame = theory.rename(columns={"theory_phase": "phase"})
    segments = []
    for other in ("H_D", "parents_only", "suppressed"):
        segments += phase_boundary_segments(frame, lambda0s, ratios, x_edges, y_edges,
                                            pair=("H_C", other))
    if segments:
        lines = LineCollection(segments, colors="white", linewidths=1.3, linestyles=(0, (3, 2)),
                               zorder=4)
        lines.set_path_effects([path_effects.Stroke(linewidth=2.6, foreground="#0b0b0b"),
                                path_effects.Normal()])
        ax.add_collection(lines)


def style(axes, lambda0s) -> None:
    for ax in np.ravel(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for ax in np.atleast_2d(axes)[-1, :]:
        ax.set_xlabel(r"common decay scale $\lambda_0$", fontsize=8)
    for ax in np.atleast_2d(axes)[:, 0]:
        ax.set_ylabel(r"block-1 decay ratio $\lambda_1/\lambda_2$", fontsize=8)


def lower_objective_selection(runs: pd.DataFrame) -> pd.DataFrame:
    """Per grid cell, the protocol run with the lower final regularized objective J."""
    chosen = runs.loc[runs.groupby(["lambda0", "ratio"]).objective.idxmin()].copy()
    chosen["selected_from"] = chosen.protocol
    return chosen


def figure_theory_vs_annealed(theory, protocols, lambda0s, ratios, temperature_label,
                              selected: pd.DataFrame | None = None,
                              task: TaskSpec = TASKS["d2d4"]):
    x_edges, y_edges = log_edges(lambda0s), log_edges(ratios)
    xc, yc = np.sqrt(x_edges[:-1] * x_edges[1:]), np.sqrt(y_edges[:-1] * y_edges[1:])
    phase_cmap = ListedColormap(PHASE_COLORS)
    phase_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), phase_cmap.N)
    share_cmap = plt.get_cmap("viridis").copy()
    share_cmap.set_bad("#d9d8d4")
    share_norm = Normalize(0, 1)
    rows = [("Theory", theory.assign(phase=theory.theory_phase, C_abs_share=theory.theory_C_share,
                                     R=theory.theory_R, phi_out=theory.theory_phi))]
    rows += [(label, frame) for label, frame in protocols]
    if selected is not None:
        rows.append(("Lower objective", selected))
    r_values = np.concatenate([grid_values(f, "R", lambda0s, ratios).ravel() for _, f in rows])
    r_norm = Normalize(np.nanmin(r_values), np.nanmax(r_values))
    r_cmap = plt.get_cmap("magma")

    fig, axes = plt.subplots(len(rows), 4, figsize=(15.5, 3.45 * len(rows) + 0.6),
                             sharex=True, sharey=True, layout="constrained")
    images = {}
    for row, (label, frame) in enumerate(rows):
        codes = grid_values(frame.assign(code=frame.phase.map(PHASE_CODE)), "code", lambda0s, ratios)
        axes[row, 0].pcolormesh(x_edges, y_edges, codes, cmap=phase_cmap, norm=phase_norm,
                                shading="flat")
        if "selected_from" in frame:   # mark which run won each cell: W = parent-warm, J = joint
            winners = grid_values(frame.assign(won=frame.selected_from.eq("joint").astype(float)),
                                  "won", lambda0s, ratios)
            for i, y in enumerate(yc):
                for j, x in enumerate(xc):
                    if np.isfinite(winners[i, j]):
                        red, green, blue, _ = phase_cmap(phase_norm(codes[i, j]))
                        dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.45
                        axes[row, 0].text(x, y, "J" if winners[i, j] else "W", ha="center",
                                          va="center", fontsize=6.5,
                                          color="white" if dark else "#0b0b0b")
        for col, column, cmap, norm in ((1, "C_abs_share", share_cmap, share_norm),
                                        (2, "R", r_cmap, r_norm),
                                        (3, "phi_out", share_cmap, share_norm)):
            values = grid_values(frame, column, lambda0s, ratios)
            images[col] = axes[row, col].pcolormesh(
                x_edges, y_edges, np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                shading="flat")
            annotate(axes[row, col], values, xc, yc, cmap, norm,
                     formatter=(lambda v: "1" if v == 1 else short_number(v))
                     if (row == 0 and col == 3) else short_number)
            if (col == 2 and min(values.shape) >= 2
                    and np.nanmin(values) <= task.rc <= np.nanmax(values)):
                axes[row, col].contour(lambda0s, ratios, values, levels=[task.rc], colors="cyan",
                                       linewidths=1.1)
        if row > 0:
            for col in range(4):
                theory_outline(axes[row, col], theory, lambda0s, ratios, x_edges, y_edges)
        for col, title in enumerate(("phase", r"$f_C$", r"$R$", r"$\varphi_{\rm out}$")):
            axes[row, col].set_title(f"{label}: {title}", fontsize=9)
    axes[0, 0].legend(handles=[Patch(color=c, label=p) for c, p in zip(PHASE_COLORS, PHASES)],
                      fontsize=6.5, loc="upper right")
    for col in (1, 2, 3):
        fig.colorbar(images[col], ax=axes[:, col], shrink=0.6, aspect=30)
    style(axes, lambda0s)
    fig.suptitle(f"Stationary theory versus {temperature_label}\n"
                 r"$f_C$: readout-weighted compositional share;  $R$: accessibility "
                 f"(cyan $R={task.rc_label}$);  " r"$\varphi_{\rm out}$: block-2 "
                 f"${task.compose}$ share of the readout-weighted ${task.target}$ amplitude;  "
                 "dashed: theory $H_C$ region"
                 + ("\nlower objective: per cell, the run with the lower final objective $J$ "
                    "(W = parent-warm, J = joint)" if selected is not None else ""),
                 fontsize=10)
    return fig


def figure_block_intrinsic(theory, protocols, lambda0s, ratios, temperature_label,
                           task: TaskSpec = TASKS["d2d4"]):
    x_edges, y_edges = log_edges(lambda0s), log_edges(ratios)
    xc, yc = np.sqrt(x_edges[:-1] * x_edges[1:]), np.sqrt(y_edges[:-1] * y_edges[1:])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d8d4")
    norm = Normalize(0, 1)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.4), sharex=True, sharey=True,
                             layout="constrained")
    image = None
    for row, (label, frame) in enumerate(protocols):
        for col, (column, title) in enumerate((
                ("f_C_sq", r"block-intrinsic $f_C^{\,\rm sq}$ (squared amplitudes)"),
                ("phi_sq", f"block-2 ${task.compose}$ share " r"$\varphi_{\rm sq}$"))):
            values = grid_values(frame, column, lambda0s, ratios)
            image = axes[row, col].pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values),
                                              cmap=cmap, norm=norm, shading="flat")
            annotate(axes[row, col], values, xc, yc, cmap, norm, fontsize=6)
            theory_outline(axes[row, col], theory, lambda0s, ratios, x_edges, y_edges)
            axes[row, col].set_title(f"{label}: {title}", fontsize=8.5)
    fig.colorbar(image, ax=axes, shrink=0.7, aspect=30)
    style(axes, lambda0s)
    fig.suptitle(f"Readout-free shares from squared neuron ${task.target}$ amplitudes: "
                 f"{temperature_label}"
                 "\ndashed: theory $H_C$ region (theory $\\varphi=1$ inside, 0 outside)",
                 fontsize=10)
    return fig


def figure_losses(theory, protocols, lambda0s, ratios, temperature_label,
                  task: TaskSpec = TASKS["d2d4"]):
    x_edges, y_edges = log_edges(lambda0s), log_edges(ratios)
    xc, yc = np.sqrt(x_edges[:-1] * x_edges[1:]), np.sqrt(y_edges[:-1] * y_edges[1:])
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    columns = (("mse_parent_mean", f"final MSE, ${task.parent}$ parents (mean of both)"),
               ("mse_target", f"final MSE, ${task.target}$ target"))
    stacked = np.concatenate([grid_values(f, c, lambda0s, ratios).ravel()
                              for _, f in protocols for c, _ in columns])
    positive = stacked[np.isfinite(stacked) & (stacked > 0)]
    norm = LogNorm(max(positive.min(), 1e-12), max(positive.max(), 1.0))
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.4), sharex=True, sharey=True,
                             layout="constrained")
    image = None
    for row, (label, frame) in enumerate(protocols):
        for col, (column, title) in enumerate(columns):
            values = grid_values(frame, column, lambda0s, ratios)
            image = axes[row, col].pcolormesh(x_edges, y_edges, values, cmap=cmap, norm=norm,
                                              shading="flat")
            annotate(axes[row, col], values, xc, yc, cmap, norm, fontsize=6)
            theory_outline(axes[row, col], theory, lambda0s, ratios, x_edges, y_edges)
            axes[row, col].set_title(f"{label}: {title}", fontsize=8.5)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.7, aspect=30)
    colorbar.set_label("MSE (log scale)", fontsize=8)
    style(axes, lambda0s)
    fig.suptitle(f"Final target losses: {temperature_label}"
                 "\ndashed: theory $H_C$ region", fontsize=10)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None,
                        help="deterministic sweep results (results_merged.csv) for comparison")
    parser.add_argument("--tau", type=float, default=1e-6,
                        help="activity threshold for dead units and 2+2->4 production")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    task_name = read_arguments(args.grid_dir)["task"]
    task = TASKS[task_name]
    results = load_results(args.grid_dir)
    results = results[results.seed == args.seed].copy()
    shares = []
    for cell_id in results.cell_id:
        model, _ = load_model(args.grid_dir / "checkpoints" / f"{cell_id}.pt",
                              torch.device(args.device))
        shares.append({"cell_id": cell_id, **task.shares(model, args.tau)})
    cells = results.merge(pd.DataFrame(shares), on="cell_id", validate="one_to_one")
    cells["mse_parent_mean"] = 0.5 * (cells[task.parent_mse[0]] + cells[task.parent_mse[1]])
    cells["mse_target"] = cells[task.target_mse]
    lambda0s = np.sort(cells.lambda0.unique())
    ratios = np.sort(cells.ratio.unique())
    theory = theory_table(lambda0s, ratios, task)
    cells = cells.merge(theory, on=["lambda0", "ratio"], how="left")
    selected = lower_objective_selection(cells)
    cells["lower_objective"] = cells.index.isin(selected.index)
    cells.to_csv(args.out_dir / "cell_metrics.csv", index=False)

    temperature_label, noisy = run_description(args.grid_dir, args.seed)
    protocols = [(label + (" + anneal" if noisy else ""), cells[cells.protocol == protocol])
                 for protocol, label in PROTOCOL_ROWS]

    metrics: dict[str, dict] = {"task": task_name, "annealed": {}, "deterministic_seed": {}}
    for protocol, _ in PROTOCOL_ROWS:
        frame = cells[cells.protocol == protocol]
        active = frame[frame.phase.isin(["H_C", "H_D"]) & frame.phi_out.notna()]
        metrics["annealed"][protocol] = {
            **protocol_metrics(frame, task.rc),
            "phi_out_vs_theory_mae_active": float((active.phi_out - active.theory_phi).abs().mean()),
            "phi_out_gt_half_matches_theory_H_C_active": float(
                ((active.phi_out > 0.5) == (active.theory_phase == "H_C")).mean()),
            "phi_out_minus_phi_sq_mean_abs": float((frame.phi_out - frame.phi_sq).abs().mean()),
            "f_C_minus_f_C_sq_mean_abs": float((frame.C_abs_share - frame.f_C_sq).abs().mean()),
        }
        inside = frame[frame.theory_phase == "H_C"]
        metrics["annealed"][protocol]["phi_out_mean_by_lambda0_in_theory_H_C"] = {
            f"{lam:g}": round(float(value), 3)
            for lam, value in inside.groupby("lambda0").phi_out.mean().items()}
        peak = frame.loc[frame.C_abs_share.idxmax()]
        metrics["annealed"][protocol]["peak_f_C"] = {
            "f_C": float(peak.C_abs_share), "lambda0": float(peak.lambda0),
            "ratio": float(peak.ratio), "theory_f_C": float(peak.theory_C_share),
            "phi_out": float(peak.phi_out), "R": float(peak.R)}
    paired = cells.pivot(index=["lambda0", "ratio"], columns="protocol", values="phase")
    metrics["annealed"]["protocol_disagreement"] = float(
        (paired["parent_warm"] != paired["joint"]).mean())
    metrics["annealed"]["lower_objective"] = selection_metrics(selected, task.rc)
    if args.reference is not None:
        reference = pd.read_csv(args.reference)
        reference = reference[reference.seed == args.seed].merge(theory, on=["lambda0", "ratio"])
        for protocol, _ in PROTOCOL_ROWS:
            metrics["deterministic_seed"][protocol] = protocol_metrics(
                reference[reference.protocol == protocol], task.rc)
        paired = reference.pivot(index=["lambda0", "ratio"], columns="protocol", values="phase")
        metrics["deterministic_seed"]["protocol_disagreement"] = float(
            (paired["parent_warm"] != paired["joint"]).mean())
        metrics["deterministic_seed"]["lower_objective"] = selection_metrics(
            lower_objective_selection(reference), task.rc)
        changes = {}
        for protocol, _ in PROTOCOL_ROWS:
            before = reference[reference.protocol == protocol].set_index(["lambda0", "ratio"]).phase
            after = cells[cells.protocol == protocol].set_index(["lambda0", "ratio"]).phase
            joined = pd.concat([before.rename("deterministic"), after.rename("annealed")], axis=1)
            changed = joined[joined.deterministic != joined.annealed]
            changes[protocol] = {
                "changed_cells": int(len(changed)),
                "transitions": {f"{a}->{b}": int(n) for (a, b), n in
                                changed.groupby(["deterministic", "annealed"]).size().items()},
                "cells": [f"l={lam:g},r={ratio:g}: {row.deterministic}->{row.annealed}"
                          for (lam, ratio), row in changed.iterrows()],
            }
        metrics["phase_changes_vs_deterministic"] = changes
    (args.out_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2))

    figures = {
        "fig1_theory_vs_annealed": figure_theory_vs_annealed(theory, protocols, lambda0s, ratios,
                                                             temperature_label, selected,
                                                             task),
        "fig2_block_intrinsic_shares": figure_block_intrinsic(theory, protocols, lambda0s,
                                                              ratios, temperature_label, task),
        "fig3_final_mse": figure_losses(theory, protocols, lambda0s, ratios, temperature_label,
                                        task),
    }
    with PdfPages(args.out_dir / "figures.pdf") as pdf:
        for stem, fig in figures.items():
            fig.savefig(args.out_dir / f"{stem}.pdf", bbox_inches="tight")
            fig.savefig(args.out_dir / f"{stem}.png", dpi=170, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
