"""Merge phase-diagram workers and compare theory with both training protocols.

Parent-warm and joint results are aggregated separately.  No post-hoc selection
between protocols is used in the main empirical/theory comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, SymLogNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


RC = 1.0 / np.sqrt(8.0)
PHASES = ("H_C", "H_D", "parents_only", "suppressed")
PHASE_CODE = {phase: index for index, phase in enumerate(PHASES)}
PHASE_COLORS = ("#2a9d8f", "#e9c46a", "#8d99ae", "#343a40")
GAMMA2 = GAMMA4 = 2.0
S_GRID = np.geomspace(2e-3, 6.0, 180)[:, None]
Q_GRID = np.geomspace(2e-3, 6.0, 180)[None, :]
F_ZERO = GAMMA2 + GAMMA4 / 2


def target_free_energy(zeta: np.ndarray) -> np.ndarray:
    result = np.full_like(zeta, GAMMA4 / 2)
    active = GAMMA4 * zeta > 1
    result[active] = (
        1 / zeta[active] - 1 / (2 * GAMMA4 * zeta[active] ** 2)
    )
    return result


def solve_theory_cell(lambda0: float, ratio: float) -> dict[str, float | str]:
    """Evaluate the report's reduced stationary theory on one experimental cell."""
    lambda_1 = ratio * lambda0
    phi = (
        2 * lambda0 * S_GRID ** 2
        + 8 * lambda_1 * Q_GRID ** 2 / S_GRID ** 2
        + GAMMA2 * lambda0 / (GAMMA2 * Q_GRID ** 2 + lambda0)
    )
    zeta_d = np.sqrt(
        (S_GRID ** 2 / 64) / lambda_1
        + ((S_GRID ** 2 + Q_GRID ** 2) / 64) / lambda0
    ) / np.sqrt(lambda0)
    zeta_c = np.sqrt(
        (S_GRID ** 2 / 64) / lambda_1
        + (Q_GRID ** 2 / 8) / lambda0
    ) / np.sqrt(lambda0)
    free_d = phi + target_free_energy(zeta_d)
    free_c = phi + target_free_energy(zeta_c)
    idx_d = np.unravel_index(np.argmin(free_d), free_d.shape)
    idx_c = np.unravel_index(np.argmin(free_c), free_c.shape)
    value_d, value_c = float(free_d[idx_d]), float(free_c[idx_c])
    if min(value_d, value_c) >= F_ZERO - 2e-4:
        return {"theory_phase": "suppressed", "theory_C_share": 0.0,
                "theory_R": 0.0, "theory_A4": 0.0, "theory_A2": 0.0}

    compositional = value_c < value_d
    idx = idx_c if compositional else idx_d
    zeta = float((zeta_c if compositional else zeta_d)[idx])
    s, q = float(S_GRID[idx[0], 0]), float(Q_GRID[0, idx[1]])
    amplitude = max(0.0, 1 - 1 / (GAMMA4 * zeta))
    parent_amplitude = GAMMA2 * q ** 2 / (GAMMA2 * q ** 2 + lambda0)
    accessibility = q / np.sqrt(s ** 2 + q ** 2)
    if amplitude < 2e-3:
        phase, frac_c = "parents_only", 0.0
    elif not compositional:
        phase, frac_c = "H_D", 0.0
    else:
        phase = "H_C"
        frac_c = float(
            (q ** 2 / (8 * lambda0))
            / (s ** 2 / (64 * lambda_1) + q ** 2 / (8 * lambda0))
        )
    return {"theory_phase": phase, "theory_C_share": frac_c,
            "theory_R": accessibility, "theory_A4": amplitude,
            "theory_A2": parent_amplitude}


def log_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted(values), dtype=float)
    if len(values) == 1:
        return np.array([values[0] / np.sqrt(2), values[0] * np.sqrt(2)])
    ratios = values[1:] / values[:-1]
    edges = np.empty(len(values) + 1)
    edges[1:-1] = np.sqrt(values[:-1] * values[1:])
    edges[0] = values[0] / np.sqrt(ratios[0])
    edges[-1] = values[-1] * np.sqrt(ratios[-1])
    return edges


def majority_phase(values: pd.Series) -> str:
    counts = values.value_counts()
    if counts.empty:
        return "suppressed"
    # Stable tie-break follows the theory figure's phase order.
    return min(counts[counts == counts.max()].index, key=PHASE_CODE.get)


def grid(frame: pd.DataFrame, column: str, lambda0s: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    pivot = frame.pivot(index="ratio", columns="lambda0", values=column)
    return pivot.reindex(index=ratios, columns=lambda0s).to_numpy()


def aggregate_protocol(results: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Aggregate one prespecified training protocol over initialization seeds."""
    subset = results[results.protocol == protocol]
    return (
        subset.groupby(["lambda0", "ratio"])
        .agg(
            phase=("phase", majority_phase),
            C_abs_share=("C_abs_share", "mean"),
            D1_abs_share=("D1_abs_share", "mean"),
            D2_abs_share=("D2_abs_share", "mean"),
            block2_pair_purity=("block2_pair_purity", "mean"),
            R=("R", "mean"),
            A4=("A4", "mean"),
            A2_mean=("A2_mean", "mean"),
            objective=("objective", "mean"),
            gradient_norm=("gradient_norm", "mean"),
            theory_C_probability=("theory_C_from_R", "mean"),
            empirical_C_probability=(
                "phase", lambda x: np.mean(np.asarray(x) == "H_C")
            ),
        )
        .reset_index()
        .assign(protocol=protocol)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="/root/parity/analysis_output/two_block_phase_diagram")
    parser.add_argument("--tags", default="g0,g1,g2,g3,g4,g5,g6,g7")
    args = parser.parse_args()
    out = Path(args.dir)
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    paths = [out / f"results_{tag}.csv" for tag in tags]
    missing = [path for path in paths if not path.exists()]
    merged_path = out / "results_merged.csv"
    if missing:
        if not merged_path.exists():
            raise FileNotFoundError(
                f"missing worker outputs and merged results: {missing}"
            )
        results = pd.read_csv(merged_path)
    else:
        results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["protocol", "seed", "lambda0", "ratio"]
    if results.duplicated(key).any():
        duplicate = results.loc[results.duplicated(key, keep=False), key]
        raise RuntimeError(f"duplicate cells found:\n{duplicate.head()}")
    results.to_csv(merged_path, index=False)

    basin = (
        results.groupby(["seed", "lambda0", "ratio"])
        .agg(phase_count=("phase", "nunique"), objective_span=("objective", lambda x: x.max() - x.min()))
        .reset_index()
    )
    basin["protocol_disagreement"] = (basin.phase_count > 1).astype(float)

    aggregates = {
        protocol: aggregate_protocol(results, protocol)
        for protocol in ("parent_warm", "joint")
    }
    disagreement = (
        basin.groupby(["lambda0", "ratio"])["protocol_disagreement"].mean().reset_index()
    )
    for protocol in aggregates:
        aggregates[protocol] = aggregates[protocol].merge(
            disagreement, on=["lambda0", "ratio"], how="left"
        )

    theory_rows = []
    for ratio in sorted(results.ratio.unique()):
        for lambda0 in sorted(results.lambda0.unique()):
            theory_rows.append({"lambda0": lambda0, "ratio": ratio,
                                **solve_theory_cell(lambda0, ratio)})
    theory = pd.DataFrame(theory_rows)
    theory.to_csv(out / "theory_grid.csv", index=False)
    for protocol in aggregates:
        summary = aggregates[protocol].merge(
            theory, on=["lambda0", "ratio"], how="left"
        )
        summary["phase_agrees_with_theory"] = (
            summary.phase == summary.theory_phase
        ).astype(float)
        aggregates[protocol] = summary
        summary.to_csv(out / f"phase_summary_{protocol}.csv", index=False)
    # Backward-compatible filename now explicitly contains parent-warm only.
    aggregates["parent_warm"].to_csv(out / "phase_summary.csv", index=False)
    aggregate = aggregates["parent_warm"]
    joint_aggregate = aggregates["joint"]

    lambda0s = np.asarray(sorted(aggregate.lambda0.unique()), dtype=float)
    ratios = np.asarray(sorted(aggregate.ratio.unique()), dtype=float)
    x_edges, y_edges = log_edges(lambda0s), log_edges(ratios)

    phase_values = grid(aggregate.assign(phase_code=aggregate.phase.map(PHASE_CODE)),
                        "phase_code", lambda0s, ratios)
    c_share = grid(aggregate, "C_abs_share", lambda0s, ratios)
    accessibility = grid(aggregate, "R", lambda0s, ratios)
    joint_phase_values = grid(
        joint_aggregate.assign(
            phase_code=joint_aggregate.phase.map(PHASE_CODE)
        ),
        "phase_code", lambda0s, ratios,
    )
    joint_c_share = grid(joint_aggregate, "C_abs_share", lambda0s, ratios)
    joint_accessibility = grid(joint_aggregate, "R", lambda0s, ratios)
    basin_disagreement = grid(aggregate, "protocol_disagreement", lambda0s, ratios)
    theory_phase_values = grid(
        aggregate.assign(theory_phase_code=aggregate.theory_phase.map(PHASE_CODE)),
        "theory_phase_code", lambda0s, ratios,
    )
    theory_c_share = grid(aggregate, "theory_C_share", lambda0s, ratios)
    theory_accessibility = grid(aggregate, "theory_R", lambda0s, ratios)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7), sharex=True, sharey=True)
    phase_cmap = ListedColormap(PHASE_COLORS)
    phase_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), phase_cmap.N)
    axes[0, 0].pcolormesh(x_edges, y_edges, phase_values, cmap=phase_cmap,
                          norm=phase_norm, shading="flat")
    axes[0, 0].set_title("(a) Parent-warm majority phase")
    axes[0, 0].legend(
        handles=[Patch(color=color, label=phase) for color, phase in zip(PHASE_COLORS, PHASES)],
        fontsize=7,
        loc="upper right",
    )

    heat = axes[0, 1].pcolormesh(x_edges, y_edges, c_share, cmap="viridis",
                                 vmin=0, vmax=1, shading="flat")
    axes[0, 1].set_title(r"(b) Mean total $2+2\to4$ share")
    fig.colorbar(heat, ax=axes[0, 1], fraction=0.046, pad=0.03)

    heat = axes[1, 0].pcolormesh(x_edges, y_edges, accessibility, cmap="magma",
                                 shading="flat")
    if np.nanmin(accessibility) <= RC <= np.nanmax(accessibility):
        axes[1, 0].contour(lambda0s, ratios, accessibility, levels=[RC],
                           colors="cyan", linewidths=1.2)
    axes[1, 0].set_title(r"(c) Final $R=Q_2/Q_1^{(1)}$; cyan: $1/\sqrt{8}$")
    fig.colorbar(heat, ax=axes[1, 0], fraction=0.046, pad=0.03)

    heat = axes[1, 1].pcolormesh(x_edges, y_edges, basin_disagreement,
                                 cmap="Reds", vmin=0, vmax=1, shading="flat")
    axes[1, 1].set_title("(d) Fraction of seeds with protocol disagreement")
    fig.colorbar(heat, ax=axes[1, 1], fraction=0.046, pad=0.03)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, color="white", ls="--", lw=0.7, alpha=0.8)
        ax.tick_params(labelsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel(r"common decay scale $\lambda_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"relative block-1 decay $\lambda_1/\lambda_2$")
    fig.suptitle("Parent-warm two-block empirical phase diagram", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "empirical_phase_diagram.pdf", bbox_inches="tight")
    fig.savefig(out / "empirical_phase_diagram.png", dpi=180, bbox_inches="tight")

    comparison, compare_axes = plt.subplots(
        3, 3, figsize=(9.8, 7.3), sharex=True, sharey=True
    )
    compare_axes[0, 0].pcolormesh(
        x_edges, y_edges, theory_phase_values, cmap=phase_cmap,
        norm=phase_norm, shading="flat",
    )
    compare_axes[1, 0].pcolormesh(
        x_edges, y_edges, phase_values, cmap=phase_cmap,
        norm=phase_norm, shading="flat",
    )
    compare_axes[2, 0].pcolormesh(
        x_edges, y_edges, joint_phase_values, cmap=phase_cmap,
        norm=phase_norm, shading="flat",
    )
    for ax, values in ((compare_axes[0, 1], theory_c_share),
                       (compare_axes[1, 1], c_share),
                       (compare_axes[2, 1], joint_c_share)):
        image = ax.pcolormesh(x_edges, y_edges, values, cmap="viridis",
                              vmin=0, vmax=1, shading="flat")
    comparison.colorbar(image, ax=compare_axes[:, 1], fraction=0.046, pad=0.025)
    r_min = np.nanmin(
        [theory_accessibility, accessibility, joint_accessibility]
    )
    r_max = np.nanmax(
        [theory_accessibility, accessibility, joint_accessibility]
    )
    for ax, values in ((compare_axes[0, 2], theory_accessibility),
                       (compare_axes[1, 2], accessibility),
                       (compare_axes[2, 2], joint_accessibility)):
        image_r = ax.pcolormesh(x_edges, y_edges, values, cmap="magma",
                                vmin=r_min, vmax=r_max, shading="flat")
        if np.nanmin(values) <= RC <= np.nanmax(values):
            ax.contour(lambda0s, ratios, values, levels=[RC], colors="cyan",
                       linewidths=1.1)
    comparison.colorbar(image_r, ax=compare_axes[:, 2], fraction=0.046, pad=0.025)
    titles = ("phase", r"compositional share $f_C$", r"accessibility $R$")
    row_names = ("Theory", "Parent-warm", "Joint")
    for row, row_name in enumerate(row_names):
        for column, title in enumerate(titles):
            compare_axes[row, column].set_title(
                f"{row_name}: {title}", fontsize=9
            )
    compare_axes[0, 0].legend(
        handles=[Patch(color=color, label=phase) for color, phase in zip(PHASE_COLORS, PHASES)],
        fontsize=6.5, loc="upper right",
    )
    for ax in compare_axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for ax in compare_axes[-1, :]:
        ax.set_xlabel(r"common decay scale $\lambda_0$")
    for ax in compare_axes[:, 0]:
        ax.set_ylabel(r"relative block-1 decay $\lambda_1/\lambda_2$")
    comparison.suptitle(
        "Stationary theory versus prespecified training protocols", fontsize=12
    )
    comparison.subplots_adjust(left=0.085, right=0.91, bottom=0.08, top=0.92,
                               wspace=0.20, hspace=0.28)
    comparison.savefig(out / "theory_empirical_comparison.pdf", bbox_inches="tight")
    comparison.savefig(out / "theory_empirical_comparison.png", dpi=180,
                       bbox_inches="tight")

    paired_objective = results.pivot(
        index=["seed", "lambda0", "ratio"],
        columns="protocol",
        values="objective",
    )
    objective_delta = (
        paired_objective["parent_warm"] - paired_objective["joint"]
    ).rename("delta_objective").reset_index()
    objective_delta = (
        objective_delta.groupby(["lambda0", "ratio"])["delta_objective"]
        .mean()
        .reset_index()
    )
    objective_delta_values = grid(
        objective_delta, "delta_objective", lambda0s, ratios
    )
    objective_norm = SymLogNorm(
        linthresh=1e-5, linscale=0.8, vmin=-2.0, vmax=2.0, base=10
    )
    objective_figure, objective_axis = plt.subplots(figsize=(5.4, 3.35))
    objective_image = objective_axis.pcolormesh(
        x_edges, y_edges, objective_delta_values,
        cmap="coolwarm", norm=objective_norm, shading="flat",
    )
    objective_axis.contour(
        lambda0s, ratios, objective_delta_values, levels=[0],
        colors="black", linewidths=0.9,
    )
    objective_axis.set_xscale("log")
    objective_axis.set_yscale("log")
    objective_axis.set_xlabel(r"common decay scale $\lambda_0$")
    objective_axis.set_ylabel(
        r"relative block-1 decay $\lambda_1/\lambda_2$"
    )
    objective_axis.set_title(
        r"Full objective: $J_{warm}-J_{joint}$ (mean over seeds)",
        fontsize=10,
    )
    objective_colorbar = objective_figure.colorbar(
        objective_image, ax=objective_axis, fraction=0.05, pad=0.035,
        ticks=[-1, -1e-2, -1e-4, 0, 1e-4, 1e-2, 1],
    )
    objective_colorbar.set_label(
        "parent-warm minus joint (signed-log scale)", fontsize=8
    )
    objective_colorbar.ax.tick_params(labelsize=7)
    objective_figure.tight_layout()
    objective_figure.savefig(
        out / "full_objective_protocol_difference.pdf", bbox_inches="tight"
    )
    objective_figure.savefig(
        out / "full_objective_protocol_difference.png", dpi=190,
        bbox_inches="tight",
    )
    plt.close(objective_figure)

    protocol_metrics = {}
    for protocol, summary in aggregates.items():
        active = summary[summary.phase.isin(["H_C", "H_D"])]
        protocol_metrics[protocol] = {
            "phase_agreement_fraction": float(
                summary.phase_agrees_with_theory.mean()
            ),
            "active_grid_cells": int(len(active)),
            "measured_R_gate_agreement_active": (
                float(np.mean((active.phase == "H_C") == (active.R > RC)))
                if len(active) else None
            ),
            "C_share_mae": float(np.mean(np.abs(
                summary.C_abs_share - summary.theory_C_share
            ))),
            "compositional_grid_cells": int((summary.phase == "H_C").sum()),
        }
    comparison_metrics = {
        "runs": int(len(results)),
        "grid_cells": int(len(aggregate)),
        "compositional_runs": int((results.phase == "H_C").sum()),
        "joint_compositional_runs": int(
            ((results.protocol == "joint") & (results.phase == "H_C")).sum()
        ),
        "warm_compositional_runs": int(
            ((results.protocol == "parent_warm") & (results.phase == "H_C")).sum()
        ),
        "suppressed_runs": int((results.phase == "suppressed").sum()),
        "converged_runs": int(results.converged.sum()),
        "capped_runs": int((results.converged == 0).sum()),
        "protocol_metrics": protocol_metrics,
        "mean_protocol_disagreement": float(aggregate.protocol_disagreement.mean()),
    }
    (out / "comparison_metrics.json").write_text(
        json.dumps(comparison_metrics, indent=2, sort_keys=True)
    )

    print(
        f"merged {len(results)} runs; protocols=parent_warm,joint; "
        f"grid={len(ratios)}x{len(lambda0s)}; "
        f"wrote {out / 'theory_empirical_comparison.pdf'}"
    )


if __name__ == "__main__":
    main()
