"""Plot protocol, seed, and convergence diagnostics for the two-block sweep."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, SymLogNorm, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PHASES = ("H_C", "H_D", "parents_only", "suppressed")
PHASE_CODE = {phase: index for index, phase in enumerate(PHASES)}
PHASE_COLORS = ("#2a9d8f", "#e9c46a", "#8d99ae", "#343a40")


def log_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted(values), dtype=float)
    ratios = values[1:] / values[:-1]
    edges = np.empty(len(values) + 1)
    edges[1:-1] = np.sqrt(values[:-1] * values[1:])
    edges[0] = values[0] / np.sqrt(ratios[0])
    edges[-1] = values[-1] * np.sqrt(ratios[-1])
    return edges


def majority_phase(values: pd.Series) -> str:
    counts = values.value_counts()
    return min(counts[counts == counts.max()].index, key=PHASE_CODE.get)


def grid(frame: pd.DataFrame, column: str, lambda0s: np.ndarray,
         ratios: np.ndarray) -> np.ndarray:
    pivot = frame.pivot(index="ratio", columns="lambda0", values=column)
    return pivot.reindex(index=ratios, columns=lambda0s).to_numpy()


def style_axes(axes: np.ndarray) -> None:
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"common decay $\lambda_0$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\lambda_1/\lambda_2$", fontsize=8)


def plot_phase(ax: plt.Axes, values: np.ndarray, x_edges: np.ndarray,
               y_edges: np.ndarray, cmap: ListedColormap,
               norm: BoundaryNorm) -> None:
    ax.pcolormesh(x_edges, y_edges, values, cmap=cmap, norm=norm,
                  shading="flat")


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="analysis_output/two_block_phase_diagram/results_merged.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output/two_block_phase_diagram/supplementary",
    )
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lambda0s = np.asarray(sorted(results.lambda0.unique()), dtype=float)
    ratios = np.asarray(sorted(results.ratio.unique()), dtype=float)
    x_edges, y_edges = log_edges(lambda0s), log_edges(ratios)
    phase_cmap = ListedColormap(PHASE_COLORS)
    phase_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), phase_cmap.N)

    protocol_maps: dict[str, np.ndarray] = {}
    for protocol in ("joint", "parent_warm"):
        summary = (
            results[results.protocol == protocol]
            .groupby(["lambda0", "ratio"])
            .phase.agg(majority_phase)
            .reset_index()
        )
        protocol_maps[protocol] = grid(
            summary.assign(code=summary.phase.map(PHASE_CODE)),
            "code", lambda0s, ratios,
        )

    best_idx = results.groupby(["seed", "lambda0", "ratio"]).objective.idxmin()
    best = results.loc[best_idx]
    best_summary = (
        best.groupby(["lambda0", "ratio"]).phase.agg(majority_phase).reset_index()
    )
    best_map = grid(
        best_summary.assign(code=best_summary.phase.map(PHASE_CODE)),
        "code", lambda0s, ratios,
    )

    paired = results.pivot(
        index=["seed", "lambda0", "ratio"], columns="protocol",
        values=["phase", "objective", "C_abs_share", "loss4", "reg_w4"],
    )
    loss4_joint = pd.to_numeric(paired["loss4"]["joint"])
    loss4_warm = pd.to_numeric(paired["loss4"]["parent_warm"])
    reg_w4_joint = pd.to_numeric(paired["reg_w4"]["joint"])
    reg_w4_warm = pd.to_numeric(paired["reg_w4"]["parent_warm"])
    paired_summary = pd.DataFrame({
        "protocol_disagreement": (
            paired["phase"]["joint"] != paired["phase"]["parent_warm"]
        ).astype(float),
        "delta_objective": (
            pd.to_numeric(paired["objective"]["parent_warm"])
            - pd.to_numeric(paired["objective"]["joint"])
        ),
        "delta_C_share": (
            pd.to_numeric(paired["C_abs_share"]["parent_warm"])
            - pd.to_numeric(paired["C_abs_share"]["joint"])
        ),
        "delta_loss4": loss4_warm - loss4_joint,
        "delta_J4_specific": (
            loss4_warm + reg_w4_warm - loss4_joint - reg_w4_joint
        ),
    }).reset_index()
    paired_agg = paired_summary.groupby(["lambda0", "ratio"]).mean().reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8), sharex=True, sharey=True)
    for ax, values, title in (
        (axes[0, 0], protocol_maps["joint"], "Joint: majority phase"),
        (axes[0, 1], protocol_maps["parent_warm"], "Parent-warm: majority phase"),
        (axes[0, 2], best_map, "Lower-objective protocol: phase"),
    ):
        plot_phase(ax, values, x_edges, y_edges, phase_cmap, phase_norm)
        ax.set_title(title, fontsize=9)
    axes[0, 0].legend(
        handles=[Patch(color=color, label=phase)
                 for color, phase in zip(PHASE_COLORS, PHASES)],
        fontsize=6, loc="upper right",
    )

    disagreement = grid(paired_agg, "protocol_disagreement", lambda0s, ratios)
    image = axes[1, 0].pcolormesh(
        x_edges, y_edges, disagreement, cmap="Reds", vmin=0, vmax=1,
        shading="flat",
    )
    axes[1, 0].set_title("Protocol disagreement fraction", fontsize=9)
    fig.colorbar(image, ax=axes[1, 0], fraction=0.047, pad=0.025)

    delta_objective = grid(paired_agg, "delta_objective", lambda0s, ratios)
    obj_limit = max(abs(np.nanpercentile(delta_objective, 2)),
                    abs(np.nanpercentile(delta_objective, 98)), 1e-8)
    image = axes[1, 1].pcolormesh(
        x_edges, y_edges, delta_objective, cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-obj_limit, vcenter=0, vmax=obj_limit),
        shading="flat",
    )
    axes[1, 1].set_title(r"$J_{warm}-J_{joint}$ (blue favors warm)", fontsize=9)
    fig.colorbar(image, ax=axes[1, 1], fraction=0.047, pad=0.025)

    delta_c = grid(paired_agg, "delta_C_share", lambda0s, ratios)
    c_limit = max(float(np.nanmax(np.abs(delta_c))), 1e-8)
    image = axes[1, 2].pcolormesh(
        x_edges, y_edges, delta_c, cmap="PiYG",
        norm=TwoSlopeNorm(vmin=-c_limit, vcenter=0, vmax=c_limit),
        shading="flat",
    )
    axes[1, 2].set_title(r"$f_{C,warm}-f_{C,joint}$", fontsize=9)
    fig.colorbar(image, ax=axes[1, 2], fraction=0.047, pad=0.025)
    style_axes(axes)
    fig.suptitle("Training history changes the observed circuit phase", fontsize=12)
    fig.tight_layout()
    save(fig, out, "protocol_phase_comparison")

    # The degree-4 loss differences span roughly nine orders of magnitude and
    # can have either sign, so a signed logarithmic color normalization is the
    # least misleading way to show both the near-zero fitted cells and the
    # occasional order-one difference caused by task suppression.
    d4_norm = SymLogNorm(
        linthresh=1e-7, linscale=0.8, vmin=-1.0, vmax=1.0, base=10
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True, sharey=True)
    for ax, column, title in (
        (
            axes[0], "delta_loss4",
            r"Data term: $L_{4,warm}-L_{4,joint}$",
        ),
        (
            axes[1], "delta_J4_specific",
            r"Target-specific: $J^{spec}_{4,warm}-J^{spec}_{4,joint}$",
        ),
    ):
        values = grid(paired_agg, column, lambda0s, ratios)
        image = ax.pcolormesh(
            x_edges, y_edges, values, cmap="coolwarm", norm=d4_norm,
            shading="flat",
        )
        ax.set_title(title, fontsize=9)
    colorbar = fig.colorbar(
        image, ax=axes, fraction=0.035, pad=0.025,
        ticks=[-1, -1e-2, -1e-4, -1e-6, 0, 1e-6, 1e-4, 1e-2, 1],
    )
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("parent-warm minus joint (signed-log scale)", fontsize=8)
    style_axes(axes.reshape(1, -1))
    fig.suptitle("Degree-4-only objective differences", fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.18, top=0.80,
                        wspace=0.14)
    save(fig, out, "d4_only_objective_difference")

    seeds = sorted(results.seed.unique())
    fig, axes = plt.subplots(len(seeds), 2, figsize=(6.7, 7.4), sharex=True,
                             sharey=True)
    for row, seed in enumerate(seeds):
        for column, protocol in enumerate(("joint", "parent_warm")):
            subset = results[(results.seed == seed) & (results.protocol == protocol)].copy()
            values = grid(
                subset.assign(code=subset.phase.map(PHASE_CODE)),
                "code", lambda0s, ratios,
            )
            plot_phase(axes[row, column], values, x_edges, y_edges,
                       phase_cmap, phase_norm)
            axes[row, column].set_title(
                f"Seed {seed}: {'joint' if protocol == 'joint' else 'parent-warm'}",
                fontsize=9,
            )
    axes[0, 0].legend(
        handles=[Patch(color=color, label=phase)
                 for color, phase in zip(PHASE_COLORS, PHASES)],
        fontsize=6, loc="upper right",
    )
    style_axes(axes)
    fig.suptitle("Phase maps for every seed and training protocol", fontsize=12)
    fig.tight_layout()
    save(fig, out, "seed_protocol_phase_grids")

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.8), sharex=True, sharey=True)
    for column, protocol in enumerate(("joint", "parent_warm")):
        summary = (
            results[results.protocol == protocol]
            .groupby(["lambda0", "ratio"])
            .agg(step_fraction=("steps", lambda x: x.mean() / 50_000),
                 gradient_norm=("gradient_norm", "mean"))
            .reset_index()
        )
        step_values = grid(summary, "step_fraction", lambda0s, ratios)
        grad_values = np.log10(np.maximum(
            grid(summary, "gradient_norm", lambda0s, ratios), 1e-12
        ))
        image_steps = axes[0, column].pcolormesh(
            x_edges, y_edges, step_values, cmap="viridis", vmin=0.3, vmax=1,
            shading="flat",
        )
        axes[0, column].set_title(
            f"{'Joint' if protocol == 'joint' else 'Parent-warm'}: steps / 50k",
            fontsize=9,
        )
        image_grad = axes[1, column].pcolormesh(
            x_edges, y_edges, grad_values, cmap="magma", shading="flat",
        )
        axes[1, column].set_title(
            f"{'Joint' if protocol == 'joint' else 'Parent-warm'}: "
            r"mean $\log_{10}\|\nabla J\|$",
            fontsize=9,
        )
    fig.colorbar(image_steps, ax=axes[0, :], fraction=0.025, pad=0.02)
    fig.colorbar(image_grad, ax=axes[1, :], fraction=0.025, pad=0.02)
    style_axes(axes)
    fig.suptitle("Finite-time convergence diagnostics", fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.11, top=0.90,
                        wspace=0.16, hspace=0.22)
    save(fig, out, "convergence_diagnostics")

    print(f"wrote supplementary figures to {out}")


if __name__ == "__main__":
    main()
