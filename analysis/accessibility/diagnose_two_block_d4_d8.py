"""Where the d4/d8 two-block networks build their parents and target (report diagnostics).

For a d4/d8 phase grid (run_two_block_d4_d8_phase_diagram.py) this adds to every cell
  * the block-1 / block-2 split of the readout-weighted parent and d8 amplitudes (h0 = E x
    holds only singletons, so each amplitude is exactly block 1 + block 2),
  * what the block-2 neurons that make the d8 output read: the |w8 a_n|-weighted share of their
    pre-activation power on the eight singletons, the two parents, the other degree-2 and
    degree-4 modes, the d8 mode itself (relayed from block 1), and all remaining modes,
  * the stationary theory's minimal free energies F_C and F_D (Eq. free8 of
    reports/two_block_theory/two_block_d4_d8_layerwise_decay_phase_theory.tex; the zero
    network has F = gamma4 + gamma8/2) beside the measured objective J,
  * from the trajectories: the first snapshot with A8 >= 0.5 and R4 there, the first snapshot
    with R4 > 5/32, the peak 4+4->8 share, and J's relative decrease over the last 50k steps.

    python -m analysis.accessibility.diagnose_two_block_d4_d8 \
        --grid-dir runs/<run>/analysis/phase_grid

writes runs/<run>/analysis/figures/cell_diagnostics.csv and prints the grids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from analysis.accessibility.plot_two_block_d4_d8_layerwise_phase import (  # noqa: E402
    F_ZERO,
    free_energy_surfaces,
)
from analysis.accessibility.plot_two_block_langevin_phase import (  # noqa: E402
    PROTOCOL_ROWS,
    TASKS,
    annotate,
    grid_values,
    run_description,
    style,
    theory_outline,
    theory_table,
)
from analysis.accessibility.run_two_block_d4_d8_phase_diagram import (  # noqa: E402
    PARENT_MASKS,
    RC8,
    TARGET_MASK,
    build_cube8,
)
from analysis.accessibility.run_two_block_phase_diagram import TwoBlockPhaseNet  # noqa: E402
from analysis.accessibility.two_block_neuron_types import load_model, log_edges  # noqa: E402

MASKS = np.arange(1, 256)                                  # non-constant Walsh modes
DEGREE = np.array([bin(mask).count("1") for mask in MASKS])
IS_PARENT = np.isin(MASKS, PARENT_MASKS)
READ_GROUPS = {                                            # disjoint; "rest" is the remainder
    "single": DEGREE == 1,
    "parent": IS_PARENT,
    "deg2": DEGREE == 2,
    "deg4_other": (DEGREE == 4) & ~IS_PARENT,
    "d8": MASKS == TARGET_MASK,
}


@torch.no_grad()
def network_diagnostics(model: TwoBlockPhaseNet) -> dict[str, float]:
    device = next(model.parameters()).device
    data = build_cube8(device, torch.float64)
    _, _, parts = model(data.x)

    def coefficients(values: torch.Tensor) -> np.ndarray:   # (neurons, 256), column = bitmask
        return (values.T @ data.chi / data.x.shape[0]).cpu().numpy()

    b1, b2, z2 = (coefficients(parts[name]) for name in ("b1", "b2", "z2"))
    wp = model.Wp.detach().cpu().numpy()
    w8 = model.w4.detach().cpu().numpy()
    row = {
        "A4_block1": float(np.mean([wp[g] @ b1[:, m] for g, m in enumerate(PARENT_MASKS)])),
        "A4_block2": float(np.mean([wp[g] @ b2[:, m] for g, m in enumerate(PARENT_MASKS)])),
        "A8_block1": float(w8 @ b1[:, TARGET_MASK]),
        "A8_block2": float(w8 @ b2[:, TARGET_MASK]),
    }
    power = z2[:, MASKS] ** 2
    share = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-300)
    weight = np.abs(w8 * b2[:, TARGET_MASK])
    total = weight.sum()
    for name, group in READ_GROUPS.items():
        row[f"read_{name}"] = float(weight @ share[:, group].sum(axis=1) / total) if total > 0 \
            else float("nan")
    row["read_rest"] = 1.0 - sum(row[f"read_{name}"] for name in READ_GROUPS)
    return row


def theory_free_energies(lambda0: float, ratio: float) -> dict[str, float]:
    free_d, free_c, _, _ = free_energy_surfaces(lambda0, ratio)
    return {"F_C": float(free_c.min()), "F_D": float(free_d.min()), "F_zero": F_ZERO}


def trajectory_diagnostics(trajectories: pd.DataFrame, tail: int = 50_000) -> pd.DataFrame:
    rows = []
    for (protocol, lambda0, ratio), cell in trajectories.groupby(["protocol", "lambda0", "ratio"]):
        cell = cell.sort_values("step")
        fitted = cell[cell.A8 >= 0.5]
        accessible = cell[cell.R > RC8]
        final = cell.iloc[-1]
        earlier = cell[cell.step <= final.step - tail]
        rows.append({
            "protocol": protocol, "lambda0": lambda0, "ratio": ratio,
            "step_A8_fitted": fitted.step.iloc[0] if len(fitted) else np.nan,
            "R_at_A8_fitted": fitted.R.iloc[0] if len(fitted) else np.nan,
            "step_R_above_gate": accessible.step.iloc[0] if len(accessible) else np.nan,
            "R_initial": cell.R.iloc[0],
            "R_max": cell.R.max(),
            "C_share_max": cell.C_abs_share.max(),
            "J_tail_relative_drop": ((earlier.objective.iloc[-1] - final.objective)
                                     / final.objective) if len(earlier) else np.nan,
        })
    return pd.DataFrame(rows)


def figure_route_diagnostics(frame: pd.DataFrame, run_label: str):
    """Rows: protocols.  Columns: J over the theory's minimum, block-2 share of the parents,
    and what block 2's d8 producers read (the block-1 d8 mode, or the singletons)."""
    lambda0s, ratios = np.sort(frame.lambda0.unique()), np.sort(frame.ratio.unique())
    theory = theory_table(lambda0s, ratios, TASKS["d4d8"])
    x_edges, y_edges = log_edges(lambda0s), log_edges(ratios)
    xc, yc = np.sqrt(x_edges[:-1] * x_edges[1:]), np.sqrt(y_edges[:-1] * y_edges[1:])
    frame = frame.assign(
        log_J_over_F=np.log10(frame.J_over_F_min),
        parent_block2=np.where(frame.A4_mean.abs() >= 0.05,
                               frame.A4_block2 / (frame.A4_block1 + frame.A4_block2), np.nan),
        relay=frame.read_d8.where(frame.A8_block2.abs() >= 1e-3),
        direct=frame.read_single.where(frame.A8_block2.abs() >= 1e-3))
    ratio_cmap = plt.get_cmap("RdBu_r").copy()
    ratio_norm = TwoSlopeNorm(vmin=np.log10(0.25), vcenter=0.0, vmax=np.log10(64.0))
    share_cmap = plt.get_cmap("viridis").copy()
    share_cmap.set_bad("#d9d8d4")
    share_norm = Normalize(0, 1)
    columns = (
        ("log_J_over_F", r"$J\,/\min(F_C,F_D,F_0)$", ratio_cmap, ratio_norm,
         lambda v: "–" if not np.isfinite(v) else (f"{10 ** v:.0f}" if v >= 1 else
                                                   f"{10 ** v:.2f}")),
        ("parent_block2", "block-2 share of the $d4$ parents", share_cmap, share_norm, None),
        ("relay", "block 2 reads the block-1 $d8$ mode", share_cmap, share_norm, None),
        ("direct", "block 2 reads the singletons", share_cmap, share_norm, None),
    )
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True, sharey=True,
                             layout="constrained")
    images = {}
    for row, (protocol, label) in enumerate(PROTOCOL_ROWS):
        part = frame[frame.protocol == protocol]
        for col, (column, title, cmap, norm, formatter) in enumerate(columns):
            ax = axes[row, col]
            values = grid_values(part, column, lambda0s, ratios)
            images[col] = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values),
                                        cmap=cmap, norm=norm, shading="flat")
            if formatter is None:
                annotate(ax, values, xc, yc, cmap, norm, fontsize=6)
            else:
                annotate(ax, values, xc, yc, cmap, norm, formatter=formatter, fontsize=6)
            theory_outline(ax, theory, lambda0s, ratios, x_edges, y_edges)
            ax.set_title(f"{label}: {title}", fontsize=8.5)
    ticks = np.array([0.3, 1, 3, 10, 30])
    bar = fig.colorbar(images[0], ax=axes[:, 0], shrink=0.7, aspect=30, location="bottom")
    bar.set_ticks(np.log10(ticks), labels=[f"{t:g}" for t in ticks])
    fig.colorbar(images[1], ax=axes[:, 1:], shrink=0.5, aspect=40, location="bottom")
    style(axes, lambda0s)
    fig.suptitle(f"Where the parents and the $d8$ output are built: {run_label}\n"
                 r"$J/\min F$: measured objective over the theory's minimal free energy "
                 r"($F_0=3$: zero network);  block-2 reads: $|w_{8,n}a_n|$-weighted share of the "
                 "pre-activation power of block-2 neurons producing $d8$;  dashed: theory $H_C$",
                 fontsize=10)
    return fig


def show(frame: pd.DataFrame, column: str, fmt: str = "{:.3g}") -> None:
    for protocol, part in frame.groupby("protocol"):
        table = part.pivot(index="ratio", columns="lambda0", values=column)
        print(f"\n{protocol}: {column}")
        print(table.sort_index(ascending=False).map(fmt.format).to_string())


def span(values: pd.Series) -> list[float]:
    values = values.dropna()
    return [round(float(values.min()), 4), round(float(values.max()), 4)] if len(values) else []


def summary(frame: pd.DataFrame, settled_tol: float = 1e-2) -> dict:
    """Aggregates quoted in the report; a cell is settled when J fell < settled_tol in 50k."""
    theory = theory_table(np.sort(frame.lambda0.unique()), np.sort(frame.ratio.unique()),
                          TASKS["d4d8"])
    frame = frame.merge(theory[["lambda0", "ratio", "theory_phase"]], on=["lambda0", "ratio"])
    frame["settled"] = frame.J_tail_relative_drop.abs() < settled_tol
    out: dict[str, dict] = {}
    for protocol, part in frame.groupby("protocol"):
        made = part[part.phase != "suppressed"]
        settled, unsettled = made[made.settled], made[~made.settled]
        inside = settled[settled.theory_phase == "H_C"]
        h_d = part[part.phase == "H_D"]
        fitted = part[part.step_A8_fitted.notna()]
        parents = part[part.A4_mean >= 0.5]
        out[protocol] = {
            "settled_unsuppressed_cells": int(len(settled)),
            "settled_J_over_F_min": span(settled.J_over_F_min),
            "unsettled_cells": [(float(r.lambda0), float(r.ratio)) for r in unsettled.itertuples()],
            "unsettled_J_over_F_min": span(unsettled.J_over_F_min),
            "unsettled_tail_drop": span(unsettled.J_tail_relative_drop),
            "theory_H_C_settled_cells": int(len(inside)),
            "theory_H_C_settled_phases": inside.phase.value_counts().to_dict(),
            "theory_H_C_settled_J_over_F_C": span(inside.objective / inside.F_C),
            "relay_read_settled_H_D_ratio_le_1": span(
                h_d[h_d.settled & (h_d.ratio <= 1)].read_d8),
            "relay_read_settled_H_D_ratio_2": span(h_d[h_d.settled & (h_d.ratio == 2)].read_d8),
            "single_read_H_D_ratio_ge_5": span(h_d[h_d.ratio >= 5].read_single),
            "single_read_unsettled_H_D": span(h_d[~h_d.settled].read_single),
            "parent_read_max_H_D": round(float(h_d.read_parent.max()), 4),
            "deg2_read_max": round(float(part.read_deg2.max()), 4),
            "parent_block2_share_ratio_le_1": span(
                (parents.A4_block2 / parents.A4_mean)[parents.ratio <= 1]),
            "parent_block2_share_ratio_ge_10": span(
                (parents.A4_block2 / parents.A4_mean)[parents.ratio >= 10]),
            "A8_block1_share_H_D": span(h_d.A8_block1 / h_d.A8),
            "steps_A8_fitted": sorted(fitted.step_A8_fitted.unique().tolist()),
            "R_at_A8_fitted": span(fitted.R_at_A8_fitted),
            "R_at_A8_fitted_median": round(float(fitted.R_at_A8_fitted.median()), 4),
            "step_R_above_gate": span(part.step_R_above_gate),
            "C_share_max_outside_lambda0_0.3": round(
                float(part[~np.isclose(part.lambda0, 0.3)].C_share_max.max()), 4),
        }
    paired = frame.pivot_table(index=["lambda0", "ratio"], columns="protocol", values="objective")
    gap = (paired.parent_warm - paired.joint).abs() / paired.min(axis=1)
    out["protocols"] = {"relative_J_gap_max": round(float(gap.max()), 4),
                        "relative_J_gap_median": round(float(gap.median()), 6)}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: <grid-dir>/../figures/cell_diagnostics.csv)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    grid = args.grid_dir
    out = args.out or grid.parent / "figures" / "cell_diagnostics.csv"
    results = pd.concat([pd.read_csv(path) for path in sorted(grid.glob("results_*.csv"))],
                        ignore_index=True)
    trajectories = pd.concat([pd.read_csv(path)
                              for path in sorted(grid.glob("trajectories_*.csv"))],
                             ignore_index=True)
    device = torch.device(args.device)
    rows = []
    for _, cell in results.iterrows():
        model, _ = load_model(grid / "checkpoints" / f"{cell.cell_id}.pt", device)
        rows.append({"protocol": cell.protocol, "lambda0": cell.lambda0, "ratio": cell.ratio,
                     "phase": cell.phase, "objective": cell.objective, "A4_mean": cell.A4_mean,
                     "A8": cell.A8, "R_final": cell.R, "C_share_final": cell.C_abs_share,
                     **theory_free_energies(cell.lambda0, cell.ratio),
                     **network_diagnostics(model)})
    frame = pd.DataFrame(rows).merge(trajectory_diagnostics(trajectories),
                                     on=["protocol", "lambda0", "ratio"], how="left")
    frame["J_over_F_min"] = frame.objective / frame[["F_C", "F_D", "F_zero"]].min(axis=1)
    frame = frame.sort_values(["protocol", "ratio", "lambda0"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    run_label, _ = run_description(grid, int(results.seed.iloc[0]))
    figure = figure_route_diagnostics(frame, run_label)
    for suffix in (".pdf", ".png"):
        figure.savefig(out.parent / f"fig4_route_diagnostics{suffix}", dpi=160)
    plt.close(figure)

    pd.set_option("display.width", 250)
    for column, fmt in (("J_over_F_min", "{:.2f}"), ("A4_block2", "{:.2f}"),
                        ("A8_block1", "{:.2f}"), ("A8_block2", "{:.2f}"),
                        ("read_single", "{:.2f}"), ("read_parent", "{:.3f}"),
                        ("read_deg2", "{:.2f}"), ("read_deg4_other", "{:.2f}"),
                        ("read_d8", "{:.2f}"), ("read_rest", "{:.2f}"),
                        ("step_A8_fitted", "{:.0f}"), ("R_at_A8_fitted", "{:.3f}"),
                        ("step_R_above_gate", "{:.0f}"), ("C_share_max", "{:.3f}"),
                        ("J_tail_relative_drop", "{:.1e}")):
        show(frame, column, fmt)
    numbers = summary(frame)
    (out.parent / "diagnostics_summary.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
