"""Fraction of hidden neurons implementing each circuit type in the two-block model.

For every saved final checkpoint of a two-block phase-diagram grid (written by
``run_two_block_phase_diagram.py --save-checkpoints``) this evaluates the network on
the exact 4-bit cube and Walsh-decomposes, for each hidden neuron, its
pre-activation z (the input it takes) and its post-ReLU activation a (what it
produces) over all 16 modes.

Neuron types, identified by the input z (mechanisms as in
reports/d2_report_rewritten.tex and reports/d4_report_rewritten.tex):

  1+1->2      z is a balanced combination of the two singletons of a parent pair,
              (x0, x1) or (x2, x3): a corner detector that produces that pair parity.
  1+1+1+1->4  z is a balanced combination of all four singletons: a majority/corner
              gate that produces the quad parity x0x1x2x3.
  2+2->4      (block 2 only) z is a balanced combination of the two parent
              parities x0x1 and x2x3: it composes them into x0x1x2x3.

"Balanced combination of a group" = the group carries >= share_min of Var(z) (the
constant term is excluded) and min|coef| / max|coef| within the group is at least
0.8 for pairs or 0.7 for the four singletons (the d2/d4 report criteria). Block-1
pre-activations are linear in x, so only the two singleton types can occur there.

Activity: a neuron is dead at threshold tau when none of the target modes is
produced, i.e. max(|a_x0x1|, |a_x2x3|, |a_x0x1x2x3|) < tau. A typed neuron must also
produce its own type's target mode with |a_mode| >= tau. rho_type is the number of
neurons of that type divided by the block width N; dead and "other" (active but
untyped) neurons make up the rest.

Choosing tau. In the parent-warm seed-0 grid, per-neuron target production is a
continuum (~1e-4 to 1) in every non-collapsed cell; the structurally dead units
are block-2 ReLUs that are silent on all 16 inputs (production exactly 0) and the
whole collapsed ("suppressed") cells, where every neuron sits near 1e-40. Only 22
of 147,456 neurons (at most 5 per cell) fall between 1e-30 and 1e-5, so tau = 1e-6
is effectively the threshold-free "dead" criterion; larger taus (1e-4 .. 1e-2)
additionally trim the weak tail of the continuum (1e-2 already cuts live units).

Usage:
  python -m analysis.accessibility.two_block_neuron_types \
      --grid-dir runs/two_block_phase_pw_seed0/analysis/phase_grid \
      --out-dir runs/two_block_phase_pw_seed0/analysis/neuron_types
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.accessibility.run_two_block_phase_diagram import (
    NONEMPTY_SUBSETS,
    PARENTS,
    QUAD,
    TwoBlockPhaseNet,
    build_cube,
    load_torch,
)

SUBSETS = ((),) + NONEMPTY_SUBSETS          # all 16 Walsh modes; () is the constant
MODE = {subset: index for index, subset in enumerate(SUBSETS)}
PAIR_A, PAIR_B = PARENTS                    # (0, 1) and (2, 3)
SINGLES_A = ((0,), (1,))
SINGLES_B = ((2,), (3,))
SINGLES = SINGLES_A + SINGLES_B
TYPES = ("1+1->2", "1+1+1+1->4", "2+2->4")
LABELS = TYPES + ("other", "dead")
EPS = 1e-300


def walsh_matrix(x: torch.Tensor) -> torch.Tensor:
    """chi[i, k] = chi_{SUBSETS[k]}(x_i) on the cube points x (rows)."""
    columns = [torch.ones(x.shape[0], dtype=x.dtype, device=x.device)]
    columns += [x[:, list(subset)].prod(dim=1) for subset in NONEMPTY_SUBSETS]
    return torch.stack(columns, dim=1)


def load_model(path: Path, device: torch.device) -> tuple[TwoBlockPhaseNet, dict]:
    """Rebuild a TwoBlockPhaseNet from a final checkpoint or a warm-start state."""
    payload = load_torch(path, device)
    state = {name: value.to(device=device, dtype=torch.float64)
             for name, value in payload["state_dict"].items()}
    width, input_dim = state["E"].shape   # 4 inputs for d2/d4, 8 for d4/d8
    model = TwoBlockPhaseNet(width, 0, device, torch.float64, input_dim=input_dim)
    model.load_state_dict(state)
    return model, dict(payload.get("metadata", {}))


@torch.no_grad()
def walsh_coefficients(model: TwoBlockPhaseNet) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per block: (z coefficients, a coefficients), each of shape (N, 16)."""
    device = next(model.parameters()).device
    data = build_cube(device, torch.float64)
    chi = walsh_matrix(data.x)
    _, _, parts = model(data.x)
    count = data.x.shape[0]
    return {
        block: ((parts[z].T @ chi / count).cpu().numpy(),
                (parts[a].T @ chi / count).cpu().numpy())
        for block, z, a in ((1, "z1", "b1"), (2, "z2", "b2"))
    }


def neuron_features(z_coef: np.ndarray, a_coef: np.ndarray) -> pd.DataFrame:
    """Input shares/balances from z and target-mode production from a, per neuron."""
    power = z_coef ** 2
    var_z = power[:, 1:].sum(axis=1)

    def share(modes) -> np.ndarray:
        return power[:, [MODE[m] for m in modes]].sum(axis=1) / np.maximum(var_z, EPS)

    def balance(modes) -> np.ndarray:
        magnitude = np.abs(z_coef[:, [MODE[m] for m in modes]])
        return magnitude.min(axis=1) / np.maximum(magnitude.max(axis=1), EPS)

    a_pair_a = a_coef[:, MODE[PAIR_A]]
    a_pair_b = a_coef[:, MODE[PAIR_B]]
    a_quad = a_coef[:, MODE[QUAD]]
    return pd.DataFrame({
        "neuron": np.arange(z_coef.shape[0]),
        "z_const": z_coef[:, 0],
        "var_z": var_z,
        "var_a": (a_coef[:, 1:] ** 2).sum(axis=1),
        "share_singles_a": share(SINGLES_A),
        "share_singles_b": share(SINGLES_B),
        "share_singles": share(SINGLES),
        "share_parents": share((PAIR_A, PAIR_B)),
        "share_quad": share((QUAD,)),
        "balance_singles_a": balance(SINGLES_A),
        "balance_singles_b": balance(SINGLES_B),
        "balance_singles": balance(SINGLES),
        "balance_parents": balance((PAIR_A, PAIR_B)),
        "a_pair_a": a_pair_a,
        "a_pair_b": a_pair_b,
        "a_quad": a_quad,
        "production": np.max(np.abs(np.stack([a_pair_a, a_pair_b, a_quad])), axis=0),
    })


def classify(features: pd.DataFrame, block: int, tau: float, share_min: float = 0.9,
             pair_balance: float = 0.8, quad_balance: float = 0.7) -> np.ndarray:
    """Label each neuron with one of LABELS at activity threshold tau."""
    f = features
    pair_a = ((f.share_singles_a >= share_min) & (f.balance_singles_a >= pair_balance)
              & (f.a_pair_a.abs() >= tau))
    pair_b = ((f.share_singles_b >= share_min) & (f.balance_singles_b >= pair_balance)
              & (f.a_pair_b.abs() >= tau))
    quad = ((f.share_singles >= share_min) & (f.balance_singles >= quad_balance)
            & (f.a_quad.abs() >= tau))
    compose = ((block == 2) & (f.share_parents >= share_min)
               & (f.balance_parents >= pair_balance) & (f.a_quad.abs() >= tau))
    dead = f.production < tau
    return np.select([pair_a | pair_b, quad, compose, dead],
                     ["1+1->2", "1+1+1+1->4", "2+2->4", "dead"], default="other")


def cell_features(path: Path, device: torch.device) -> pd.DataFrame:
    model, metadata = load_model(path, device)
    frames = []
    for block, (z_coef, a_coef) in walsh_coefficients(model).items():
        frame = neuron_features(z_coef, a_coef)
        frame.insert(0, "block", block)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    for key in ("ratio", "lambda0", "seed", "protocol", "cell_id"):
        table.insert(0, key, metadata.get(key, np.nan))
    if pd.isna(table.cell_id.iloc[0]):
        table["cell_id"] = path.stem
    return table


def fractions(table: pd.DataFrame, taus: list[float], **criteria) -> pd.DataFrame:
    rows = []
    for (cell_id, block), group in table.groupby(["cell_id", "block"], sort=False):
        width = len(group)
        first = group.iloc[0]
        for tau in taus:
            labels = classify(group, int(block), tau, **criteria)
            counts = pd.Series(labels).value_counts()
            for label in LABELS:
                count = int(counts.get(label, 0))
                rows.append({
                    "cell_id": cell_id, "lambda0": first.lambda0, "ratio": first.ratio,
                    "block": int(block), "tau": tau, "type": label,
                    "count": count, "rho": count / width,
                })
    return pd.DataFrame(rows)


# Sequential single-hue ramp (light = 0, dark = high), chart ink, and the boundary line.
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BOUNDARY = "#eb6834"
PANELS = ((1, "1+1->2", 0, 0), (1, "1+1+1+1->4", 0, 1),
          (2, "1+1->2", 1, 0), (2, "1+1+1+1->4", 1, 1), (2, "2+2->4", 1, 2))


def log_edges(values: np.ndarray) -> np.ndarray:
    logs = np.log10(values)
    middle = 0.5 * (logs[:-1] + logs[1:])
    return 10 ** np.concatenate([[logs[0] - (middle[0] - logs[0])], middle,
                                 [logs[-1] + (logs[-1] - middle[-1])]])


def type_title(label: str) -> str:
    return label.replace("->", " → ")


def load_phases(grid_dir: Path) -> pd.DataFrame | None:
    """Per-cell phase labels from the grid's results_g*.csv, if present."""
    paths = sorted(grid_dir.glob("results_g*.csv"))
    if not paths:
        return None
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return results[["cell_id", "lambda0", "ratio", "phase"]]


def phase_boundary_segments(phases: pd.DataFrame, lambda0s: np.ndarray, ratios: np.ndarray,
                            x_edges: np.ndarray, y_edges: np.ndarray,
                            pair: tuple[str, str] = ("H_C", "H_D")) -> list:
    """Cell edges shared by an H_C cell and an H_D cell (a staircase on the grid)."""
    labels = (phases.pivot(index="ratio", columns="lambda0", values="phase")
              .reindex(index=ratios, columns=lambda0s).to_numpy())
    wanted = set(pair)
    segments = []
    rows, cols = labels.shape
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols and {labels[i, j], labels[i, j + 1]} == wanted:
                segments.append([(x_edges[j + 1], y_edges[i]), (x_edges[j + 1], y_edges[i + 1])])
            if i + 1 < rows and {labels[i, j], labels[i + 1, j]} == wanted:
                segments.append([(x_edges[j], y_edges[i + 1]), (x_edges[j + 1], y_edges[i + 1])])
    return segments


def plot_threshold(frame: pd.DataFrame, tau_label: str, vmax: dict[int, float], title: str,
                   path_stem: Path, phases: pd.DataFrame | None = None) -> None:
    """Row = block, column = circuit type; one colour scale per block (row).

    Every cell carries its rho value; if phase labels are given, the H_C | H_D
    boundary of the grid is drawn on every panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    lambda0s = np.sort(frame.lambda0.unique())
    ratios = np.sort(frame.ratio.unique())
    x_edges, y_edges = log_edges(lambda0s), log_edges(ratios)
    x_centres = np.sqrt(x_edges[:-1] * x_edges[1:])
    y_centres = np.sqrt(y_edges[:-1] * y_edges[1:])
    segments = (phase_boundary_segments(phases, lambda0s, ratios, x_edges, y_edges)
                if phases is not None else [])
    halo = [path_effects.Stroke(linewidth=4.4, foreground=SURFACE), path_effects.Normal()]

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), sharex=True, sharey=True,
                             facecolor=SURFACE)
    images = {}
    for block, label, row, col in PANELS:
        ax = axes[row, col]
        sub = frame[(frame.block == block) & (frame.type == label)]
        grid = (sub.pivot_table(index="ratio", columns="lambda0", values="rho")
                .reindex(index=ratios, columns=lambda0s).to_numpy())
        norm = Normalize(0.0, vmax[block])
        images[block] = ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, norm=norm,
                                      shading="flat", edgecolors=SURFACE, linewidth=0.6)
        for i, y in enumerate(y_centres):
            for j, x in enumerate(x_centres):
                value = grid[i, j]
                if np.isnan(value):
                    continue
                red, green, blue, _ = cmap(norm(value))
                dark_cell = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                ink = "white" if dark_cell else (INK_MUTED if value == 0 else INK_PRIMARY)
                ax.text(x, y, "0" if value == 0 else f"{value:.2f}", ha="center",
                        va="center", fontsize=6.6, color=ink)
        if segments:
            boundary = LineCollection(segments, colors=BOUNDARY, linewidths=2.2,
                                      capstyle="round", joinstyle="round", zorder=3)
            boundary.set_path_effects(halo)
            ax.add_collection(boundary)
        ax.set_title(f"block {block}:  ρ({type_title(label)})", fontsize=10,
                     color=INK_PRIMARY, loc="left")
    corner = axes[0, 2]
    corner.axis("off")
    corner.text(0.0, 0.95, "2 + 2 → 4 cannot occur in\nblock 1: its pre-activation\n"
                "z₁ = V₁E x is linear in x.\n\n"
                "Columns of 0 at large λ₀: suppressed\nphase (the network collapsed).",
                fontsize=9, color=INK_SECONDARY, transform=corner.transAxes, va="top")
    if segments:
        corner.legend([Line2D([], [], color=BOUNDARY, lw=2.2, path_effects=halo)],
                      ["phase boundary of this grid:\nH_D above, H_C below"],
                      loc="lower left", bbox_to_anchor=(-0.02, 0.0), frameon=False,
                      fontsize=9, labelcolor=INK_SECONDARY, handlelength=2.2)
    for ax in axes.flat:
        if not ax.axison:
            continue
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=8, colors=INK_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("common decay scale λ₀", fontsize=9, color=INK_SECONDARY)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("block-1 decay ratio λ₁/λ₂", fontsize=9,
                      color=INK_SECONDARY)
    for block, row_axes in ((1, axes[0, :2]), (2, axes[1, :])):
        colorbar = fig.colorbar(images[block], ax=list(row_axes) + ([axes[0, 2]] if block == 1 else []),
                                shrink=0.9, pad=0.015, aspect=18)
        colorbar.set_label(f"fraction of block-{block} neurons", fontsize=9,
                           color=INK_SECONDARY)
        colorbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
        colorbar.outline.set_visible(False)
    fig.suptitle(f"{title}  —  activity threshold {tau_label}", fontsize=11,
                 color=INK_PRIMARY, x=0.07, ha="left")
    # Append extensions explicitly: with_suffix would eat the ".0001" of "rho_tau_0.0001".
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(f"{path_stem}.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-dir", required=True, type=Path,
                        help="directory with checkpoints/<cell_id>.pt")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--protocol", default="parent_warm")
    parser.add_argument("--taus", default="1e-6,1e-4,1e-3,1e-2",
                        help="activity thresholds on |a_mode| (activation units)")
    parser.add_argument("--share-min", type=float, default=0.9)
    parser.add_argument("--pair-balance", type=float, default=0.8)
    parser.add_argument("--quad-balance", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    taus = [float(item) for item in args.taus.split(",") if item.strip()]
    device = torch.device(args.device)
    paths = sorted((args.grid_dir / "checkpoints").glob(f"{args.protocol}_*.pt"))
    if not paths:
        raise SystemExit(f"no {args.protocol}_*.pt checkpoints under {args.grid_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.concat([cell_features(path, device) for path in paths], ignore_index=True)
    table.to_csv(args.out_dir / "per_neuron.csv.gz", index=False)
    criteria = dict(share_min=args.share_min, pair_balance=args.pair_balance,
                    quad_balance=args.quad_balance)
    result = fractions(table, taus, **criteria)
    result.to_csv(args.out_dir / "fractions.csv", index=False)

    typed = result[result.type.isin(TYPES)]
    # One scale per block, shared by every threshold so the figures compare directly.
    vmax = {block: max(0.05, float(np.ceil(typed[typed.block == block].rho.max() * 20) / 20))
            for block in (1, 2)}
    seeds = sorted(table.seed.dropna().unique())
    title = (f"Neuron circuit types, {args.protocol.replace('_', '-')}"
             f" (seed {', '.join(str(int(s)) for s in seeds)}, {len(paths)} cells)")
    phases = load_phases(args.grid_dir)
    if phases is not None:
        phases = phases[phases.cell_id.isin(table.cell_id.unique())]
    for tau in taus:
        plot_threshold(result[result.tau == tau], f"|a| ≥ {tau:g}", vmax, title,
                       args.out_dir / f"rho_tau_{tau:.0e}", phases)
    print(f"{len(paths)} checkpoints -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
