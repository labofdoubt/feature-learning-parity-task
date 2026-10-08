"""Squared-amplitude block-2 route purity and p_j distributions.

This script is deliberately self-contained: it needs only NumPy, Matplotlib,
and PyTorch.  For each block-2 neuron, p_j and s_j are the canonical-parent and
singleton fractions of nonconstant preactivation Walsh power.  If
a_j = w4_j * hat{b2_j}({1,2,3,4}) is its readout-weighted d4 amplitude, then

    P_C     = sum_j a_j**2 p_j,
    P_D     = sum_j a_j**2 s_j,
    P_other = sum_j a_j**2 (1-p_j-s_j).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SINGLES = ((0,), (1,), (2,), (3,))
PARENTS = ((0, 1), (2, 3))
QUAD = (0, 1, 2, 3)
NONEMPTY_SUBSETS = tuple(
    subset
    for degree in range(1, 5)
    for subset in itertools.combinations(range(4), degree)
)
MODE = {support: index for index, support in enumerate(NONEMPTY_SUBSETS)}
EPS = 1e-30

INK = "#171717"
GRID = "#d6d3d1"
HC_COLOR = "#168f83"
HD_COLOR = "#d99a24"


def build_cube(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [[1.0 if (mask >> bit) & 1 == 0 else -1.0 for bit in range(4)]
         for mask in range(16)],
        dtype=dtype,
    )
    chi = torch.stack(
        [x[:, list(support)].prod(dim=1) for support in NONEMPTY_SUBSETS], dim=1
    )
    return x, chi


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = {name: value.to(dtype=torch.float64) for name, value in payload["state_dict"].items()}
    return state, dict(payload.get("metadata", {}))


@torch.no_grad()
def analyze_checkpoint(path: Path, x: torch.Tensor, chi: torch.Tensor) -> tuple[dict, dict[str, np.ndarray]]:
    state, metadata = load_checkpoint(path)
    e, v1, v2 = state["E"], state["V1"], state["V2"]
    wp, w4 = state["Wp"], state["w4"]
    h0 = x @ e.T
    b1 = torch.relu(h0 @ v1.T)
    h1 = h0 + b1
    z2 = h1 @ v2.T
    b2 = torch.relu(z2)
    h2 = h1 + b2
    pred2 = h2 @ wp.T
    pred4 = h2 @ w4

    count = x.shape[0]
    z_coef = z2.T @ chi / count
    target = chi[:, MODE[QUAD]]
    q4_h1 = (h1 * target[:, None]).mean(dim=0)
    q4_b2 = (b2 * target[:, None]).mean(dim=0)

    z_power = z_coef.square()
    total_input_power = z_power.sum(dim=1)
    singleton_power = z_power[:, [MODE[s] for s in SINGLES]].sum(dim=1)
    parent_power = z_power[:, [MODE[s] for s in PARENTS]].sum(dim=1)
    p_j = parent_power / (total_input_power + EPS)
    s_j = singleton_power / (total_input_power + EPS)
    other_j = torch.clamp(1.0 - p_j - s_j, min=0.0)

    output_amplitude = w4 * q4_b2
    output_power = output_amplitude.square()
    p_c = float((output_power * p_j).sum())
    p_d = float((output_power * s_j).sum())
    p_other = float((output_power * other_j).sum())
    p_total = p_c + p_d + p_other

    c1_neuron = w4 * q4_h1
    c2_neuron = output_amplitude
    c_abs = float((c2_neuron.abs() * p_j).sum())
    d_abs = float((c2_neuron.abs() * s_j).sum())
    other_abs = float((c2_neuron.abs() * other_j).sum())
    d1_abs = float(c1_neuron.abs().sum())
    block2_abs = c_abs + d_abs + other_abs
    total_abs = d1_abs + block2_abs
    block2_abs_share = block2_abs / max(total_abs, EPS)
    block2_pair_purity_abs = c_abs / max(block2_abs, EPS)

    y2 = chi[:, [MODE[s] for s in PARENTS]]
    a2a = float((pred2[:, 0] * y2[:, 0]).mean())
    a2b = float((pred2[:, 1] * y2[:, 1]).mean())
    a4 = float((pred4 * target).mean())
    parents_active = min(a2a, a2b) >= 0.5
    target_active = a4 >= 0.5
    if not target_active:
        phase = "parents_only" if parents_active else "suppressed"
    elif block2_abs_share < 0.05:
        phase = "H_D"
    else:
        phase = "H_C" if block2_pair_purity_abs >= 0.5 else "H_D"

    block2_active = target_active and block2_abs_share >= 0.05 and p_total > 1e-30
    cell = {
        "cell_id": metadata.get("cell_id", path.stem),
        "seed": int(metadata.get("seed", 0)),
        "protocol": metadata.get("protocol", "parent_warm"),
        "lambda0": float(metadata["lambda0"]),
        "ratio": float(metadata["ratio"]),
        "phase": phase,
        "A2a": a2a,
        "A2b": a2b,
        "A4": a4,
        "block2_abs_share": block2_abs_share,
        "block2_pair_purity_abs": block2_pair_purity_abs,
        "P_C": p_c,
        "P_D": p_d,
        "P_other": p_other,
        "P_total": p_total,
        "block2_pair_purity_squared": p_c / max(p_total, 1e-300),
        "block2_active": block2_active,
    }
    neurons = {
        "p_j": p_j.cpu().numpy(),
        "s_j": s_j.cpu().numpy(),
        "other_j": other_j.cpu().numpy(),
        "var_z": total_input_power.cpu().numpy(),
        "a_quad": q4_b2.cpu().numpy(),
        "w4": w4.cpu().numpy(),
        "output_amplitude": output_amplitude.cpu().numpy(),
        "output_power": output_power.cpu().numpy(),
    }
    return cell, neurons


def log_edges(values: np.ndarray) -> np.ndarray:
    logs = np.log10(values)
    middle = 0.5 * (logs[:-1] + logs[1:])
    return 10 ** np.concatenate(
        [[logs[0] - (middle[0] - logs[0])], middle,
         [logs[-1] + (logs[-1] - middle[-1])]]
    )


def plot_heatmap(cells: list[dict], output: Path) -> None:
    lambda0s = np.asarray(sorted({row["lambda0"] for row in cells}))
    ratios = np.asarray(sorted({row["ratio"] for row in cells}))
    index_x = {value: i for i, value in enumerate(lambda0s)}
    index_y = {value: i for i, value in enumerate(ratios)}
    values = np.full((len(ratios), len(lambda0s)), np.nan)
    active = np.zeros_like(values, dtype=bool)
    for row in cells:
        iy, ix = index_y[row["ratio"]], index_x[row["lambda0"]]
        values[iy, ix] = row["block2_pair_purity_squared"]
        active[iy, ix] = row["block2_active"]
    shown = np.ma.masked_where(~active, values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d6d3d1")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    image = ax.pcolormesh(log_edges(lambda0s), log_edges(ratios), shown,
                          cmap=cmap, vmin=0, vmax=1, shading="flat")
    if shown.count() and float(shown.min()) <= 0.5 <= float(shown.max()):
        ax.contour(lambda0s, ratios, shown, levels=[0.5], colors="white",
                   linewidths=1.35)
    for iy, ratio in enumerate(ratios):
        for ix, lambda0 in enumerate(lambda0s):
            if not active[iy, ix]:
                continue
            value = values[iy, ix]
            ax.text(lambda0, ratio, f"{value:.2f}", ha="center", va="center",
                    fontsize=6.1, color="white" if value < 0.42 else INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"common decay scale $\lambda_0$")
    ax.set_ylabel(r"relative block-1 decay $\lambda_1/\lambda_2$")
    ax.set_title("Parent-warm seed 0: squared block-2 pair purity")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.025)
    colorbar.set_label(r"$P_C/(P_C+P_D+P_{\rm other})$")
    ax.text(0.01, -0.17,
            r"White: purity $=0.5$. Gray: inactive target or block-2 share $<5\%$.",
            transform=ax.transAxes, fontsize=8, color="#525252")
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(records: list[tuple[dict, dict[str, np.ndarray]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), sharex=True)
    bins = np.linspace(0, 1, 51)
    for phase, label, color in (("H_C", r"$H_C$ cells", HC_COLOR),
                                ("H_D", r"$H_D$ cells", HD_COLOR)):
        selected = [(cell, neurons) for cell, neurons in records
                    if cell["block2_active"] and cell["phase"] == phase]
        if not selected:
            continue
        p_values = np.concatenate([n["p_j"][n["var_z"] > 1e-20] for _, n in selected])
        axes[0].hist(p_values, bins=bins, density=True, histtype="step",
                     linewidth=1.8, color=color,
                     label=f"{label} ({len(p_values):,} neurons)")
        p_power = np.concatenate([n["p_j"] for _, n in selected])
        weights = np.concatenate([n["output_power"] for _, n in selected])
        weights = weights / max(float(weights.sum()), 1e-300)
        axes[1].hist(p_power, bins=bins, weights=weights, histtype="step",
                     linewidth=1.8, color=color, label=label)

    axes[0].set_title(r"Unweighted distribution of $p_j$")
    axes[0].set_ylabel("probability density")
    axes[1].set_title(r"Weighted by squared $d4$ output")
    axes[1].set_ylabel("fraction of block-2 d4 power")
    for ax in axes:
        ax.set_xlabel(r"$p_j$: canonical-parent fraction of preactivation power")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Parent-warm seed 0: block-2 neuron input composition", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)


def write_tables(records: list[tuple[dict, dict[str, np.ndarray]]], out_dir: Path) -> None:
    cells = [cell for cell, _ in records]
    with (out_dir / "cell_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(sorted(cells, key=lambda row: (row["ratio"], row["lambda0"])))

    neuron_fields = [
        "cell_id", "seed", "protocol", "lambda0", "ratio", "phase", "block2_active",
        "neuron", "p_j", "s_j", "other_j", "var_z", "a_quad", "w4",
        "output_amplitude", "output_power",
    ]
    with gzip.open(out_dir / "per_neuron.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=neuron_fields)
        writer.writeheader()
        for cell, neurons in records:
            width = len(neurons["p_j"])
            for j in range(width):
                writer.writerow({
                    "cell_id": cell["cell_id"], "seed": cell["seed"],
                    "protocol": cell["protocol"], "lambda0": cell["lambda0"],
                    "ratio": cell["ratio"], "phase": cell["phase"],
                    "block2_active": cell["block2_active"], "neuron": j,
                    **{name: float(values[j]) for name, values in neurons.items()},
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir", type=Path,
        default=Path("runs/two_block_phase_pw_seed0/analysis/phase_grid"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("runs/two_block_phase_pw_seed0/analysis/squared_purity"),
    )
    args = parser.parse_args()
    paths = sorted((args.grid_dir / "checkpoints").glob("parent_warm_s0_*.pt"))
    if not paths:
        raise SystemExit(f"No parent-warm seed-0 checkpoints under {args.grid_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, chi = build_cube()
    records = [analyze_checkpoint(path, x, chi) for path in paths]
    write_tables(records, args.out_dir)
    plot_heatmap([cell for cell, _ in records], args.out_dir / "squared_pair_purity_heatmap.png")
    plot_distribution(records, args.out_dir / "p_j_distribution.png")
    print(f"{len(paths)} checkpoints -> {args.out_dir}")


if __name__ == "__main__":
    main()
