"""Plot the stationary d4-parent/d8-target two-block phase diagram.

The reduced theory compares a direct singleton route (eight degree-1 modes to
degree 8) with a compositional route (two supervised degree-4 modes to degree
8).  See reports/two_block_theory/two_block_d4_d8_layerwise_decay_phase_theory.tex.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "reports" / "two_block_theory" / "fig_two_block_d4_d8_layerwise_phase.pdf"
OUT_PNG = ROOT / "reports" / "two_block_theory" / "fig_two_block_d4_d8_layerwise_phase.png"

GAMMA4 = GAMMA8 = 2.0
S_GRID = np.geomspace(2e-3, 8.0, 200)[:, None]
Q_GRID = np.geomspace(2e-3, 8.0, 200)[None, :]
F_ZERO = GAMMA4 + GAMMA8 / 2


def target_free_energy(zeta: np.ndarray) -> np.ndarray:
    result = np.full_like(zeta, GAMMA8 / 2)
    active = GAMMA8 * zeta > 1
    result[active] = (
        1 / zeta[active] - 1 / (2 * GAMMA8 * zeta[active] ** 2)
    )
    return result


def free_energy_surfaces(lambda0: float, ratio: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """F_D, F_C of Eq. (free8) and their zeta_D, zeta_C on the (s, q) grid."""
    lambda_e = lambda_p = lambda_w8 = lambda_2 = lambda0
    lambda_1 = ratio * lambda0

    # Two direct 1+1+1+1->4 parents.  A d4 parent of norm q costs
    # 64 q^2/s^2 in hidden norm; two parents and the 1/2 penalty give 64 lambda1 q^2/s^2.
    phi4 = (
        4 * lambda_e * S_GRID ** 2
        + 64 * lambda_1 * Q_GRID ** 2 / S_GRID ** 2
        + GAMMA4 * lambda_p / (GAMMA4 * Q_GRID ** 2 + lambda_p)
    )

    # Direct d8 efficiency is 5/(32 sqrt(8)) times the singleton norm.
    # Each direct d4 parent emits singleton leakage of norm 2q, hence
    # Q1_out^2 = s^2 + 4q^2 under the orthogonal-leakage assumption.
    k1 = 25 * S_GRID ** 2 / (8192 * lambda_1)
    kd2 = 25 * (S_GRID ** 2 + 4 * Q_GRID ** 2) / (8192 * lambda_2)
    kc = Q_GRID ** 2 / (8 * lambda_2)
    zeta_d = np.sqrt((k1 + kd2) / lambda_w8)
    zeta_c = np.sqrt((k1 + kc) / lambda_w8)

    return (phi4 + target_free_energy(zeta_d), phi4 + target_free_energy(zeta_c),
            zeta_d, zeta_c)


def solve_cell(lambda0: float, ratio: float) -> tuple[int, float, float, float, float]:
    """Return phase, f_44, R4, d8 amplitude, and d4-parent amplitude."""
    lambda_p = lambda_2 = lambda0
    lambda_1 = ratio * lambda0
    free_d, free_c, zeta_d, zeta_c = free_energy_surfaces(lambda0, ratio)
    idx_d = np.unravel_index(np.argmin(free_d), free_d.shape)
    idx_c = np.unravel_index(np.argmin(free_c), free_c.shape)
    value_d, value_c = float(free_d[idx_d]), float(free_c[idx_c])
    if min(value_d, value_c) >= F_ZERO - 2e-4:
        return 3, 0.0, 0.0, 0.0, 0.0

    compositional = value_c < value_d
    idx = idx_c if compositional else idx_d
    zeta = float((zeta_c if compositional else zeta_d)[idx])
    s, q = float(S_GRID[idx[0], 0]), float(Q_GRID[0, idx[1]])
    amplitude8 = max(0.0, 1 - 1 / (GAMMA8 * zeta))
    amplitude4 = GAMMA4 * q ** 2 / (GAMMA4 * q ** 2 + lambda_p)
    accessibility = q / np.sqrt(s ** 2 + 4 * q ** 2)
    if amplitude8 < 2e-3:
        return 2, 0.0, accessibility, amplitude8, amplitude4
    if not compositional:
        return 1, 0.0, accessibility, amplitude8, amplitude4

    k1_value = 25 * s ** 2 / (8192 * lambda_1)
    kc_value = q ** 2 / (8 * lambda_2)
    fraction = kc_value / (k1_value + kc_value)
    return 0, float(fraction), accessibility, amplitude8, amplitude4


def main() -> None:
    lambda_values = np.geomspace(1e-4, 1.0, 82)
    ratio_values = np.geomspace(0.05, 20.0, 76)
    phase = np.empty((len(ratio_values), len(lambda_values)), dtype=int)
    fraction = np.empty_like(phase, dtype=float)

    for iy, decay_ratio in enumerate(ratio_values):
        for ix, lambda0 in enumerate(lambda_values):
            phase[iy, ix], fraction[iy, ix], _, _, _ = solve_cell(
                lambda0, decay_ratio
            )

    x_edges = np.geomspace(
        lambda_values[0] / np.sqrt(lambda_values[1] / lambda_values[0]),
        lambda_values[-1] * np.sqrt(lambda_values[-1] / lambda_values[-2]),
        len(lambda_values) + 1,
    )
    y_edges = np.geomspace(
        ratio_values[0] / np.sqrt(ratio_values[1] / ratio_values[0]),
        ratio_values[-1] * np.sqrt(ratio_values[-1] / ratio_values[-2]),
        len(ratio_values) + 1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    phase_colors = ["#2a9d8f", "#e9c46a", "#8d99ae", "#343a40"]
    phase_cmap = ListedColormap(phase_colors)
    phase_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], phase_cmap.N)

    axes[0].pcolormesh(
        x_edges, y_edges, phase, cmap=phase_cmap, norm=phase_norm,
        shading="flat", rasterized=True,
    )
    axes[0].axhline(1, color="white", lw=0.9, ls="--")
    axes[0].text(
        1.3e-4, 1.12, r"equal decay $\lambda_1=\lambda_2$",
        fontsize=6.6, color="white", va="bottom",
    )
    axes[0].set_title("(a) Stationary phase", fontsize=9)
    axes[0].legend(
        handles=[
            Patch(color=phase_colors[0], label=r"$H_C=D_1+C_{44}$"),
            Patch(color=phase_colors[1], label=r"$H_D=D_1+D_2$"),
            Patch(color=phase_colors[2], label=r"$d4$ parents only"),
            Patch(color=phase_colors[3], label="suppressed"),
        ],
        loc="upper right", fontsize=6.0, frameon=True, handlelength=1.0,
        borderpad=0.25, labelspacing=0.25,
    )

    fraction_masked = np.ma.masked_where(phase != 0, fraction)
    heat = axes[1].pcolormesh(
        x_edges, y_edges, fraction_masked, cmap="viridis", vmin=0, vmax=1,
        shading="flat", rasterized=True,
    )
    axes[1].contour(
        lambda_values, ratio_values, fraction_masked,
        levels=[0.6, 0.75, 0.85], colors="white", linewidths=0.7,
    )
    axes[1].set_title(r"(b) Total $4+4\to8$ share $f_{44}$", fontsize=9)
    cbar = fig.colorbar(heat, ax=axes[1], fraction=0.047, pad=0.025,
                        ticks=[0, 0.5, 1])
    cbar.ax.tick_params(labelsize=6.5)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lambda_values[0], lambda_values[-1])
        ax.set_ylim(ratio_values[0], ratio_values[-1])
        ax.set_xlabel(r"overall decay $\lambda_0$", fontsize=8)
        ax.tick_params(labelsize=7, length=2.5)
    axes[0].set_ylabel(
        r"first/second-block decay $\lambda_1/\lambda_2$", fontsize=8
    )

    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.16, top=0.89, wspace=0.22)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=210, bbox_inches="tight")
    print(OUT_PDF)


if __name__ == "__main__":
    main()
