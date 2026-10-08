"""Plot the stationary two-block phase diagram used by the parent-only report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "two_block_theory" / "fig_two_block_parent_only_layerwise_phase.pdf"

GAMMA2 = GAMMA4 = 2.0
S_GRID = np.geomspace(2e-3, 6.0, 180)[:, None]
Q_GRID = np.geomspace(2e-3, 6.0, 180)[None, :]


def target_free_energy(kappa, lambda_w4):
    result = np.full_like(kappa, GAMMA4 / 2)
    scaled_cost = kappa * lambda_w4
    active = scaled_cost < GAMMA4**2
    result[active] = (
        np.sqrt(scaled_cost[active])
        - scaled_cost[active] / (2 * GAMMA4)
    )
    return result


def solve_cell(lambda0, ratio):
    """Return phase, accessibility, target amplitude, and parent amplitude."""
    lambda_e = lambda_p = lambda_w4 = lambda_2 = lambda0
    lambda_1 = ratio * lambda0

    phi = (
        2 * lambda_e * S_GRID**2
        + 8 * lambda_1 * Q_GRID**2 / S_GRID**2
        + GAMMA2 * lambda_p / (GAMMA2 * Q_GRID**2 + lambda_p)
    )
    kappa_d = 64 * lambda_2 / (S_GRID**2 + Q_GRID**2)
    kappa_c = np.broadcast_to(8 * lambda_2 / Q_GRID**2, phi.shape)

    free_d = phi + target_free_energy(kappa_d, lambda_w4)
    free_c = phi + target_free_energy(kappa_c, lambda_w4)
    idx_d = np.unravel_index(np.argmin(free_d), free_d.shape)
    idx_c = np.unravel_index(np.argmin(free_c), free_c.shape)
    value_d, value_c = float(free_d[idx_d]), float(free_c[idx_c])
    compositional = value_c < value_d
    idx = idx_c if compositional else idx_d
    kappa = float((kappa_c if compositional else kappa_d)[idx])
    s, q = float(S_GRID[idx[0], 0]), float(Q_GRID[0, idx[1]])
    accessibility = q / np.sqrt(s**2 + q**2)
    amplitude = max(0.0, 1 - np.sqrt(kappa * lambda_w4) / GAMMA4)
    parent_amplitude = GAMMA2 * q**2 / (GAMMA2 * q**2 + lambda_p)
    if amplitude < 2e-3 and parent_amplitude < 2e-3:
        return 3, accessibility, amplitude, parent_amplitude
    if amplitude < 2e-3:
        return 2, accessibility, amplitude, parent_amplitude
    return (0 if compositional else 1), accessibility, amplitude, parent_amplitude


lambda_values = np.geomspace(1e-4, 1.0, 82)
ratio_values = np.geomspace(0.05, 20.0, 76)
phase = np.empty((len(ratio_values), len(lambda_values)), dtype=int)
accessibility = np.empty_like(phase, dtype=float)

for iy, ratio in enumerate(ratio_values):
    for ix, lambda0 in enumerate(lambda_values):
        phase[iy, ix], accessibility[iy, ix], _, _ = solve_cell(lambda0, ratio)

fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
fig.subplots_adjust(left=0.085, right=0.975, bottom=0.18, top=0.88, wspace=0.28)

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

phase_colors = ["#2a9d8f", "#e9c46a", "#8d99ae", "#343a40"]
phase_cmap = ListedColormap(phase_colors)
phase_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], phase_cmap.N)
axes[0].pcolormesh(
    x_edges, y_edges, phase, cmap=phase_cmap,
    norm=phase_norm, shading="flat", rasterized=True,
)
axes[0].axhline(1, color="white", lw=0.9, ls="--")
axes[0].text(
    1.3e-4, 1.12, r"equal decay $\lambda_1=\lambda_2$",
    fontsize=6.6, color="white", va="bottom",
)
axes[0].set_title("(a) Stationary phase", fontsize=8.5)
axes[0].legend(
    handles=[
        Patch(color=phase_colors[0], label=r"$C: 2+2\to4$"),
        Patch(color=phase_colors[1], label=r"$D: 1+1+1+1\to4$"),
        Patch(color=phase_colors[2], label="parents only"),
        Patch(color=phase_colors[3], label="suppressed"),
    ],
    loc="upper right", fontsize=5.8, frameon=True, handlelength=1.0,
    borderpad=0.25, labelspacing=0.25,
)

heat = axes[1].pcolormesh(
    x_edges, y_edges, accessibility, cmap="viridis",
    vmin=0, vmax=1, shading="flat", rasterized=True,
)
axes[1].contour(
    lambda_values, ratio_values, accessibility,
    levels=[1 / np.sqrt(8)], colors="white", linewidths=1.0,
)
axes[1].set_title(r"(b) Selected accessibility $R=Q_2/Q_1^{(1)}$", fontsize=8.5)
cbar = fig.colorbar(heat, ax=axes[1], fraction=0.047, pad=0.025, ticks=[0, 0.5, 1])
cbar.ax.tick_params(labelsize=6)

for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lambda_values[0], lambda_values[-1])
    ax.set_ylim(ratio_values[0], ratio_values[-1])
    ax.set_xlabel(r"overall decay $\lambda_0$", fontsize=7.5)
    ax.tick_params(labelsize=6.5, length=2.5)
axes[0].set_ylabel(r"first/second-block decay $\lambda_1/\lambda_2$", fontsize=7.5)
axes[1].set_yticklabels([])

fig.savefig(OUT, bbox_inches="tight")
print(OUT)
