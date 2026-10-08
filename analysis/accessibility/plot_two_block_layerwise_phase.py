"""Plot the stationary two-block phase diagram used by the concise theory report.

The plotted closure assumes balanced orthogonal Walsh modes, an orthogonal
singleton byproduct from each parent gate, aligned target features, and a
zero-temperature global minimum.  See the report for the reduced free energy.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "two_block_theory" / "fig_two_block_layerwise_phase.pdf"

GAMMA2 = GAMMA4 = 2.0
S_GRID = np.geomspace(2e-3, 6.0, 180)[:, None]
Q_GRID = np.geomspace(2e-3, 6.0, 180)[None, :]
F_ZERO = GAMMA2 + GAMMA4 / 2


def target_free_energy(zeta):
    result = np.full_like(zeta, GAMMA4 / 2)
    active = GAMMA4 * zeta > 1
    result[active] = (1 / zeta[active]
                      - 1 / (2 * GAMMA4 * zeta[active] ** 2))
    return result


def solve_cell(lambda0, ratio):
    """Return phase, f_2+2, R, A, B for one decay-scale/ratio cell."""
    lambda_e = lambda_p = lambda_w4 = lambda_2 = lambda0
    lambda_1 = ratio * lambda0

    phi = (2 * lambda_e * S_GRID ** 2
           + 8 * lambda_1 * Q_GRID ** 2 / S_GRID ** 2
           + GAMMA2 * lambda_p / (GAMMA2 * Q_GRID ** 2 + lambda_p))
    zeta_d = np.sqrt(
        (S_GRID ** 2 / 64) / lambda_1
        + ((S_GRID ** 2 + Q_GRID ** 2) / 64) / lambda_2
    ) / np.sqrt(lambda_w4)
    zeta_c = np.sqrt(
        (S_GRID ** 2 / 64) / lambda_1
        + (Q_GRID ** 2 / 8) / lambda_2
    ) / np.sqrt(lambda_w4)

    free_d = phi + target_free_energy(zeta_d)
    free_c = phi + target_free_energy(zeta_c)
    idx_d = np.unravel_index(np.argmin(free_d), free_d.shape)
    idx_c = np.unravel_index(np.argmin(free_c), free_c.shape)
    value_d, value_c = float(free_d[idx_d]), float(free_c[idx_c])
    if min(value_d, value_c) >= F_ZERO - 2e-4:
        return 3, 0.0, 0.0, 0.0, 0.0

    compositional = value_c < value_d
    idx = idx_c if compositional else idx_d
    zeta = float((zeta_c if compositional else zeta_d)[idx])
    s, q = float(S_GRID[idx[0], 0]), float(Q_GRID[0, idx[1]])
    amplitude = max(0.0, 1 - 1 / (GAMMA4 * zeta))
    parent_amplitude = GAMMA2 * q ** 2 / (GAMMA2 * q ** 2 + lambda_p)
    if amplitude < 2e-3:
        return 2, 0.0, q / np.sqrt(s ** 2 + q ** 2), amplitude, parent_amplitude
    if not compositional:
        return 1, 0.0, q / np.sqrt(s ** 2 + q ** 2), amplitude, parent_amplitude

    frac_c = ((q ** 2 / (8 * lambda_2))
              / (s ** 2 / (64 * lambda_1) + q ** 2 / (8 * lambda_2)))
    return 0, frac_c, q / np.sqrt(s ** 2 + q ** 2), amplitude, parent_amplitude


lambda_values = np.geomspace(1e-4, 1.0, 82)
ratio_values = np.geomspace(0.05, 20.0, 76)
phase = np.empty((len(ratio_values), len(lambda_values)), dtype=int)
frac = np.empty_like(phase, dtype=float)

for iy, ratio in enumerate(ratio_values):
    for ix, lambda0 in enumerate(lambda_values):
        phase[iy, ix], frac[iy, ix], _, _, _ = solve_cell(lambda0, ratio)

fig = plt.figure(figsize=(7.15, 5.35))
grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72],
                        left=0.075, right=0.985, bottom=0.12, top=0.94,
                        wspace=0.30, hspace=0.52)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :])]

x_edges = np.geomspace(lambda_values[0] / np.sqrt(lambda_values[1] / lambda_values[0]),
                       lambda_values[-1] * np.sqrt(lambda_values[-1] / lambda_values[-2]),
                       len(lambda_values) + 1)
y_edges = np.geomspace(ratio_values[0] / np.sqrt(ratio_values[1] / ratio_values[0]),
                       ratio_values[-1] * np.sqrt(ratio_values[-1] / ratio_values[-2]),
                       len(ratio_values) + 1)

phase_colors = ["#2a9d8f", "#e9c46a", "#8d99ae", "#343a40"]
phase_cmap = ListedColormap(phase_colors)
phase_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], phase_cmap.N)
axes[0].pcolormesh(x_edges, y_edges, phase, cmap=phase_cmap,
                   norm=phase_norm, shading="flat", rasterized=True)
axes[0].axhline(1, color="white", lw=0.9, ls="--")
axes[0].text(1.3e-4, 1.12, r"equal decay $\lambda_1=\lambda_2$",
             fontsize=6.6, color="white", va="bottom")
axes[0].set_title("(a) Stationary phase", fontsize=8.5)
axes[0].legend(handles=[Patch(color=phase_colors[0], label=r"$H_C=D_1+C$"),
                       Patch(color=phase_colors[1], label=r"$H_D=D_1+D_2$"),
                       Patch(color=phase_colors[2], label="parents only"),
                       Patch(color=phase_colors[3], label="suppressed")],
               loc="upper right", fontsize=5.8, frameon=True, handlelength=1.0,
               borderpad=0.25, labelspacing=0.25)

frac_masked = np.ma.masked_where(phase != 0, frac)
heat = axes[1].pcolormesh(x_edges, y_edges, frac_masked, cmap="viridis",
                          vmin=0, vmax=1, shading="flat", rasterized=True)
axes[1].contour(lambda_values, ratio_values, frac_masked,
                levels=[0.5, 0.75, 0.9], colors="white", linewidths=0.65)
axes[1].set_title(r"(b) Total $2+2\to4$ share $f_C$", fontsize=8.5)
cbar = fig.colorbar(heat, ax=axes[1], fraction=0.047, pad=0.025,
                    ticks=[0, 0.5, 1])
cbar.ax.tick_params(labelsize=6)

for ax in axes[:2]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lambda_values[0], lambda_values[-1])
    ax.set_ylim(ratio_values[0], ratio_values[-1])
    ax.set_xlabel(r"overall decay $\lambda_0$", fontsize=7.5)
    ax.tick_params(labelsize=6.5, length=2.5)
axes[0].set_ylabel(r"first/second-block decay $\lambda_1/\lambda_2$", fontsize=7.5)
axes[1].set_yticklabels([])

decay_ratios = np.geomspace(0.05, 20, 200)
for accessibility, style in [(0.36, "-"), (0.45, "--"), (0.60, ":")]:
    t2 = accessibility ** 2 / (1 - accessibility ** 2)
    curve = 8 * decay_ratios * t2 / (1 + 8 * decay_ratios * t2)
    axes[2].plot(decay_ratios, curve, style, lw=1.25,
                 label=rf"$R={accessibility:.2f}$")
axes[2].axhline(0.5, color="0.55", lw=0.7)
axes[2].set_xscale("log")
axes[2].set_xlim(decay_ratios[0], decay_ratios[-1])
axes[2].set_ylim(0, 1.02)
axes[2].set_xlabel(r"$\lambda_1/\lambda_2$", fontsize=7.5)
axes[2].set_ylabel(r"$f_C$ at fixed $R$", fontsize=7.5)
axes[2].set_title("(c) Purity control", fontsize=8.5)
axes[2].tick_params(labelsize=6.5, length=2.5)
axes[2].legend(fontsize=6.2, loc="lower right", frameon=False)
axes[2].grid(alpha=0.18, lw=0.5)

fig.savefig(OUT, bbox_inches="tight")
print(OUT)
