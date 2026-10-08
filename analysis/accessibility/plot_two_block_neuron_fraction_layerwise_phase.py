"""Plot the neuron-population extension of the two-block phase theory.

The upper-left panel restricts the occupation-aware free energy to zero
first-block 4x1 occupation.  The remaining panels make both first-block
circuit families and all three second-block circuit families available.
Ordinary L2 decay removes literal neuron fractions from the free energy;
positive fractions only make a route available.  See the report for the
derivation and interpretation.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "two_block_theory" / "fig_two_block_neuron_fraction_phase.pdf"

GAMMA2 = GAMMA4 = 2.0
LAMBDA_VALUES = np.geomspace(1e-4, 1.0, 62)
RATIO_VALUES = np.geomspace(0.05, 20.0, 58)
TARGET_ACTIVE_TOL = 2e-3


def optimize_occupied_theory(seed=7, restarts=18, steps=3100, allow_block1_d4=True):
    """Batched Adam minimization of F_C and F_D over (s,q,r,p,Q).

    Dimensions are restart x route x lambda1/lambda2 x lambda0 x variable.
    Multiple deliberately different initializations are used because the
    reduced free energy has boundary minima and competing basins.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    ny, nx = len(RATIO_VALUES), len(LAMBDA_VALUES)
    l0 = torch.as_tensor(LAMBDA_VALUES)[None, None, None, :]
    ratio = torch.as_tensor(RATIO_VALUES)[None, None, :, None]

    base = torch.empty((restarts, 2, ny, nx, 5))
    base.uniform_(-5.5, 0.7)
    templates = [
        [0.45, 0.45, 1e-5, 1e-5, 0.45],  # parent + block-2 target
        [0.70, 0.20, 1e-5, 0.45, 0.35],  # block-2 parent refinement
        [0.55, 0.25, 0.35, 1e-5, 1e-5],  # block-1 target
        [0.35, 0.55, 0.25, 0.20, 0.30],  # fully mixed
        [0.90, 0.04, 0.04, 0.04, 0.50],  # direct-friendly singleton sector
        [0.12, 0.80, 0.08, 0.04, 0.50],  # composition-friendly parent sector
        [1e-5, 1e-5, 1e-5, 1e-5, 1e-5],  # suppressed boundary
    ]
    for i, values in enumerate(templates[:restarts]):
        v = torch.log(torch.as_tensor(values))[None, None, None, :]
        base[i] = v.expand(2, ny, nx, 5)
    z = torch.nn.Parameter(base)
    optimizer = torch.optim.Adam([z], lr=0.045)

    def objective_and_vars():
        values = torch.exp(torch.clamp(z, -15.0, 3.0))
        s, q, r, p, target_q = values.unbind(dim=-1)
        if not allow_block1_d4:
            r = torch.zeros_like(r)
        s2 = s.square() + 1e-20
        singleton2 = s.square() + q.square() + 4 * r.square()
        parent2 = q.square() + r.square()
        final_parent2 = parent2 + p.square() + target_q.square()
        hidden_target2 = r.square() + target_q.square()

        kappa_d = 64 * l0 / (singleton2 + 1e-20)
        kappa_c = 8 * l0 / (parent2 + 1e-20)
        # route index 0 is C and route index 1 is D
        route_is_d = torch.as_tensor([False, True])[None, :, None, None]
        kappa = torch.where(route_is_d, kappa_d, kappa_c)
        free = (
            2 * l0 * s.square()
            + 8 * (ratio * l0) * q.square() / s2
            + 32 * (ratio * l0) * r.square() / s2
            + 8 * l0 * p.square() / (singleton2 + 1e-20)
            + 0.5 * kappa * target_q.square()
            + GAMMA2 * l0 / (GAMMA2 * final_parent2 + l0)
            + GAMMA4 * l0 / (2 * (GAMMA4 * hidden_target2 + l0))
        )
        return free, values

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        free, _ = objective_and_vars()
        free.sum().backward()
        optimizer.step()
        if step == 1700:
            for group in optimizer.param_groups:
                group["lr"] = 0.018
        if step == 2250:
            for group in optimizer.param_groups:
                group["lr"] = 0.007

    with torch.no_grad():
        free, values = objective_and_vars()
        best_restart = free.argmin(dim=0)  # route x y x x
        gather_idx = best_restart[None, ..., None].expand(1, 2, ny, nx, 5)
        best_values = torch.gather(values, 0, gather_idx).squeeze(0)
        best_free = torch.gather(free, 0, best_restart[None, ...]).squeeze(0)
        if not allow_block1_d4:
            best_values[..., 2] = 0
    return best_free.cpu().numpy(), best_values.cpu().numpy()


def classify_occupied(best_free, best_values):
    """Choose the global route and return phase and diagnostic fields."""
    use_c = best_free[0] < best_free[1]
    selected = np.where(use_c[..., None], best_values[0], best_values[1])
    s, q, r, p, target_q = np.moveaxis(selected, -1, 0)
    singleton2 = s**2 + q**2 + 4 * r**2
    parent2 = q**2 + r**2
    final_parent2 = parent2 + p**2 + target_q**2
    hidden_target2 = r**2 + target_q**2
    l0 = LAMBDA_VALUES[None, :]
    parent_amplitude = GAMMA2 * final_parent2 / (GAMMA2 * final_parent2 + l0)
    target_amplitude = GAMMA4 * hidden_target2 / (GAMMA4 * hidden_target2 + l0)
    accessibility = np.sqrt(parent2 / np.maximum(singleton2, 1e-30))
    block1_share = r**2 / np.maximum(hidden_target2, 1e-30)

    phase = np.where(use_c, 0, 1).astype(int)
    target_inactive = target_amplitude < TARGET_ACTIVE_TOL
    parent_inactive = parent_amplitude < TARGET_ACTIVE_TOL
    phase[target_inactive & ~parent_inactive] = 2
    phase[target_inactive & parent_inactive] = 3
    block2_share = target_q**2 / np.maximum(hidden_target2, 1e-30)
    phase[(~target_inactive) & (block2_share < 1e-3)] = 4
    block1_share[target_inactive] = np.nan
    return phase, accessibility, block1_share


def cell_edges(values):
    ratio = values[1] / values[0]
    return np.geomspace(values[0] / np.sqrt(ratio), values[-1] * np.sqrt(ratio), len(values) + 1)


reference_free, reference_values = optimize_occupied_theory(seed=11, allow_block1_d4=False)
reference_phase, _, _ = classify_occupied(reference_free, reference_values)
free, values = optimize_occupied_theory(seed=7, allow_block1_d4=True)
phase, accessibility, block1_share = classify_occupied(free, values)

x_edges = cell_edges(LAMBDA_VALUES)
y_edges = cell_edges(RATIO_VALUES)
phase_colors = ["#168f83", "#e9b949", "#9aa4b2", "#30343b", "#8055a5"]
phase_cmap = ListedColormap(phase_colors)
phase_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), phase_cmap.N)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), sharex=True, sharey=True)
fig.subplots_adjust(left=0.095, right=0.93, bottom=0.10, top=0.93, wspace=0.16, hspace=0.25)

for ax, data, title in [
    (axes[0, 0], reference_phase, r"(a) Availability face: $\rho^{(1)}_{4\times1}=0$"),
    (axes[0, 1], phase, r"(b) All circuit populations available"),
]:
    ax.pcolormesh(x_edges, y_edges, data, cmap=phase_cmap, norm=phase_norm,
                  shading="flat", rasterized=True)
    ax.set_title(title, fontsize=8.5)

legend_handles = [
    Patch(color=phase_colors[0], label=r"$C:2+2\to4$ in block 2"),
    Patch(color=phase_colors[1], label=r"$D:4\times1\to4$ in block 2"),
    Patch(color=phase_colors[4], label="block 1 only"),
    Patch(color=phase_colors[2], label="parents only"),
    Patch(color=phase_colors[3], label="suppressed"),
]
axes[0, 1].legend(handles=legend_handles, loc="upper right", fontsize=5.4,
                  frameon=True, handlelength=0.9, borderpad=0.25, labelspacing=0.18)

access_heat = axes[1, 0].pcolormesh(
    x_edges, y_edges, accessibility, cmap="viridis", vmin=0, vmax=0.72,
    shading="flat", rasterized=True,
)
axes[1, 0].contour(LAMBDA_VALUES, RATIO_VALUES, accessibility,
                   levels=[1 / np.sqrt(8)], colors="white", linewidths=1.1)
axes[1, 0].set_title(r"(c) Selected accessibility $P/S$", fontsize=8.5)
cbar1 = fig.colorbar(access_heat, ax=axes[1, 0], fraction=0.046, pad=0.025,
                     ticks=[0, 1 / np.sqrt(8), 0.7])
cbar1.ax.tick_params(labelsize=6)
cbar1.ax.set_yticklabels(["0", r"$1/\sqrt{8}$", "0.7"])

share_map = plt.get_cmap("magma").copy()
share_map.set_bad("#d7d7d7")
share_heat = axes[1, 1].pcolormesh(
    x_edges, y_edges, np.ma.masked_invalid(block1_share), cmap=share_map,
    vmin=0, vmax=1, shading="flat", rasterized=True,
)
axes[1, 1].set_title(r"(d) Block-1 share $r^2/(r^2+Q^2)$", fontsize=8.5)
cbar2 = fig.colorbar(share_heat, ax=axes[1, 1], fraction=0.046, pad=0.025,
                     ticks=[0, 0.5, 1])
cbar2.ax.tick_params(labelsize=6)

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LAMBDA_VALUES[0], LAMBDA_VALUES[-1])
    ax.set_ylim(RATIO_VALUES[0], RATIO_VALUES[-1])
    ax.tick_params(labelsize=6.5, length=2.5)
    ax.axhline(1, color="white", lw=0.65, ls="--", alpha=0.75)
for ax in axes[1, :]:
    ax.set_xlabel(r"overall decay $\lambda_0$", fontsize=7.5)
for ax in axes[:, 0]:
    ax.set_ylabel(r"block-1/block-2 decay $\lambda_1/\lambda_2$", fontsize=7.5)

fig.savefig(OUT, bbox_inches="tight")
print(OUT)
