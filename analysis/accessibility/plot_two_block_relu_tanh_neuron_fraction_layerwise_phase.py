"""Phase diagrams for the two-block ReLU(tanh) circuit theory.

The script compares the scale-invariant pure-ReLU limit with the saturating
activation phi(z) = max(tanh(z), 0).  In the latter case the neuron counts are
part of the reduced free energy.  All minimizations are batched over the phase
plane and use several deterministic restarts.
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
OUT = ROOT / "reports" / "two_block_theory" / "fig_two_block_relu_tanh_neuron_fraction_phase.pdf"

GAMMA2 = GAMMA4 = 2.0
LAMBDA_VALUES = np.geomspace(1e-4, 1.0, 54)
RATIO_VALUES = np.geomspace(0.05, 20.0, 50)
TARGET_ACTIVE_TOL = 2e-3
MIX_TOL = 0.10

# A transparent finite-width slice.  At N=64 these fractions correspond to
# integer counts (32, 24) in block 1 and (4, 8, 16) in block 2.
RHO12, RHO14 = 1 / 2, 3 / 8
RHO22, RHOD, RHOC = 1 / 16, 1 / 8, 1 / 4


def _cell_edges(values):
    ratio = values[1] / values[0]
    return np.geomspace(values[0] / np.sqrt(ratio), values[-1] * np.sqrt(ratio), len(values) + 1)


def optimize_relu(seed=17, restarts=15, steps=2700):
    """Pure-ReLU reference: minimize the C and D branches separately."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    ny, nx = len(RATIO_VALUES), len(LAMBDA_VALUES)
    l0 = torch.as_tensor(LAMBDA_VALUES)[None, None, None, :]
    ratio = torch.as_tensor(RATIO_VALUES)[None, None, :, None]

    z0 = torch.empty((restarts, 2, ny, nx, 5)).uniform_(-5.5, 0.5)
    templates = [
        [0.45, 0.45, 1e-6, 1e-6, 0.45],
        [0.70, 0.20, 1e-6, 0.45, 0.35],
        [0.55, 0.25, 0.35, 1e-6, 1e-6],
        [0.35, 0.55, 0.25, 0.20, 0.30],
        [0.90, 0.04, 0.04, 0.04, 0.50],
        [0.12, 0.80, 0.08, 0.04, 0.50],
        [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
    ]
    for i, vals in enumerate(templates[:restarts]):
        z0[i] = torch.log(torch.as_tensor(vals))[None, None, None, :].expand(2, ny, nx, 5)
    z = torch.nn.Parameter(z0)
    optimizer = torch.optim.Adam([z], lr=0.045)

    def evaluate():
        vals = torch.exp(torch.clamp(z, -18.0, 3.0))
        s, q, r, p, target_q = vals.unbind(dim=-1)
        s2 = s.square() + 1e-24
        singleton2 = s.square() + q.square() + 4 * r.square()
        parent2 = q.square() + r.square()
        final_parent2 = parent2 + p.square() + target_q.square()
        hidden_target2 = r.square() + target_q.square()
        kappa_d = 64 * l0 / (singleton2 + 1e-24)
        kappa_c = 8 * l0 / (parent2 + 1e-24)
        route_is_d = torch.as_tensor([False, True])[None, :, None, None]
        kappa = torch.where(route_is_d, kappa_d, kappa_c)
        free = (
            2 * l0 * s.square()
            + 8 * ratio * l0 * q.square() / s2
            + 32 * ratio * l0 * r.square() / s2
            + 8 * l0 * p.square() / (singleton2 + 1e-24)
            + 0.5 * kappa * target_q.square()
            + GAMMA2 * l0 / (GAMMA2 * final_parent2 + l0)
            + GAMMA4 * l0 / (2 * (GAMMA4 * hidden_target2 + l0))
        )
        return free, vals

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        free, _ = evaluate()
        free.sum().backward()
        optimizer.step()
        if step == 1450:
            optimizer.param_groups[0]["lr"] = 0.017
        elif step == 2050:
            optimizer.param_groups[0]["lr"] = 0.006

    with torch.no_grad():
        free, vals = evaluate()
        best_restart = free.argmin(dim=0)
        free_best = torch.gather(free, 0, best_restart[None]).squeeze(0)
        gather = best_restart[None, ..., None].expand(1, 2, ny, nx, 5)
        vals_best = torch.gather(vals, 0, gather).squeeze(0)
    return free_best.cpu().numpy(), vals_best.cpu().numpy()


def classify_relu(free, vals):
    use_c = free[0] < free[1]
    selected = np.where(use_c[..., None], vals[0], vals[1])
    s, q, r, p, target_q = np.moveaxis(selected, -1, 0)
    parent2 = q**2 + r**2
    final_parent2 = parent2 + p**2 + target_q**2
    target2 = r**2 + target_q**2
    l0 = LAMBDA_VALUES[None, :]
    b_amp = GAMMA2 * final_parent2 / (GAMMA2 * final_parent2 + l0)
    a_amp = GAMMA4 * target2 / (GAMMA4 * target2 + l0)
    phase = np.where(use_c, 0, 1).astype(int)
    target_off = a_amp < TARGET_ACTIVE_TOL
    parent_off = b_amp < TARGET_ACTIVE_TOL
    phase[target_off & ~parent_off] = 4
    phase[target_off & parent_off] = 5
    phase[(~target_off) & (target_q**2 / np.maximum(target2, 1e-30) < 1e-3)] = 3
    return phase


def _a(g):
    """Magnitude of every nonconstant coefficient of a balanced two-input gate."""
    return torch.tanh(2 * g) / 4


def _b1(g):
    return (torch.tanh(4 * g) + 2 * torch.tanh(2 * g)) / 16


def _b2(g):
    return torch.tanh(4 * g) / 16


def _b4(g):
    return (4 * torch.tanh(2 * g) - torch.tanh(4 * g)) / 16


def optimize_relu_tanh(width, seed=23, restarts=18, steps=3300):
    """Minimize the joint ReLU(tanh) free energy over six positive variables.

    The variables are (s, gamma_12, gamma_14, gamma_22, gamma_D, gamma_C).
    Direct and compositional block-2 target populations are optimized together,
    so an interior target-strength mixture is allowed.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed + int(width))
    ny, nx = len(RATIO_VALUES), len(LAMBDA_VALUES)
    l0 = torch.as_tensor(LAMBDA_VALUES)[None, None, :]
    ratio = torch.as_tensor(RATIO_VALUES)[None, :, None]
    m12, m14 = width * RHO12, width * RHO14
    m22, m_d, m_c = width * RHO22, width * RHOD, width * RHOC

    z0 = torch.empty((restarts, ny, nx, 6)).uniform_(-6.5, 0.2)
    templates = [
        [0.50, 0.05, 1e-7, 1e-7, 1e-7, 0.10],
        [0.50, 0.05, 1e-7, 1e-7, 0.16, 1e-7],
        [0.55, 0.04, 0.12, 1e-7, 1e-7, 1e-7],
        [0.40, 0.07, 0.08, 0.06, 0.08, 0.08],
        [0.85, 0.01, 0.01, 0.01, 0.12, 0.02],
        [0.18, 0.13, 0.02, 0.01, 0.02, 0.12],
        [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7],
        [0.60, 0.03, 0.18, 0.03, 0.12, 0.12],
    ]
    for i, vals in enumerate(templates[:restarts]):
        z0[i] = torch.log(torch.as_tensor(vals))[None, None, :].expand(ny, nx, 6)
    z = torch.nn.Parameter(z0)
    optimizer = torch.optim.Adam([z], lr=0.042)

    def evaluate():
        vals = torch.exp(torch.clamp(z, -18.0, 2.5))
        s, g12, g14, g22, g_d, g_c = vals.unbind(dim=-1)
        q = np.sqrt(m12 / 2) * _a(g12)
        r = np.sqrt(m14) * _b4(g14)
        p = np.sqrt(m22 / 2) * _a(g22)
        q_d = np.sqrt(m_d) * _b4(g_d)
        q_c = np.sqrt(m_c) * _a(g_c)
        leak1 = np.sqrt(m14) * _b1(g14)
        leak2 = np.sqrt(m14) * _b2(g14)
        leak_d = np.sqrt(m_d) * _b2(g_d)

        s2 = s.square() + 1e-24
        singleton2 = s.square() + q.square() + leak1.square()
        parent2 = q.square() + leak2.square()
        final_parent2 = parent2 + p.square() + leak_d.square() + q_c.square()
        hidden_target2 = r.square() + q_d.square() + q_c.square()
        free = (
            2 * l0 * s.square()
            + ratio * l0 * (m12 * g12.square() + 2 * m14 * g14.square()) / s2
            + l0
            * (
                m22 * g22.square() / (singleton2 + 1e-24)
                + 2 * m_d * g_d.square() / (singleton2 + 1e-24)
                + m_c * g_c.square() / (parent2 + 1e-24)
            )
            + GAMMA2 * l0 / (GAMMA2 * final_parent2 + l0)
            + GAMMA4 * l0 / (2 * (GAMMA4 * hidden_target2 + l0))
        )
        diagnostics = torch.stack(
            [q, r, p, q_d, q_c, torch.sqrt(singleton2), torch.sqrt(parent2),
             torch.sqrt(final_parent2), torch.sqrt(hidden_target2)], dim=-1
        )
        return free, vals, diagnostics

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        free, _, _ = evaluate()
        free.sum().backward()
        optimizer.step()
        if step == 1750:
            optimizer.param_groups[0]["lr"] = 0.016
        elif step == 2550:
            optimizer.param_groups[0]["lr"] = 0.005

    with torch.no_grad():
        free, vals, diagnostics = evaluate()
        best_restart = free.argmin(dim=0)
        free_best = torch.gather(free, 0, best_restart[None]).squeeze(0)
        gather_v = best_restart[None, ..., None].expand(1, ny, nx, vals.shape[-1])
        vals_best = torch.gather(vals, 0, gather_v).squeeze(0)
        gather_d = best_restart[None, ..., None].expand(1, ny, nx, diagnostics.shape[-1])
        diag_best = torch.gather(diagnostics, 0, gather_d).squeeze(0)
    return free_best.cpu().numpy(), vals_best.cpu().numpy(), diag_best.cpu().numpy()


def classify_relu_tanh(vals, diag):
    s, g12, g14, g22, g_d, g_c = np.moveaxis(vals, -1, 0)
    q, r, p, q_d, q_c, singleton, parent, final_parent, hidden_target = np.moveaxis(diag, -1, 0)
    l0 = LAMBDA_VALUES[None, :]
    b_amp = GAMMA2 * final_parent**2 / (GAMMA2 * final_parent**2 + l0)
    a_amp = GAMMA4 * hidden_target**2 / (GAMMA4 * hidden_target**2 + l0)
    block2_target2 = q_d**2 + q_c**2
    alpha_c = q_c**2 / np.maximum(block2_target2, 1e-30)
    accessibility = parent / np.maximum(singleton, 1e-30)

    # 0 C-dominant, 1 D-dominant, 2 genuine mixture, 3 block-1-only,
    # 4 parents only, and 5 fully suppressed.
    phase = np.full(alpha_c.shape, 2, dtype=int)
    phase[alpha_c >= 1 - MIX_TOL] = 0
    phase[alpha_c <= MIX_TOL] = 1
    target_off = a_amp < TARGET_ACTIVE_TOL
    parent_off = b_amp < TARGET_ACTIVE_TOL
    block2_share = block2_target2 / np.maximum(hidden_target**2, 1e-30)
    phase[(~target_off) & (block2_share < 1e-3)] = 3
    phase[target_off & ~parent_off] = 4
    phase[target_off & parent_off] = 5
    alpha_c[(phase == 3) | target_off] = np.nan
    return phase, alpha_c, accessibility


relu_free, relu_vals = optimize_relu()
relu_phase = classify_relu(relu_free, relu_vals)

_, tanh64_vals, tanh64_diag = optimize_relu_tanh(64, seed=23)
tanh64_phase, tanh64_alpha, tanh64_access = classify_relu_tanh(tanh64_vals, tanh64_diag)

_, tanh1024_vals, tanh1024_diag = optimize_relu_tanh(1024, seed=31, restarts=16, steps=3000)
tanh1024_phase, _, _ = classify_relu_tanh(tanh1024_vals, tanh1024_diag)

x_edges = _cell_edges(LAMBDA_VALUES)
y_edges = _cell_edges(RATIO_VALUES)
phase_colors = ["#168f83", "#e9b949", "#d96d5f", "#8055a5", "#9aa4b2", "#30343b"]
phase_cmap = ListedColormap(phase_colors)
phase_norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), phase_cmap.N)

fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.55), sharex=True, sharey=True)
fig.subplots_adjust(left=0.095, right=0.93, bottom=0.10, top=0.93, wspace=0.17, hspace=0.27)

for ax, data, title in [
    (axes[0, 0], relu_phase, r"(a) Pure ReLU (scale-invariant limit)"),
    (axes[0, 1], tanh64_phase, r"(b) ReLU(tanh), $N_1=N_2=64$"),
    (axes[1, 1], tanh1024_phase, r"(d) ReLU(tanh), $N_1=N_2=1024$"),
]:
    ax.pcolormesh(x_edges, y_edges, data, cmap=phase_cmap, norm=phase_norm,
                  shading="flat", rasterized=True)
    ax.set_title(title, fontsize=8.5)

legend_handles = [
    Patch(color=phase_colors[0], label=r"$C$-dominant: $2+2\to4$"),
    Patch(color=phase_colors[1], label=r"$D$-dominant: $4\times1\to4$"),
    Patch(color=phase_colors[2], label="mixed block-2 target"),
    Patch(color=phase_colors[3], label="block 1 only"),
    Patch(color=phase_colors[4], label="parents only"),
    Patch(color=phase_colors[5], label="suppressed"),
]
axes[0, 1].legend(handles=legend_handles, loc="upper right", fontsize=5.15,
                  frameon=True, handlelength=0.9, borderpad=0.25, labelspacing=0.16)

alpha_cmap = plt.get_cmap("BrBG").copy()
alpha_cmap.set_bad("#d7d7d7")
alpha_heat = axes[1, 0].pcolormesh(
    x_edges, y_edges, np.ma.masked_invalid(tanh64_alpha), cmap=alpha_cmap,
    vmin=0, vmax=1, shading="flat", rasterized=True,
)
axes[1, 0].contour(LAMBDA_VALUES, RATIO_VALUES, tanh64_access,
                   levels=[1 / np.sqrt(8)], colors="white", linewidths=1.05)
axes[1, 0].set_title(r"(c) Composition share $\alpha_C$ at $N=64$", fontsize=8.5)
cbar = fig.colorbar(alpha_heat, ax=axes[1, 0], fraction=0.046, pad=0.025, ticks=[0, 0.5, 1])
cbar.ax.tick_params(labelsize=6)
cbar.ax.set_yticklabels(["direct", "mixed", "composite"])

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LAMBDA_VALUES[0], LAMBDA_VALUES[-1])
    ax.set_ylim(RATIO_VALUES[0], RATIO_VALUES[-1])
    ax.tick_params(labelsize=6.5, length=2.5)
    ax.axhline(1, color="white", lw=0.6, ls="--", alpha=0.75)
for ax in axes[1, :]:
    ax.set_xlabel(r"overall decay $\lambda_0$", fontsize=7.5)
for ax in axes[:, 0]:
    ax.set_ylabel(r"block-1/block-2 decay $\lambda_1/\lambda_2$", fontsize=7.5)

fig.savefig(OUT, bbox_inches="tight")
print(OUT)
