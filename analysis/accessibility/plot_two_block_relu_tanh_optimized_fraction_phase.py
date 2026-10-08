"""Optimize ReLU(tanh) circuit fractions and plot the resulting phase map.

This is the homogeneous-population closure from the accompanying report.  It
jointly minimizes the reduced zero-temperature free energy over the six mode
amplitudes and over two layerwise simplices of active circuit allocations plus
an idle category.
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
OUT = ROOT / "reports" / "two_block_theory" / "fig_two_block_relu_tanh_optimized_fraction_phase.pdf"
DATA_OUT = ROOT / "reports" / "two_block_theory" / "data_two_block_relu_tanh_optimized_fractions.npz"

GAMMA2 = GAMMA4 = 2.0
WIDTH1 = WIDTH2 = 64.0
LAMBDA_VALUES = np.geomspace(1e-4, 1.0, 52)
RATIO_VALUES = np.geomspace(0.05, 20.0, 48)
TARGET_ACTIVE_TOL = 2e-3
MIX_TOL = 0.10
WEIGHT_ENERGY_TOL = 1e-7


def a(gamma):
    return torch.tanh(2 * gamma) / 4


def b1(gamma):
    return (torch.tanh(4 * gamma) + 2 * torch.tanh(2 * gamma)) / 16


def b2(gamma):
    return torch.tanh(4 * gamma) / 16


def b4(gamma):
    return (4 * torch.tanh(2 * gamma) - torch.tanh(4 * gamma)) / 16


def optimize(seed=47, restarts=24, steps=3900):
    """Batched multistart minimization over amplitudes and circuit fractions."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    ny, nx = len(RATIO_VALUES), len(LAMBDA_VALUES)
    l0 = torch.as_tensor(LAMBDA_VALUES)[None, None, :]
    ratio = torch.as_tensor(RATIO_VALUES)[None, :, None]

    # Positive variables: s and the five common within-population amplitudes.
    z0 = torch.empty((restarts, ny, nx, 6)).uniform_(-6.5, 0.2)
    # Layer-1 categories: 2x1, 4x1, idle.  Layer-2 categories: 2x1, D, C, idle.
    logits1_0 = torch.randn((restarts, ny, nx, 3))
    logits2_0 = torch.randn((restarts, ny, nx, 4))

    amplitude_templates = [
        [0.50, 0.05, 1e-7, 1e-7, 1e-7, 0.10],
        [0.50, 0.05, 1e-7, 1e-7, 0.16, 1e-7],
        [0.55, 0.04, 0.12, 1e-7, 1e-7, 1e-7],
        [0.40, 0.07, 0.08, 0.06, 0.08, 0.08],
        [0.85, 0.01, 0.01, 0.01, 0.12, 0.02],
        [0.18, 0.13, 0.02, 0.01, 0.02, 0.12],
        [1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7],
        [0.60, 0.03, 0.18, 0.03, 0.12, 0.12],
        [0.45, 0.08, 0.04, 0.08, 0.03, 0.15],
        [0.45, 0.08, 0.04, 0.08, 0.15, 0.03],
    ]
    fraction_templates = [
        ([0.70, 0.20, 0.10], [0.20, 0.10, 0.55, 0.15]),
        ([0.70, 0.20, 0.10], [0.20, 0.55, 0.10, 0.15]),
        ([0.15, 0.75, 0.10], [0.10, 0.10, 0.10, 0.70]),
        ([0.45, 0.45, 0.10], [0.20, 0.30, 0.35, 0.15]),
        ([0.80, 0.05, 0.15], [0.10, 0.70, 0.05, 0.15]),
        ([0.80, 0.05, 0.15], [0.10, 0.05, 0.70, 0.15]),
        ([0.05, 0.05, 0.90], [0.03, 0.03, 0.03, 0.91]),
        ([0.20, 0.70, 0.10], [0.10, 0.35, 0.35, 0.20]),
        ([0.65, 0.25, 0.10], [0.55, 0.05, 0.30, 0.10]),
        ([0.65, 0.25, 0.10], [0.55, 0.30, 0.05, 0.10]),
    ]
    for i, (amps, (frac1, frac2)) in enumerate(zip(amplitude_templates, fraction_templates)):
        if i >= restarts:
            break
        z0[i] = torch.log(torch.as_tensor(amps))[None, None, :].expand(ny, nx, 6)
        logits1_0[i] = torch.log(torch.as_tensor(frac1))[None, None, :].expand(ny, nx, 3)
        logits2_0[i] = torch.log(torch.as_tensor(frac2))[None, None, :].expand(ny, nx, 4)

    z = torch.nn.Parameter(z0)
    logits1 = torch.nn.Parameter(logits1_0)
    logits2 = torch.nn.Parameter(logits2_0)
    optimizer = torch.optim.Adam(
        [
            {"params": [z], "lr": 0.040},
            {"params": [logits1, logits2], "lr": 0.065},
        ]
    )

    def evaluate():
        vals = torch.exp(torch.clamp(z, -18.0, 2.5))
        s, g12, g14, g22, g_d, g_c = vals.unbind(dim=-1)
        fractions1 = torch.softmax(logits1, dim=-1)
        fractions2 = torch.softmax(logits2, dim=-1)
        rho12, rho14, _ = fractions1.unbind(dim=-1)
        rho22, rho_d, rho_c, _ = fractions2.unbind(dim=-1)
        m12, m14 = WIDTH1 * rho12, WIDTH1 * rho14
        m22, m_d, m_c = WIDTH2 * rho22, WIDTH2 * rho_d, WIDTH2 * rho_c

        # Squared mode norms; this form is smooth even when a fraction vanishes.
        q2 = 0.5 * m12 * a(g12).square()
        r2 = m14 * b4(g14).square()
        p2 = 0.5 * m22 * a(g22).square()
        qd2 = m_d * b4(g_d).square()
        qc2 = m_c * a(g_c).square()
        l12 = m14 * b1(g14).square()
        l22 = m14 * b2(g14).square()
        ld2 = m_d * b2(g_d).square()

        s2 = s.square() + 1e-24
        singleton2 = s.square() + q2 + l12
        parent2 = q2 + l22
        final_parent2 = parent2 + p2 + ld2 + qc2
        hidden_target2 = r2 + qd2 + qc2
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
            [q2, r2, p2, qd2, qc2, l12, l22, ld2,
             singleton2, parent2, final_parent2, hidden_target2],
            dim=-1,
        )
        weight_energies = torch.stack(
            [m12 * g12.square(), m14 * g14.square(), m22 * g22.square(),
             m_d * g_d.square(), m_c * g_c.square()], dim=-1
        )
        return free, vals, fractions1, fractions2, diagnostics, weight_energies

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        free, *_ = evaluate()
        free.sum().backward()
        optimizer.step()
        if step == 2000:
            optimizer.param_groups[0]["lr"] = 0.015
            optimizer.param_groups[1]["lr"] = 0.024
        elif step == 3000:
            optimizer.param_groups[0]["lr"] = 0.0045
            optimizer.param_groups[1]["lr"] = 0.007

    with torch.no_grad():
        free, vals, frac1, frac2, diag, energies = evaluate()
        best_restart = free.argmin(dim=0)

        def take_best(tensor):
            gather = best_restart[None, ..., None].expand(1, ny, nx, tensor.shape[-1])
            return torch.gather(tensor, 0, gather).squeeze(0)

        free_best = torch.gather(free, 0, best_restart[None]).squeeze(0)
        outputs = [take_best(t) for t in (vals, frac1, frac2, diag, energies)]
    return (free_best.cpu().numpy(), *(out.cpu().numpy() for out in outputs))


def classify(vals, frac1, frac2, diag, energies):
    q2, r2, p2, qd2, qc2, l12, l22, ld2, singleton2, parent2, final_parent2, target2 = np.moveaxis(diag, -1, 0)
    l0 = LAMBDA_VALUES[None, :]
    parent_amplitude = GAMMA2 * final_parent2 / (GAMMA2 * final_parent2 + l0)
    target_amplitude = GAMMA4 * target2 / (GAMMA4 * target2 + l0)
    block2_target2 = qd2 + qc2
    alpha_c = qc2 / np.maximum(block2_target2, 1e-30)

    phase = np.full(alpha_c.shape, 2, dtype=int)
    phase[alpha_c >= 1 - MIX_TOL] = 0
    phase[alpha_c <= MIX_TOL] = 1
    target_off = target_amplitude < TARGET_ACTIVE_TOL
    parent_off = parent_amplitude < TARGET_ACTIVE_TOL
    block2_share = block2_target2 / np.maximum(target2, 1e-30)
    phase[(~target_off) & (block2_share < 1e-3)] = 3
    phase[target_off & ~parent_off] = 4
    phase[target_off & parent_off] = 5

    # Circuit labels are physically meaningless when their incoming weights are
    # zero.  Report active fractions only and absorb inactive allocations into
    # the idle category.
    rho_active = np.stack(
        [frac1[..., 0], frac1[..., 1], frac2[..., 0], frac2[..., 1], frac2[..., 2]],
        axis=-1,
    )
    # Allocations whose propagated modes contribute less than 0.5% are
    # functionally indistinguishable from idle labels and are displayed as zero.
    relative_tol = 5e-3
    signal_floor = 1e-12
    active = np.stack(
        [
            q2 > relative_tol * np.maximum.reduce([singleton2, final_parent2, np.full_like(q2, signal_floor)]),
            (r2 > relative_tol * np.maximum(target2, signal_floor))
            | (l22 > relative_tol * np.maximum(final_parent2, signal_floor)),
            p2 > relative_tol * np.maximum(final_parent2, signal_floor),
            (qd2 > relative_tol * np.maximum(target2, signal_floor))
            | (ld2 > relative_tol * np.maximum(final_parent2, signal_floor)),
            qc2 > relative_tol * np.maximum.reduce([target2, final_parent2, np.full_like(qc2, signal_floor)]),
        ],
        axis=-1,
    )
    active &= energies >= WEIGHT_ENERGY_TOL
    rho_active[~active] = 0.0
    return phase, alpha_c, rho_active, parent_amplitude, target_amplitude


def cell_edges(values):
    ratio = values[1] / values[0]
    return np.geomspace(values[0] / np.sqrt(ratio), values[-1] * np.sqrt(ratio), len(values) + 1)


free, vals, frac1, frac2, diag, energies = optimize()
phase, alpha_c, rho_active, parent_amp, target_amp = classify(vals, frac1, frac2, diag, energies)

np.savez_compressed(
    DATA_OUT,
    lambda0=LAMBDA_VALUES,
    lambda1_over_lambda2=RATIO_VALUES,
    objective=free,
    phase=phase,
    alpha_c=alpha_c,
    rho1_2x1=rho_active[..., 0],
    rho1_4x1=rho_active[..., 1],
    rho2_2x1=rho_active[..., 2],
    rho2_4x1=rho_active[..., 3],
    rho2_2x2=rho_active[..., 4],
    parent_amplitude=parent_amp,
    target_amplitude=target_amp,
)

x_edges = cell_edges(LAMBDA_VALUES)
y_edges = cell_edges(RATIO_VALUES)
phase_colors = ["#168f83", "#e9b949", "#d96d5f", "#8055a5", "#9aa4b2", "#30343b"]
phase_cmap = ListedColormap(phase_colors)
phase_norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), phase_cmap.N)

fig, axes = plt.subplots(2, 3, figsize=(8.3, 5.25), sharex=True, sharey=True)
fig.subplots_adjust(left=0.075, right=0.91, bottom=0.105, top=0.93, wspace=0.16, hspace=0.27)

axes[0, 0].pcolormesh(x_edges, y_edges, phase, cmap=phase_cmap, norm=phase_norm,
                      shading="flat", rasterized=True)
axes[0, 0].set_title(r"(a) Optimized-fraction phase", fontsize=8.5)
legend_handles = [
    Patch(color=phase_colors[0], label=r"$C$-dominant: $2+2\to4$"),
    Patch(color=phase_colors[1], label=r"$D$-dominant: $4\times1\to4$"),
    Patch(color=phase_colors[2], label="mixed block-2 target"),
    Patch(color=phase_colors[3], label="block 1 only"),
    Patch(color=phase_colors[4], label="parents only"),
    Patch(color=phase_colors[5], label="suppressed"),
]
axes[0, 0].legend(handles=legend_handles, loc="upper right", fontsize=4.65,
                  frameon=True, handlelength=0.8, borderpad=0.22, labelspacing=0.12)

fraction_panels = [
    (axes[0, 1], rho_active[..., 0], r"(b) $\rho^{(1)}_{2\times1}$"),
    (axes[0, 2], rho_active[..., 1], r"(c) $\rho^{(1)}_{4\times1}$"),
    (axes[1, 0], rho_active[..., 2], r"(d) $\rho^{(2)}_{2\times1}$"),
    (axes[1, 1], rho_active[..., 3], r"(e) $\rho^{(2)}_{4\times1}$ (direct)"),
    (axes[1, 2], rho_active[..., 4], r"(f) $\rho^{(2)}_{2\times2}$ (composite)"),
]
fraction_cmap = "viridis"
last_heat = None
for ax, data, title in fraction_panels:
    last_heat = ax.pcolormesh(x_edges, y_edges, data, cmap=fraction_cmap, vmin=0, vmax=1,
                              shading="flat", rasterized=True)
    ax.set_title(title, fontsize=8.5)

cbar_ax = fig.add_axes([0.925, 0.18, 0.014, 0.64])
cbar = fig.colorbar(last_heat, cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar.ax.tick_params(labelsize=6)
cbar.set_label("active neuron fraction", fontsize=7)

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LAMBDA_VALUES[0], LAMBDA_VALUES[-1])
    ax.set_ylim(RATIO_VALUES[0], RATIO_VALUES[-1])
    ax.tick_params(labelsize=6.3, length=2.4)
    ax.axhline(1, color="white", lw=0.6, ls="--", alpha=0.75)
for ax in axes[1, :]:
    ax.set_xlabel(r"overall decay $\lambda_0$", fontsize=7.3)
for ax in axes[:, 0]:
    ax.set_ylabel(r"block-1/block-2 decay $\lambda_1/\lambda_2$", fontsize=7.3)

fig.savefig(OUT, bbox_inches="tight")
print(OUT)
print(DATA_OUT)
