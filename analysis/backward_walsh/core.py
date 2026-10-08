"""Core Walsh-spectrum computations for the backward analysis.

Two data passes per checkpoint:

Pass 1 — forward + backward:
  Collects hat_h_l(T) and hat_g_l(T) for every layer l and every mode T.
  Uses a degree-16-only MSE loss for the gradients.

Pass 2 — R matrices + VJP:
  For every block l:
    - Accumulates R_l(T) = E[chi_T * phi'(z_l) * h_in_l^T]  (N×N, float64)
      for every degree-16 mode T  -> Q_l(T), U_l(T), Lambda_l(T).
    - Injects a pure-S cotangent and back-props through the block alone
      -> hat_tilde_g(T) for every degree-8 mode T
      -> M_l(T <- S) = ||hat_tilde_g(T)|| / ||ghat_S||.
"""
from __future__ import annotations

import torch

from .modes import ModeInfo, compute_chi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _compute_dphi(block, h_in: torch.Tensor) -> torch.Tensor:
    """Element-wise phi'(z) where z = block.linear(h_in). Returns (batch, N) float32."""
    with torch.no_grad():
        z = block.linear(h_in)
    with torch.enable_grad():
        z_g = z.detach().requires_grad_(True)
        phi_z = block.activation(z_g)
        dphi = torch.autograd.grad(phi_z.sum(), z_g, create_graph=False)[0]
    return dphi.detach()


# ---------------------------------------------------------------------------
# Pass 1: forward + backward Walsh coefficients
# ---------------------------------------------------------------------------

def pass1_walsh_coefficients(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    modes: list[ModeInfo],
    d16_col: int,
    batch_size: int,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Return (h_coeffs, g_coeffs).

    h_coeffs[l][t] = (N,) float64 = E_x[h_l(x) * chi_T(x)]
    g_coeffs[l][t] = (N,) float64 = E_x[g_l(x) * chi_T(x)]
                     where g_l = d L_{d16} / d h_l

    l indexes residual-stream positions: 0 = after embedding, L = after last block.
    """
    model.eval()
    L = len(model.blocks)
    N = model.config.N
    n_total = x.shape[0]
    n_modes = len(modes)
    dev = x.device

    h_acc = [[torch.zeros(N, dtype=torch.float64, device=dev) for _ in range(n_modes)]
             for _ in range(L + 1)]
    g_acc = [[torch.zeros(N, dtype=torch.float64, device=dev) for _ in range(n_modes)]
             for _ in range(L + 1)]

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        xb = x[start:stop]
        yb = y[start:stop]

        # Make h_0 a leaf so it accumulates .grad during backward.
        with torch.no_grad():
            h0_val = model.embedding(xb)
        h_leaf = h0_val.detach().requires_grad_(True)

        activations: list[torch.Tensor] = [h_leaf]
        with torch.enable_grad():
            h = h_leaf
            for block in model.blocks:
                h = block(h)
                h.retain_grad()
                activations.append(h)

            y_pred = model.readout(activations[-1])
            y_d16 = yb[:, d16_col]
            loss = 0.5 * ((y_pred[:, d16_col] - y_d16) ** 2).mean()
            loss.backward()

        with torch.no_grad():
            for l, act in enumerate(activations):
                h_l = act.detach()
                g_l = act.grad
                if g_l is None:
                    g_l = torch.zeros_like(h_l)
                g_l = g_l.detach()

                for t, mode in enumerate(modes):
                    chi = compute_chi(xb, mode.indices).double()
                    h_acc[l][t].add_((chi.unsqueeze(1) * h_l.double()).sum(0))
                    g_acc[l][t].add_((chi.unsqueeze(1) * g_l.double()).sum(0))

    for l in range(L + 1):
        for t in range(n_modes):
            h_acc[l][t].div_(n_total)
            g_acc[l][t].div_(n_total)

    return h_acc, g_acc


# ---------------------------------------------------------------------------
# Pass 2: R matrices (tangent accessibility) + VJP (reverse transfer)
# ---------------------------------------------------------------------------

def pass2_r_and_vjp(
    model,
    x: torch.Tensor,
    modes_d16: list[ModeInfo],
    modes_d8: list[ModeInfo],
    S_indices: tuple[int, ...],
    ghat_S_by_layer: list[torch.Tensor],
    batch_size: int,
) -> tuple[list[list[dict[str, float]]], list[list[torch.Tensor]]]:
    """
    For each block l (0..L-1):
      R matrices  -> Q_l(T), U_l(T), Lambda_l(T)  for T in modes_d16
      VJP accumulators -> hat_tilde_g_l^{(S)}(T)   for T in modes_d8

    Args:
      ghat_S_by_layer[l]: (N,) float64 Walsh coefficient of g_{l+1}(x) on mode S.
                           Used both for VJP injection and for U/Lambda.

    Returns:
      R_results[l][t_d16]: dict with keys Q, U, Lambda
      vjp_acc[l][t_d8]:    (N,) float64 — hat_tilde_g_l^{(S)}(T), un-normalized by ||ghat_S||
    """
    model.eval()
    L = len(model.blocks)
    N = model.config.N
    n_total = x.shape[0]
    n_d16 = len(modes_d16)
    n_d8 = len(modes_d8)
    dev = x.device
    eps = 1e-30

    R_acc = [
        [torch.zeros(N, N, dtype=torch.float64, device=dev) for _ in range(n_d16)]
        for _ in range(L)
    ]
    vjp_acc_raw = [
        [torch.zeros(N, dtype=torch.float64, device=dev) for _ in range(n_d8)]
        for _ in range(L)
    ]

    ghat_S_f32 = [ghat_S_by_layer[l].float().to(dev) for l in range(L)]

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        xb = x[start:stop]

        # Cache h_in for every block
        h_ins: list[torch.Tensor] = []
        with torch.no_grad():
            h = model.embedding(xb)
            for blk in model.blocks:
                h_ins.append(h)
                h = blk(h)

        chi_S = compute_chi(xb, S_indices).float()  # (batch,)

        for l, block in enumerate(model.blocks):
            h_in = h_ins[l]  # (batch, N) float32

            dphi = _compute_dphi(block, h_in)  # (batch, N) float32

            # R matrix accumulation
            h_in_d = h_in.double()
            dphi_d = dphi.double()
            for t, mode in enumerate(modes_d16):
                chi = compute_chi(xb, mode.indices).double()
                w_dphi = dphi_d * chi.unsqueeze(1)      # (batch, N)
                R_acc[l][t].add_(w_dphi.t() @ h_in_d)  # (N, N)

            # VJP (skipped when no projection modes are requested)
            if n_d8 == 0:
                continue
            with torch.enable_grad():
                h_in_g = h_in.detach().requires_grad_(True)
                h_out_g = block(h_in_g)

            grad_out = chi_S.unsqueeze(1) * ghat_S_f32[l].unsqueeze(0)  # (batch, N)
            grad_in = torch.autograd.grad(
                outputs=h_out_g,
                inputs=h_in_g,
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
            )[0].detach()

            grad_in_d = grad_in.double()
            for t, mode in enumerate(modes_d8):
                chi = compute_chi(xb, mode.indices).double()
                vjp_acc_raw[l][t].add_((chi.unsqueeze(1) * grad_in_d).sum(0))

    # Normalize R and compute scalars
    R_results: list[list[dict[str, float]]] = []
    for l in range(L):
        block_results: list[dict[str, float]] = []
        ghat_S_d = ghat_S_by_layer[l].double()
        ghat_S_sq = (ghat_S_d * ghat_S_d).sum().item()
        for t in range(n_d16):
            R = R_acc[l][t] / n_total
            Q = (R * R).sum().item()
            param_grad = ghat_S_d.unsqueeze(1) * R  # (N, N)
            U = (param_grad * param_grad).sum().item()
            Lambda = U / (ghat_S_sq + eps)
            block_results.append({"Q": Q, "U": U, "Lambda": Lambda})
        R_results.append(block_results)

    # Normalize VJP
    for l in range(L):
        for t in range(n_d8):
            vjp_acc_raw[l][t].div_(n_total)

    return R_results, vjp_acc_raw


# ---------------------------------------------------------------------------
# Compute M values from raw VJP accumulators
# ---------------------------------------------------------------------------

def compute_m_values(
    vjp_acc: list[list[torch.Tensor]],
    ghat_S_by_layer: list[torch.Tensor],
    eps: float = 1e-30,
) -> list[list[float]]:
    """M_l(T <- S) = ||hat_tilde_g_l^{(S)}(T)|| / (||ghat_S_{l+1}|| + eps)."""
    L = len(vjp_acc)
    M_vals = []
    for l in range(L):
        ghat_norm = ghat_S_by_layer[l].norm().item()
        row = [
            vjp_acc[l][t].norm().item() / (ghat_norm + eps)
            for t in range(len(vjp_acc[l]))
        ]
        M_vals.append(row)
    return M_vals
