"""Pure-numpy metrics over the neuron-by-mode Walsh coefficient matrix.

Everything operates on C with shape [num_modes M, num_neurons N], where
C[S, i] = a_iS = E_uniform[h_i(x) * chi_S(x)], plus per-neuron activation
variance var[N]. Raw values are always returned; noise thresholds are applied
only in the UI.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-30


# ── mode localization: how many neurons carry a mode? ────────────────────────

def mode_power(C: np.ndarray) -> np.ndarray:
    """H(S) = sum_i a_iS^2, shape [M]."""
    return (C ** 2).sum(axis=1)


def n_eff(C: np.ndarray) -> np.ndarray:
    """N_eff(S) = 1 / sum_i p(i|S)^2 with p(i|S) = a_iS^2 / sum_j a_jS^2."""
    P = C ** 2
    p = P / (P.sum(axis=1, keepdims=True) + EPS)
    return 1.0 / ((p ** 2).sum(axis=1) + EPS)


def cumulative_power(C: np.ndarray, mode_idx: int) -> np.ndarray:
    """C_S(k): cumulative fraction of H(S) captured by top-k neurons, shape [N]."""
    P = np.sort(C[mode_idx] ** 2)[::-1]
    tot = P.sum() + EPS
    return np.cumsum(P) / tot


def topk_fractions(C: np.ndarray, ks=(1, 5, 10)) -> dict[int, np.ndarray]:
    """{k: [M] array of top-k captured power fraction} (k may exceed N safely)."""
    P = np.sort(C ** 2, axis=1)[:, ::-1]
    tot = P.sum(axis=1) + EPS
    cum = np.cumsum(P, axis=1)
    return {k: cum[:, min(k, P.shape[1]) - 1] / tot for k in ks}


def neurons_for_fraction(C: np.ndarray, mode_idx: int, frac: float) -> int:
    """Smallest k with C_S(k) >= frac."""
    cum = cumulative_power(C, mode_idx)
    return int(np.searchsorted(cum, frac) + 1)


# ── neuron semanticity: how many modes does one neuron carry? ────────────────

def neuron_tracked_power(C: np.ndarray) -> np.ndarray:
    """sum_S a_iS^2, shape [N]."""
    return (C ** 2).sum(axis=0)


def m_eff(C: np.ndarray) -> np.ndarray:
    """M_eff(i) = 1 / sum_S p(S|i)^2 with p(S|i) = a_iS^2 / sum_T a_iT^2."""
    P = C ** 2
    p = P / (P.sum(axis=0, keepdims=True) + EPS)
    return 1.0 / ((p ** 2).sum(axis=0) + EPS)


def purity(C: np.ndarray) -> np.ndarray:
    """max_S p(S|i), shape [N]."""
    P = C ** 2
    p = P / (P.sum(axis=0, keepdims=True) + EPS)
    return p.max(axis=0)


def coverage(C: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Fraction of neuron variance explained by tracked modes:
    sum_S a_iS^2 / (var_i + eps). Walsh modes are orthonormal under uniform."""
    return neuron_tracked_power(C) / (var + EPS)


def dominant_mode(C: np.ndarray) -> np.ndarray:
    """argmax_S a_iS^2, shape [N]."""
    return (C ** 2).argmax(axis=0)


# ── mode x mode neuron co-localization ───────────────────────────────────────

def mode_colocalization(C: np.ndarray, signed: bool = False) -> np.ndarray:
    """[M, M] cosine similarity of neuron-space profiles.

    signed=False (default): power profiles v_S = (a_iS^2)_i — sign-invariant
        'do these modes live in the same neurons?'
    signed=True: raw coefficient vectors hat h(S) — 'signed coefficient
        alignment'."""
    V = C if signed else C ** 2
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    Vn = V / (norms + EPS)
    return Vn @ Vn.T
