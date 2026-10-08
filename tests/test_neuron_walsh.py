"""Sanity tests for the neuron-level Walsh analysis (spec section 24).

Run with:  python -m pytest tests/test_neuron_walsh.py -v
Or:        python tests/test_neuron_walsh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.neuron_walsh import metrics as M
from analysis.neuron_walsh.modes import build_mode_registry

RNG = np.random.default_rng(0)


def _uniform_x(n, d):
    return RNG.integers(0, 2, (n, d)).astype(np.float64) * 2 - 1


def _chi(x, support):
    return np.prod(x[:, list(support)], axis=1)


def _walsh_matrix(x, modes):
    return np.stack([_chi(x, m["support"]) for m in modes], axis=1)  # (n, M)


def test_mode_registry_is_the_15_tree_targets():
    modes = build_mode_registry(16)
    assert len(modes) == 15
    by_deg = {}
    for m in modes:
        by_deg.setdefault(m["degree"], []).append(m)
    assert {d: len(v) for d, v in by_deg.items()} == {2: 8, 4: 4, 8: 2, 16: 1}
    d4_1 = next(m for m in modes if m["name"] == "d4_1")
    assert d4_1["support"] == (4, 5, 6, 7)
    assert d4_1["parent"] == "d8_0"
    assert d4_1["children"] == ["d2_2", "d2_3"]


def test_coefficients_match_direct_and_pearson():
    n, d = 200_000, 16
    x = _uniform_x(n, d)
    modes = build_mode_registry(d)
    Xw = _walsh_matrix(x, modes)
    # H: 3 neurons with known Walsh content + noise via an untracked mode
    h0 = 0.7 * _chi(x, (0, 1)) - 0.2 * _chi(x, (0, 1, 2, 3))
    h1 = 0.5 * _chi(x, tuple(range(16)))
    h2 = 0.3 * _chi(x, (4, 5)) + 0.4 * _chi(x, (0, 5, 10, 15))  # 2nd is untracked
    H = np.stack([h0, h1, h2], axis=1)
    C = Xw.T @ H / n                                     # [M, 3]

    # direct per-neuron recomputation (spec 24a)
    for mi, ni in [(0, 0), (14, 1), (2, 2)]:
        direct = float((H[:, ni] * Xw[:, mi]).mean())
        assert abs(C[mi, ni] - direct) < 1e-9

    # recovery of planted coefficients (sampling noise ~ 1/sqrt(n))
    name = {m["name"]: m["index"] for m in modes}
    assert abs(C[name["d2_0"], 0] - 0.7) < 0.01
    assert abs(C[name["d4_0"], 0] + 0.2) < 0.01
    assert abs(C[name["d16_0"], 1] - 0.5) < 0.01

    # layer-level recovery (24b): sum_i C^2 == ||E[H chi_S]||^2 by definition
    hS = (H * Xw[:, [0]]).mean(0)
    # identical math, different fp summation order (BLAS matmul vs mean)
    assert abs((C[0] ** 2).sum() - (hS ** 2).sum()) < 1e-9

    # normalized correlation == Pearson (24c)
    var = H.var(axis=0)
    r_analytic = C[name["d2_2"], 2] / (np.sqrt(var[2]) + 1e-30)
    r_direct = np.corrcoef(H[:, 2], Xw[:, name["d2_2"]])[0, 1]
    # Pearson centers by SAMPLE means (E_n[chi] ~ 1/sqrt(n), not exactly 0),
    # so analytic a/sigma and direct Pearson agree only to O(1/sqrt(n)).
    assert abs(r_analytic - r_direct) < 0.01

    # coverage (24e): tracked fraction of variance for neuron 2 is
    # 0.3^2 / (0.3^2 + 0.4^2) = 0.36 (the 0.4 mode is untracked)
    covg = M.coverage(C, var)
    assert abs(covg[2] - 0.09 / 0.25) < 0.02
    assert abs(covg[0] - 1.0) < 0.02       # fully tracked
    assert abs(covg[1] - 1.0) < 0.02


def test_effective_numbers_synthetic():
    Mn, N = 15, 64
    # one mode entirely in one neuron -> N_eff = 1
    C = np.zeros((Mn, N)); C[3, 7] = 2.0
    assert abs(M.n_eff(C)[3] - 1.0) < 1e-9
    # equally distributed across k neurons -> N_eff = k
    C = np.zeros((Mn, N)); C[5, :16] = 0.5
    assert abs(M.n_eff(C)[5] - 16.0) < 1e-6
    # one neuron equally carrying k tracked modes -> M_eff = k
    C = np.zeros((Mn, N)); C[:4, 9] = 0.3
    assert abs(M.m_eff(C)[9] - 4.0) < 1e-6
    assert abs(M.purity(C)[9] - 0.25) < 1e-9
    # only one mode -> M_eff = 1, purity = 1
    C = np.zeros((Mn, N)); C[2, 11] = 0.4
    assert abs(M.m_eff(C)[11] - 1.0) < 1e-9
    assert abs(M.purity(C)[11] - 1.0) < 1e-9


def test_cumulative_and_topk():
    C = np.zeros((3, 10))
    C[0] = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    cum = M.cumulative_power(C, 0)
    tot = 9 + 4 + 1
    assert abs(cum[0] - 9 / tot) < 1e-12
    assert abs(cum[1] - 13 / tot) < 1e-12
    assert abs(cum[-1] - 1.0) < 1e-12
    tk = M.topk_fractions(C, (1, 5))
    assert abs(tk[1][0] - 9 / tot) < 1e-12
    assert M.neurons_for_fraction(C, 0, 0.5) == 1
    assert M.neurons_for_fraction(C, 0, 0.9) == 2


def test_colocalization():
    C = np.zeros((2, 8))
    C[0, :4] = 1.0
    C[1, :4] = -1.0        # same neurons, opposite signs
    K = M.mode_colocalization(C, signed=False)
    assert abs(K[0, 1] - 1.0) < 1e-9   # power profiles identical
    Ks = M.mode_colocalization(C, signed=True)
    assert abs(Ks[0, 1] + 1.0) < 1e-9  # signed alignment = -1
    C[1] = 0.0
    C[1, 4:] = 1.0          # disjoint neurons
    K = M.mode_colocalization(C, signed=False)
    assert abs(K[0, 1]) < 1e-9


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
