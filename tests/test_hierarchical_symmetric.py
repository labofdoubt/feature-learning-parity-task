"""Symmetry verification for hierarchical_symmetric and hierarchical_symmetric_degree2.

Run with:  python -m pytest tests/test_hierarchical_symmetric.py -v
Or:        python tests/test_hierarchical_symmetric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parity_net.data import (
    sample_hierarchical_symmetric_inputs,
    sample_hierarchical_symmetric_degree2_inputs,
    sample_hierarchical_inputs,
)

CPU = torch.device("cpu")
F32 = torch.float32
N = 400_000
RHO = 0.5
K = 16
INPUT_DIM = 32


def _root_corr(S: torch.Tensor, y: torch.Tensor) -> float:
    return (S * y).mean().item()


def _parity_block(x: torch.Tensor, start: int, length: int) -> torch.Tensor:
    return x[:, start:start + length].prod(dim=1)


# ---------------------------------------------------------------------------
# Helper: compute all internal tree node parities for k=16
# ---------------------------------------------------------------------------

def _tree_nodes(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute degree-2/4/8 parities for the 16 relevant bits."""
    nodes: dict[str, torch.Tensor] = {}
    # degree-2: pairs (0,1),(2,3),...,(14,15)
    for i in range(8):
        nodes[f"d2_{i}"] = x[:, 2*i] * x[:, 2*i+1]
    # degree-4: quads (0-3),(4-7),(8-11),(12-15)
    for i in range(4):
        nodes[f"d4_{i}"] = x[:, 4*i:4*i+4].prod(dim=1)
    # degree-8: halves (0-7),(8-15)
    nodes["d8_0"] = x[:, :8].prod(dim=1)
    nodes["d8_1"] = x[:, 8:16].prod(dim=1)
    return nodes


# ============================================================
# A. Target identity
# ============================================================

def test_target_identity_symmetric():
    x, root = sample_hierarchical_symmetric_inputs(20_000, K, INPUT_DIM, CPU, F32, RHO)
    computed = x[:, :K].prod(dim=1)
    assert torch.allclose(computed, root, atol=1e-5), "root != product of relevant bits"


def test_target_identity_symmetric_degree2():
    x, root = sample_hierarchical_symmetric_degree2_inputs(20_000, K, INPUT_DIM, CPU, F32, RHO)
    computed = x[:, :K].prod(dim=1)
    assert torch.allclose(computed, root, atol=1e-5), "root != product of relevant bits"


# ============================================================
# B. LR = P for every internal node (exact per-sample check)
# ============================================================

def test_internal_product_constraint_symmetric():
    x, root = sample_hierarchical_symmetric_inputs(5_000, K, INPUT_DIM, CPU, F32, RHO)
    nodes = _tree_nodes(x)
    S = root
    for i in range(4):
        d4 = nodes[f"d4_{i}"]
        d2_l = nodes[f"d2_{2*i}"]
        d2_r = nodes[f"d2_{2*i+1}"]
        assert torch.allclose(d4, d2_l * d2_r, atol=1e-5), f"d4_{i} != d2_{2*i}*d2_{2*i+1}"
    for i in range(2):
        d8 = nodes[f"d8_{i}"]
        d4_l = nodes[f"d4_{2*i}"]
        d4_r = nodes[f"d4_{2*i+1}"]
        assert torch.allclose(d8, d4_l * d4_r, atol=1e-5), f"d8_{i} != d4_{2*i}*d4_{2*i+1}"
    assert torch.allclose(S, nodes["d8_0"] * nodes["d8_1"], atol=1e-5), "root != d8_0*d8_1"


def test_degree2_pair_product_symmetric_degree2():
    x, root = sample_hierarchical_symmetric_degree2_inputs(5_000, K, INPUT_DIM, CPU, F32, RHO)
    for i in range(8):
        z = x[:, 2*i] * x[:, 2*i+1]
        d2 = _tree_nodes(x)[f"d2_{i}"]
        assert torch.allclose(z, d2, atol=1e-5)


# ============================================================
# C. Degree-8 symmetry: E[S*d8_0] == E[S*d8_1] ~ rho/2
# ============================================================

def test_degree8_symmetry_symmetric():
    x, root = sample_hierarchical_symmetric_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    nodes = _tree_nodes(x)
    c0 = _root_corr(S, nodes["d8_0"])
    c1 = _root_corr(S, nodes["d8_1"])
    expected = RHO / 2
    tol = 0.01
    assert abs(c0 - expected) < tol, f"E[S*d8_0]={c0:.4f}, expected~{expected}"
    assert abs(c1 - expected) < tol, f"E[S*d8_1]={c1:.4f}, expected~{expected}"
    assert abs(c0 - c1) < tol, f"d8 nodes differ: {c0:.4f} vs {c1:.4f}"
    print(f"  d8: c0={c0:.4f} c1={c1:.4f} expected={expected:.4f}  [OK]")


def test_degree8_symmetry_symmetric_degree2():
    x, root = sample_hierarchical_symmetric_degree2_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    nodes = _tree_nodes(x)
    c0 = _root_corr(S, nodes["d8_0"])
    c1 = _root_corr(S, nodes["d8_1"])
    expected = RHO / 2
    tol = 0.01
    assert abs(c0 - expected) < tol, f"E[S*d8_0]={c0:.4f}, expected~{expected}"
    assert abs(c1 - expected) < tol, f"E[S*d8_1]={c1:.4f}, expected~{expected}"
    assert abs(c0 - c1) < tol, f"d8 nodes differ: {c0:.4f} vs {c1:.4f}"


# ============================================================
# D. Degree-4 symmetry: all 4 nodes ~ (rho/2)^2
# ============================================================

def test_degree4_symmetry_symmetric():
    x, root = sample_hierarchical_symmetric_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    nodes = _tree_nodes(x)
    corrs = [_root_corr(S, nodes[f"d4_{i}"]) for i in range(4)]
    expected = (RHO / 2) ** 2
    tol = 0.01
    for i, c in enumerate(corrs):
        assert abs(c - expected) < tol, f"E[S*d4_{i}]={c:.4f}, expected~{expected:.4f}"
    vals = torch.tensor(corrs)
    assert vals.std().item() < tol, f"d4 variance too large: {corrs}"
    print(f"  d4: {[f'{c:.4f}' for c in corrs]}  expected={expected:.4f}  [OK]")


def test_degree4_symmetry_symmetric_degree2():
    x, root = sample_hierarchical_symmetric_degree2_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    nodes = _tree_nodes(x)
    corrs = [_root_corr(S, nodes[f"d4_{i}"]) for i in range(4)]
    expected = (RHO / 2) ** 2
    tol = 0.01
    for i, c in enumerate(corrs):
        assert abs(c - expected) < tol, f"E[S*d4_{i}]={c:.4f}, expected~{expected:.4f}"


# ============================================================
# E. Degree-2 symmetry: all 8 nodes ~ (rho/2)^3
# ============================================================

def test_degree2_symmetry_symmetric():
    x, root = sample_hierarchical_symmetric_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    nodes = _tree_nodes(x)
    corrs = [_root_corr(S, nodes[f"d2_{i}"]) for i in range(8)]
    expected = (RHO / 2) ** 3
    tol = 0.01
    for i, c in enumerate(corrs):
        assert abs(c - expected) < tol, f"E[S*d2_{i}]={c:.4f}, expected~{expected:.4f}"
    vals = torch.tensor(corrs)
    assert vals.std().item() < tol, f"d2 variance too large: {corrs}"
    print(f"  d2: {[f'{c:.4f}' for c in corrs]}  expected={expected:.4f}  [OK]")


def test_degree2_symmetry_symmetric_degree2():
    x, root = sample_hierarchical_symmetric_degree2_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    nodes = _tree_nodes(x)
    corrs = [_root_corr(S, nodes[f"d2_{i}"]) for i in range(8)]
    expected = (RHO / 2) ** 3
    tol = 0.01
    for i, c in enumerate(corrs):
        assert abs(c - expected) < tol, f"E[S*d2_{i}]={c:.4f}, expected~{expected:.4f}"


# ============================================================
# F. Degree-1 correlations E[S*x_j]
# ============================================================

def test_degree1_corr_symmetric():
    """Full symmetric: E[S*x_j] should be ~ (rho/2)^4 (nonzero but equal across bits)."""
    x, root = sample_hierarchical_symmetric_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    corrs = [_root_corr(S, x[:, j]) for j in range(K)]
    expected = (RHO / 2) ** 4
    tol = 0.012
    vals = torch.tensor(corrs)
    assert vals.std().item() < tol, f"d1 variance too large across bits: {corrs}"
    # Mean should be close to expected
    assert abs(vals.mean().item() - expected) < tol, (
        f"E[S*x_j] mean={vals.mean():.4f}, expected~{expected:.4f}"
    )
    print(f"  d1 (full sym): mean={vals.mean():.5f} std={vals.std():.5f}  expected={expected:.5f}")


def test_degree1_corr_zero_symmetric_degree2():
    """Truncated symmetric: E[S*x_j] must be ~0 for all relevant bits."""
    x, root = sample_hierarchical_symmetric_degree2_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    S = root
    for j in range(K):
        c = _root_corr(S, x[:, j])
        assert abs(c) < 0.015, f"E[S*x_{j}]={c:.4f}, expected ~0"
    print(f"  d1 (sym_deg2): all |E[S*x_j]| < 0.015  [OK]")


def test_bit_means_zero_symmetric_degree2():
    """Truncated symmetric: E[x_j] must be ~0 for all relevant bits."""
    x, _ = sample_hierarchical_symmetric_degree2_inputs(N, K, INPUT_DIM, CPU, F32, RHO)
    for j in range(K):
        m = x[:, j].mean().item()
        assert abs(m) < 0.015, f"E[x_{j}]={m:.4f}, expected ~0"


# ============================================================
# G. rho=0 recovers uniform distribution
# ============================================================

def test_rho0_symmetric():
    x_sym, _ = sample_hierarchical_symmetric_inputs(N, K, INPUT_DIM, CPU, F32, rho=0.0)
    for j in range(K):
        m = x_sym[:, j].mean().item()
        assert abs(m) < 0.015, f"bit {j} mean={m:.4f} at rho=0"
    # Check parity product constraint still holds
    root = x_sym[:, :K].prod(dim=1)
    assert abs(root.mean().item()) < 0.015


def test_rho0_symmetric_degree2():
    x_sym, _ = sample_hierarchical_symmetric_degree2_inputs(N, K, INPUT_DIM, CPU, F32, rho=0.0)
    for j in range(K):
        m = x_sym[:, j].mean().item()
        assert abs(m) < 0.015, f"bit {j} mean={m:.4f} at rho=0"


# ============================================================
# H. Compare old and new asymmetric vs symmetric:
#    old asymmetric should FAIL the equal-variance test
# ============================================================

def test_old_asymmetric_NOT_symmetric():
    """The existing hierarchical generator should NOT pass the degree-2 symmetry check
    (this verifies that the new generator is actually different)."""
    x, root = sample_hierarchical_inputs(N, K, INPUT_DIM, CPU, F32, rho=RHO)
    S = root
    nodes = _tree_nodes(x)
    corrs = [_root_corr(S, nodes[f"d2_{i}"]) for i in range(8)]
    vals = torch.tensor(corrs)
    # The old generator should have large std across d2 nodes
    assert vals.std().item() > 0.02, (
        f"Old asymmetric has unexpectedly small d2 std={vals.std():.4f}; "
        "symmetric and asymmetric may be the same implementation."
    )
    print(f"  old asymmetric d2 std={vals.std():.4f} (confirmed asymmetric)  [OK]")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_target_identity_symmetric,
        test_target_identity_symmetric_degree2,
        test_internal_product_constraint_symmetric,
        test_degree2_pair_product_symmetric_degree2,
        test_degree8_symmetry_symmetric,
        test_degree8_symmetry_symmetric_degree2,
        test_degree4_symmetry_symmetric,
        test_degree4_symmetry_symmetric_degree2,
        test_degree2_symmetry_symmetric,
        test_degree2_symmetry_symmetric_degree2,
        test_degree1_corr_symmetric,
        test_degree1_corr_zero_symmetric_degree2,
        test_bit_means_zero_symmetric_degree2,
        test_rho0_symmetric,
        test_rho0_symmetric_degree2,
        test_old_asymmetric_NOT_symmetric,
    ]
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
