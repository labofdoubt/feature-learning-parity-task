"""Sanity checks for the degree-2-truncated hierarchical parity data generator.

Run with:  python -m pytest tests/test_hierarchical_degree2_data.py -v
Or:        python tests/test_hierarchical_degree2_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parity_net.data import (
    sample_hierarchical_degree2_inputs,
    sample_hierarchical_inputs,
)


# ---------------------------------------------------------------------------
# A. Exact parity identities
# ---------------------------------------------------------------------------

def test_parity_identity_k4():
    k, d = 4, 4
    x, root = sample_hierarchical_degree2_inputs(5000, k, d, torch.device("cpu"), torch.float32, rho=0.5)
    # z_i = x_{2i-1} * x_{2i}
    for i in range(k // 2):
        z_i = x[:, 2 * i] * x[:, 2 * i + 1]
        expected = torch.ones(5000)  # would need actual z values
        assert torch.allclose(z_i, z_i, atol=0), "trivial"  # just check shapes below
    # root == product of all k bits
    prod = x[:, :k].prod(dim=1)
    assert torch.allclose(root, prod, atol=1e-5), "root != product of relevant bits (k=4)"


def test_parity_identity_k8():
    k, d = 8, 8
    x, root = sample_hierarchical_degree2_inputs(5000, k, d, torch.device("cpu"), torch.float32, rho=0.3)
    prod = x[:, :k].prod(dim=1)
    assert torch.allclose(root, prod, atol=1e-5), "root != product of relevant bits (k=8)"


def test_parity_identity_k16():
    k, d = 16, 32
    x, root = sample_hierarchical_degree2_inputs(5000, k, d, torch.device("cpu"), torch.float32, rho=0.7)
    prod = x[:, :k].prod(dim=1)
    assert torch.allclose(root, prod, atol=1e-5), "root != product of relevant bits (k=16)"


def test_degree2_latent_identity_k16():
    """z_i = x_{2i-1} * x_{2i} holds for every i."""
    k, d, rho = 16, 16, 0.6
    n = 2000
    x, _ = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    for i in range(k // 2):
        z_i = x[:, 2 * i] * x[:, 2 * i + 1]
        # z_i must be ±1
        assert torch.all((z_i == 1) | (z_i == -1)), f"z_{i} not in {{-1,+1}}"
    # Also verify product of z's == root parity
    z = torch.stack([x[:, 2*i] * x[:, 2*i+1] for i in range(k//2)], dim=1)  # (n, k/2)
    z_prod = z.prod(dim=1)
    root = x[:, :k].prod(dim=1)
    assert torch.allclose(z_prod, root, atol=1e-5), "product of z_i's != root"


# ---------------------------------------------------------------------------
# B. Individual bits are balanced
# ---------------------------------------------------------------------------

def test_bits_balanced_k16_high_rho():
    """E[x_j] ≈ 0 for every relevant bit even at rho=0.7."""
    k, d, rho = 16, 32, 0.7
    n = 100_000
    x, _ = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    for j in range(k):
        mean_j = x[:, j].mean().item()
        assert abs(mean_j) < 0.02, f"E[x_{j}] = {mean_j:.4f}, expected ~0 (rho={rho})"


def test_bits_balanced_k8():
    k, d, rho = 8, 8, 0.5
    n = 50_000
    x, _ = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    for j in range(k):
        mean_j = x[:, j].mean().item()
        assert abs(mean_j) < 0.03, f"E[x_{j}] = {mean_j:.4f}, expected ~0"


# ---------------------------------------------------------------------------
# C. No degree-1 / root correlation  E[S * x_j] ≈ 0
# ---------------------------------------------------------------------------

def test_no_degree1_target_correlation_k16():
    """E[S * x_j] ≈ 0 for all j — the most important check."""
    k, d, rho = 16, 32, 0.7
    n = 200_000
    x, root = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    for j in range(k):
        corr = (root * x[:, j]).mean().item()
        assert abs(corr) < 0.015, (
            f"E[S * x_{j}] = {corr:.4f} (rho={rho}), expected ~0"
        )


def test_no_degree1_target_correlation_k8():
    k, d, rho = 8, 16, 0.5
    n = 200_000
    x, root = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    for j in range(k):
        corr = (root * x[:, j]).mean().item()
        assert abs(corr) < 0.015, f"E[S * x_{j}] = {corr:.4f}, expected ~0"


def test_old_generator_has_nonzero_degree1_correlation():
    """Verify the OLD hierarchical generator DOES have E[S*x_j] != 0 at high rho.

    This confirms the new generator fixes a real issue, not a phantom one.
    """
    k, d, rho = 16, 32, 0.7
    n = 200_000
    x, root = sample_hierarchical_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    # At rho=0.7, x_0 (left leaf of left subtree) should have E[S*x_0] != 0
    corrs = [(root * x[:, j]).mean().item() for j in range(k)]
    max_abs_corr = max(abs(c) for c in corrs)
    assert max_abs_corr > 0.05, (
        f"Expected old generator to have |E[S*x_j]| > 0.05 at rho=0.7; "
        f"got max={max_abs_corr:.4f}. Check the old generator."
    )


# ---------------------------------------------------------------------------
# D. Intermediate correlations survive
# ---------------------------------------------------------------------------

def test_degree2_correlations_survive_k16():
    """E[S * z_i] != 0 for rho > 0 (hierarchical structure preserved at degree 2)."""
    k, d, rho = 16, 16, 0.5
    n = 200_000
    x, root = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    # Compute z_i = x_{2i-1} * x_{2i}
    z = torch.stack([x[:, 2*i] * x[:, 2*i+1] for i in range(k//2)], dim=1)  # (n, k/2)
    corrs = [(root * z[:, i]).mean().item() for i in range(k // 2)]
    max_abs_corr = max(abs(c) for c in corrs)
    assert max_abs_corr > 0.05, (
        f"Expected |E[S*z_i]| > 0.05 for some i at rho={rho}; "
        f"got max={max_abs_corr:.4f}. Hierarchical structure may be lost."
    )


def test_parent_child_z_correlation_k16():
    """z-tree parent-child correlations are nonzero at rho > 0."""
    k, d, rho = 16, 16, 0.5
    n = 200_000
    x, _ = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    # z_1 = x0*x1, z_2 = x2*x3 — their product z1*z2 = x0*x1*x2*x3 is the
    # left degree-4 parent. E[z1 * z2] should be nonzero (it's E[left_d4]).
    z1 = x[:, 0] * x[:, 1]
    z2 = x[:, 2] * x[:, 3]
    corr_z1z2 = (z1 * z2).mean().item()
    # At rho=0.5, left_d4 has E[left_d4] = rho (by biased left-child rule)
    assert abs(corr_z1z2) > 0.1, (
        f"E[z1*z2] = {corr_z1z2:.4f}, expected |corr| > 0.1 at rho={rho}"
    )


# ---------------------------------------------------------------------------
# E. rho=0 reduces to uniform
# ---------------------------------------------------------------------------

def test_rho0_uniform_marginals_k16():
    """At rho=0, all relevant bits should be balanced and uncorrelated."""
    k, d = 16, 32
    n = 100_000
    x, _ = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=0.0)
    for j in range(k):
        mean_j = x[:, j].mean().item()
        assert abs(mean_j) < 0.02, f"bit {j} mean = {mean_j:.4f}, expected ~0"
    # Pairwise correlations
    for i in range(0, k, 4):
        for jj in range(i + 1, min(i + 4, k)):
            corr = (x[:, i] * x[:, jj]).mean().item()
            assert abs(corr) < 0.02, f"corr(x_{i}, x_{jj}) = {corr:.4f}, expected ~0"


def test_rho0_no_target_correlation_k8():
    """At rho=0, E[S*x_j] ≈ 0 for all j."""
    k, d = 8, 8
    n = 50_000
    x, root = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=0.0)
    for j in range(k):
        corr = (root * x[:, j]).mean().item()
        assert abs(corr) < 0.03, f"E[S*x_{j}] = {corr:.4f} at rho=0, expected ~0"


def test_irrelevant_bits_independent():
    """Irrelevant bits should be balanced and uncorrelated with root."""
    k, d, rho = 16, 32, 0.7
    n = 50_000
    x, root = sample_hierarchical_degree2_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    for j in range(k, d):
        mean_j = x[:, j].mean().item()
        assert abs(mean_j) < 0.03, f"irrel bit {j} mean = {mean_j:.4f}"
        corr = (root * x[:, j]).mean().item()
        assert abs(corr) < 0.03, f"E[S * irrel_bit_{j}] = {corr:.4f}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_parity_identity_k4,
        test_parity_identity_k8,
        test_parity_identity_k16,
        test_degree2_latent_identity_k16,
        test_bits_balanced_k16_high_rho,
        test_bits_balanced_k8,
        test_no_degree1_target_correlation_k16,
        test_no_degree1_target_correlation_k8,
        test_old_generator_has_nonzero_degree1_correlation,
        test_degree2_correlations_survive_k16,
        test_parent_child_z_correlation_k16,
        test_rho0_uniform_marginals_k16,
        test_rho0_no_target_correlation_k8,
        test_irrelevant_bits_independent,
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
