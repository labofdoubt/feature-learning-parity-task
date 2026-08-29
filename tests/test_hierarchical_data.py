"""Sanity checks for the hierarchical non-uniform parity data generator.

Run with:  python -m pytest tests/test_hierarchical_data.py -v
Or:        python tests/test_hierarchical_data.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parity_net.data import (
    make_hierarchical_dataset,
    make_uniform_eval_dataset,
    sample_hierarchical_inputs,
    tree_parity_specs,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prod_relevant(x: torch.Tensor, k: int) -> torch.Tensor:
    """Product of the first k bits for each sample."""
    return x[:, :k].prod(dim=1)


def _check_parent_child(x: torch.Tensor, k: int) -> None:
    """For every internal node in the parity tree, verify parent == left * right."""
    # tree_parity_specs gives targets in ascending degree order.
    # For each non-leaf level, a target's value = product of two half-degree children.
    specs = tree_parity_specs(k)
    spec_map = {s.indices: s for s in specs}
    vals = {s.indices: x[:, list(s.indices)].prod(dim=1) for s in specs}

    for spec in specs:
        if len(spec.indices) == 1:
            continue
        half = len(spec.indices) // 2
        left_idx = spec.indices[:half]
        right_idx = spec.indices[half:]
        parent_val = vals[spec.indices]
        left_val = vals.get(left_idx) if left_idx in vals else x[:, list(left_idx)].prod(dim=1)
        right_val = vals.get(right_idx) if right_idx in vals else x[:, list(right_idx)].prod(dim=1)
        product = left_val * right_val
        assert torch.allclose(parent_val, product, atol=1e-5), (
            f"Node {spec.name}: parent != left*right  (max diff "
            f"{(parent_val - product).abs().max():.2e})"
        )


# ---------------------------------------------------------------------------
# Test: label correctness (y == product of leaves)
# ---------------------------------------------------------------------------

def test_label_correctness_k4():
    k, d = 4, 4
    n = 5000
    x, root = sample_hierarchical_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=0.5)
    assert x.shape == (n, d)
    prod = _prod_relevant(x, k)
    assert torch.allclose(root, prod, atol=1e-5), "root parity != product of leaves (k=4)"


def test_label_correctness_k8():
    k, d = 8, 8
    n = 5000
    x, root = sample_hierarchical_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=0.3)
    prod = _prod_relevant(x, k)
    assert torch.allclose(root, prod, atol=1e-5), "root parity != product of leaves (k=8)"


def test_label_correctness_k16():
    k, d = 16, 32
    n = 5000
    x, root = sample_hierarchical_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=0.7)
    assert x.shape == (n, d)
    prod = _prod_relevant(x, k)
    assert torch.allclose(root, prod, atol=1e-5), "root parity != product of leaves (k=16)"


# ---------------------------------------------------------------------------
# Test: parent == left * right at every internal node
# ---------------------------------------------------------------------------

def test_parent_child_consistency_k4():
    k = 4
    n = 2000
    x, _ = sample_hierarchical_inputs(n, k, k, torch.device("cpu"), torch.float32, rho=0.6)
    _check_parent_child(x, k)


def test_parent_child_consistency_k8():
    k = 8
    n = 2000
    x, _ = sample_hierarchical_inputs(n, k, k, torch.device("cpu"), torch.float32, rho=0.4)
    _check_parent_child(x, k)


def test_parent_child_consistency_k16():
    k = 16
    n = 2000
    x, _ = sample_hierarchical_inputs(n, k, k, torch.device("cpu"), torch.float32, rho=0.5)
    _check_parent_child(x, k)


# ---------------------------------------------------------------------------
# Test: biased child – E[L] ≈ rho at each internal node
# ---------------------------------------------------------------------------

def test_biased_child_mean_k4():
    """E[L] ≈ rho where L is the left child at each internal node.

    The top-level left child for k=4 is L_0 = x0*x1 (the d2_0 target).
    Its marginal expectation is rho because P(L_0=+1) = (1+rho)/2 unconditionally.
    The right child R_0 = root*L_0 has E[R_0] = E[root]*E[L_0] = 0, since root is
    uniform and independent of L_0's coin flip.
    """
    k, d, rho = 4, 4, 0.6
    n = 200_000
    x, _ = sample_hierarchical_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    # Left child at level 1 (top-level split): product of first k/2 bits
    L_top = x[:, 0] * x[:, 1]
    mean_L = L_top.mean().item()
    assert abs(mean_L - rho) < 0.015, f"E[top-left child] = {mean_L:.4f}, expected {rho:.4f}"

    # Left children at level 2 are the individual bits x0 and x2 (left children of L_0 and R_0)
    mean_x0 = x[:, 0].mean().item()
    mean_x2 = x[:, 2].mean().item()
    assert abs(mean_x0 - rho) < 0.015, f"E[x0] = {mean_x0:.4f}, expected {rho:.4f}"
    assert abs(mean_x2 - rho) < 0.015, f"E[x2] = {mean_x2:.4f}, expected {rho:.4f}"

    # Right children: x1 = L_0*x0  =>  E[x1] = rho^2;  x3 = R_0*x2  =>  E[x3] = 0
    mean_x1 = x[:, 1].mean().item()
    assert abs(mean_x1 - rho ** 2) < 0.015, f"E[x1] = {mean_x1:.4f}, expected {rho**2:.4f}"
    mean_x3 = x[:, 3].mean().item()
    assert abs(mean_x3) < 0.015, f"E[x3] = {mean_x3:.4f}, expected 0"


def test_biased_child_mean_k8():
    """E[level-1 left child] ≈ rho for k=8."""
    k, d, rho = 8, 8, 0.4
    n = 200_000
    x, _ = sample_hierarchical_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    # Top-level left child = product of first k/2 = 4 bits
    L_top = x[:, :4].prod(dim=1)
    mean_L = L_top.mean().item()
    assert abs(mean_L - rho) < 0.015, f"E[top-left child] = {mean_L:.4f}, expected {rho:.4f}"
    # Top-level right child E[R] = 0 (root uniform, independent of L coin)
    R_top = x[:, 4:8].prod(dim=1)
    mean_R = R_top.mean().item()
    assert abs(mean_R) < 0.015, f"E[top-right child] = {mean_R:.4f}, expected 0"


# ---------------------------------------------------------------------------
# Test: rho=0 gives uniform marginals on leaves
# ---------------------------------------------------------------------------

def test_rho0_uniform_marginals_k4():
    k = 4
    n = 50_000
    x, root = sample_hierarchical_inputs(n, k, k, torch.device("cpu"), torch.float32, rho=0.0)
    # Each bit should be mean ~0 (balanced)
    for i in range(k):
        mean_i = x[:, i].mean().item()
        assert abs(mean_i) < 0.03, f"bit {i} mean = {mean_i:.4f}, expected ~0 for rho=0"
    # Pairwise correlations should be ~0
    for i in range(k):
        for j in range(i + 1, k):
            corr = (x[:, i] * x[:, j]).mean().item()
            assert abs(corr) < 0.03, f"corr(bit {i}, bit {j}) = {corr:.4f}, expected ~0 for rho=0"


def test_rho0_uniform_marginals_k8():
    k = 8
    n = 100_000
    x, _ = sample_hierarchical_inputs(n, k, k, torch.device("cpu"), torch.float32, rho=0.0)
    for i in range(k):
        mean_i = x[:, i].mean().item()
        assert abs(mean_i) < 0.02, f"bit {i} mean = {mean_i:.4f}"


# ---------------------------------------------------------------------------
# Test: rho=0 generator output agrees statistically with uniform distribution
# ---------------------------------------------------------------------------

def test_rho0_matches_uniform_k4():
    """With rho=0, bit product distribution should match uniform parity."""
    k = 4
    n = 50_000
    x_hier, root_hier = sample_hierarchical_inputs(n, k, k, torch.device("cpu"), torch.float32, rho=0.0)
    # Root should be balanced ±1
    frac_pos = (root_hier > 0).float().mean().item()
    assert abs(frac_pos - 0.5) < 0.02, f"root +1 fraction = {frac_pos:.4f}, expected 0.5"
    # All 2^4 = 16 leaf patterns should appear with roughly equal frequency
    keys = ((x_hier + 1) / 2).long()  # {0,1}^4
    packed = (keys * torch.tensor([1, 2, 4, 8])).sum(dim=1)
    counts = torch.bincount(packed, minlength=16).float()
    expected = n / 16.0
    # Chi-squared-like: max relative deviation
    max_rel = ((counts - expected).abs() / expected).max().item()
    assert max_rel < 0.1, f"max relative freq deviation = {max_rel:.4f} (rho=0 should be uniform)"


# ---------------------------------------------------------------------------
# Test: irrelevant bits are always independent uniform
# ---------------------------------------------------------------------------

def test_irrelevant_bits_uniform_k4():
    k, d, rho = 4, 16, 0.9
    n = 50_000
    x, _ = sample_hierarchical_inputs(n, k, d, torch.device("cpu"), torch.float32, rho=rho)
    for j in range(k, d):
        mean_j = x[:, j].mean().item()
        assert abs(mean_j) < 0.03, f"irrel bit {j} mean = {mean_j:.4f}"
        # Also check independence from relevant bits
        corr = (x[:, j] * x[:, 0]).mean().item()
        assert abs(corr) < 0.03, f"irrel bit {j} corr with bit 0 = {corr:.4f}"


# ---------------------------------------------------------------------------
# Test: make_uniform_eval_dataset – exhaustive relevant coverage
# ---------------------------------------------------------------------------

def test_uniform_eval_exhaustive_k4():
    k, d = 4, 8
    noise_rep = 3
    ds = make_uniform_eval_dataset(k, d, torch.device("cpu"), torch.float32,
                                   eval_noise_repeats=noise_rep, seed=0)
    expected_n = (2 ** k) * noise_rep
    assert ds.x.shape == (expected_n, d), f"shape {ds.x.shape}, expected ({expected_n}, {d})"
    # All relevant configs appear exactly noise_rep times
    rel = ds.x[:, :k]
    packed = ((rel + 1) / 2).long()
    packed_keys = (packed * torch.tensor([1, 2, 4, 8])).sum(dim=1)
    counts = torch.bincount(packed_keys, minlength=2 ** k)
    assert (counts == noise_rep).all(), f"some configs appear != {noise_rep} times: {counts}"


def test_uniform_eval_exhaustive_k8():
    k, d = 8, 16
    noise_rep = 2
    ds = make_uniform_eval_dataset(k, d, torch.device("cpu"), torch.float32,
                                   eval_noise_repeats=noise_rep, seed=0)
    assert ds.x.shape[0] == (2 ** k) * noise_rep


def test_uniform_eval_label_correctness():
    k, d = 4, 4
    ds = make_uniform_eval_dataset(k, d, torch.device("cpu"), torch.float32,
                                   eval_noise_repeats=1, seed=0,
                                   exclude_targets=["d2"])  # only d4, d8, d16 if applicable for k=4
    # With k=4 and no d2, targets are d4 (the root for k=4)
    prod = ds.x[:, :k].prod(dim=1)
    # y last column = d4 root
    assert torch.allclose(ds.y[:, -1], prod, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: make_hierarchical_dataset label correctness via tree_parity_specs
# ---------------------------------------------------------------------------

def test_hierarchical_dataset_labels_k8():
    k, d, rho = 8, 16, 0.5
    ds = make_hierarchical_dataset(500, rho, k, d, torch.device("cpu"), torch.float32)
    # Last target column = d8 root parity
    prod = ds.x[:, :k].prod(dim=1)
    assert torch.allclose(ds.y[:, -1], prod, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: rho validation
# ---------------------------------------------------------------------------

def test_rho_validation():
    import pytest
    with pytest.raises(ValueError, match="rho"):
        sample_hierarchical_inputs(10, 4, 4, torch.device("cpu"), torch.float32, rho=1.0)
    with pytest.raises(ValueError, match="rho"):
        sample_hierarchical_inputs(10, 4, 4, torch.device("cpu"), torch.float32, rho=-0.1)


def test_nonpower2_relevant_dim_raises():
    import pytest
    with pytest.raises(ValueError, match="power of 2"):
        sample_hierarchical_inputs(10, 6, 8, torch.device("cpu"), torch.float32, rho=0.3)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest dependency)
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_label_correctness_k4,
        test_label_correctness_k8,
        test_label_correctness_k16,
        test_parent_child_consistency_k4,
        test_parent_child_consistency_k8,
        test_parent_child_consistency_k16,
        test_biased_child_mean_k4,
        test_biased_child_mean_k8,
        test_rho0_uniform_marginals_k4,
        test_rho0_uniform_marginals_k8,
        test_rho0_matches_uniform_k4,
        test_irrelevant_bits_uniform_k4,
        test_uniform_eval_exhaustive_k4,
        test_uniform_eval_exhaustive_k8,
        test_uniform_eval_label_correctness,
        test_hierarchical_dataset_labels_k8,
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
