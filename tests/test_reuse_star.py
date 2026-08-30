"""Sanity checks for the reuse-star task data generator.

Run with:  python -m pytest tests/test_reuse_star.py -v
Or:        python tests/test_reuse_star.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parity_net.data import (
    REUSE_STAR_PRIVATE_SUPPORTS,
    REUSE_STAR_SHARED_SUPPORT,
    REUSE_STAR_TARGET_SUPPORTS,
    sample_reuse_star_inputs,
)

CPU = torch.device("cpu")
F32 = torch.float32


# ---------------------------------------------------------------------------
# A. Target parity identities
# ---------------------------------------------------------------------------

def test_target_identities():
    """S1=A*B, S2=A*C, S3=A*D hold exactly for every sample."""
    n, rho = 5000, 0.5
    x, targets = sample_reuse_star_inputs(n, rho, "hierarchical_degree2", CPU, F32)
    A = x[:, :8].prod(dim=1)
    B = x[:, 8:16].prod(dim=1)
    C = x[:, 16:24].prod(dim=1)
    D = x[:, 24:32].prod(dim=1)
    assert torch.allclose(targets[:, 0], A * B, atol=1e-5), "S1 != A*B"
    assert torch.allclose(targets[:, 1], A * C, atol=1e-5), "S2 != A*C"
    assert torch.allclose(targets[:, 2], A * D, atol=1e-5), "S3 != A*D"


def test_target_identities_uniform():
    """Same identities hold at rho=0."""
    n = 5000
    x, targets = sample_reuse_star_inputs(n, 0.0, "uniform", CPU, F32)
    A = x[:, :8].prod(dim=1)
    assert torch.allclose(targets[:, 0], A * x[:, 8:16].prod(dim=1), atol=1e-5)


# ---------------------------------------------------------------------------
# B. Degree: each target involves exactly 16 distinct bits
# ---------------------------------------------------------------------------

def test_target_degree():
    for j, supp in enumerate(REUSE_STAR_TARGET_SUPPORTS):
        assert len(set(supp)) == 16, f"S{j+1} support has {len(set(supp))} bits, expected 16"
        n, rho = 3000, 0.4
        x, targets = sample_reuse_star_inputs(n, rho, "hierarchical_degree2", CPU, F32)
        idx = torch.tensor(list(supp), dtype=torch.long)
        computed = x[:, idx].prod(dim=1)
        assert torch.allclose(computed, targets[:, j], atol=1e-5), f"S{j+1} != product of support bits"


# ---------------------------------------------------------------------------
# C. Shared overlap: every pair of targets shares exactly the A-block
# ---------------------------------------------------------------------------

def test_shared_overlap():
    for j in range(3):
        for k in range(j + 1, 3):
            sj = set(REUSE_STAR_TARGET_SUPPORTS[j])
            sk = set(REUSE_STAR_TARGET_SUPPORTS[k])
            shared = sj & sk
            assert shared == set(REUSE_STAR_SHARED_SUPPORT), (
                f"S{j+1} ∩ S{k+1} = {sorted(shared)}, expected A-block {sorted(REUSE_STAR_SHARED_SUPPORT)}"
            )


# ---------------------------------------------------------------------------
# D. Private groups are pairwise disjoint
# ---------------------------------------------------------------------------

def test_private_disjoint():
    for j in range(3):
        for k in range(j + 1, 3):
            pj = set(REUSE_STAR_PRIVATE_SUPPORTS[j])
            pk = set(REUSE_STAR_PRIVATE_SUPPORTS[k])
            assert pj.isdisjoint(pk), f"B{j+1} and B{k+1} share bits: {pj & pk}"


# ---------------------------------------------------------------------------
# E. Statistical properties
# ---------------------------------------------------------------------------

def test_zero_means_rho0():
    """At rho=0: E[A]=E[B]=E[C]=E[D]=E[Sj]=0."""
    n = 200_000
    x, targets = sample_reuse_star_inputs(n, 0.0, "uniform", CPU, F32)
    for name, vals in [
        ("A", x[:, :8].prod(dim=1)),
        ("B", x[:, 8:16].prod(dim=1)),
        ("C", x[:, 16:24].prod(dim=1)),
        ("D", x[:, 24:32].prod(dim=1)),
        ("S1", targets[:, 0]),
        ("S2", targets[:, 1]),
        ("S3", targets[:, 2]),
    ]:
        mean = vals.mean().item()
        assert abs(mean) < 0.015, f"E[{name}] = {mean:.4f} at rho=0, expected ~0"


def test_A_biased_rho_pos():
    """E[A] ≈ rho for rho > 0."""
    n, rho = 200_000, 0.5
    x, _ = sample_reuse_star_inputs(n, rho, "hierarchical_degree2", CPU, F32)
    A = x[:, :8].prod(dim=1)
    mean_A = A.mean().item()
    assert abs(mean_A - rho) < 0.02, f"E[A] = {mean_A:.4f}, expected ≈ {rho}"


def test_private_constituents_balanced():
    """E[B]=E[C]=E[D]≈0 even when rho>0 (because S_j are balanced)."""
    n, rho = 200_000, 0.6
    x, _ = sample_reuse_star_inputs(n, rho, "hierarchical_degree2", CPU, F32)
    for name, sl in [("B", slice(8, 16)), ("C", slice(16, 24)), ("D", slice(24, 32))]:
        mean = x[:, sl].prod(dim=1).mean().item()
        assert abs(mean) < 0.02, f"E[{name}] = {mean:.4f}, expected ~0"


def test_shared_has_no_root_correlation():
    """E[Sj * A] ≈ 0 for all j (A has no direct root correlation)."""
    n, rho = 200_000, 0.6
    x, targets = sample_reuse_star_inputs(n, rho, "hierarchical_degree2", CPU, F32)
    A = x[:, :8].prod(dim=1)
    for j in range(3):
        corr = (targets[:, j] * A).mean().item()
        assert abs(corr) < 0.02, f"E[S{j+1}*A] = {corr:.4f}, expected ~0 (rho={rho})"


def test_private_directly_correlated_with_root():
    """E[S1*B] ≈ rho, E[S2*C] ≈ rho, E[S3*D] ≈ rho."""
    n, rho = 200_000, 0.5
    x, targets = sample_reuse_star_inputs(n, rho, "hierarchical_degree2", CPU, F32)
    pairs = [
        ("S1", "B", targets[:, 0], x[:, 8:16].prod(dim=1)),
        ("S2", "C", targets[:, 1], x[:, 16:24].prod(dim=1)),
        ("S3", "D", targets[:, 2], x[:, 24:32].prod(dim=1)),
    ]
    for sname, bname, s_vals, b_vals in pairs:
        corr = (s_vals * b_vals).mean().item()
        assert abs(corr - rho) < 0.02, f"E[{sname}*{bname}] = {corr:.4f}, expected ≈ {rho}"


def test_cross_correlations_zero():
    """E[S1*C] ≈ 0, E[S1*D] ≈ 0, etc. (cross-root/private correlations)."""
    n, rho = 200_000, 0.6
    x, targets = sample_reuse_star_inputs(n, rho, "hierarchical_degree2", CPU, F32)
    B = x[:, 8:16].prod(dim=1)
    C = x[:, 16:24].prod(dim=1)
    D = x[:, 24:32].prod(dim=1)
    cross = [
        ("S1", "C", targets[:, 0], C),
        ("S1", "D", targets[:, 0], D),
        ("S2", "B", targets[:, 1], B),
        ("S2", "D", targets[:, 1], D),
        ("S3", "B", targets[:, 2], B),
        ("S3", "C", targets[:, 2], C),
    ]
    for sn, bn, sv, bv in cross:
        corr = (sv * bv).mean().item()
        assert abs(corr) < 0.02, f"E[{sn}*{bn}] = {corr:.4f}, expected ~0"


def test_roots_orthogonal():
    """E[Si*Sj] ≈ 0 for i≠j (targets are orthogonal under uniform measure)."""
    n = 200_000
    x, targets = sample_reuse_star_inputs(n, 0.0, "uniform", CPU, F32)
    for j in range(3):
        for k in range(j + 1, 3):
            corr = (targets[:, j] * targets[:, k]).mean().item()
            assert abs(corr) < 0.02, f"E[S{j+1}*S{k+1}] = {corr:.4f}, expected ~0"


# ---------------------------------------------------------------------------
# E. rho=0 gives uniform iid inputs (all bits balanced, pairwise uncorrelated)
# ---------------------------------------------------------------------------

def test_rho0_uniform_bits():
    n = 100_000
    x, _ = sample_reuse_star_inputs(n, 0.0, "uniform", CPU, F32)
    for j in range(32):
        mean = x[:, j].mean().item()
        assert abs(mean) < 0.02, f"bit {j} mean = {mean:.4f} at rho=0"
    # Spot-check pairwise correlations (same block and cross-block)
    for (i, j) in [(0, 1), (0, 8), (0, 16), (7, 15), (4, 20)]:
        corr = (x[:, i] * x[:, j]).mean().item()
        assert abs(corr) < 0.02, f"corr(bit_{i}, bit_{j}) = {corr:.4f} at rho=0"


# ---------------------------------------------------------------------------
# F. Loss normalization: increasing m doesn't grow total gradient scale
# ---------------------------------------------------------------------------

def test_loss_scale_independent_of_m():
    """At init, MSE per active target should be ~1 regardless of m."""
    n_batch = 1024
    for m in [1, 2, 3]:
        x, y = sample_reuse_star_inputs(n_batch, 0.0, "uniform", CPU, F32)
        # Simulate near-zero init predictions
        pred = torch.zeros(n_batch, 3)
        import torch.nn.functional as F
        per_target = F.mse_loss(pred, y, reduction="none").mean(dim=0)
        active_mask = torch.zeros(3)
        active_mask[:m] = 1.0
        loss = (per_target * active_mask).sum() / active_mask.sum()
        # MSE of 0 vs ±1 Rademacher = 1.0
        assert abs(loss.item() - 1.0) < 0.05, f"m={m}: loss={loss.item():.4f}, expected ~1.0"


# ---------------------------------------------------------------------------
# Independence from m: same (x, targets) regardless of num_reuse_targets
# ---------------------------------------------------------------------------

def test_data_independent_of_m():
    """Same seed produces identical data regardless of m (m only affects mask)."""
    gen1 = torch.Generator(CPU)
    gen1.manual_seed(42)
    x1, y1 = sample_reuse_star_inputs(1000, 0.3, "hierarchical_degree2", CPU, F32, gen1)

    gen2 = torch.Generator(CPU)
    gen2.manual_seed(42)
    x2, y2 = sample_reuse_star_inputs(1000, 0.3, "hierarchical_degree2", CPU, F32, gen2)

    assert torch.allclose(x1, x2, atol=0), "x differs between runs with same seed"
    assert torch.allclose(y1, y2, atol=0), "y differs between runs with same seed"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_target_identities,
        test_target_identities_uniform,
        test_target_degree,
        test_shared_overlap,
        test_private_disjoint,
        test_zero_means_rho0,
        test_A_biased_rho_pos,
        test_private_constituents_balanced,
        test_shared_has_no_root_correlation,
        test_private_directly_correlated_with_root,
        test_cross_correlations_zero,
        test_roots_orthogonal,
        test_rho0_uniform_bits,
        test_loss_scale_independent_of_m,
        test_data_independent_of_m,
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
