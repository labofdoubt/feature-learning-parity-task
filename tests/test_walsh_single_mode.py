"""Tests for the walsh_single_mode data distribution.

Target density: p(x) = 2^{-d} (1 + rho * chi_A(x)), chi_A(x) = prod_{i in A} x_i.
The only nonconstant Walsh moment is E[chi_A] = rho; every other nonempty mode
has mean 0.

Run with:  python -m pytest tests/test_walsh_single_mode.py -v
Or:        python tests/test_walsh_single_mode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parity_net.data import (
    check_walsh_mode_args,
    labels_from_inputs,
    make_walsh_single_mode_dataset,
    sample_walsh_single_mode_inputs,
)

CPU = torch.device("cpu")
F32 = torch.float32
N = 400_000
K = 16
INPUT_DIM = 32
# The example experiment's mode: A = {2, ..., 15} (zero-based), |A| = 14.
MODE = list(range(2, 16))
TOL = 0.01  # ±1 mean over 400k samples has std ~0.0016; 0.01 is > 6 sigma


def _sample(n=N, rho=0.5, mode=MODE, input_dim=INPUT_DIM, seed=None):
    gen = None
    if seed is not None:
        gen = torch.Generator(device=CPU)
        gen.manual_seed(seed)
    return sample_walsh_single_mode_inputs(
        n, K, input_dim, CPU, F32, rho, gen, mode_indices=mode
    )


def _chi(x: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return x[:, indices].prod(dim=1)


# ============================================================
# 1. Exact parity constraint: chi_A(x) == z per sample
# ============================================================

def test_parity_constraint_exact():
    x, z = _sample(n=20_000, rho=0.5)
    chi_A = _chi(x, MODE)
    assert torch.allclose(chi_A, z, atol=1e-6), "chi_A(x) != sampled z"


def test_parity_constraint_exact_singleton():
    x, z = _sample(n=20_000, rho=0.5, mode=[5])
    assert torch.allclose(x[:, 5], z, atol=1e-6), "singleton mode: x_5 != z"


# ============================================================
# 2. Selected mode mean: E[chi_A] ~ rho
# ============================================================

def test_mode_mean_matches_rho():
    for rho in (-0.8, 0.0, 0.3, 0.9):
        x, _ = _sample(rho=rho)
        m = _chi(x, MODE).mean().item()
        assert abs(m - rho) < TOL, f"E[chi_A]={m:.4f}, expected~{rho}"
        print(f"  rho={rho:+.1f}: E[chi_A]={m:+.4f}  [OK]")


# ============================================================
# 3. No unwanted Walsh moments
# ============================================================

OTHER_MODES = {
    "singleton_in_A": [2],
    "singleton_outside_A": [0],
    "pair_inside_A": [2, 3],
    "proper_subset_of_A": list(range(3, 15)),       # x3..x14, 12 of A's 14 indices
    "partial_overlap": [0, 1, 2],                   # one index in A, two outside
    "disjoint": [0, 1],                             # the S-triangle-A mode for S=d16
    "superset_of_A": [0] + MODE,                    # A plus one extra coordinate
    "irrelevant_region": [16, 17],
}


def test_no_unwanted_moments():
    x, _ = _sample(rho=0.9)
    for name, mode_b in OTHER_MODES.items():
        m = _chi(x, mode_b).mean().item()
        assert abs(m) < TOL, f"E[chi_B]={m:.4f} for B={name}, expected ~0"
    print(f"  all {len(OTHER_MODES)} other moments |E[chi_B]| < {TOL}  [OK]")


# ============================================================
# 4. rho = 0 is exactly uniform (statistically)
# ============================================================

def test_rho0_all_moments_zero():
    x, _ = _sample(rho=0.0)
    for name, mode_b in {**OTHER_MODES, "A_itself": MODE}.items():
        m = _chi(x, mode_b).mean().item()
        assert abs(m) < TOL, f"rho=0: E[chi_{name}]={m:.4f}, expected ~0"


# ============================================================
# 5-6. rho = ±1: parity constraint saturates exactly
# ============================================================

def test_rho_plus_one_exact():
    x, _ = _sample(n=50_000, rho=1.0)
    chi_A = _chi(x, MODE)
    assert torch.all(chi_A == 1.0), "rho=1: chi_A must be +1 on every sample"


def test_rho_minus_one_exact():
    x, _ = _sample(n=50_000, rho=-1.0)
    chi_A = _chi(x, MODE)
    assert torch.all(chi_A == -1.0), "rho=-1: chi_A must be -1 on every sample"


# ============================================================
# 7. Individual marginals stay balanced (|A| > 1), incl. bits in A
# ============================================================

def test_individual_marginals_balanced():
    x, _ = _sample(rho=0.9)
    for j in range(INPUT_DIM):
        m = x[:, j].mean().item()
        assert abs(m) < TOL, f"E[x_{j}]={m:.4f}, expected ~0 (|A|>1)"
    print(f"  all {INPUT_DIM} marginals balanced at rho=0.9  [OK]")


def test_singleton_marginal_is_rho():
    rho = 0.7
    x, _ = _sample(rho=rho, mode=[5])
    m = x[:, 5].mean().item()
    assert abs(m - rho) < TOL, f"singleton A: E[x_5]={m:.4f}, expected~{rho}"
    # other coordinates stay balanced
    for j in (0, 4, 6, 31):
        assert abs(x[:, j].mean().item()) < TOL


# ============================================================
# 8. Reproducibility with seeded generator
# ============================================================

def test_seeded_reproducibility():
    x1, z1 = _sample(n=10_000, rho=0.5, seed=123)
    x2, z2 = _sample(n=10_000, rho=0.5, seed=123)
    assert torch.equal(x1, x2) and torch.equal(z1, z2), "same seed must reproduce exactly"
    x3, _ = _sample(n=10_000, rho=0.5, seed=124)
    assert not torch.equal(x1, x3), "different seed should differ"


# ============================================================
# Exactness of the full density on a small input space
# ============================================================

def test_exact_density_small_d():
    """d=4, A={1,2}: empirical frequency of each of the 16 configs must match
    p(x) = (1 + rho*x_1*x_2)/16. There are 2^{d-1}=8 configs per parity class and
    the sampler is uniform within each class, so P(x) = P(z=chi_A(x))/8."""
    d, rho, mode = 4, 0.6, [1, 2]
    n = 400_000
    gen = torch.Generator(device=CPU)
    gen.manual_seed(0)
    x, _ = sample_walsh_single_mode_inputs(n, 4, d, CPU, F32, rho, gen, mode_indices=mode)
    bits = (x > 0).long()
    keys = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2] + 8 * bits[:, 3]
    counts = torch.bincount(keys, minlength=16).float() / n
    max_err = 0.0
    for key in range(16):
        xb = torch.tensor([(key >> i) & 1 for i in range(4)]).float() * 2 - 1
        p_target = (1 + rho * xb[1] * xb[2]) / 16
        max_err = max(max_err, abs(counts[key].item() - p_target))
    assert max_err < 0.003, f"max |empirical - target| = {max_err:.4f}"
    print(f"  exact density d=4: max deviation {max_err:.5f}  [OK]")


# ============================================================
# Dataset factory: labels are the usual tree parities
# ============================================================

def test_dataset_factory_labels():
    ds = make_walsh_single_mode_dataset(
        5_000, 0.9, K, INPUT_DIM, CPU, F32, mode_indices=MODE
    )
    y_expected = labels_from_inputs(ds.x, K)
    assert torch.allclose(ds.y, y_expected, atol=1e-6)
    # d16 root target: E[S] = E[chi_{0..15}] = 0 (S != A), while E[chi_A] = 0.9
    root = ds.x[:, :K].prod(dim=1)
    assert abs(root.mean().item()) < 0.05  # loose: only 5k samples


# ============================================================
# Validation
# ============================================================

def test_validation_errors():
    checks = [
        (dict(mode_indices=None, input_dim=32, rho=0.5), "missing mode"),
        (dict(mode_indices=[], input_dim=32, rho=0.5), "empty mode"),
        (dict(mode_indices=[1, 1], input_dim=32, rho=0.5), "duplicates"),
        (dict(mode_indices=[0, 32], input_dim=32, rho=0.5), "out of range"),
        (dict(mode_indices=[-1], input_dim=32, rho=0.5), "negative index"),
        (dict(mode_indices=[0, 1.5], input_dim=32, rho=0.5), "non-integer"),
        (dict(mode_indices=[0], input_dim=32, rho=1.5), "rho > 1"),
        (dict(mode_indices=[0], input_dim=32, rho=-1.5), "rho < -1"),
    ]
    for kwargs, label in checks:
        try:
            check_walsh_mode_args(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"validation should reject: {label}")
    # No power-of-two restriction: arbitrary sizes are valid
    for mode in ([0], [0, 1], [2, 5, 11, 14], list(range(2, 16)), [0, 1, 2]):
        check_walsh_mode_args(mode, 32, 0.5)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_parity_constraint_exact,
        test_parity_constraint_exact_singleton,
        test_mode_mean_matches_rho,
        test_no_unwanted_moments,
        test_rho0_all_moments_zero,
        test_rho_plus_one_exact,
        test_rho_minus_one_exact,
        test_individual_marginals_balanced,
        test_singleton_marginal_is_rho,
        test_seeded_reproducibility,
        test_exact_density_small_d,
        test_dataset_factory_labels,
        test_validation_errors,
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
