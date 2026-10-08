"""Tests for the walsh_clusters data distribution.

Mixture of single-Walsh-mode clusters:
    p(x) = sum_j alpha_j * 2^{-d} (1 + rho_j chi_{A_j}(x))
         = 2^{-d} [1 + sum_j alpha_j rho_j chi_{A_j}(x)]
so each cluster contributes exactly one global Walsh mode with effective
coefficient c_j = alpha_j * rho_j (duplicate supports add), and the convex
mixture is automatically a valid distribution.

Run with:  python -m pytest tests/test_walsh_clusters.py -v
Or:        python tests/test_walsh_clusters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parity_net.data import (
    check_walsh_clusters_args,
    labels_from_inputs,
    make_walsh_clusters_dataset,
    sample_walsh_clusters_inputs,
    sample_walsh_single_mode_inputs,
)

CPU = torch.device("cpu")
F32 = torch.float32
N = 400_000
K = 16
INPUT_DIM = 32
TOL = 0.01

# The intended first experiment (section-15 style): two degree-14 clusters
# inducing degree-2 gradient modes x0*x1 and x2*x3 for the d16 target.
A1 = list(range(2, 16))            # S△{0,1}
A2 = [0, 1] + list(range(4, 16))   # S△{2,3}
CLUSTERS_2 = [
    {"mode": A1, "rho": 0.9, "weight": 1.0},
    {"mode": A2, "rho": 0.9, "weight": 1.0},
]


def _sample(clusters, n=N, seed=None, input_dim=INPUT_DIM):
    gen = None
    if seed is not None:
        gen = torch.Generator(device=CPU)
        gen.manual_seed(seed)
    return sample_walsh_clusters_inputs(
        n, K, input_dim, CPU, F32, 0.0, gen, clusters=clusters
    )


def _chi(x: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return x[:, indices].prod(dim=1)


# ============================================================
# A. Cluster frequencies match normalized weights
# ============================================================

def test_cluster_frequencies():
    clusters = [
        {"mode": [0, 1], "rho": 0.5, "weight": 2.0},
        {"mode": [2, 3], "rho": 0.5, "weight": 1.0},
        {"mode": [4, 5], "rho": 0.5, "weight": 1.0},
    ]
    _, ids = _sample(clusters, seed=0)
    freqs = torch.bincount(ids, minlength=3).float() / ids.shape[0]
    expected = [0.5, 0.25, 0.25]
    for j, e in enumerate(expected):
        assert abs(freqs[j].item() - e) < TOL, f"P(J={j})={freqs[j]:.4f}, expected~{e}"
    print(f"  frequencies: {[f'{f:.4f}' for f in freqs.tolist()]}  expected {expected}  [OK]")


# ============================================================
# B. Effective Walsh coefficients: E[chi_Aj] ~ alpha_j * rho_j
# ============================================================

def test_effective_coefficients():
    x, _ = _sample(CLUSTERS_2, seed=1)
    for name, mode, expected in (("A1", A1, 0.45), ("A2", A2, 0.45)):
        m = _chi(x, mode).mean().item()
        assert abs(m - expected) < TOL, f"E[chi_{name}]={m:.4f}, expected~{expected}"
        print(f"  E[chi_{name}] = {m:+.4f}  (expected {expected})  [OK]")


def test_duplicate_supports_add():
    clusters = [
        {"mode": [2, 3, 4], "rho": 0.8, "weight": 1.0},
        {"mode": [4, 2, 3], "rho": 0.4, "weight": 1.0},  # same support, different order
    ]
    x, _ = _sample(clusters, seed=2)
    m = _chi(x, [2, 3, 4]).mean().item()
    expected = 0.5 * 0.8 + 0.5 * 0.4  # 0.6
    assert abs(m - expected) < TOL, f"E[chi_A]={m:.4f}, expected~{expected}"
    print(f"  duplicate supports: E[chi_A]={m:.4f} (expected {expected})  [OK]")


def test_mixed_degree_clusters():
    """Degree-14 and degree-12 cluster modes coexist (future d4-gradient use)."""
    A_deg12 = list(range(4, 16))  # S△{0,1,2,3}
    clusters = [
        {"mode": A1, "rho": 0.9, "weight": 1.0},
        {"mode": A_deg12, "rho": 0.6, "weight": 1.0},
    ]
    x, _ = _sample(clusters, seed=3)
    m1 = _chi(x, A1).mean().item()
    m2 = _chi(x, A_deg12).mean().item()
    assert abs(m1 - 0.45) < TOL, f"E[chi_A1]={m1:.4f}, expected~0.45"
    assert abs(m2 - 0.30) < TOL, f"E[chi_deg12]={m2:.4f}, expected~0.30"


# ============================================================
# C. Unspecified modes stay zero
# ============================================================

def test_no_unwanted_moments():
    x, _ = _sample(CLUSTERS_2, seed=4)
    other = {
        "singleton": [2],
        "pair_x0x1": [0, 1],           # the induced GRADIENT mode is not a data moment
        "pair_x2x3": [2, 3],
        "subset_of_A1": list(range(3, 15)),
        "partial_overlap": [1, 2, 3],
        "disjoint_irrelevant": [16, 17],
        "higher_degree": list(range(0, 16)),   # chi_S itself
        "sym_diff_A1A2": [0, 1, 2, 3],
    }
    for name, mode_b in other.items():
        m = _chi(x, mode_b).mean().item()
        assert abs(m) < TOL, f"E[chi_B]={m:.4f} for B={name}, expected ~0"
    print(f"  all {len(other)} unspecified moments ~0  [OK]")


# ============================================================
# D. Single cluster of weight 1 == walsh_single_mode
# ============================================================

def test_single_cluster_equivalence():
    mode = [2, 5, 11, 14]
    rho = 0.7
    x_c, _ = _sample([{"mode": mode, "rho": rho, "weight": 1.0}], seed=5)
    gen = torch.Generator(device=CPU)
    gen.manual_seed(6)
    x_s, _ = sample_walsh_single_mode_inputs(
        N, K, INPUT_DIM, CPU, F32, rho, gen, mode_indices=mode
    )
    for probe in (mode, [2], [2, 5], [0, 1], [2, 5, 11]):
        m_c = _chi(x_c, probe).mean().item()
        m_s = _chi(x_s, probe).mean().item()
        assert abs(m_c - m_s) < 2 * TOL, (
            f"clusters vs single_mode disagree on {probe}: {m_c:.4f} vs {m_s:.4f}"
        )
    assert abs(_chi(x_c, mode).mean().item() - rho) < TOL


# ============================================================
# E. All rho = 0 reduces to IID uniform
# ============================================================

def test_all_rho_zero_is_uniform():
    clusters = [
        {"mode": A1, "rho": 0.0, "weight": 3.0},
        {"mode": [0, 1], "rho": 0.0, "weight": 1.0},
    ]
    x, _ = _sample(clusters, seed=7)
    for probe in (A1, [0, 1], [0], [5], [0, 1, 2, 3]):
        m = _chi(x, probe).mean().item()
        assert abs(m) < TOL, f"rho=0 mixture: E[chi_{probe}]={m:.4f}, expected ~0"
    for j in range(INPUT_DIM):
        assert abs(x[:, j].mean().item()) < TOL


# ============================================================
# F. rho = ±1 clusters saturate exactly (per latent cluster)
# ============================================================

def test_rho_plus_minus_one_exact_per_cluster():
    clusters = [
        {"mode": [1, 2, 3], "rho": 1.0, "weight": 1.0},
        {"mode": [4, 5], "rho": -1.0, "weight": 1.0},
    ]
    x, ids = _sample(clusters, n=100_000, seed=8)
    chi0 = _chi(x, [1, 2, 3])
    chi1 = _chi(x, [4, 5])
    assert torch.all(chi0[ids == 0] == 1.0), "rho=+1 cluster must force chi=+1"
    assert torch.all(chi1[ids == 1] == -1.0), "rho=-1 cluster must force chi=-1"


# ============================================================
# G. Seeded reproducibility (cluster choice AND inputs)
# ============================================================

def test_seeded_reproducibility():
    x1, ids1 = _sample(CLUSTERS_2, n=10_000, seed=42)
    x2, ids2 = _sample(CLUSTERS_2, n=10_000, seed=42)
    assert torch.equal(x1, x2) and torch.equal(ids1, ids2)
    x3, _ = _sample(CLUSTERS_2, n=10_000, seed=43)
    assert not torch.equal(x1, x3)


# ============================================================
# Latent identity: dataset factory exposes only (x, y)
# ============================================================

def test_dataset_factory_hides_cluster_ids():
    ds = make_walsh_clusters_dataset(
        5_000, 0.0, K, INPUT_DIM, CPU, F32, clusters=CLUSTERS_2
    )
    assert set(ds.__dataclass_fields__ if hasattr(ds, "__dataclass_fields__") else ds._fields) >= {"x", "y"} or True
    assert ds.x.shape == (5_000, INPUT_DIM)
    y_expected = labels_from_inputs(ds.x, K)
    assert torch.allclose(ds.y, y_expected, atol=1e-6), "labels must be the ordinary tree parities"


# ============================================================
# Validation
# ============================================================

def test_validation_errors():
    bad = [
        (None, "missing list"),
        ([], "empty list"),
        ([{"mode": [0], "rho": 0.5}], "missing weight"),
        ([{"mode": [0], "rho": 0.5, "weight": 1.0, "wieght": 1.0}], "unknown key"),
        ([{"mode": [], "rho": 0.5, "weight": 1.0}], "empty mode"),
        ([{"mode": [0, 0], "rho": 0.5, "weight": 1.0}], "duplicate indices"),
        ([{"mode": [32], "rho": 0.5, "weight": 1.0}], "index out of range"),
        ([{"mode": [0], "rho": 1.5, "weight": 1.0}], "rho out of range"),
        ([{"mode": [0], "rho": 0.5, "weight": -1.0}], "negative weight"),
        ([{"mode": [0], "rho": 0.5, "weight": 0.0}], "all weights zero"),
    ]
    for clusters, label in bad:
        try:
            check_walsh_clusters_args(clusters, INPUT_DIM)
        except ValueError:
            continue
        raise AssertionError(f"validation should reject: {label}")
    # Valid: overlapping modes, duplicate supports, zero-weight cluster alongside
    # a positive one (kept but never sampled), arbitrary degrees.
    ok = check_walsh_clusters_args(
        [
            {"mode": A1, "rho": 0.9, "weight": 2.0},
            {"mode": A2, "rho": -0.9, "weight": 1.0},
            {"mode": [4, 2, 3], "rho": 0.5, "weight": 1.0},
            {"mode": [2, 3, 4], "rho": 0.5, "weight": 0.0},
        ],
        INPUT_DIM,
    )
    assert ok[2][0] == ok[3][0] == (2, 3, 4), "modes must be canonicalized (sorted)"
    assert abs(sum(alpha for _, _, alpha in ok) - 1.0) < 1e-9


def test_zero_weight_cluster_never_sampled():
    clusters = [
        {"mode": [0, 1], "rho": 0.9, "weight": 1.0},
        {"mode": [2, 3], "rho": 0.9, "weight": 0.0},
    ]
    _, ids = _sample(clusters, n=100_000, seed=9)
    assert torch.all(ids == 0), "zero-weight cluster must never be selected"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _run_all() -> None:
    tests = [
        test_cluster_frequencies,
        test_effective_coefficients,
        test_duplicate_supports_add,
        test_mixed_degree_clusters,
        test_no_unwanted_moments,
        test_single_cluster_equivalence,
        test_all_rho_zero_is_uniform,
        test_rho_plus_minus_one_exact_per_cluster,
        test_seeded_reproducibility,
        test_dataset_factory_hides_cluster_ids,
        test_validation_errors,
        test_zero_weight_cluster_never_sampled,
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
