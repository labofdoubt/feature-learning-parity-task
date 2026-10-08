"""Walsh mode utilities: chi functions and control-mode generation."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModeInfo:
    indices: tuple[int, ...]   # sorted input-bit indices
    label: str                 # human-readable name
    role: str                  # "target", "child_a", "child_b", "ctrl_d8_partition",
                               # "ctrl_d8_random", "ctrl_d16_replace1/2/4", "ctrl_d16_random"
    ctrl_index: int = -1       # index within its control group, -1 for primary modes


def compute_chi(x: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    """Chi_T(x) = prod_{i in T} x_i.  x is (batch, input_dim), returns (batch,)."""
    idx = torch.tensor(indices, device=x.device, dtype=torch.long)
    return x[:, idx].prod(dim=1)


def _canonical_pair(C: frozenset, D: frozenset) -> tuple[frozenset, frozenset]:
    return (min(C, D, key=sorted), max(C, D, key=sorted))


def generate_alternative_partitions(
    S: tuple[int, ...],
    A: tuple[int, ...],
    B: tuple[int, ...],
    n: int,
    seed: int = 42,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return n pairs (C_k, D_k): C ∪ D = S, |C|=|D|=|S|//2, (C,D) ≠ (A,B).

    Each pair is returned with C_k ≠ A; D_k = S \\ C_k.
    """
    S_set = frozenset(S)
    A_set = frozenset(A)
    B_set = frozenset(B)
    S_list = sorted(S)
    half = len(S_list) // 2

    rng = random.Random(seed)
    seen = {_canonical_pair(A_set, B_set)}
    result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    attempts = 0
    while len(result) < n and attempts < 100_000:
        attempts += 1
        C_list = sorted(rng.sample(S_list, half))
        C = frozenset(C_list)
        D = S_set - C
        key = _canonical_pair(C, D)
        if key not in seen:
            seen.add(key)
            result.append((tuple(C_list), tuple(sorted(D))))

    if len(result) < n:
        raise RuntimeError(
            f"Could only generate {len(result)} alternative partitions (needed {n})."
        )
    return result


def generate_degree16_controls(
    S: tuple[int, ...],
    input_dim: int,
    relevant_dim: int,
    n_per_class: int,
    seed: int = 42,
) -> list[tuple[tuple[int, ...], str]]:
    """Return degree-16 controls by replacing k relevant bits with k noise bits.

    Returns list of (mode_indices, label) where label is "replace1", "replace2", "replace4".
    """
    noise_vars = list(range(relevant_dim, input_dim))
    S_list = list(S)
    rng = random.Random(seed)
    controls: list[tuple[tuple[int, ...], str]] = []

    for n_replace in [1, 2, 4]:
        label = f"replace{n_replace}"
        seen: set[tuple[int, ...]] = set()
        class_controls = 0
        for _ in range(n_per_class * 200):
            if class_controls >= n_per_class:
                break
            remove = sorted(rng.sample(S_list, n_replace))
            add = sorted(rng.sample(noise_vars, n_replace))
            new_mode = tuple(sorted([i for i in S_list if i not in remove] + add))
            if new_mode not in seen:
                seen.add(new_mode)
                controls.append((new_mode, label))
                class_controls += 1

    return controls


def build_mode_list(
    target_mode: tuple[int, ...],
    child_a: tuple[int, ...],
    child_b: tuple[int, ...],
    n_partition_controls: int,
    n_d16_controls_per_class: int,
    input_dim: int,
    relevant_dim: int,
) -> list[ModeInfo]:
    """Build the full list of ModeInfo objects used throughout the analysis."""
    modes: list[ModeInfo] = [
        ModeInfo(child_a, "A", "child_a"),
        ModeInfo(child_b, "B", "child_b"),
        ModeInfo(target_mode, "S", "target"),
    ]

    # Alternative partition controls (d8)
    alt_pairs = generate_alternative_partitions(
        target_mode, child_a, child_b, n_partition_controls
    )
    for k, (C, D) in enumerate(alt_pairs):
        modes.append(ModeInfo(C, f"C{k}", "ctrl_d8_partition", ctrl_index=k))
        modes.append(ModeInfo(D, f"D{k}", "ctrl_d8_partition", ctrl_index=k))

    # Degree-16 controls
    d16_ctrls = generate_degree16_controls(
        target_mode, input_dim, relevant_dim, n_d16_controls_per_class
    )
    for k, (mode, cls) in enumerate(d16_ctrls):
        modes.append(ModeInfo(mode, f"S'_{k}_{cls}", f"ctrl_d16_{cls}", ctrl_index=k))

    return modes
