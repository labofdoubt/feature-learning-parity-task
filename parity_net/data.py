from __future__ import annotations

import warnings
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


DEGREE_SLICES = {
    2: slice(0, 8),
    4: slice(8, 12),
    8: slice(12, 14),
    16: slice(14, 15),
}


@dataclass(frozen=True)
class ParityDataset:
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class TargetSpec:
    name: str
    degree: int
    indices: tuple[int, ...]


def input_key_powers(input_dim: int, device: torch.device) -> torch.Tensor:
    if input_dim > 62:
        raise ValueError("Packed input keys support input_dim <= 62")
    return 2 ** torch.arange(input_dim, device=device, dtype=torch.long)


def input_keys(x: torch.Tensor) -> torch.Tensor:
    powers = input_key_powers(x.shape[1], x.device)
    bits = (x > 0).to(dtype=torch.long)
    return torch.sum(bits * powers, dim=1)


def exclusion_keys(x: torch.Tensor) -> torch.Tensor:
    return torch.unique(input_keys(x), sorted=True)


def tree_parity_indices(relevant_dim: int = 16) -> list[tuple[int, ...]]:
    return [spec.indices for spec in tree_parity_specs(relevant_dim)]


def _validate_task_shape(input_dim: int, relevant_dim: int) -> None:
    if input_dim <= 0 or relevant_dim <= 0:
        raise ValueError("input_dim and relevant_dim must be positive")
    if input_dim % 2 or relevant_dim % 2:
        raise ValueError("input_dim and relevant_dim must both be even")
    if relevant_dim > input_dim:
        raise ValueError("relevant_dim must be <= input_dim")


def _target_is_excluded(spec: TargetSpec, exclude_targets: list[str] | tuple[str, ...]) -> bool:
    degree_name = f"d{spec.degree}"
    for pattern in exclude_targets:
        if pattern in {spec.name, degree_name} or fnmatch(spec.name, pattern):
            return True
    return False


def _degree_is_excluded(degree: int, exclude_targets: list[str] | tuple[str, ...]) -> bool:
    degree_name = f"d{degree}"
    dummy_name = f"{degree_name}_0"
    return any(pattern == degree_name or fnmatch(dummy_name, pattern) for pattern in exclude_targets)


def tree_parity_specs(
    relevant_dim: int = 16,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    max_degree: int | None = None,
) -> list[TargetSpec]:
    _validate_task_shape(relevant_dim, relevant_dim)
    exclude_targets = exclude_targets or []
    specs: list[TargetSpec] = []
    degree = 2
    while degree <= relevant_dim and (max_degree is None or degree <= max_degree):
        if relevant_dim % degree:
            if _degree_is_excluded(degree, exclude_targets):
                degree *= 2
                continue
            raise ValueError(
                f"Cannot include d{degree} targets because relevant_dim={relevant_dim} "
                f"is not divisible by {degree}; exclude d{degree} targets or choose "
                "a compatible relevant_dim"
            )
        for start in range(0, relevant_dim, degree):
            spec = TargetSpec(
                name=f"d{degree}_{start // degree}",
                degree=degree,
                indices=tuple(range(start, start + degree)),
            )
            if not _target_is_excluded(spec, exclude_targets):
                specs.append(spec)
        degree *= 2
    if not specs:
        raise ValueError("Task has no targets after applying exclude_targets")
    return specs


def target_names(
    relevant_dim: int = 16,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    max_degree: int | None = None,
) -> list[str]:
    return [spec.name for spec in tree_parity_specs(relevant_dim, exclude_targets, max_degree)]


def degree_slices_for_targets(target_names_: list[str]) -> dict[int, slice]:
    slices: dict[int, slice] = {}
    start = 0
    while start < len(target_names_):
        degree = int(target_names_[start].split("_", 1)[0][1:])
        stop = start + 1
        while stop < len(target_names_):
            next_degree = int(target_names_[stop].split("_", 1)[0][1:])
            if next_degree != degree:
                break
            stop += 1
        slices[degree] = slice(start, stop)
        start = stop
    return slices


def sample_inputs(n: int, input_dim: int, device: torch.device) -> torch.Tensor:
    bits = torch.randint(0, 2, (n, input_dim), device=device)
    return bits.float().mul_(2).sub_(1)


def inputs_from_keys(keys: torch.Tensor, input_dim: int, device: torch.device) -> torch.Tensor:
    bit_positions = torch.arange(input_dim, device=device, dtype=torch.long)
    bits = (keys.to(device=device, dtype=torch.long).unsqueeze(1) >> bit_positions) & 1
    return bits.float().mul_(2).sub_(1)


def sample_unique_inputs(n: int, input_dim: int, device: torch.device) -> torch.Tensor:
    if input_dim > 62:
        raise ValueError("Unique input sampling supports input_dim <= 62")
    input_space_size = 2**input_dim
    target_n = min(n, input_space_size)
    if target_n < n:
        warnings.warn(
            f"Requested {n} unique inputs, but input_dim={input_dim} gives only "
            f"{input_space_size} possible inputs. Truncating dataset to {target_n} "
            "unique samples.",
            RuntimeWarning,
            stacklevel=2,
        )
    if target_n == 0:
        return torch.empty((0, input_dim), device=device)

    enumerate_space_limit = 5_000_000
    if input_space_size <= enumerate_space_limit and target_n > input_space_size // 4:
        keys = torch.randperm(input_space_size, device=device, dtype=torch.long)[:target_n]
        return inputs_from_keys(keys, input_dim, device)

    seen = torch.empty(0, device=device, dtype=torch.long)
    while seen.numel() < target_n:
        remaining = target_n - seen.numel()
        candidate_count = max(remaining * 2, 1024)
        candidates = torch.randint(
            0,
            input_space_size,
            (candidate_count,),
            device=device,
            dtype=torch.long,
        )
        candidates = torch.unique(candidates, sorted=True)
        if seen.numel():
            positions = torch.searchsorted(seen, candidates)
            safe_positions = positions.clamp(max=seen.numel() - 1)
            already_seen = (positions < seen.numel()) & (seen[safe_positions] == candidates)
            candidates = candidates[~already_seen]
        if candidates.numel() == 0:
            continue
        take = min(remaining, candidates.numel())
        seen = torch.unique(torch.cat([seen, candidates[:take]]), sorted=True)

    return inputs_from_keys(seen[:target_n], input_dim, device)


def sample_unique_inputs_excluding(
    n: int,
    input_dim: int,
    device: torch.device,
    excluded_keys: torch.Tensor,
) -> torch.Tensor:
    """`n` distinct inputs, none of them in `excluded_keys`.

    `sample_inputs_excluding` avoids the excluded set but may repeat inputs, which is
    fine for a fresh batch and wrong for a fixed training pool of a stated size.
    """
    if excluded_keys.numel() == 0:
        return sample_unique_inputs(n, input_dim, device)
    if input_dim > 62:
        raise ValueError("Unique input sampling supports input_dim <= 62")

    input_space_size = 2**input_dim
    available = input_space_size - int(excluded_keys.numel())
    target_n = min(n, available)
    if target_n < n:
        warnings.warn(
            f"Requested {n} unique inputs outside the excluded set, but only "
            f"{available} of the {input_space_size} possible inputs are available. "
            f"Truncating to {target_n}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if target_n <= 0:
        return torch.empty((0, input_dim), device=device)

    enumerate_space_limit = 5_000_000
    if input_space_size <= enumerate_space_limit:
        keep = torch.ones(input_space_size, device=device, dtype=torch.bool)
        keep[excluded_keys.to(device=device, dtype=torch.long)] = False
        candidates = torch.nonzero(keep, as_tuple=False).flatten()
        chosen = torch.randperm(candidates.numel(), device=device)[:target_n]
        return inputs_from_keys(candidates[chosen], input_dim, device)

    excluded_sorted = torch.unique(excluded_keys.to(device=device, dtype=torch.long), sorted=True)
    seen = torch.empty(0, device=device, dtype=torch.long)
    while seen.numel() < target_n:
        remaining = target_n - seen.numel()
        candidates = torch.randint(
            0,
            input_space_size,
            (max(remaining * 2, 1024),),
            device=device,
            dtype=torch.long,
        )
        candidates = torch.unique(candidates, sorted=True)
        for blocked in (excluded_sorted, seen):
            if blocked.numel() and candidates.numel():
                positions = torch.searchsorted(blocked, candidates)
                safe = positions.clamp(max=blocked.numel() - 1)
                candidates = candidates[~((positions < blocked.numel()) & (blocked[safe] == candidates))]
        if candidates.numel() == 0:
            continue
        seen = torch.unique(torch.cat([seen, candidates[:remaining]]), sorted=True)

    return inputs_from_keys(seen[:target_n], input_dim, device)


def sample_inputs_excluding(
    n: int,
    input_dim: int,
    device: torch.device,
    excluded_keys: torch.Tensor,
) -> torch.Tensor:
    if excluded_keys.numel() == 0:
        return sample_inputs(n, input_dim, device)
    if excluded_keys.numel() >= 2**input_dim:
        warnings.warn(
            "Cannot avoid overlap with the test set because the excluded set covers "
            "the full input space; sampling training inputs without exclusion.",
            RuntimeWarning,
            stacklevel=2,
        )
        return sample_inputs(n, input_dim, device)

    chunks = []
    total = 0
    while total < n:
        remaining = n - total
        candidate_count = max(remaining * 2, 1024)
        candidates = sample_inputs(candidate_count, input_dim, device)
        candidate_keys = input_keys(candidates)
        positions = torch.searchsorted(excluded_keys, candidate_keys)
        safe_positions = positions.clamp(max=excluded_keys.numel() - 1)
        is_excluded = (positions < excluded_keys.numel()) & (
            excluded_keys[safe_positions] == candidate_keys
        )
        accepted = candidates[~is_excluded]
        if accepted.numel() == 0:
            continue
        take = min(remaining, accepted.shape[0])
        chunks.append(accepted[:take])
        total += take
    return torch.cat(chunks, dim=0)


def labels_from_inputs(
    x: torch.Tensor,
    relevant_dim: int = 16,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    max_degree: int | None = None,
) -> torch.Tensor:
    outputs = []
    for spec in tree_parity_specs(relevant_dim, exclude_targets, max_degree):
        idx = torch.tensor(spec.indices, device=x.device, dtype=torch.long)
        outputs.append(torch.prod(x[:, idx], dim=1))
    return torch.stack(outputs, dim=1).to(dtype=x.dtype)


def make_dataset(
    n: int,
    input_dim: int,
    relevant_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    max_degree: int | None = None,
) -> ParityDataset:
    _validate_task_shape(input_dim, relevant_dim)
    x = sample_unique_inputs(n, input_dim, device).to(dtype=dtype)
    y = labels_from_inputs(x, relevant_dim, exclude_targets, max_degree).to(dtype=dtype)
    return ParityDataset(x=x, y=y)


def save_dataset(dataset: ParityDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": dataset.x.detach().cpu(),
            "y": dataset.y.detach().cpu(),
        },
        path,
    )


def load_dataset(path: str | Path, device: torch.device, dtype: torch.dtype) -> ParityDataset:
    payload = torch.load(path, map_location=device)
    return ParityDataset(
        x=payload["x"].to(device=device, dtype=dtype),
        y=payload["y"].to(device=device, dtype=dtype),
    )


def make_loader(dataset: ParityDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        TensorDataset(dataset.x, dataset.y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Hierarchical non-uniform distribution (rho > 0)
# ---------------------------------------------------------------------------

def _check_hierarchical_args(relevant_dim: int, rho: float) -> None:
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    if relevant_dim < 2 or (relevant_dim & (relevant_dim - 1)) != 0:
        raise ValueError(
            f"relevant_dim must be a power of 2 >= 2 for hierarchical sampling, got {relevant_dim}"
        )


def sample_hierarchical_inputs(
    n: int,
    relevant_dim: int,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    rho: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n inputs from the hierarchical correlated parity distribution.

    Generates inputs top-down through the same balanced binary parity tree used
    by tree_parity_specs.  At each internal node with value P, the left child L
    is drawn as +1 with probability (1+rho)/2 and the right child is R = P*L,
    so L*R = P exactly.  rho=0 recovers the uniform distribution.

    Returns:
        x     : (n, input_dim) float tensor in {-1, +1}
        root  : (n,) float tensor, the root parity value (= product of relevant bits)
    """
    _check_hierarchical_args(relevant_dim, rho)

    # Sample root uniformly in {-1, +1}
    root = (
        torch.randint(0, 2, (n,), device=device, generator=generator)
        .float()
        .mul_(2)
        .sub_(1)
    )

    # nodes shape: (n, num_nodes_at_current_level)
    nodes = root.unsqueeze(1)  # level 0: one node (the root)
    num_levels = (relevant_dim - 1).bit_length()  # log2(relevant_dim)
    p_left = (1.0 + rho) / 2.0

    for _ in range(num_levels):
        num_nodes = nodes.shape[1]
        u = torch.rand(n, num_nodes, device=device, generator=generator)
        L = torch.where(
            u < p_left,
            torch.ones(n, num_nodes, device=device),
            -torch.ones(n, num_nodes, device=device),
        )
        R = nodes * L  # P = L*R  =>  R = P/L = P*L  (since L in {-1,+1})
        # Interleave: left child of node i at 2*i, right child at 2*i+1
        nodes = torch.stack([L, R], dim=2).reshape(n, -1)  # (n, 2*num_nodes)

    leaves = nodes  # (n, relevant_dim), the leaf values

    irrel_dim = input_dim - relevant_dim
    if irrel_dim > 0:
        irrel = (
            torch.randint(0, 2, (n, irrel_dim), device=device, generator=generator)
            .float()
            .mul_(2)
            .sub_(1)
        )
        x = torch.cat([leaves, irrel], dim=1)
    else:
        x = leaves

    return x.to(dtype=dtype), root.to(dtype=dtype)


def make_hierarchical_dataset(
    n: int,
    rho: float,
    relevant_dim: int,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    max_degree: int | None = None,
    generator: torch.Generator | None = None,
) -> ParityDataset:
    """Dataset of n samples from the hierarchical distribution p_rho.

    Labels are computed from x using the same tree_parity_specs convention as
    the rest of the codebase, so exclude_targets and max_degree apply normally.
    """
    x, _root = sample_hierarchical_inputs(
        n, relevant_dim, input_dim, device, dtype, rho, generator
    )
    y = labels_from_inputs(x, relevant_dim, exclude_targets, max_degree).to(dtype=dtype)
    return ParityDataset(x=x, y=y)


# ---------------------------------------------------------------------------
# Exhaustive uniform evaluation dataset
# ---------------------------------------------------------------------------

def make_uniform_eval_dataset(
    relevant_dim: int,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    eval_noise_repeats: int = 2,
    seed: int = 0,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    max_degree: int | None = None,
) -> ParityDataset:
    """Fixed uniform evaluation dataset with exact relevant-bit coverage.

    For relevant_dim <= 20, enumerates all 2^relevant_dim relevant-bit
    configurations and pairs each with eval_noise_repeats independent draws of
    the irrelevant bits.  For relevant_dim > 20, falls back to a large
    Monte Carlo uniform sample of equivalent total size.

    The irrelevant-bit draws are seeded for reproducibility.
    """
    irrel_dim = input_dim - relevant_dim
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if relevant_dim <= 20:
        k = relevant_dim
        n_configs = 2 ** k
        all_keys = torch.arange(n_configs, device=device, dtype=torch.long)
        bit_positions = torch.arange(k, device=device, dtype=torch.long)
        # rel_bits[i, j] = bit j of config i, mapped to {-1, +1}
        rel_bits = (
            ((all_keys.unsqueeze(1) >> bit_positions) & 1)
            .float()
            .mul_(2)
            .sub_(1)
        )  # (n_configs, k)
        # Repeat each relevant config eval_noise_repeats times
        rel_bits = rel_bits.repeat_interleave(eval_noise_repeats, dim=0)  # (n, k)
        n = rel_bits.shape[0]
        if irrel_dim > 0:
            irrel = (
                torch.randint(0, 2, (n, irrel_dim), device=device, generator=gen)
                .float()
                .mul_(2)
                .sub_(1)
            )
            x = torch.cat([rel_bits, irrel], dim=1)
        else:
            x = rel_bits
    else:
        # Monte Carlo fallback
        n = eval_noise_repeats * (2 ** 17)
        x = (
            torch.randint(0, 2, (n, input_dim), device=device, generator=gen)
            .float()
            .mul_(2)
            .sub_(1)
        )

    y = labels_from_inputs(x, relevant_dim, exclude_targets, max_degree).to(dtype=dtype)
    return ParityDataset(x=x.to(dtype=dtype), y=y)


# ---------------------------------------------------------------------------
# Hierarchical-degree-2 distribution: bias stops at degree-2 latents
# ---------------------------------------------------------------------------

def sample_hierarchical_degree2_inputs(
    n: int,
    relevant_dim: int,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    rho: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical distribution truncated at degree-2 latents.

    Generates z_1..z_{k/2} using the same top-down biased binary tree as
    sample_hierarchical_inputs (with the same rho), then expands each z_i
    into two individual input bits via:
        x_{2i-1} = r_i,   x_{2i} = z_i * r_i,   r_i ~ Unif{-1,+1}

    This guarantees E[S * x_j] = 0 for every individual bit x_j regardless
    of rho, while preserving hierarchical correlations at degree 2 and above.
    rho=0 reduces to the ordinary uniform distribution.

    Returns:
        x    : (n, input_dim) float tensor in {-1, +1}
        root : (n,) float tensor, the root parity (= product of relevant bits)
    """
    _check_hierarchical_args(relevant_dim, rho)

    num_z = relevant_dim // 2  # number of degree-2 latent variables

    # Sample root uniformly in {-1, +1}
    root = (
        torch.randint(0, 2, (n,), device=device, generator=generator)
        .float()
        .mul_(2)
        .sub_(1)
    )

    if num_z == 1:
        # k=2 special case: the single z is the root itself
        z = root.unsqueeze(1)  # (n, 1)
    else:
        # Run the biased top-down tree to produce num_z leaves (the z's).
        # num_z is a power of 2 >= 2 since relevant_dim is a power of 2 >= 4.
        nodes = root.unsqueeze(1)
        num_levels = (num_z - 1).bit_length()  # log2(num_z)
        p_left = (1.0 + rho) / 2.0
        for _ in range(num_levels):
            num_nodes = nodes.shape[1]
            u = torch.rand(n, num_nodes, device=device, generator=generator)
            L = torch.where(
                u < p_left,
                torch.ones(n, num_nodes, device=device),
                -torch.ones(n, num_nodes, device=device),
            )
            R = nodes * L
            nodes = torch.stack([L, R], dim=2).reshape(n, -1)
        z = nodes  # (n, num_z)

    # Expand each z_i into two bits: x_{2i-1} = r_i, x_{2i} = z_i * r_i
    r = (
        torch.randint(0, 2, (n, num_z), device=device, generator=generator)
        .float()
        .mul_(2)
        .sub_(1)
    )
    x_left = r          # (n, num_z)
    x_right = z * r     # (n, num_z)
    # Interleave into (n, relevant_dim): positions 2i -> x_left[:,i], 2i+1 -> x_right[:,i]
    x_rel = torch.stack([x_left, x_right], dim=2).reshape(n, relevant_dim)

    irrel_dim = input_dim - relevant_dim
    if irrel_dim > 0:
        irrel = (
            torch.randint(0, 2, (n, irrel_dim), device=device, generator=generator)
            .float()
            .mul_(2)
            .sub_(1)
        )
        x = torch.cat([x_rel, irrel], dim=1)
    else:
        x = x_rel

    return x.to(dtype=dtype), root.to(dtype=dtype)


def make_hierarchical_degree2_dataset(
    n: int,
    rho: float,
    relevant_dim: int,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    max_degree: int | None = None,
    generator: torch.Generator | None = None,
) -> ParityDataset:
    """Dataset of n samples from the degree-2-truncated hierarchical distribution."""
    x, _root = sample_hierarchical_degree2_inputs(
        n, relevant_dim, input_dim, device, dtype, rho, generator
    )
    y = labels_from_inputs(x, relevant_dim, exclude_targets, max_degree).to(dtype=dtype)
    return ParityDataset(x=x, y=y)


# ---------------------------------------------------------------------------
# Reuse-star task: three degree-16 targets sharing one degree-8 constituent
# ---------------------------------------------------------------------------
# Fixed layout for d=32:
#   x[0:8]   → A-block  (shared constituent)
#   x[8:16]  → B-block  (private partner for S1 = A·B)
#   x[16:24] → C-block  (private partner for S2 = A·C)
#   x[24:32] → D-block  (private partner for S3 = A·D)
#
# Latent generation (rho > 0):
#   S1, S2, S3 ~ Unif{-1,+1} independently
#   A ~ P(A=+1) = (1+rho)/2                  (biased shared feature)
#   B = S1·A,  C = S2·A,  D = S3·A           (derived private features)
#
# At rho=0 the full 32-bit distribution is exactly i.i.d. Unif{-1,+1}.

REUSE_STAR_INPUT_DIM   = 32
REUSE_STAR_BLOCK_SIZE  = 8
REUSE_STAR_NUM_TARGETS = 3

# Target supports (0-indexed bit positions); useful for spectral analysis later.
REUSE_STAR_SHARED_SUPPORT: tuple[int, ...] = tuple(range(8))
REUSE_STAR_PRIVATE_SUPPORTS: list[tuple[int, ...]] = [
    tuple(range(8,  16)),
    tuple(range(16, 24)),
    tuple(range(24, 32)),
]
REUSE_STAR_TARGET_SUPPORTS: list[tuple[int, ...]] = [
    REUSE_STAR_SHARED_SUPPORT + priv
    for priv in REUSE_STAR_PRIVATE_SUPPORTS
]
REUSE_STAR_TARGET_NAMES = ["s1", "s2", "s3"]


def _expand_block_from_root(
    root: torch.Tensor,
    rho: float,
    data_distribution: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    block_size: int = 8,
) -> torch.Tensor:
    """Expand a block of `block_size` bits conditional on their product equalling `root`.

    data_distribution selects the internal sub-parity structure:
      "hierarchical"         – rho-biased top-down tree down to individual bits.
      "hierarchical_degree2" – rho-biased tree stops at degree-2 latents; each z_i is
                               then split uniformly into two bits.
      "uniform"              – uniform conditional expansion (product = root, rho ignored).

    root : (n,) tensor of ±1 values.
    Returns: (n, block_size) float tensor in {-1, +1}.
    """
    n = root.shape[0]
    eff_rho = 0.0 if data_distribution == "uniform" else rho

    if data_distribution == "hierarchical_degree2":
        num_z = block_size // 2  # 4 for block_size=8
        nodes = root.float().unsqueeze(1)
        num_levels = (num_z - 1).bit_length()
        p_left = (1.0 + eff_rho) / 2.0
        for _ in range(num_levels):
            m = nodes.shape[1]
            u = torch.rand(n, m, device=device, generator=generator)
            L = torch.where(u < p_left, torch.ones(n, m, device=device), -torch.ones(n, m, device=device))
            R = nodes * L
            nodes = torch.stack([L, R], dim=2).reshape(n, -1)
        z = nodes  # (n, num_z)
        r = torch.randint(0, 2, (n, num_z), device=device, generator=generator).float().mul_(2).sub_(1)
        bits = torch.stack([r, z * r], dim=2).reshape(n, block_size)
    else:
        # "hierarchical" (with eff_rho) or "uniform" (eff_rho=0)
        num_levels = (block_size - 1).bit_length()
        nodes = root.float().unsqueeze(1)
        p_left = (1.0 + eff_rho) / 2.0
        for _ in range(num_levels):
            m = nodes.shape[1]
            u = torch.rand(n, m, device=device, generator=generator)
            L = torch.where(u < p_left, torch.ones(n, m, device=device), -torch.ones(n, m, device=device))
            R = nodes * L
            nodes = torch.stack([L, R], dim=2).reshape(n, -1)
        bits = nodes  # (n, block_size)

    return bits.to(dtype=dtype)


def sample_reuse_star_inputs(
    n: int,
    rho: float,
    data_distribution: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n inputs for the reuse-star task.

    Always generates the full d=32 input and all three targets [S1, S2, S3].
    The active-target mask in training controls which targets contribute to the loss;
    the data distribution itself never depends on num_reuse_targets.

    Returns:
        x       : (n, 32) float tensor in {-1, +1}
        targets : (n, 3)  float tensor – [S1, S2, S3] recomputed from x
    """
    # Sample three roots uniformly
    S_raw = (
        torch.randint(0, 2, (n, 3), device=device, generator=generator)
        .float().mul_(2).sub_(1)
    )
    S1, S2, S3 = S_raw[:, 0], S_raw[:, 1], S_raw[:, 2]

    # Shared constituent A: biased by rho
    p_A = (1.0 + rho) / 2.0
    u_A = torch.rand(n, device=device, generator=generator)
    A = torch.where(u_A < p_A, torch.ones(n, device=device), -torch.ones(n, device=device))

    # Private constituents determined by roots and A
    B = S1 * A
    C = S2 * A
    D = S3 * A

    # Expand each macro-variable into 8 input bits
    x_A = _expand_block_from_root(A, rho, data_distribution, device, dtype, generator)
    x_B = _expand_block_from_root(B, rho, data_distribution, device, dtype, generator)
    x_C = _expand_block_from_root(C, rho, data_distribution, device, dtype, generator)
    x_D = _expand_block_from_root(D, rho, data_distribution, device, dtype, generator)
    x = torch.cat([x_A, x_B, x_C, x_D], dim=1)  # (n, 32)

    # Recompute targets from x (ground truth, not from latents)
    A_x = x[:, :8].prod(dim=1)
    B_x = x[:, 8:16].prod(dim=1)
    C_x = x[:, 16:24].prod(dim=1)
    D_x = x[:, 24:32].prod(dim=1)
    targets = torch.stack([A_x * B_x, A_x * C_x, A_x * D_x], dim=1)

    return x.to(dtype=dtype), targets.to(dtype=dtype)


def make_reuse_star_dataset(
    n: int,
    rho: float,
    data_distribution: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> ParityDataset:
    """Fixed dataset for the reuse-star task; y has shape (n, 3)."""
    x, targets = sample_reuse_star_inputs(n, rho, data_distribution, device, dtype, generator)
    return ParityDataset(x=x, y=targets)
