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
    x = sample_inputs(n, input_dim, device).to(dtype=dtype)
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
