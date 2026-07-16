from __future__ import annotations

from dataclasses import dataclass
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


def input_key_powers(input_dim: int, device: torch.device) -> torch.Tensor:
    if input_dim > 62:
        raise ValueError("Packed input keys support input_dim <= 62")
    return 2 ** torch.arange(input_dim, device=device, dtype=torch.long)


def input_keys(x: torch.Tensor) -> torch.Tensor:
    powers = input_key_powers(x.shape[1], x.device)
    bits = (x > 0).to(dtype=torch.long)
    return bits @ powers


def exclusion_keys(x: torch.Tensor) -> torch.Tensor:
    return torch.unique(input_keys(x), sorted=True)


def tree_parity_indices(relevant_dim: int = 16) -> list[tuple[int, ...]]:
    if relevant_dim != 16:
        raise ValueError("The binary-tree staircase is currently defined for relevant_dim=16")
    groups: list[tuple[int, ...]] = []
    for degree in (2, 4, 8, 16):
        for start in range(0, relevant_dim, degree):
            groups.append(tuple(range(start, start + degree)))
    return groups


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
        raise ValueError("Cannot sample training inputs: excluded set covers the full input space")

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


def labels_from_inputs(x: torch.Tensor, relevant_dim: int = 16) -> torch.Tensor:
    outputs = []
    for indices in tree_parity_indices(relevant_dim):
        idx = torch.tensor(indices, device=x.device, dtype=torch.long)
        outputs.append(torch.prod(x[:, idx], dim=1))
    return torch.stack(outputs, dim=1).to(dtype=x.dtype)


def make_dataset(
    n: int,
    input_dim: int,
    relevant_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityDataset:
    x = sample_inputs(n, input_dim, device).to(dtype=dtype)
    y = labels_from_inputs(x, relevant_dim).to(dtype=dtype)
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
