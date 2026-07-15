from __future__ import annotations

from dataclasses import dataclass

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


def make_loader(dataset: ParityDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        TensorDataset(dataset.x, dataset.y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
