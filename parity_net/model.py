from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from .config import ModelConfig


class HalfTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.tanh(x))


def activation_from_name(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "half-tanh":
        return HalfTanh()
    raise ValueError(f"Unknown activation: {name}")


def orthonormal_embedding(input_dim: int, N: int) -> torch.Tensor:
    if N < input_dim:
        raise ValueError("N must be at least input_dim for W.T @ W = I")
    q, _ = torch.linalg.qr(torch.randn(N, input_dim), mode="reduced")
    return q


def scaled_embedding(input_dim: int, N: int, variance: float | None) -> torch.Tensor:
    embedding = orthonormal_embedding(input_dim, N)
    if variance is None:
        return embedding
    if variance < 0:
        raise ValueError("embedding_weight_variance must be non-negative")
    return embedding * math.sqrt(N * variance)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        activation: str,
        variance: float,
        bias: bool,
        use_post_activation_linear: bool,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=bias)
        self.activation = activation_from_name(activation)
        self.post_activation_linear = (
            nn.Linear(width, width, bias=bias) if use_post_activation_linear else None
        )

        nn.init.normal_(self.linear.weight, mean=0.0, std=math.sqrt(variance))
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.post_activation_linear is not None:
            nn.init.normal_(
                self.post_activation_linear.weight,
                mean=0.0,
                std=math.sqrt(variance),
            )
            if self.post_activation_linear.bias is not None:
                nn.init.zeros_(self.post_activation_linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        activation_intervention: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        update = self.activation(self.linear(x))
        if activation_intervention is not None:
            update = activation_intervention(update)
        if self.post_activation_linear is not None:
            update = self.post_activation_linear(update)
        return x + update


class ParityResidualNet(nn.Module):
    def __init__(self, config: ModelConfig, output_dim: int = 15) -> None:
        super().__init__()
        self.config = config
        self.output_dim = output_dim

        embedding = nn.Linear(config.input_dim, config.N, bias=False)
        with torch.no_grad():
            embedding.weight.copy_(
                scaled_embedding(
                    config.input_dim,
                    config.N,
                    config.embedding_weight_variance,
                )
            )
        embedding.weight.requires_grad_(not config.freeze_embedding)
        self.embedding = embedding

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    config.N,
                    config.activation,
                    config.hidden_weight_variance,
                    config.bias,
                    config.use_post_activation_linear,
                )
                for _ in range(config.L)
            ]
        )
        self.readout = nn.Linear(config.N, output_dim, bias=config.bias)
        nn.init.normal_(
            self.readout.weight,
            mean=0.0,
            std=math.sqrt(config.readout_weight_variance),
        )
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_activations: bool = False,
        intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        block_intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.embedding(x)
        activations = [h]
        for layer_idx, block in enumerate(self.blocks):
            if intervention is not None and layer_idx == intervention[0]:
                h = intervention[1](h)
            activation_intervention = None
            if block_intervention is not None and layer_idx == block_intervention[0]:
                activation_intervention = block_intervention[1]
            h = block(h, activation_intervention=activation_intervention)
            activations.append(h)
        if intervention is not None and intervention[0] == len(self.blocks):
            h = intervention[1](h)
        y = self.readout(h)
        if return_activations:
            return y, activations
        return y

    def readout_barrier(self, c: float, barrier_lambda: float) -> torch.Tensor:
        excess = torch.relu(self.readout.weight.abs() - c)
        return barrier_lambda * torch.sum(excess.square())

    def weight_variances(self) -> dict[str, float]:
        variances = {
            "embedding.weight": self.embedding.weight.detach().float().var(unbiased=False).item()
        }
        for i, block in enumerate(self.blocks):
            variances[f"blocks.{i}.linear.weight"] = (
                block.linear.weight.detach().float().var(unbiased=False).item()
            )
            if block.post_activation_linear is not None:
                variances[f"blocks.{i}.post_activation_linear.weight"] = (
                    block.post_activation_linear.weight.detach().float().var(unbiased=False).item()
                )
        variances["readout.weight"] = self.readout.weight.detach().float().var(unbiased=False).item()
        return variances
