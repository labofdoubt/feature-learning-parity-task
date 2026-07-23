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
    def __init__(
        self,
        config: ModelConfig,
        output_dim: int = 15,
        target_names_: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.target_names = target_names_

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
        self.readout = None
        self.layerwise_readouts = nn.ModuleDict()
        self.layerwise_readout_order: list[tuple[int, str]] = []
        if config.use_layerwise_readouts:
            if target_names_ is None:
                raise ValueError("target_names_ is required when use_layerwise_readouts=True")
            targets_by_block: dict[int, list[str]] = {}
            for target_name in target_names_:
                degree = int(target_name.split("_", 1)[0][1:])
                block_idx = int(math.log2(degree)) - 1
                if 2 ** (block_idx + 1) != degree:
                    raise ValueError(f"Layerwise readouts require power-of-two target degrees: {target_name}")
                if block_idx < 0 or block_idx >= config.L:
                    raise ValueError(
                        f"Target {target_name} requires block {block_idx}, but model has L={config.L}"
                    )
                targets_by_block.setdefault(block_idx, []).append(target_name)
            for block_idx in sorted(targets_by_block):
                key = str(block_idx)
                readout = nn.Linear(config.N, len(targets_by_block[block_idx]), bias=config.bias)
                self._init_readout(readout)
                self.layerwise_readouts[key] = readout
                self.layerwise_readout_order.append((block_idx, key))
        else:
            self.readout = nn.Linear(config.N, output_dim, bias=config.bias)
            self._init_readout(self.readout)

    def _init_readout(self, readout: nn.Linear) -> None:
        nn.init.normal_(
            readout.weight,
            mean=0.0,
            std=math.sqrt(self.config.readout_weight_variance),
        )
        if readout.bias is not None:
            nn.init.zeros_(readout.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_activations: bool = False,
        intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        block_intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.embedding(x)
        if intervention is not None and intervention[0] == 0:
            h = intervention[1](h)
        activations = [h]
        layerwise_outputs = []
        next_readout_idx = 0
        for layer_idx, block in enumerate(self.blocks):
            activation_intervention = None
            if block_intervention is not None and layer_idx == block_intervention[0]:
                activation_intervention = block_intervention[1]
            h = block(h, activation_intervention=activation_intervention)
            residual_stream_idx = layer_idx + 1
            if intervention is not None and intervention[0] == residual_stream_idx:
                h = intervention[1](h)
            activations.append(h)
            while (
                self.config.use_layerwise_readouts
                and next_readout_idx < len(self.layerwise_readout_order)
                and self.layerwise_readout_order[next_readout_idx][0] == layer_idx
            ):
                _, key = self.layerwise_readout_order[next_readout_idx]
                layerwise_outputs.append(self.layerwise_readouts[key](h))
                next_readout_idx += 1
        if self.config.use_layerwise_readouts:
            y = torch.cat(layerwise_outputs, dim=1)
        else:
            assert self.readout is not None
            y = self.readout(h)
        if return_activations:
            return y, activations
        return y

    def readout_barrier(self, c: float, barrier_lambda: float) -> torch.Tensor:
        penalties = []
        for readout in self.readout_modules():
            excess = torch.relu(readout.weight.abs() - c)
            penalties.append(torch.sum(excess.square()))
        if not penalties:
            return torch.zeros((), device=self.embedding.weight.device, dtype=self.embedding.weight.dtype)
        return barrier_lambda * torch.stack(penalties).sum()

    def readout_modules(self) -> list[nn.Linear]:
        if self.config.use_layerwise_readouts:
            return list(self.layerwise_readouts.values())
        assert self.readout is not None
        return [self.readout]

    def readout_parameters(self):
        for readout in self.readout_modules():
            yield from readout.parameters()

    def readout_weight_matrix(self) -> torch.Tensor:
        if self.config.use_layerwise_readouts:
            return torch.cat(
                [self.layerwise_readouts[key].weight for _, key in self.layerwise_readout_order],
                dim=0,
            )
        assert self.readout is not None
        return self.readout.weight

    def readout_bias_vector(self) -> torch.Tensor | None:
        if self.config.use_layerwise_readouts:
            biases = [self.layerwise_readouts[key].bias for _, key in self.layerwise_readout_order]
            if any(bias is None for bias in biases):
                return None
            return torch.cat([bias for bias in biases if bias is not None], dim=0)
        assert self.readout is not None
        return self.readout.bias

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
        if self.config.use_layerwise_readouts:
            for block_idx, key in self.layerwise_readout_order:
                variances[f"layerwise_readouts.block_{block_idx}.weight"] = (
                    self.layerwise_readouts[key].weight.detach().float().var(unbiased=False).item()
                )
        else:
            assert self.readout is not None
            variances["readout.weight"] = self.readout.weight.detach().float().var(unbiased=False).item()
        return variances
