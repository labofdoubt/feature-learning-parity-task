from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
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
        targets: torch.Tensor | None = None,
        return_activations: bool = False,
        intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        block_intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        del targets  # Accepted for interface parity with ParityTransformer; unused here.
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


class LayerKVCache:
    """Keys and values already computed for one attention layer, shape
    (batch, heads, seq, head_dim). Used to avoid recomputing the prefix at every
    autoregressive decoding step."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention, no normalization, no dropout."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        variance: float,
        bias: bool,
        logit_scale: str,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if width % num_heads:
            raise ValueError(f"N={width} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        if logit_scale == "1/sqrt(d)":
            # Standard parameterization.
            self.logit_scale = 1.0 / math.sqrt(self.head_dim)
        elif logit_scale == "1/d":
            # muP: query-key logits stay Theta(1) as head_dim grows with width.
            self.logit_scale = 1.0 / self.head_dim
        else:
            raise ValueError(f"Unknown attention_logit_scale: {logit_scale}")

        self.q_proj = nn.Linear(width, width, bias=bias)
        self.k_proj = nn.Linear(width, width, bias=bias)
        self.v_proj = nn.Linear(width, width, bias=bias)
        self.out_proj = nn.Linear(width, width, bias=bias)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=math.sqrt(variance))
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def new_cache(self) -> LayerKVCache:
        return LayerKVCache()

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = projected.shape
        return projected.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, *, cache: LayerKVCache | None = None) -> torch.Tensor:
        batch, seq, width = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        is_causal = True
        if cache is not None:
            extending = cache.length > 0
            k, v = cache.append(k, v)
            if extending:
                # Decoding one position at a time: every cached key precedes the
                # single query, so no mask applies. (is_causal would be wrong here
                # because it aligns the mask to the top-left of a 1 x kv_len grid.)
                if seq != 1:
                    raise ValueError(
                        "KV-cached decoding appends one position at a time; "
                        f"got {seq} positions with a cache of length {cache.length - seq}"
                    )
                is_causal = False
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=self.logit_scale
        )
        return self.out_proj(attended.transpose(1, 2).reshape(batch, seq, width))


class PrefixMeanCache:
    """Running sum and count for uniform causal mixing, the analogue of a KV cache."""

    def __init__(self) -> None:
        self.total: torch.Tensor | None = None
        self.count = 0

    def append(self, v: torch.Tensor) -> torch.Tensor:
        running = v.cumsum(dim=1)
        if self.total is not None:
            running = running + self.total.unsqueeze(1)
        self.total = running[:, -1, :]
        positions = torch.arange(
            self.count + 1, self.count + 1 + v.shape[1], device=v.device, dtype=v.dtype
        )
        self.count += v.shape[1]
        return running / positions.view(1, -1, 1)


class UniformCausalMixing(nn.Module):
    """Attention with the softmax frozen to uniform: every position averages the value
    vectors of the whole causal prefix. Keeps the V and O projections so it ablates the
    query-key selectivity specifically, not the ability to mix across positions."""

    def __init__(self, width: int, variance: float, bias: bool) -> None:
        super().__init__()
        self.v_proj = nn.Linear(width, width, bias=bias)
        self.out_proj = nn.Linear(width, width, bias=bias)
        for projection in (self.v_proj, self.out_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=math.sqrt(variance))
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def new_cache(self) -> PrefixMeanCache:
        return PrefixMeanCache()

    def forward(self, x: torch.Tensor, *, cache: PrefixMeanCache | None = None) -> torch.Tensor:
        v = self.v_proj(x)
        if cache is None:
            positions = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=v.dtype)
            pooled = v.cumsum(dim=1) / positions.view(1, -1, 1)
        else:
            pooled = cache.append(v)
        return self.out_proj(pooled)


def build_sequence_mixing(
    sequence_mixing: str,
    width: int,
    num_heads: int,
    variance: float,
    bias: bool,
    attention_logit_scale: str,
) -> nn.Module | None:
    if sequence_mixing == "attention":
        return CausalSelfAttention(width, num_heads, variance, bias, attention_logit_scale)
    if sequence_mixing == "uniform":
        return UniformCausalMixing(width, variance, bias)
    if sequence_mixing == "none":
        return None
    raise ValueError(
        f'sequence_mixing must be "attention", "uniform", or "none", got {sequence_mixing!r}'
    )


class TransformerBlock(nn.Module):
    """A sequence-mixing sub-layer with a residual connection, then the same MLP block
    the residual net uses (which carries its own residual connection)."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        activation: str,
        variance: float,
        bias: bool,
        use_post_activation_linear: bool,
        attention_logit_scale: str,
        sequence_mixing: str = "attention",
    ) -> None:
        super().__init__()
        self.mixing = build_sequence_mixing(
            sequence_mixing, width, num_heads, variance, bias, attention_logit_scale
        )
        self.mlp = ResidualBlock(width, activation, variance, bias, use_post_activation_linear)

    @property
    def attention(self) -> nn.Module | None:
        """Alias kept so analysis code can keep reaching for block.attention."""
        return self.mixing

    def new_cache(self):
        return None if self.mixing is None else self.mixing.new_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        cache=None,
        activation_intervention: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.mixing is not None:
            x = x + self.mixing(x, cache=cache)
        return self.mlp(x, activation_intervention=activation_intervention)


class ParityTransformer(nn.Module):
    """Next-position parity prediction with causal attention.

    The sequence is the `input_dim` input bits followed by the target parities in
    binary-tree order (all degree-2 targets, then degree-4, degree-8, ...). Position
    `input_dim - 1` — the last input bit — predicts the first target, position
    `input_dim` predicts the second, and so on. Every position, input and answer
    alike, owns a learnable embedding vector; the value at that position (a +/-1 bit
    or parity) scales it, exactly as `W x` scales the columns of the residual net's
    embedding. The unembedding is a single position-independent N -> 1 map.

    Training is teacher-forced (pass `targets`); evaluation without `targets`
    generates autoregressively from the input bits alone.
    """

    def __init__(
        self,
        config: ModelConfig,
        output_dim: int = 15,
        target_names_: list[str] | None = None,
    ) -> None:
        super().__init__()
        if config.use_layerwise_readouts:
            raise ValueError("use_layerwise_readouts is not supported with use_attention=True")
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        if config.sequence_mixing == "none" and output_dim > 1:
            warnings.warn(
                'sequence_mixing="none" removes every path between positions, so each '
                "position sees only its own value and cannot compute a parity of others. "
                'Use "uniform" to keep the mixing but drop the query-key selectivity.',
                RuntimeWarning,
                stacklevel=2,
            )
        self.config = config
        self.output_dim = output_dim
        self.target_names = target_names_
        # The final target is never fed back in, so it needs no input position.
        self.num_positions = config.input_dim + output_dim - 1
        self.first_prediction_position = config.input_dim - 1

        embedding = nn.Linear(self.num_positions, config.N, bias=False)
        with torch.no_grad():
            embedding.weight.copy_(
                scaled_embedding(
                    self.num_positions,
                    config.N,
                    config.embedding_weight_variance,
                )
            )
        embedding.weight.requires_grad_(not config.freeze_embedding)
        self.embedding = embedding

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.N,
                    config.num_heads,
                    config.activation,
                    config.hidden_weight_variance,
                    config.bias,
                    config.use_post_activation_linear,
                    config.attention_logit_scale,
                    config.sequence_mixing,
                )
                for _ in range(config.L)
            ]
        )

        self.readout = nn.Linear(config.N, 1, bias=config.bias)
        nn.init.normal_(self.readout.weight, mean=0.0, std=math.sqrt(config.readout_weight_variance))
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def embed(self, values: torch.Tensor, start_position: int = 0) -> torch.Tensor:
        """(batch, seq) sequence values -> (batch, seq, N) residual stream, taking
        embedding vectors from `start_position` onwards."""
        stop = start_position + values.shape[1]
        if start_position < 0 or stop > self.num_positions:
            raise ValueError(
                f"Positions [{start_position}, {stop}) fall outside the "
                f"{self.num_positions} embedded positions"
            )
        position_embeddings = self.embedding.weight[:, start_position:stop].transpose(0, 1)
        return values.unsqueeze(-1) * position_embeddings.unsqueeze(0)

    def run_blocks(
        self,
        h: torch.Tensor,
        *,
        caches: list[LayerKVCache] | None = None,
        intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        block_intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if intervention is not None and intervention[0] == 0:
            h = intervention[1](h)
        activations = [h]
        for layer_idx, block in enumerate(self.blocks):
            activation_intervention = None
            if block_intervention is not None and layer_idx == block_intervention[0]:
                activation_intervention = block_intervention[1]
            h = block(
                h,
                cache=None if caches is None else caches[layer_idx],
                activation_intervention=activation_intervention,
            )
            if intervention is not None and intervention[0] == layer_idx + 1:
                h = intervention[1](h)
            activations.append(h)
        return h, activations

    def forward(
        self,
        x: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        return_activations: bool = False,
        intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        block_intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} input positions, got {x.shape[1]}"
            )
        if targets is None:
            if return_activations:
                raise ValueError(
                    "return_activations requires teacher forcing; pass targets=... "
                    "to run a single forward pass over the full sequence"
                )
            return self.generate(
                x,
                intervention=intervention,
                block_intervention=block_intervention,
            )

        if targets.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected {self.output_dim} targets, got {targets.shape[1]}"
            )
        values = x if self.output_dim == 1 else torch.cat([x, targets[:, :-1]], dim=1)
        h, activations = self.run_blocks(
            self.embed(values),
            intervention=intervention,
            block_intervention=block_intervention,
        )
        y = self.readout(h[:, self.first_prediction_position :, :]).squeeze(-1)
        if return_activations:
            return y, activations
        return y

    def feedback_value(self, prediction: torch.Tensor) -> torch.Tensor:
        """The value written into the next sequence position during generation."""
        if self.config.autoregressive_feedback == "sign":
            ones = torch.ones_like(prediction)
            return torch.where(prediction >= 0, ones, -ones)
        return prediction

    def generate(
        self,
        x: torch.Tensor,
        *,
        use_cache: bool | None = None,
        intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        block_intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Predict every target from the input bits alone, feeding each prediction back.

        With the key/value cache the input bits are run once and each later position
        costs a single-position forward pass, so the whole generation costs about as
        much as one full-sequence pass. `use_cache` defaults to `config.use_kv_cache`;
        the uncached path recomputes the prefix at every step and produces the same
        numbers. Interventions force the uncached path, since an intervention may be a
        function of the whole prefix rather than of one position at a time.
        """
        if use_cache is None:
            use_cache = self.config.use_kv_cache
        if intervention is not None or block_intervention is not None:
            use_cache = False
        if not use_cache:
            values = x
            predictions = []
            for step in range(self.output_dim):
                h, _ = self.run_blocks(
                    self.embed(values),
                    intervention=intervention,
                    block_intervention=block_intervention,
                )
                predictions.append(self.readout(h[:, -1, :]).squeeze(-1))
                if step + 1 < self.output_dim:
                    values = torch.cat(
                        [values, self.feedback_value(predictions[-1]).unsqueeze(1)], dim=1
                    )
            return torch.stack(predictions, dim=1)

        caches = [block.new_cache() for block in self.blocks]
        h, _ = self.run_blocks(self.embed(x), caches=caches)
        predictions = [self.readout(h[:, -1, :]).squeeze(-1)]
        for position in range(self.config.input_dim, self.num_positions):
            feedback = self.feedback_value(predictions[-1]).unsqueeze(1)
            h, _ = self.run_blocks(self.embed(feedback, position), caches=caches)
            predictions.append(self.readout(h[:, -1, :]).squeeze(-1))
        return torch.stack(predictions, dim=1)

    def readout_barrier(self, c: float, barrier_lambda: float) -> torch.Tensor:
        excess = torch.relu(self.readout.weight.abs() - c)
        return barrier_lambda * torch.sum(excess.square())

    def readout_modules(self) -> list[nn.Linear]:
        return [self.readout]

    def readout_parameters(self):
        yield from self.readout.parameters()

    def readout_weight_matrix(self) -> torch.Tensor:
        return self.readout.weight

    def readout_bias_vector(self) -> torch.Tensor | None:
        return self.readout.bias

    def weight_variances(self) -> dict[str, float]:
        variances = {
            "embedding.weight": self.embedding.weight.detach().float().var(unbiased=False).item()
        }
        for i, block in enumerate(self.blocks):
            if block.mixing is not None:
                for name, parameter in block.mixing.named_parameters():
                    variances[f"blocks.{i}.mixing.{name}"] = (
                        parameter.detach().float().var(unbiased=False).item()
                    )
            variances[f"blocks.{i}.mlp.linear.weight"] = (
                block.mlp.linear.weight.detach().float().var(unbiased=False).item()
            )
            if block.mlp.post_activation_linear is not None:
                variances[f"blocks.{i}.mlp.post_activation_linear.weight"] = (
                    block.mlp.post_activation_linear.weight.detach().float().var(unbiased=False).item()
                )
        variances["readout.weight"] = self.readout.weight.detach().float().var(unbiased=False).item()
        return variances


def build_model(
    config: ModelConfig,
    output_dim: int = 15,
    target_names_: list[str] | None = None,
) -> ParityResidualNet | ParityTransformer:
    """Construct the architecture selected by `config.use_attention`."""
    if config.use_attention:
        return ParityTransformer(config, output_dim=output_dim, target_names_=target_names_)
    return ParityResidualNet(config, output_dim=output_dim, target_names_=target_names_)
