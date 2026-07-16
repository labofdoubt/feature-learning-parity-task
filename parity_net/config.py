from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelConfig:
    input_dim: int = 32
    relevant_dim: int = 16
    N: int = 1024
    L: int = 4
    activation: Literal["relu", "gelu", "tanh", "silu"] = "silu"
    use_readout_barrier: bool = True
    embedding_weight_variance: float | None = None
    freeze_embedding: bool = True
    hidden_weight_variance: float = 1.0
    readout_weight_variance: float = 1e-4
    use_post_activation_linear: bool = False
    bias: bool = False


@dataclass
class OptimizerConfig:
    name: Literal["sgd", "adamw"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class TrainingConfig:
    num_steps: int = 10_000
    test_samples: int = 20_000
    batch_size: int = 512
    seed: int = 0
    device: str = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    log_every: int = 100
    checkpoint_every: int = 1_000
    output_dir: str = "runs/parity"
    barrier_c: float | None = None
    barrier_lambda: float = 10.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def to_dict(config: Any) -> dict[str, Any]:
    data = asdict(config)
    if "training" in data:
        data["training"]["optimizer"]["betas"] = list(data["training"]["optimizer"]["betas"])
    return data


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(to_dict(config), f, sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    opt = OptimizerConfig(**raw["training"].pop("optimizer"))
    if isinstance(opt.betas, list):
        opt.betas = tuple(opt.betas)
    return ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(optimizer=opt, **raw["training"]),
    )


def write_default_config(path: str | Path) -> None:
    save_config(ExperimentConfig(), path)
