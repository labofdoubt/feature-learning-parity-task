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
    activation: Literal["relu", "gelu", "tanh", "silu", "half-tanh"] = "silu"
    use_readout_barrier: bool = True
    embedding_weight_variance: float | None = None
    freeze_embedding: bool = True
    hidden_weight_variance: float = 1.0
    readout_weight_variance: float = 1e-4
    use_layerwise_readouts: bool = False
    use_post_activation_linear: bool = False
    bias: bool = False


@dataclass
class TaskConfig:
    input_dim: int = 32
    relevant_dim: int = 16
    exclude_targets: list[str] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    name: Literal["sgd", "adamw"] = "adamw"
    lr: float = 1e-3
    lr_embedding: float | None = None
    lr_hidden: float | None = None
    lr_readout: float | None = None
    weight_decay: float = 0.0
    wd_embedding: float | None = None
    wd_hidden: float | None = None
    wd_readout: float | None = None
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
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        default_task = TaskConfig()
        if self.task == default_task and (
            self.model.input_dim != default_task.input_dim
            or self.model.relevant_dim != default_task.relevant_dim
        ):
            self.task = TaskConfig(
                input_dim=self.model.input_dim,
                relevant_dim=self.model.relevant_dim,
            )
        else:
            self.model.input_dim = self.task.input_dim
            self.model.relevant_dim = self.task.relevant_dim


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
    task_raw = raw.pop("task", None)
    model_raw = raw["model"]
    if task_raw is None:
        task = TaskConfig(
            input_dim=model_raw.get("input_dim", 32),
            relevant_dim=model_raw.get("relevant_dim", 16),
        )
    else:
        task = TaskConfig(**task_raw)
        model_raw["input_dim"] = task.input_dim
        model_raw["relevant_dim"] = task.relevant_dim
    opt = OptimizerConfig(**raw["training"].pop("optimizer"))
    if isinstance(opt.betas, list):
        opt.betas = tuple(opt.betas)
    return ExperimentConfig(
        model=ModelConfig(**model_raw),
        task=task,
        training=TrainingConfig(optimizer=opt, **raw["training"]),
    )


def write_default_config(path: str | Path) -> None:
    save_config(ExperimentConfig(), path)
