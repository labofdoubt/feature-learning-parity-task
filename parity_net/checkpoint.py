from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy

import torch

from .config import ExperimentConfig, to_dict
from .model import ParityResidualNet


def save_checkpoint(
    path: str | Path,
    *,
    model: ParityResidualNet,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    step: int,
    config: ExperimentConfig,
    metrics: dict[str, Any] | None = None,
    test_data_path: str | Path | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "config": to_dict(config),
        "metrics": metrics or {},
        "test_data_path": str(test_data_path) if test_data_path is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    load_optimizer: bool = False,
) -> tuple[ParityResidualNet, dict[str, Any], torch.optim.Optimizer | None]:
    from .train import build_optimizer

    payload = torch.load(path, map_location=device)
    raw = deepcopy(payload["config"])
    from .config import ModelConfig, OptimizerConfig, TaskConfig, TrainingConfig
    from .data import target_names

    task_raw = raw.pop("task", None)
    model_raw = raw["model"]
    if task_raw is None:
        task_config = TaskConfig(
            input_dim=model_raw.get("input_dim", 32),
            relevant_dim=model_raw.get("relevant_dim", 16),
        )
    else:
        task_config = TaskConfig(**task_raw)
        model_raw["input_dim"] = task_config.input_dim
        model_raw["relevant_dim"] = task_config.relevant_dim
    opt_cfg = OptimizerConfig(**raw["training"].pop("optimizer"))
    if isinstance(opt_cfg.betas, list):
        opt_cfg.betas = tuple(opt_cfg.betas)
    config = ExperimentConfig(
        model=ModelConfig(**model_raw),
        task=task_config,
        training=TrainingConfig(optimizer=opt_cfg, **raw["training"]),
    )
    output_dim = len(target_names(config.task.relevant_dim, config.task.exclude_targets))
    model = ParityResidualNet(config.model, output_dim=output_dim).to(device)
    model.load_state_dict(payload["model_state"])

    optimizer = None
    if load_optimizer and payload.get("optimizer_state") is not None:
        optimizer = build_optimizer(model, config.training.optimizer)
        optimizer.load_state_dict(payload["optimizer_state"])
    return model, payload, optimizer
