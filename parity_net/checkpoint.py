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
    from .config import ModelConfig, OptimizerConfig, TrainingConfig

    opt_cfg = OptimizerConfig(**raw["training"].pop("optimizer"))
    if isinstance(opt_cfg.betas, list):
        opt_cfg.betas = tuple(opt_cfg.betas)
    config = ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(optimizer=opt_cfg, **raw["training"]),
    )
    model = ParityResidualNet(config.model).to(device)
    model.load_state_dict(payload["model_state"])

    optimizer = None
    if load_optimizer and payload.get("optimizer_state") is not None:
        optimizer = build_optimizer(model, config.training.optimizer)
        optimizer.load_state_dict(payload["optimizer_state"])
    return model, payload, optimizer
