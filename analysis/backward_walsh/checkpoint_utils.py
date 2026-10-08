"""Utilities for listing and loading checkpoints."""
from __future__ import annotations

import re
from pathlib import Path

import torch

from parity_net.checkpoint import load_checkpoint
from parity_net.train import resolve_dtype


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return (step, path) pairs for all step_*.pt files, sorted by step."""
    result = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = re.match(r"step_0*(\d+)\.pt$", p.name)
        if m:
            result.append((int(m.group(1)), p))
    result.sort(key=lambda t: t[0])
    return result


def load_model_at_checkpoint(ckpt_path: Path, device: torch.device):
    """Load model and return (model_eval, payload, dtype)."""
    model, payload, _ = load_checkpoint(ckpt_path, device)
    cfg = payload["config"]
    dtype = resolve_dtype(cfg["training"]["dtype"])
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, payload, dtype


def find_d16_output_col(model) -> int:
    """Column index of the first d16 target in the model output."""
    if model.target_names is None:
        raise ValueError("model.target_names is None")
    for i, name in enumerate(model.target_names):
        if name.startswith("d16_"):
            return i
    raise ValueError("No d16 target found in model.target_names")
