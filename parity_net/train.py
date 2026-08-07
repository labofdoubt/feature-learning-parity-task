from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint
from .config import ExperimentConfig, OptimizerConfig, load_config, save_config, write_default_config
from .data import (
    ParityDataset,
    degree_slices_for_targets,
    exclusion_keys,
    labels_from_inputs,
    make_dataset,
    sample_inputs_excluding,
    sample_unique_inputs_excluding,
    save_dataset,
    target_names,
)
from .model import build_model

if TYPE_CHECKING:
    from .model import ParityResidualNet, ParityTransformer

    ParityModel = ParityResidualNet | ParityTransformer


def max_target_degree_for_model(model_config) -> int | None:
    if not model_config.use_layerwise_readouts:
        return None
    return min(model_config.relevant_dim, 2**model_config.L)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


MATMUL_PRECISIONS = ("highest", "high", "medium")


def apply_matmul_precision(precision: str) -> None:
    """Choose whether float32 matmuls may run on tensor cores.

    "highest" is true float32 and is PyTorch's default. "high" enables TF32, which
    rounds matmul inputs to a 10-bit mantissa while still accumulating in float32;
    on Ampere and later this is typically several times faster, at roughly 1e-3
    relative precision per matmul instead of 1e-7. "medium" additionally allows
    bfloat16 inputs. Only matmuls are affected - the optimizer, loss, and
    elementwise ops stay in float32.

    This sets global process state, so it also applies to anything run after
    training in the same session.
    """
    if precision not in MATMUL_PRECISIONS:
        raise ValueError(
            f"matmul_precision must be one of {MATMUL_PRECISIONS}, got {precision!r}"
        )
    torch.set_float32_matmul_precision(precision)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def build_optimizer(model: ParityModel, config: OptimizerConfig) -> torch.optim.Optimizer:
    param_groups = []
    group_specs = [
        ("embedding", model.embedding.parameters(), config.lr_embedding, config.wd_embedding),
        ("hidden", model.blocks.parameters(), config.lr_hidden, config.wd_hidden),
        ("readout", model.readout_parameters(), config.lr_readout, config.wd_readout),
    ]
    for name, params, group_lr, group_wd in group_specs:
        trainable = [p for p in params if p.requires_grad]
        if trainable:
            param_groups.append(
                {
                    "params": trainable,
                    "lr": config.lr if group_lr is None else group_lr,
                    "weight_decay": config.weight_decay if group_wd is None else group_wd,
                    "name": name,
                }
            )
    if not param_groups:
        raise ValueError("No trainable parameters found")

    if config.name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {config.name}")


@torch.no_grad()
def evaluate(
    model: ParityModel,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    target_names_: list[str] | None = None,
) -> dict[str, float]:
    model.eval()
    # Attention models are scored the way they are used at test time: only the input
    # bits are given, and each prediction is fed back into the next position.
    preds = []
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        preds.append(model(x[start:stop]))
    pred = torch.cat(preds, dim=0)
    metrics = {"test_mse": F.mse_loss(pred, y).item()}
    if target_names_ is None:
        target_names_ = target_names()
    for degree, slc in degree_slices_for_targets(target_names_).items():
        metrics[f"test_mse_d{degree}"] = F.mse_loss(pred[:, slc], y[:, slc]).item()
    return metrics


def train_set_metrics(
    model: ParityModel,
    train_data: ParityDataset | None,
    batch_size: int,
    target_names_: list[str],
) -> dict[str, float]:
    """Same metrics as `evaluate`, over the fixed training pool. Empty when training
    draws a fresh sample every step, since then there is no fixed pool to score."""
    if train_data is None:
        return {}
    metrics = evaluate(model, train_data.x, train_data.y, batch_size, target_names_)
    return {name.replace("test_", "train_set_", 1): value for name, value in metrics.items()}


def train(config: ExperimentConfig) -> Path:
    training = config.training
    model_config = config.model
    task_config = config.task
    device = resolve_device(training.device)
    dtype = resolve_dtype(training.dtype)
    apply_matmul_precision(training.matmul_precision)
    torch.manual_seed(training.seed)
    max_degree = max_target_degree_for_model(model_config)
    target_names_ = target_names(task_config.relevant_dim, task_config.exclude_targets, max_degree)
    if model_config.input_dim != task_config.input_dim:
        model_config.input_dim = task_config.input_dim
    if model_config.relevant_dim != task_config.relevant_dim:
        model_config.relevant_dim = task_config.relevant_dim

    output_dir = Path(training.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    test_data = make_dataset(
        training.test_samples,
        task_config.input_dim,
        task_config.relevant_dim,
        device,
        dtype,
        task_config.exclude_targets,
        max_degree,
    )
    test_data_path = output_dir / "test_data.pt"
    save_dataset(test_data, test_data_path)
    test_exclusion_keys = exclusion_keys(test_data.x)
    input_space_size = 2**task_config.input_dim
    if test_exclusion_keys.numel() >= input_space_size:
        warnings.warn(
            "The saved test set covers the full input space, so training samples "
            "cannot be drawn while avoiding every test input. Training will continue "
            "with ordinary random samples, and train/test overlap is possible.",
            RuntimeWarning,
            stacklevel=2,
        )
        test_exclusion_keys = torch.empty(0, device=device, dtype=torch.long)

    # With train_samples set, training draws a fixed pool of that many distinct inputs
    # once and never sees anything else; otherwise every step gets a fresh sample.
    train_data = None
    train_data_path = None
    train_batch_size = training.batch_size
    if training.train_samples is not None:
        if training.train_samples <= 0:
            raise ValueError("train_samples must be positive or null")
        train_x = sample_unique_inputs_excluding(
            training.train_samples,
            task_config.input_dim,
            device,
            test_exclusion_keys,
        ).to(dtype=dtype)
        train_y = labels_from_inputs(
            train_x,
            task_config.relevant_dim,
            task_config.exclude_targets,
            max_degree,
        ).to(dtype=dtype)
        train_data = ParityDataset(x=train_x, y=train_y)
        train_data_path = output_dir / "train_data.pt"
        save_dataset(train_data, train_data_path)
        train_pool_size = train_data.x.shape[0]
        if train_pool_size < train_batch_size:
            warnings.warn(
                f"train_samples={train_pool_size} is smaller than batch_size="
                f"{train_batch_size}; every step will use the whole training pool.",
                RuntimeWarning,
                stacklevel=2,
            )
            train_batch_size = train_pool_size

    model = build_model(
        model_config,
        output_dim=len(target_names_),
        target_names_=target_names_,
    ).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, training.optimizer)
    barrier_c = training.barrier_c
    if barrier_c is None:
        barrier_c = 7.0 / model_config.N

    history = []
    # Shuffled epochs over the fixed pool, so every training input is seen equally often.
    epoch_order = torch.empty(0, device=device, dtype=torch.long)
    epoch_cursor = 0
    start_time = time.perf_counter()
    progress = tqdm(
        range(1, training.num_steps + 1),
        total=training.num_steps,
        desc="training",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
        model.train()
        if train_data is None:
            x_batch = sample_inputs_excluding(
                training.batch_size,
                task_config.input_dim,
                device,
                test_exclusion_keys,
            ).to(dtype=dtype)
            y_batch = labels_from_inputs(
                x_batch,
                task_config.relevant_dim,
                task_config.exclude_targets,
                max_degree,
            ).to(dtype=dtype)
        else:
            if epoch_cursor + train_batch_size > epoch_order.numel():
                epoch_order = torch.randperm(train_data.x.shape[0], device=device)
                epoch_cursor = 0
            batch_idx = epoch_order[epoch_cursor : epoch_cursor + train_batch_size]
            epoch_cursor += train_batch_size
            x_batch = train_data.x[batch_idx]
            y_batch = train_data.y[batch_idx]

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch, targets=y_batch)
        mse = F.mse_loss(pred, y_batch)
        barrier = torch.zeros((), device=device, dtype=dtype)
        if model_config.use_readout_barrier:
            barrier = model.readout_barrier(barrier_c, training.barrier_lambda)
        loss = mse + barrier
        loss.backward()
        optimizer.step()
        progress.set_postfix(
            train_mse=f"{mse.item():.4g}",
            barrier=f"{barrier.item():.4g}",
            loss=f"{loss.item():.4g}",
        )

        if training.log_every and step % training.log_every == 0:
            metrics = evaluate(model, test_data.x, test_data.y, training.batch_size, target_names_)
            elapsed_seconds = time.perf_counter() - start_time
            row = {
                "step": step,
                "elapsed_seconds": elapsed_seconds,
                "train_mse": mse.item(),
                "barrier": barrier.item(),
                "loss": loss.item(),
                **metrics,
                # "train_mse" above is the current batch; these cover the whole fixed
                # pool, so the gap against the test columns measures memorization.
                **train_set_metrics(model, train_data, training.batch_size, target_names_),
            }
            history.append(row)
            progress.set_postfix(
                train_mse=f"{mse.item():.4g}",
                test_mse=f"{metrics['test_mse']:.4g}",
                barrier=f"{barrier.item():.4g}",
                loss=f"{loss.item():.4g}",
            )
            tqdm.write(str(row))
            pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)

        if training.checkpoint_every and step % training.checkpoint_every == 0:
            metrics = evaluate(model, test_data.x, test_data.y, training.batch_size, target_names_)
            save_checkpoint(
                ckpt_dir / f"step_{step:08d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=0,
                step=step,
                config=config,
                metrics=metrics,
                test_data_path=test_data_path,
            )

    final_metrics = evaluate(model, test_data.x, test_data.y, training.batch_size, target_names_)
    final_row = {
        "step": training.num_steps,
        "elapsed_seconds": time.perf_counter() - start_time,
        **final_metrics,
        **train_set_metrics(model, train_data, training.batch_size, target_names_),
    }
    history.append(final_row)
    pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)

    final_path = ckpt_dir / "final.pt"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        step=training.num_steps,
        config=config,
        metrics=final_metrics,
        test_data_path=test_data_path,
    )
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--write-default-config", type=str, required=False)
    args = parser.parse_args()

    if args.write_default_config:
        write_default_config(args.write_default_config)
        return
    if not args.config:
        raise SystemExit("Provide --config or --write-default-config")
    train(load_config(args.config))


if __name__ == "__main__":
    main()
