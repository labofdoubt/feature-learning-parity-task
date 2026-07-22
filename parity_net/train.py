from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint
from .config import ExperimentConfig, OptimizerConfig, load_config, save_config, write_default_config
from .data import (
    degree_slices_for_targets,
    exclusion_keys,
    labels_from_inputs,
    make_dataset,
    sample_inputs_excluding,
    save_dataset,
    target_names,
)
from .model import ParityResidualNet


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def build_optimizer(model: ParityResidualNet, config: OptimizerConfig) -> torch.optim.Optimizer:
    param_groups = []
    group_specs = [
        ("embedding", model.embedding.parameters(), config.lr_embedding, config.wd_embedding),
        ("hidden", model.blocks.parameters(), config.lr_hidden, config.wd_hidden),
        ("readout", model.readout.parameters(), config.lr_readout, config.wd_readout),
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
    model: ParityResidualNet,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    target_names_: list[str] | None = None,
) -> dict[str, float]:
    model.eval()
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


def train(config: ExperimentConfig) -> Path:
    training = config.training
    model_config = config.model
    task_config = config.task
    device = resolve_device(training.device)
    dtype = resolve_dtype(training.dtype)
    torch.manual_seed(training.seed)
    target_names_ = target_names(task_config.relevant_dim, task_config.exclude_targets)
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
    )
    test_data_path = output_dir / "test_data.pt"
    save_dataset(test_data, test_data_path)
    test_exclusion_keys = exclusion_keys(test_data.x)

    model = ParityResidualNet(model_config, output_dim=len(target_names_)).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, training.optimizer)
    barrier_c = training.barrier_c
    if barrier_c is None:
        barrier_c = 7.0 / model_config.N

    history = []
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
        ).to(dtype=dtype)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
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
