from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .checkpoint import save_checkpoint
from .config import ExperimentConfig, OptimizerConfig, load_config, save_config, write_default_config
from .data import DEGREE_SLICES, make_dataset, make_loader
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    if config.name == "sgd":
        return torch.optim.SGD(
            trainable,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.name == "adamw":
        return torch.optim.AdamW(
            trainable,
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
) -> dict[str, float]:
    model.eval()
    preds = []
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        preds.append(model(x[start:stop]))
    pred = torch.cat(preds, dim=0)
    metrics = {"test_mse": F.mse_loss(pred, y).item()}
    for degree, slc in DEGREE_SLICES.items():
        metrics[f"test_mse_d{degree}"] = F.mse_loss(pred[:, slc], y[:, slc]).item()
    return metrics


def train(config: ExperimentConfig) -> Path:
    training = config.training
    model_config = config.model
    device = resolve_device(training.device)
    dtype = resolve_dtype(training.dtype)
    torch.manual_seed(training.seed)

    output_dir = Path(training.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    train_data = make_dataset(
        training.train_samples,
        model_config.input_dim,
        model_config.relevant_dim,
        device,
        dtype,
    )
    test_data = make_dataset(
        training.test_samples,
        model_config.input_dim,
        model_config.relevant_dim,
        device,
        dtype,
    )
    loader = make_loader(train_data, training.batch_size, shuffle=True)

    model = ParityResidualNet(model_config).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, training.optimizer)
    barrier_c = training.barrier_c
    if barrier_c is None:
        barrier_c = 7.0 / model_config.N

    history = []
    step = 0
    for epoch in range(1, training.epochs + 1):
        model.train()
        for x_batch, y_batch in loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            mse = F.mse_loss(pred, y_batch)
            barrier = torch.zeros((), device=device, dtype=dtype)
            if model_config.use_readout_barrier:
                barrier = model.readout_barrier(barrier_c, training.barrier_lambda)
            loss = mse + barrier
            loss.backward()
            optimizer.step()

            if training.log_every and step % training.log_every == 0:
                row = {
                    "epoch": epoch,
                    "step": step,
                    "train_mse": mse.item(),
                    "barrier": barrier.item(),
                    "loss": loss.item(),
                }
                history.append(row)
                print(row)

        metrics = evaluate(model, test_data.x, test_data.y, training.batch_size)
        row = {"epoch": epoch, "step": step, **metrics}
        history.append(row)
        print(row)
        pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)

        if training.checkpoint_every and epoch % training.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                config=config,
                metrics=metrics,
            )

    final_path = ckpt_dir / "final.pt"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        epoch=training.epochs,
        step=step,
        config=config,
        metrics=evaluate(model, test_data.x, test_data.y, training.batch_size),
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
