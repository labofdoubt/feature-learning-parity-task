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

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False

from .checkpoint import save_checkpoint
from .config import ExperimentConfig, OptimizerConfig, load_config, save_config, write_default_config
from .data import (
    ParityDataset,
    REUSE_STAR_TARGET_NAMES,
    REUSE_STAR_TARGET_SUPPORTS,
    REUSE_STAR_SHARED_SUPPORT,
    REUSE_STAR_PRIVATE_SUPPORTS,
    degree_slices_for_targets,
    exclusion_keys,
    labels_from_inputs,
    make_dataset,
    make_hierarchical_dataset,
    make_hierarchical_degree2_dataset,
    make_reuse_star_dataset,
    make_uniform_eval_dataset,
    sample_hierarchical_degree2_inputs,
    sample_hierarchical_inputs,
    sample_inputs_excluding,
    sample_reuse_star_inputs,
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
    metrics = {
        "test_mse": F.mse_loss(pred, y).item(),
        "test_acc": ((pred * y) > 0).float().mean().item(),
    }
    if target_names_ is None:
        target_names_ = target_names()
    for degree, slc in degree_slices_for_targets(target_names_).items():
        metrics[f"test_mse_d{degree}"] = F.mse_loss(pred[:, slc], y[:, slc]).item()
        metrics[f"test_acc_d{degree}"] = ((pred[:, slc] * y[:, slc]) > 0).float().mean().item()
    return metrics


def per_target_batch_mse(
    pred: torch.Tensor,
    y_batch: torch.Tensor,
    target_names_: list[str],
) -> dict[str, float]:
    """MSE of the current batch split by individual parity target."""
    values = (pred.detach() - y_batch).square().mean(dim=0)
    return dict(zip(target_names_, values.tolist()))


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
    if config.task.task_type == "reuse_star":
        return _train_reuse_star(config)
    training = config.training
    model_config = config.model
    task_config = config.task
    device = resolve_device(training.device)
    dtype = resolve_dtype(training.dtype)
    apply_matmul_precision(training.matmul_precision)
    torch.manual_seed(training.seed)
    max_degree = max_target_degree_for_model(model_config)
    # When train_only_root is set, exclude all parity degrees below the root so
    # the loss covers only the degree-relevant_dim target (explicit guard against
    # accidentally retaining intermediate supervision in non-uniform experiments).
    effective_exclude = list(task_config.exclude_targets)
    if task_config.train_only_root:
        deg = 2
        while deg < task_config.relevant_dim:
            if f"d{deg}" not in effective_exclude:
                effective_exclude.append(f"d{deg}")
            deg *= 2
    target_names_ = target_names(task_config.relevant_dim, effective_exclude, max_degree)
    if model_config.input_dim != task_config.input_dim:
        model_config.input_dim = task_config.input_dim
    if model_config.relevant_dim != task_config.relevant_dim:
        model_config.relevant_dim = task_config.relevant_dim

    output_dir = Path(training.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    writer = None
    if _HAS_TENSORBOARD:
        writer = _SummaryWriter(str(output_dir / "tb_logs"))
        degree_slices_ = degree_slices_for_targets(target_names_)
        degrees_ = sorted(degree_slices_.keys())
        train_degree_tags = [f"train/d{d}" for d in degrees_]
        test_degree_tags  = [f"test/d{d}"  for d in degrees_]
        custom_layout: dict = {
            "MSE by degree": {
                "train": ["Multiline", train_degree_tags],
                "test":  ["Multiline", test_degree_tags],
            },
            "Total MSE": {
                "train": ["Multiline", ["train/total"]],
                "test":  ["Multiline", ["test/total"]],
            },
            "Eval accuracy": {
                "uniform":    ["Multiline", ["eval_uniform/acc"]],
                "nonuniform": ["Multiline", ["eval_nonuniform/acc"]],
            },
            "Eval loss": {
                "uniform":    ["Multiline", ["eval_uniform/loss"]],
                "nonuniform": ["Multiline", ["eval_nonuniform/loss"]],
            },
            "Generalization gap": {
                "loss_gap": ["Multiline", ["eval_gap/loss"]],
                "acc_gap":  ["Multiline", ["eval_gap/acc"]],
            },
        }
        writer.add_custom_scalars(custom_layout)

    test_data = make_dataset(
        training.test_samples,
        task_config.input_dim,
        task_config.relevant_dim,
        device,
        dtype,
        effective_exclude,
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

    task_rho = task_config.data_rho
    task_distribution = task_config.data_distribution
    use_nonuniform = task_rho > 0.0 and task_distribution != "uniform"
    use_degree2 = use_nonuniform and task_distribution == "hierarchical_degree2"
    eval_noise_repeats = task_config.eval_noise_repeats

    # --- Uniform exhaustive eval dataset (fixed for the whole run) ---------------
    # Enumerates all 2^relevant_dim relevant-bit configs (for k<=20) with
    # eval_noise_repeats independent irrelevant-bit draws each.
    eval_uniform = make_uniform_eval_dataset(
        task_config.relevant_dim,
        task_config.input_dim,
        device,
        dtype,
        eval_noise_repeats=eval_noise_repeats,
        seed=training.seed,
        exclude_targets=effective_exclude,
        max_degree=max_degree,
    )
    eval_uniform_path = output_dir / "eval_uniform.pt"
    torch.save(
        {
            "x": eval_uniform.x.cpu(),
            "y": eval_uniform.y.cpu(),
            "rho": 0.0,
            "relevant_dim": task_config.relevant_dim,
            "input_dim": task_config.input_dim,
            "eval_noise_repeats": eval_noise_repeats,
            "seed": training.seed,
            "kind": "uniform_exhaustive",
        },
        eval_uniform_path,
    )

    # --- Non-uniform eval dataset (only when rho > 0 and distribution != uniform) -
    eval_nonuniform: ParityDataset | None = None
    eval_nonuniform_path: Path | None = None
    if use_nonuniform:
        n_nonuniform = eval_uniform.x.shape[0]
        gen_nonuniform = torch.Generator(device=device)
        gen_nonuniform.manual_seed(training.seed + 2_000_003)
        if use_degree2:
            eval_nonuniform = make_hierarchical_degree2_dataset(
                n_nonuniform,
                task_rho,
                task_config.relevant_dim,
                task_config.input_dim,
                device,
                dtype,
                effective_exclude,
                max_degree,
                generator=gen_nonuniform,
            )
            kind_str = "nonuniform_hierarchical_degree2"
        else:
            eval_nonuniform = make_hierarchical_dataset(
                n_nonuniform,
                task_rho,
                task_config.relevant_dim,
                task_config.input_dim,
                device,
                dtype,
                effective_exclude,
                max_degree,
                generator=gen_nonuniform,
            )
            kind_str = "nonuniform_hierarchical"
        eval_nonuniform_path = output_dir / "eval_nonuniform.pt"
        torch.save(
            {
                "x": eval_nonuniform.x.cpu(),
                "y": eval_nonuniform.y.cpu(),
                "rho": task_rho,
                "relevant_dim": task_config.relevant_dim,
                "input_dim": task_config.input_dim,
                "eval_noise_repeats": eval_noise_repeats,
                "seed": training.seed + 2_000_003,
                "kind": kind_str,
            },
            eval_nonuniform_path,
        )

    # With train_samples set, training draws a fixed pool of that many distinct inputs
    # once and never sees anything else; otherwise every step gets a fresh sample.
    # When task_rho > 0, pool samples come from the hierarchical distribution rather
    # than the uniform one; uniqueness is not enforced (not meaningful for large n).
    train_data = None
    train_data_path = None
    train_batch_size = training.batch_size
    if training.train_samples is not None:
        if training.train_samples <= 0:
            raise ValueError("train_samples must be positive or null")
        if use_nonuniform:
            gen_train = torch.Generator(device=device)
            gen_train.manual_seed(training.seed)
            if use_degree2:
                train_x, _ = sample_hierarchical_degree2_inputs(
                    training.train_samples,
                    task_config.relevant_dim,
                    task_config.input_dim,
                    device,
                    dtype,
                    task_rho,
                    generator=gen_train,
                )
            else:
                train_x, _ = sample_hierarchical_inputs(
                    training.train_samples,
                    task_config.relevant_dim,
                    task_config.input_dim,
                    device,
                    dtype,
                    task_rho,
                    generator=gen_train,
                )
        else:
            train_x = sample_unique_inputs_excluding(
                training.train_samples,
                task_config.input_dim,
                device,
                test_exclusion_keys,
            ).to(dtype=dtype)
        train_y = labels_from_inputs(
            train_x,
            task_config.relevant_dim,
            effective_exclude,
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
    if not 0.0 <= training.teacher_forcing_ratio <= 1.0:
        raise ValueError("teacher_forcing_ratio must be in [0, 1]")
    uses_scheduled_sampling = training.teacher_forcing_ratio < 1.0 and hasattr(model, "generate")
    if training.teacher_forcing_ratio < 1.0 and not uses_scheduled_sampling:
        warnings.warn(
            "teacher_forcing_ratio only applies to autoregressive models; the residual "
            "MLP emits every target in one pass, so it is ignored here.",
            RuntimeWarning,
            stacklevel=2,
        )
    barrier_c = training.barrier_c
    if barrier_c is None:
        barrier_c = 7.0 / model_config.N

    # Curriculum: start with only the lowest-degree targets in the loss and unlock the
    # next degree once the current highest one is trained. Targets are ordered by
    # degree, so the active set is always a prefix of the columns.
    curriculum_slices = degree_slices_for_targets(target_names_)
    curriculum_degrees = sorted(curriculum_slices)
    active_degree_count = 1 if training.curriculum else len(curriculum_degrees)
    if training.curriculum:
        if training.curriculum_mse_threshold <= 0:
            raise ValueError("curriculum_mse_threshold must be positive")
        if len(curriculum_degrees) < 2:
            warnings.warn(
                f"curriculum is on but the task has a single degree ({curriculum_degrees}), "
                "so nothing is ever gated.",
                RuntimeWarning,
                stacklevel=2,
            )
        print(
            f"Curriculum on: degrees {curriculum_degrees} unlock in order once the "
            f"current top degree's train MSE drops below {training.curriculum_mse_threshold}. "
            f"Starting with d{curriculum_degrees[0]} only."
        )

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
            if use_nonuniform:
                if use_degree2:
                    x_batch, _ = sample_hierarchical_degree2_inputs(
                        training.batch_size,
                        task_config.relevant_dim,
                        task_config.input_dim,
                        device,
                        dtype,
                        task_rho,
                    )
                else:
                    x_batch, _ = sample_hierarchical_inputs(
                        training.batch_size,
                        task_config.relevant_dim,
                        task_config.input_dim,
                        device,
                        dtype,
                        task_rho,
                    )
            else:
                x_batch = sample_inputs_excluding(
                    training.batch_size,
                    task_config.input_dim,
                    device,
                    test_exclusion_keys,
                ).to(dtype=dtype)
            y_batch = labels_from_inputs(
                x_batch,
                task_config.relevant_dim,
                effective_exclude,
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

        # Teacher forcing puts the true parities in the answer positions. Below the
        # ratio, the model instead rolls out its own predictions first and trains on
        # that context; the loss targets stay the true parities either way, so this is
        # scheduled sampling, not self-distillation. The rollout is detached, so no
        # gradient flows back through it.
        context_targets = y_batch
        if uses_scheduled_sampling and float(torch.rand(())) >= training.teacher_forcing_ratio:
            with torch.no_grad():
                context_targets = model.feedback_value(model.generate(x_batch))

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch, targets=context_targets)
        active_stop = curriculum_slices[curriculum_degrees[active_degree_count - 1]].stop
        mse = F.mse_loss(pred[:, :active_stop], y_batch[:, :active_stop])
        barrier = torch.zeros((), device=device, dtype=dtype)
        if model_config.use_readout_barrier:
            barrier = model.readout_barrier(barrier_c, training.barrier_lambda)
        loss = mse + barrier
        loss.backward()
        optimizer.step()

        if training.curriculum and active_degree_count < len(curriculum_degrees):
            top_degree = curriculum_degrees[active_degree_count - 1]
            with torch.no_grad():
                top_mse = F.mse_loss(
                    pred[:, curriculum_slices[top_degree]],
                    y_batch[:, curriculum_slices[top_degree]],
                ).item()
            if top_mse < training.curriculum_mse_threshold:
                active_degree_count += 1
                tqdm.write(
                    f"step {step}: d{top_degree} train MSE {top_mse:.4g} < "
                    f"{training.curriculum_mse_threshold} -> unlocking "
                    f"d{curriculum_degrees[active_degree_count - 1]}"
                )

        progress.set_postfix(
            train_mse=f"{mse.item():.4g}",
            barrier=f"{barrier.item():.4g}",
            loss=f"{loss.item():.4g}",
        )

        should_validate = training.validate_every and step % training.validate_every == 0
        should_checkpoint = training.checkpoint_every and step % training.checkpoint_every == 0
        # One evaluation serves both, so coinciding schedules do not validate twice.
        metrics = (
            evaluate(model, test_data.x, test_data.y, training.batch_size, target_names_)
            if should_validate or should_checkpoint
            else None
        )

        if training.progress_every and step % training.progress_every == 0 and not should_validate:
            # Cheap heartbeat: the current batch only, split by parity. No evaluation.
            breakdown = " ".join(
                f"{name}={value:.4g}"
                for name, value in per_target_batch_mse(pred, y_batch, target_names_).items()
            )
            tqdm.write(
                f"step {step}: train_mse={mse.item():.4g} loss={loss.item():.4g} "
                f"elapsed={time.perf_counter() - start_time:.1f}s | {breakdown}"
            )

        if should_validate:
            elapsed_seconds = time.perf_counter() - start_time
            train_pool_metrics = train_set_metrics(model, train_data, training.batch_size, target_names_)

            # Evaluate on the two fixed held-out datasets
            m_uniform = evaluate(
                model, eval_uniform.x, eval_uniform.y, training.batch_size, target_names_
            )
            eval_uniform_metrics = {
                k.replace("test_", "eval_uniform_", 1): v for k, v in m_uniform.items()
            }

            eval_nonuniform_metrics: dict = {}
            if eval_nonuniform is not None:
                m_nonuniform = evaluate(
                    model, eval_nonuniform.x, eval_nonuniform.y,
                    training.batch_size, target_names_,
                )
                eval_nonuniform_metrics = {
                    k.replace("test_", "eval_nonuniform_", 1): v for k, v in m_nonuniform.items()
                }

            row = {
                "step": step,
                "elapsed_seconds": elapsed_seconds,
                # train_mse is the loss actually optimized, so under a curriculum it
                # covers only the unlocked degrees.
                "train_mse": mse.item(),
                "curriculum_max_degree": curriculum_degrees[active_degree_count - 1],
                "barrier": barrier.item(),
                "loss": loss.item(),
                **metrics,
                # "train_mse" above is the current batch; these cover the whole fixed
                # pool, so the gap against the test columns measures memorization.
                **train_pool_metrics,
                **eval_uniform_metrics,
                **eval_nonuniform_metrics,
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

            if writer is not None:
                writer.add_scalar("train/total", mse.item(), step)
                writer.add_scalar("test/total", metrics["test_mse"], step)
                for degree, slc in degree_slices_for_targets(target_names_).items():
                    writer.add_scalar(f"train/d{degree}",
                                      F.mse_loss(pred[:, slc], y_batch[:, slc]).item(), step)
                    writer.add_scalar(f"test/d{degree}",
                                      metrics.get(f"test_mse_d{degree}", float("nan")), step)

                # Uniform eval
                writer.add_scalar("eval_uniform/loss", m_uniform["test_mse"], step)
                writer.add_scalar("eval_uniform/acc",  m_uniform["test_acc"], step)

                # Non-uniform eval (only when rho > 0)
                if eval_nonuniform is not None:
                    writer.add_scalar("eval_nonuniform/loss", m_nonuniform["test_mse"], step)
                    writer.add_scalar("eval_nonuniform/acc",  m_nonuniform["test_acc"], step)
                    # Generalization gap
                    writer.add_scalar(
                        "eval_gap/loss",
                        m_uniform["test_mse"] - m_nonuniform["test_mse"],
                        step,
                    )
                    writer.add_scalar(
                        "eval_gap/acc",
                        m_nonuniform["test_acc"] - m_uniform["test_acc"],
                        step,
                    )

        if should_checkpoint:
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
    m_uniform_final = evaluate(
        model, eval_uniform.x, eval_uniform.y, training.batch_size, target_names_
    )
    final_eval_uniform = {k.replace("test_", "eval_uniform_", 1): v for k, v in m_uniform_final.items()}
    final_eval_nonuniform: dict = {}
    if eval_nonuniform is not None:
        m_nonuniform_final = evaluate(
            model, eval_nonuniform.x, eval_nonuniform.y, training.batch_size, target_names_
        )
        final_eval_nonuniform = {
            k.replace("test_", "eval_nonuniform_", 1): v for k, v in m_nonuniform_final.items()
        }
    final_row = {
        "step": training.num_steps,
        "elapsed_seconds": time.perf_counter() - start_time,
        **final_metrics,
        **train_set_metrics(model, train_data, training.batch_size, target_names_),
        **final_eval_uniform,
        **final_eval_nonuniform,
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
    if writer is not None:
        writer.close()
    return final_path


def _train_reuse_star(config: ExperimentConfig) -> Path:
    """Training loop for the reuse-star task (three degree-16 targets sharing A).

    Architecture: 3-output head, fixed for all m. Loss is averaged over the m
    active targets only (controlled by task_config.num_reuse_targets).
    Data generation never depends on m — only the loss mask changes.
    """
    training = config.training
    model_config = config.model
    task_config = config.task
    device = resolve_device(training.device)
    dtype = resolve_dtype(training.dtype)
    apply_matmul_precision(training.matmul_precision)
    torch.manual_seed(training.seed)

    m = task_config.num_reuse_targets
    if not 1 <= m <= 3:
        raise ValueError(f"num_reuse_targets must be 1, 2, or 3; got {m}")

    rho = task_config.data_rho
    data_dist = task_config.data_distribution
    use_nonuniform = rho > 0.0 and data_dist != "uniform"

    # Active-target mask: first m targets are active, the rest get zero weight.
    active_mask = torch.zeros(3, device=device, dtype=dtype)
    active_mask[:m] = 1.0

    output_dir = Path(training.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    writer = None
    if _HAS_TENSORBOARD:
        writer = _SummaryWriter(str(output_dir / "tb_logs"))
        layout = {
            "Active-target MSE": {
                "train": ["Multiline", ["train/active"]],
                "eval_uniform": ["Multiline", ["eval_uniform/active/loss"]],
                "eval_nonuniform": ["Multiline", ["eval_nonuniform/active/loss"]],
            },
            "Active-target accuracy": {
                "eval_uniform": ["Multiline", ["eval_uniform/active/acc"]],
                "eval_nonuniform": ["Multiline", ["eval_nonuniform/active/acc"]],
            },
            "Per-target loss (uniform eval)": {
                k: ["Multiline", [f"eval_uniform/{k}/loss"]]
                for k in REUSE_STAR_TARGET_NAMES
            },
            "Generalization gap": {
                "loss_gap": ["Multiline", ["eval_gap/active/loss"]],
                "acc_gap":  ["Multiline", ["eval_gap/active/acc"]],
            },
        }
        writer.add_custom_scalars(layout)

    # --- Evaluation datasets (fixed for the whole run) -------------------------
    gen_eval_unif = torch.Generator(device=device)
    gen_eval_unif.manual_seed(training.seed + 1_000_001)
    n_eval = max(training.test_samples, 2 ** 15)
    eval_uniform_ds = make_reuse_star_dataset(n_eval, 0.0, "uniform", device, dtype, gen_eval_unif)
    eval_uniform_path = output_dir / "eval_uniform.pt"
    torch.save(
        {
            "x": eval_uniform_ds.x.cpu(), "y": eval_uniform_ds.y.cpu(),
            "rho": 0.0, "kind": "reuse_star_uniform",
            "num_reuse_targets": m,
            "target_supports": REUSE_STAR_TARGET_SUPPORTS,
            "shared_support": REUSE_STAR_SHARED_SUPPORT,
            "private_supports": REUSE_STAR_PRIVATE_SUPPORTS,
        },
        eval_uniform_path,
    )

    eval_nonuniform_ds: ParityDataset | None = None
    eval_nonuniform_path: Path | None = None
    if use_nonuniform:
        gen_eval_nu = torch.Generator(device=device)
        gen_eval_nu.manual_seed(training.seed + 2_000_003)
        eval_nonuniform_ds = make_reuse_star_dataset(n_eval, rho, data_dist, device, dtype, gen_eval_nu)
        eval_nonuniform_path = output_dir / "eval_nonuniform.pt"
        torch.save(
            {
                "x": eval_nonuniform_ds.x.cpu(), "y": eval_nonuniform_ds.y.cpu(),
                "rho": rho, "kind": f"reuse_star_{data_dist}",
                "num_reuse_targets": m,
                "target_supports": REUSE_STAR_TARGET_SUPPORTS,
                "shared_support": REUSE_STAR_SHARED_SUPPORT,
                "private_supports": REUSE_STAR_PRIVATE_SUPPORTS,
            },
            eval_nonuniform_path,
        )

    # --- Test dataset (standard iid sample for training evaluation) ------------
    gen_test = torch.Generator(device=device)
    gen_test.manual_seed(training.seed + 3_000_007)
    test_ds = make_reuse_star_dataset(training.test_samples, rho if use_nonuniform else 0.0,
                                      data_dist if use_nonuniform else "uniform", device, dtype, gen_test)
    test_data_path = output_dir / "test_data.pt"
    save_dataset(test_ds, test_data_path)

    # --- Fixed training pool (when train_samples is set) ----------------------
    train_data: ParityDataset | None = None
    train_batch_size = training.batch_size
    if training.train_samples is not None:
        gen_train = torch.Generator(device=device)
        gen_train.manual_seed(training.seed)
        train_x, train_y = sample_reuse_star_inputs(
            training.train_samples, rho if use_nonuniform else 0.0,
            data_dist if use_nonuniform else "uniform", device, dtype, gen_train,
        )
        train_data = ParityDataset(x=train_x, y=train_y)
        save_dataset(train_data, output_dir / "train_data.pt")
        if train_data.x.shape[0] < train_batch_size:
            train_batch_size = train_data.x.shape[0]

    # --- Model -----------------------------------------------------------------
    model = build_model(
        model_config,
        output_dim=3,
        target_names_=REUSE_STAR_TARGET_NAMES,
    ).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, training.optimizer)

    barrier_c = training.barrier_c
    if barrier_c is None:
        barrier_c = 7.0 / model_config.N

    def _eval_reuse(ds: ParityDataset) -> dict[str, float]:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, ds.x.shape[0], training.batch_size):
                preds.append(model(ds.x[start:start + training.batch_size]))
        pred = torch.cat(preds, dim=0)  # (n, 3)
        y = ds.y
        out: dict[str, float] = {}
        for j, name in enumerate(REUSE_STAR_TARGET_NAMES):
            out[f"{name}/loss"] = F.mse_loss(pred[:, j], y[:, j]).item()
            out[f"{name}/acc"]  = ((pred[:, j] * y[:, j]) > 0).float().mean().item()
        # Active mean
        active_losses = torch.tensor([out[f"{k}/loss"] for k in REUSE_STAR_TARGET_NAMES[:m]])
        active_accs   = torch.tensor([out[f"{k}/acc"]  for k in REUSE_STAR_TARGET_NAMES[:m]])
        out["active/loss"] = active_losses.mean().item()
        out["active/acc"]  = active_accs.mean().item()
        return out

    history = []
    epoch_order = torch.empty(0, device=device, dtype=torch.long)
    epoch_cursor = 0
    start_time = time.perf_counter()
    progress = tqdm(
        range(1, training.num_steps + 1),
        total=training.num_steps, desc="training", unit="step", dynamic_ncols=True,
    )
    for step in progress:
        model.train()
        if train_data is None:
            x_batch, y_batch = sample_reuse_star_inputs(
                training.batch_size,
                rho if use_nonuniform else 0.0,
                data_dist if use_nonuniform else "uniform",
                device, dtype,
            )
        else:
            if epoch_cursor + train_batch_size > epoch_order.numel():
                epoch_order = torch.randperm(train_data.x.shape[0], device=device)
                epoch_cursor = 0
            idx = epoch_order[epoch_cursor:epoch_cursor + train_batch_size]
            epoch_cursor += train_batch_size
            x_batch = train_data.x[idx]
            y_batch = train_data.y[idx]

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)  # (batch, 3)
        barrier = torch.zeros((), device=device, dtype=dtype)
        if model_config.use_readout_barrier:
            barrier = model.readout_barrier(barrier_c, training.barrier_lambda)
        # Mean MSE over active targets only
        per_target_mse = F.mse_loss(pred, y_batch, reduction="none").mean(dim=0)  # (3,)
        mse = (per_target_mse * active_mask).sum() / active_mask.sum()
        loss = mse + barrier
        loss.backward()
        optimizer.step()

        progress.set_postfix(train_mse=f"{mse.item():.4g}", barrier=f"{barrier.item():.4g}")

        should_validate   = training.validate_every   and step % training.validate_every   == 0
        should_checkpoint = training.checkpoint_every and step % training.checkpoint_every == 0

        if training.progress_every and step % training.progress_every == 0 and not should_validate:
            tqdm.write(
                f"step {step}: train_mse={mse.item():.4g} loss={loss.item():.4g} "
                f"elapsed={time.perf_counter() - start_time:.1f}s"
            )

        metrics = None
        if should_validate or should_checkpoint:
            m_test = _eval_reuse(test_ds)
            m_uniform = _eval_reuse(eval_uniform_ds)
            m_nonuniform = _eval_reuse(eval_nonuniform_ds) if eval_nonuniform_ds is not None else {}

            row = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - start_time,
                "train_mse": mse.item(),
                "barrier": barrier.item(),
                "loss": loss.item(),
                "num_reuse_targets": m,
                "rho": rho,
                **{f"test_{k}": v for k, v in m_test.items()},
                **{f"eval_uniform_{k}": v for k, v in m_uniform.items()},
                **{f"eval_nonuniform_{k}": v for k, v in m_nonuniform.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
            tqdm.write(str(row))
            metrics = {f"test_{k}": v for k, v in m_test.items()}

            if writer is not None:
                writer.add_scalar("train/active", mse.item(), step)
                for name in REUSE_STAR_TARGET_NAMES:
                    writer.add_scalar(f"eval_uniform/{name}/loss", m_uniform[f"{name}/loss"], step)
                    writer.add_scalar(f"eval_uniform/{name}/acc",  m_uniform[f"{name}/acc"],  step)
                writer.add_scalar("eval_uniform/active/loss", m_uniform["active/loss"], step)
                writer.add_scalar("eval_uniform/active/acc",  m_uniform["active/acc"],  step)
                if eval_nonuniform_ds is not None:
                    for name in REUSE_STAR_TARGET_NAMES:
                        writer.add_scalar(f"eval_nonuniform/{name}/loss", m_nonuniform[f"{name}/loss"], step)
                        writer.add_scalar(f"eval_nonuniform/{name}/acc",  m_nonuniform[f"{name}/acc"],  step)
                    writer.add_scalar("eval_nonuniform/active/loss", m_nonuniform["active/loss"], step)
                    writer.add_scalar("eval_nonuniform/active/acc",  m_nonuniform["active/acc"],  step)
                    writer.add_scalar("eval_gap/active/loss",
                                      m_uniform["active/loss"] - m_nonuniform["active/loss"], step)
                    writer.add_scalar("eval_gap/active/acc",
                                      m_nonuniform["active/acc"] - m_uniform["active/acc"], step)

        if should_checkpoint:
            save_checkpoint(
                ckpt_dir / f"step_{step:08d}.pt",
                model=model, optimizer=optimizer, epoch=0, step=step,
                config=config, metrics=metrics, test_data_path=test_data_path,
            )

    final_path = ckpt_dir / "final.pt"
    m_final = _eval_reuse(test_ds)
    m_uniform_final = _eval_reuse(eval_uniform_ds)
    m_nonuniform_final = _eval_reuse(eval_nonuniform_ds) if eval_nonuniform_ds is not None else {}
    final_row = {
        "step": training.num_steps,
        "elapsed_seconds": time.perf_counter() - start_time,
        "num_reuse_targets": m, "rho": rho,
        **{f"test_{k}": v for k, v in m_final.items()},
        **{f"eval_uniform_{k}": v for k, v in m_uniform_final.items()},
        **{f"eval_nonuniform_{k}": v for k, v in m_nonuniform_final.items()},
    }
    history.append(final_row)
    pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
    save_checkpoint(
        final_path, model=model, optimizer=optimizer, epoch=0,
        step=training.num_steps, config=config,
        metrics={f"test_{k}": v for k, v in m_final.items()},
        test_data_path=test_data_path,
    )
    if writer is not None:
        writer.close()
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
