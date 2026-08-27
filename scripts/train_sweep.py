"""Width-sweep training script.

Run a sweep over one or more model widths, writing each run to its own
subdirectory under --run-dir.  Mirrors the training cells of the notebooks
but works from the command line on a remote GPU server.

Example (notebook-2 config, widths 1024 and 2048):
    python scripts/train_sweep.py --run-dir runs/my_exp --widths 1024 2048

TensorBoard logs are written to <run-dir>/N_<width>/tb_logs/ automatically
when tensorboard is installed (`pip install tensorboard`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def make_config(N: int, output_dir: Path) -> dict:
    return {
        "model": {
            "input_dim": 32,
            "relevant_dim": 16,
            "N": N,
            "L": 4,
            "use_skip_connections": True,
            "activation": "half-tanh",
            "activation_scale": 1.0,
            "use_readout_barrier": False,
            "embedding_weight_variance": 1.0 / 32,
            "freeze_embedding": False,
            "hidden_weight_variance": 1.0 / N,
            "readout_weight_variance": 1.0 / N,
            "use_layerwise_readouts": True,
            "use_post_activation_linear": False,
            "post_activation_linear_variance": 1.0 / N,
            "bias": False,
            "use_attention": False,
        },
        "task": {
            "input_dim": 32,
            "relevant_dim": 16,
            "exclude_targets": [],
        },
        "training": {
            "curriculum": False,
            "curriculum_mse_threshold": 0.001,
            "num_steps": 500_000,
            "train_samples": 100_000,
            "test_samples": 100_000,
            "batch_size": 512,
            "seed": 0,
            "device": "cuda",
            "dtype": "float32",
            "matmul_precision": "high",
            "progress_every": 100,
            "validate_every": 1_000,
            "checkpoint_every": 10_000,
            "output_dir": str(output_dir),
            "barrier_c": None,
            "barrier_lambda": 10.0,
            "optimizer": {
                "name": "sgd",
                "lr": 1e-3,
                "lr_embedding": None,
                "lr_hidden": None,
                "lr_readout": None,
                "weight_decay": 1e-3,
                "wd_embedding": None,
                "wd_hidden": None,
                "wd_readout": None,
                "momentum": 0.9,
                "betas": [0.9, 0.999],
            },
        },
    }


def make_config_mup(N: int, output_dir: Path) -> dict:
    """muP-style per-layer lr/wd scaling relative to base width 256."""
    config = make_config(N, output_dir)
    opt = config["training"]["optimizer"]
    base_lr = opt["lr"]
    base_wd = opt["weight_decay"]
    config["model"]["readout_weight_variance"] = 1.0 / N**2

    opt["lr_embedding"] = base_lr * N / 256
    opt["lr_hidden"] = base_lr
    opt["lr_readout"] = base_lr * 256 / N

    opt["wd_embedding"] = base_wd * 256 / N
    opt["wd_hidden"] = base_wd
    opt["wd_readout"] = base_wd * N / 256
    return config


CONFIG_FACTORIES = {
    "mup": make_config_mup,
    "standard": make_config,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Root directory for all runs")
    parser.add_argument("--widths", type=int, nargs="+", default=[2048], metavar="N")
    parser.add_argument(
        "--config-preset",
        choices=list(CONFIG_FACTORIES),
        default="mup",
        help="Which config factory to use (default: mup)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if final.pt already exists",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_factory = CONFIG_FACTORIES[args.config_preset]

    # Late import so errors from missing deps surface clearly.
    from parity_net.config import load_config
    from parity_net.train import train

    for N in args.widths:
        output_dir = run_dir / f"N_{N}"
        final_pt = output_dir / "checkpoints" / "final.pt"
        if final_pt.exists() and not args.force_retrain:
            print(f"[skip] N={N}: final checkpoint exists at {final_pt}")
            continue

        config_path = output_dir / "config.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = config_factory(N, output_dir)
        with config_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"\n{'='*60}")
        print(f"Training N={N}  ({args.config_preset} config)")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        train(load_config(config_path))
        print(f"\nDone N={N}. Checkpoint: {final_pt}\n")


if __name__ == "__main__":
    main()
