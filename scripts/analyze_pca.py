"""Per-block PCA intervention analysis.

For each residual-stream position (embedding output + each block output),
sweeps over how many PCs to keep (0 to --keep-pcs-max), measures per-degree
test MSE with the PCA projection applied, and saves a plot and CSV.

Usage:
    python scripts/analyze_pca.py --run-dir runs/my_exp/N_2048
    python scripts/analyze_pca.py --run-dir runs/my_exp/N_2048 --pca-samples 20000 --keep-pcs-max 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from parity_net.analysis import (
    collect_layer_activations,
    make_pca_intervention,
    pca_from_activations,
    per_degree_mse,
    predict_in_batches,
    rank_for_threshold,
)
from parity_net.checkpoint import load_checkpoint
from parity_net.data import load_dataset, target_names
from parity_net.train import max_target_degree_for_model, resolve_device, resolve_dtype


def run(run_dir: Path, pca_samples: int, keep_pcs_max: int, keep_pcs_step: int, batch_size: int) -> None:
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    model, payload, _ = load_checkpoint(ckpt_path, device)
    config = payload["config"]
    training = config["training"]
    task = config.get("task") or config["model"]
    dtype = resolve_dtype(training["dtype"])
    model = model.to(device=device, dtype=dtype)

    target_names_ = target_names(
        int(task["relevant_dim"]),
        list(task.get("exclude_targets", [])),
        max_target_degree_for_model(model.config),
    )

    test_data = load_dataset(run_dir / "test_data.pt", device, dtype)
    if pca_samples < test_data.x.shape[0]:
        test_data = type(test_data)(x=test_data.x[:pca_samples], y=test_data.y[:pca_samples])

    layer_acts = collect_layer_activations(model, test_data.x, batch_size)
    pcas = [pca_from_activations(acts) for acts in layer_acts]

    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    analysis_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    num_layers = len(pcas)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4), squeeze=False)

    all_rows = []
    rank_rows = []
    for layer_idx, (pca, acts) in enumerate(zip(pcas, layer_acts)):
        cum = pca["cumulative_explained_variance"]
        rank_rows.append({
            "layer_idx": layer_idx,
            "rank_90": rank_for_threshold(cum, 0.90),
            "rank_99": rank_for_threshold(cum, 0.99),
            "N": acts.shape[1],
        })

        ax = axes[0][layer_idx]
        layer_name = "embedding" if layer_idx == 0 else f"block_{layer_idx - 1}"
        degree_series: dict[str, list[float]] = {}
        keep_pcs_values = range(0, keep_pcs_max + 1, keep_pcs_step)

        for keep_pcs in keep_pcs_values:
            if keep_pcs == 0:
                # All output is mean - every degree → its marginal MSE
                pred_zero = test_data.y.mean(dim=0, keepdim=True).expand_as(test_data.y)
                mses = per_degree_mse(pred_zero, test_data.y, target_names_)
            else:
                intervention_fn = make_pca_intervention(pca, keep_pcs)
                pred = predict_in_batches(
                    model, test_data.x, batch_size,
                    intervention=(layer_idx, intervention_fn),
                )
                mses = per_degree_mse(pred, test_data.y, target_names_)
            row = {"layer_idx": layer_idx, "layer_name": layer_name, "keep_pcs": keep_pcs, **mses}
            all_rows.append(row)
            for k, v in mses.items():
                degree_series.setdefault(k, []).append(v)

        x_vals = list(keep_pcs_values)
        for label, ys in degree_series.items():
            ax.semilogy(x_vals, ys, label=label)
        ax.set_title(layer_name)
        ax.set_xlabel("PCs kept")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PCA Intervention MSE  –  {run_dir.name}")
    fig.tight_layout()
    plot_path = plots_dir / "pca_intervention_sweep.png"
    fig.savefig(plot_path, dpi=200)
    fig.savefig(plot_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    df = pd.DataFrame(all_rows)
    csv_path = analysis_dir / "pca_intervention_sweep.csv"
    df.to_csv(csv_path, index=False)

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(analysis_dir / "pca_ranks.csv", index=False)
    print(rank_df.to_string(index=False))
    print(f"CSVs saved: {analysis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pca-samples", type=int, default=20_000)
    parser.add_argument("--keep-pcs-max", type=int, default=80)
    parser.add_argument("--keep-pcs-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    run(Path(args.run_dir), args.pca_samples, args.keep_pcs_max, args.keep_pcs_step, args.batch_size)


if __name__ == "__main__":
    main()
