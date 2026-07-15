from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .data import DEGREE_SLICES, make_dataset
from .train import resolve_device, resolve_dtype


@torch.no_grad()
def predict_in_batches(
    model,
    x: torch.Tensor,
    batch_size: int,
    intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> torch.Tensor:
    preds = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        preds.append(model(x[start:stop], intervention=intervention))
    return torch.cat(preds, dim=0)


def per_degree_mse(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    metrics = {"mse_all": F.mse_loss(pred, y).item()}
    for degree, slc in DEGREE_SLICES.items():
        metrics[f"mse_d{degree}"] = F.mse_loss(pred[:, slc], y[:, slc]).item()
    return metrics


@torch.no_grad()
def collect_layer_activations(model, x: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    all_layers: list[list[torch.Tensor]] | None = None
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        _, activations = model(x[start:stop], return_activations=True)
        if all_layers is None:
            all_layers = [[] for _ in activations]
        for layer_idx, activation in enumerate(activations):
            all_layers[layer_idx].append(activation.detach())
    assert all_layers is not None
    return [torch.cat(chunks, dim=0) for chunks in all_layers]


def pca_from_activations(activations: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    variances = singular_values.square() / max(activations.shape[0] - 1, 1)
    explained = variances / variances.sum().clamp_min(torch.finfo(variances.dtype).eps)
    cumulative = torch.cumsum(explained, dim=0)
    return {
        "mean": mean.squeeze(0),
        "components": vh,
        "explained_variance": explained,
        "cumulative_explained_variance": cumulative,
    }


def rank_for_threshold(cumulative: torch.Tensor, threshold: float) -> int:
    return int(torch.searchsorted(cumulative, torch.tensor(threshold, device=cumulative.device)).item() + 1)


def make_pca_intervention(pca: dict[str, torch.Tensor], keep_pcs: int):
    mean = pca["mean"]
    components = pca["components"][:keep_pcs]

    def intervention(h: torch.Tensor) -> torch.Tensor:
        centered = h - mean
        coeffs = centered @ components.T
        return mean + coeffs @ components

    return intervention


def run_analysis(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    pca_samples: int | None = None,
    batch_size: int | None = None,
    intervention_layer: int | None = None,
    keep_pcs: int | None = None,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload, _ = load_checkpoint(checkpoint, device)
    config = payload["config"]
    training = config["training"]
    model_config = config["model"]
    device = resolve_device(training["device"])
    dtype = resolve_dtype(training["dtype"])
    model = model.to(device=device, dtype=dtype)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = batch_size or training["batch_size"]
    pca_samples = pca_samples or training["test_samples"]
    torch.manual_seed(training["seed"] + 10_000)
    heldout = make_dataset(
        pca_samples,
        model_config["input_dim"],
        model_config["relevant_dim"],
        device,
        dtype,
    )

    weight_variances = model.weight_variances()
    pd.DataFrame(
        [{"layer": layer, "variance": variance} for layer, variance in weight_variances.items()]
    ).to_csv(output_dir / "weight_variances.csv", index=False)

    pred = predict_in_batches(model, heldout.x, batch_size)
    baseline_metrics = per_degree_mse(pred, heldout.y)
    pd.DataFrame([baseline_metrics]).to_csv(output_dir / "baseline_mse.csv", index=False)

    activations = collect_layer_activations(model, heldout.x, batch_size)
    pcas = [pca_from_activations(layer_acts) for layer_acts in activations]
    rank_rows = []
    for layer_idx, pca in enumerate(pcas):
        cumulative = pca["cumulative_explained_variance"]
        rank_rows.append(
            {
                "layer_idx": layer_idx,
                "rank_90": rank_for_threshold(cumulative, 0.90),
                "rank_99": rank_for_threshold(cumulative, 0.99),
                "num_dimensions": cumulative.numel(),
            }
        )
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(output_dir / "pca_rank_thresholds.csv", index=False)

    intervention_metrics = None
    if intervention_layer is not None and keep_pcs is not None:
        if intervention_layer < 0 or intervention_layer >= len(pcas):
            raise ValueError(f"intervention_layer must be in [0, {len(pcas) - 1}]")
        intervention = make_pca_intervention(pcas[intervention_layer], keep_pcs)
        pred_intervened = predict_in_batches(
            model,
            heldout.x,
            batch_size,
            intervention=(intervention_layer, intervention),
        )
        intervention_metrics = {
            "intervention_layer": intervention_layer,
            "keep_pcs": keep_pcs,
            **per_degree_mse(pred_intervened, heldout.y),
        }
        pd.DataFrame([intervention_metrics]).to_csv(
            output_dir / "pca_intervention_mse.csv", index=False
        )

    summary = {
        "weight_variances": weight_variances,
        "baseline_metrics": baseline_metrics,
        "pca_rank_thresholds": rank_rows,
        "intervention_metrics": intervention_metrics,
    }
    with (output_dir / "analysis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--intervention-layer", type=int, default=None)
    parser.add_argument("--keep-pcs", type=int, default=None)
    args = parser.parse_args()
    summary = run_analysis(
        args.checkpoint,
        args.output_dir,
        pca_samples=args.pca_samples,
        batch_size=args.batch_size,
        intervention_layer=args.intervention_layer,
        keep_pcs=args.keep_pcs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
