from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .data import ParityDataset, degree_slices_for_targets, make_dataset, load_dataset, target_names
from .train import max_target_degree_for_model, resolve_device, resolve_dtype


@torch.no_grad()
def predict_in_batches(
    model,
    x: torch.Tensor,
    batch_size: int,
    intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    block_intervention: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> torch.Tensor:
    preds = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        preds.append(
            model(
                x[start:stop],
                intervention=intervention,
                block_intervention=block_intervention,
            )
        )
    return torch.cat(preds, dim=0)


def per_degree_mse(
    pred: torch.Tensor,
    y: torch.Tensor,
    target_names_: list[str] | None = None,
) -> dict[str, float]:
    metrics = {"mse_all": F.mse_loss(pred, y).item()}
    if target_names_ is None:
        target_names_ = target_names()
    for degree, slc in degree_slices_for_targets(target_names_).items():
        stop = min(slc.stop or pred.shape[1], pred.shape[1], y.shape[1])
        start = min(slc.start or 0, stop)
        if stop <= start:
            continue
        metrics[f"mse_d{degree}"] = F.mse_loss(pred[:, start:stop], y[:, start:stop]).item()
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


def resolve_test_data_path(
    checkpoint: str | Path,
    training: dict[str, object],
    test_data_path_hint: str | None = None,
) -> Path | None:
    candidates = []
    if test_data_path_hint:
        candidates.append(Path(test_data_path_hint))
    output_dir = training.get("output_dir")
    if output_dir:
        candidates.append(Path(str(output_dir)) / "test_data.pt")
    checkpoint_path = Path(checkpoint)
    candidates.append(checkpoint_path.parent.parent / "test_data.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_or_make_heldout(
    checkpoint: str | Path,
    training: dict[str, object],
    model_config: dict[str, object],
    task_config: dict[str, object],
    max_degree: int | None,
    *,
    pca_samples: int | None,
    device: torch.device,
    dtype: torch.dtype,
    test_data_path_hint: str | None = None,
):
    test_data_path = resolve_test_data_path(checkpoint, training, test_data_path_hint)
    if test_data_path is not None:
        heldout = load_dataset(test_data_path, device, dtype)
        if pca_samples is not None and pca_samples < heldout.x.shape[0]:
            heldout = ParityDataset(
                x=heldout.x[:pca_samples],
                y=heldout.y[:pca_samples],
            )
        elif pca_samples is not None and pca_samples > heldout.x.shape[0]:
            raise ValueError(
                f"Requested pca_samples={pca_samples}, but saved test set has "
                f"{heldout.x.shape[0]} samples"
            )
        return heldout, test_data_path

    pca_samples = pca_samples or int(training["test_samples"])
    torch.manual_seed(int(training["seed"]) + 10_000)
    heldout = make_dataset(
        pca_samples,
        int(task_config["input_dim"]),
        int(task_config["relevant_dim"]),
        device,
        dtype,
        list(task_config.get("exclude_targets", [])),
        max_degree,
    )
    return heldout, None


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
    task_config = config.get("task") or {
        "input_dim": model_config["input_dim"],
        "relevant_dim": model_config["relevant_dim"],
        "exclude_targets": [],
    }
    target_names_ = target_names(
        int(task_config["relevant_dim"]),
        list(task_config.get("exclude_targets", [])),
        max_target_degree_for_model(model.config),
    )
    device = resolve_device(training["device"])
    dtype = resolve_dtype(training["dtype"])
    model = model.to(device=device, dtype=dtype)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = batch_size or training["batch_size"]
    heldout, test_data_path = load_or_make_heldout(
        checkpoint,
        training,
        model_config,
        task_config,
        pca_samples=pca_samples,
        device=device,
        dtype=dtype,
        test_data_path_hint=payload.get("test_data_path"),
        max_degree=max_target_degree_for_model(model.config),
    )

    weight_variances = model.weight_variances()
    pd.DataFrame(
        [{"layer": layer, "variance": variance} for layer, variance in weight_variances.items()]
    ).to_csv(output_dir / "weight_variances.csv", index=False)

    pred = predict_in_batches(model, heldout.x, batch_size)
    baseline_metrics = per_degree_mse(pred, heldout.y, target_names_)
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
            **per_degree_mse(pred_intervened, heldout.y, target_names_),
        }
        pd.DataFrame([intervention_metrics]).to_csv(
            output_dir / "pca_intervention_mse.csv", index=False
        )

    summary = {
        "weight_variances": weight_variances,
        "baseline_metrics": baseline_metrics,
        "pca_rank_thresholds": rank_rows,
        "intervention_metrics": intervention_metrics,
        "test_data_path": str(test_data_path) if test_data_path is not None else None,
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
