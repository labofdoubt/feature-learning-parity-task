"""Forward-pass Walsh composition analysis for the d16 target.

For each run in a group (e.g. all seeds of wd3_k0 and wd3_k4), loads the final
checkpoint and computes H_l(T) = ||E_x[h_l(x)*chi_T(x)]||^2 at every residual-
stream layer l for:
  - A, B       : the canonical degree-8 children of S
  - S          : the degree-16 target
  - C_k, D_k   : n_alt_partitions alternative 8+8 splits of S

The key question is: does h_{l*} (input to the block that computes S) always
represent A and B, or does it sometimes represent other degree-8 pairs?

Usage:
  python -m analysis.forward_composition.run_analysis \\
    --runs wd3_k0_N2048 wd3_k0_s1_N2048 ... wd3_k4_s5_N2048 \\
    --runs-root /root/parity/runs \\
    --output-dir /root/parity/analysis_output/forward_composition \\
    [--n-alt-partitions 20] [--batch-size 2048]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from parity_net.data import load_dataset, tree_parity_specs
from parity_net.train import resolve_device

from analysis.backward_walsh.checkpoint_utils import (
    list_checkpoints,
    load_model_at_checkpoint,
)
from analysis.backward_walsh.modes import (
    ModeInfo,
    compute_chi,
    generate_alternative_partitions,
)


# ---------------------------------------------------------------------------
# core computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_h_walsh_power(
    model,
    x: torch.Tensor,
    layer_idx: int,
    indices: tuple[int, ...],
    batch_size: int,
) -> float:
    """H_l(T) = ||E_x[h_l(x) * chi_T(x)]||^2, float64 accumulation."""
    N = model.config.N
    acc = torch.zeros(N, dtype=torch.float64, device=x.device)
    n = x.shape[0]
    model.eval()
    for start in range(0, n, batch_size):
        xb = x[start:min(start + batch_size, n)]
        with torch.no_grad():
            _, acts = model(xb, return_activations=True)
        h = acts[layer_idx].detach()
        chi = compute_chi(xb, indices).double()
        acc.add_((chi.unsqueeze(1) * h.double()).sum(0))
    return (acc / n).square().sum().item()


def _last_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    """Return (step, path) for the final checkpoint: prefers final.pt, falls back to last step_N.pt."""
    ckpt_dir = run_dir / "checkpoints"
    final_pt = ckpt_dir / "final.pt"
    if final_pt.exists():
        payload = torch.load(final_pt, map_location="cpu", weights_only=False)
        step = payload.get("step", -1)
        return step, final_pt
    ckpts = list_checkpoints(ckpt_dir)
    if not ckpts:
        return None
    return ckpts[-1]


def _detect_modes(run_dir: Path):
    """Return (S, A, B) tuples from config.yaml."""
    import yaml
    with (run_dir / "config.yaml").open() as f:
        cfg_raw = yaml.safe_load(f)
    task = cfg_raw.get("task", cfg_raw.get("model", {}))
    relevant_dim = int(task.get("relevant_dim", 16))
    exclude = task.get("exclude_targets", [])
    specs = tree_parity_specs(relevant_dim, exclude)
    d16_specs = [s for s in specs if s.degree == 16]
    d8_specs  = [s for s in specs if s.degree == 8]
    S = d16_specs[0].indices
    A = d8_specs[0].indices
    B = d8_specs[1].indices
    return S, A, B


def analyze_run(
    run_name: str,
    run_dir: Path,
    ckpt_path: Path,
    step: int,
    device: torch.device,
    n_alt_partitions: int,
    batch_size: int,
) -> pd.DataFrame:
    model, payload, dtype = load_model_at_checkpoint(ckpt_path, device)
    L = len(model.blocks)

    # Load dataset in float32 (same as backward_walsh convention)
    x = load_dataset(run_dir / "test_data.pt", device, torch.float32).x
    x = x.to(dtype=dtype)

    S, A, B = _detect_modes(run_dir)

    # Build mode list: A, B, S + alternatives
    alt_pairs = generate_alternative_partitions(S, A, B, n_alt_partitions)
    modes: list[ModeInfo] = [
        ModeInfo(A, "A", "child_a"),
        ModeInfo(B, "B", "child_b"),
        ModeInfo(S, "S", "target"),
    ]
    for k, (C, D) in enumerate(alt_pairs):
        modes.append(ModeInfo(C, f"C{k}", "ctrl_d8_partition", ctrl_index=k))
        modes.append(ModeInfo(D, f"D{k}", "ctrl_d8_partition", ctrl_index=k))

    records = []
    for l in range(L + 1):
        for mode in modes:
            H = compute_h_walsh_power(model, x, l, mode.indices, batch_size)
            records.append(dict(
                run=run_name,
                step=step,
                layer=l,
                mode_label=mode.label,
                role=mode.role,
                value=H,
            ))
        print(f"  layer {l}: H(A)={records[-len(modes)+0]['value']:.3e}  "
              f"H(B)={records[-len(modes)+1]['value']:.3e}  "
              f"H(S)={records[-len(modes)+2]['value']:.3e}")

    del model
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_composition_comparison(df: pd.DataFrame, out_path: str) -> None:
    """Bar chart: for each run, H_{l*}(T) for A, B, and alternative partitions."""
    runs = df["run"].unique()

    # For each run, find l* = layer with largest H(S) delta
    L_max = df["layer"].max()
    results = []
    for run in runs:
        sub = df[df["run"] == run]
        # l* = layer where H(S) is largest (output of star block)
        s_rows = sub[(sub["mode_label"] == "S")].groupby("layer")["value"].mean()
        l_star_out = int(s_rows.idxmax())
        l_star_in  = max(l_star_out - 1, 0)

        H_A = sub[(sub["layer"] == l_star_in) & (sub["mode_label"] == "A")]["value"].values[0]
        H_B = sub[(sub["layer"] == l_star_in) & (sub["mode_label"] == "B")]["value"].values[0]
        ctrl = sub[(sub["layer"] == l_star_in) & (sub["role"] == "ctrl_d8_partition")]["value"]
        H_ctrl_med = ctrl.median()
        H_ctrl_max = ctrl.max()
        H_S = sub[(sub["layer"] == l_star_out) & (sub["mode_label"] == "S")]["value"].values[0]
        results.append(dict(run=run, H_A=H_A, H_B=H_B,
                            H_ctrl_med=H_ctrl_med, H_ctrl_max=H_ctrl_max, H_S=H_S,
                            l_star_in=l_star_in))

    res_df = pd.DataFrame(results)
    n = len(res_df)
    x = np.arange(n)
    w = 0.2

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    ax.bar(x - 1.5*w, res_df["H_A"],        w, label="H(A) at l*")
    ax.bar(x - 0.5*w, res_df["H_B"],        w, label="H(B) at l*")
    ax.bar(x + 0.5*w, res_df["H_ctrl_med"], w, label="H(ctrl) median")
    ax.bar(x + 1.5*w, res_df["H_ctrl_max"], w, label="H(ctrl) max")
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["run"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("H (Walsh power at layer l*)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("d16 composition: A,B vs. alternative degree-8 pairs at input to star block")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary table
    print("\n=== Composition summary ===")
    print(res_df[["run", "l_star_in", "H_A", "H_B", "H_ctrl_med", "H_ctrl_max", "H_S"]].to_string(index=False))


def plot_h_by_layer(df: pd.DataFrame, out_path: str) -> None:
    """One subplot per run: H_l(A), H_l(B), H_l(S), ctrl envelope vs. layer."""
    runs = df["run"].unique()
    ncols = min(3, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["run"] == run]
        layers = sorted(sub["layer"].unique())
        H_A = [sub[(sub["layer"]==l)&(sub["mode_label"]=="A")]["value"].values[0] for l in layers]
        H_B = [sub[(sub["layer"]==l)&(sub["mode_label"]=="B")]["value"].values[0] for l in layers]
        H_S = [sub[(sub["layer"]==l)&(sub["mode_label"]=="S")]["value"].values[0] for l in layers]
        ctrl = sub[sub["role"] == "ctrl_d8_partition"]
        ctrl_med = [ctrl[ctrl["layer"]==l]["value"].median() for l in layers]
        ctrl_max = [ctrl[ctrl["layer"]==l]["value"].max() for l in layers]

        ax.semilogy(layers, np.maximum(H_A, 1e-12), "b-o", label="H(A)", markersize=4)
        ax.semilogy(layers, np.maximum(H_B, 1e-12), "r-o", label="H(B)", markersize=4)
        ax.semilogy(layers, np.maximum(H_S, 1e-12), "g-s", label="H(S)", markersize=4)
        ax.semilogy(layers, np.maximum(ctrl_med, 1e-12), "k--", label="ctrl med", linewidth=0.8)
        ax.semilogy(layers, np.maximum(ctrl_max, 1e-12), "k:", label="ctrl max", linewidth=0.8)
        ax.set_title(run, fontsize=8)
        ax.set_xlabel("layer")
        ax.set_ylabel("H")
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(len(runs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("H_l(T) by layer for each run", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="Run names to analyze")
    parser.add_argument("--runs-root", default="/root/parity/runs")
    parser.add_argument("--output-dir", default="/root/parity/analysis_output/forward_composition")
    parser.add_argument("--n-alt-partitions", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for run_name in args.runs:
        run_dir = Path(args.runs_root) / run_name
        last = _last_checkpoint(run_dir)
        if last is None:
            print(f"[SKIP] {run_name}: no checkpoints found")
            continue
        step, ckpt_path = last
        print(f"\n=== {run_name}  (step {step}) ===")
        df = analyze_run(run_name, run_dir, ckpt_path, step, device,
                         args.n_alt_partitions, args.batch_size)
        all_dfs.append(df)

    if not all_dfs:
        print("No runs to analyze.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    csv_path = out_dir / "results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    plot_composition_comparison(df_all, str(out_dir / "composition_comparison.pdf"))
    plot_h_by_layer(df_all, str(out_dir / "h_by_layer.pdf"))

    # Upload to Drive
    remote = "gdrive:parity/analysis_output/forward_composition"
    print(f"\nUploading to {remote} ...")
    r = subprocess.run(["rclone", "copy", str(out_dir), remote, "--progress"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("Upload complete.")
    else:
        print(f"[WARN] rclone failed:\n{r.stderr}")


if __name__ == "__main__":
    main()
