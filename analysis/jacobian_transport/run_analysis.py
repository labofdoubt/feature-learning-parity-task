"""Constant Walsh mode of the block Jacobian analysis.

Computes hat_J_l(empty) = E_x[J_l(x)] for every block across training checkpoints.

For a non-residual block  h_{l+1} = phi(W_l h_l):
    J_branch = diag(E_x[phi'(W_l h_l(x))]) @ W_l
    J0 = J_branch

For a residual block  h_{l+1} = h_l + phi(W_l h_l):
    J0 = I + J_branch

Quantities recorded per checkpoint/block:
  - Frobenius norm of J0 and J_branch
  - Distance from identity: ||J0 - I||_F / ||I||_F
  - SVD spectrum statistics (min, max, mean, median, std(log sv), condition, frac in [0.5,2], [0.8,1.2])
  - Task-relevant mode preservation: P(hhat) = ||J0^T hhat|| / ||hhat||
                                      C(hhat) = cos(hhat, J0^T hhat)
    for modes A, B, S at their respective layers
  - Cumulative backward transport through all blocks

Usage:
  python -m analysis.jacobian_transport.run_analysis \\
    --runs no_skip_N2048 wd3_k0_N2048 \\
    --runs-root /root/parity/runs \\
    --output-dir /root/parity/analysis_output/jacobian_transport
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from parity_net.data import load_dataset, tree_parity_specs
from parity_net.train import resolve_device

from analysis.backward_walsh.checkpoint_utils import list_checkpoints, load_model_at_checkpoint
from analysis.backward_walsh.modes import compute_chi


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def _compute_dphi(block, h_in: torch.Tensor) -> torch.Tensor:
    """Element-wise phi'(z) where z = block.linear(h_in). Returns (batch, N) float32."""
    with torch.no_grad():
        z = block.linear(h_in)
    with torch.enable_grad():
        z_g = z.detach().requires_grad_(True)
        phi_z = block.activation(z_g)
        dphi = torch.autograd.grad(phi_z.sum(), z_g, create_graph=False)[0]
    return dphi.detach()


@torch.no_grad()
def compute_mean_dphi(
    model,
    x: torch.Tensor,
    block_idx: int,
    batch_size: int,
) -> torch.Tensor:
    """Compute E_x[phi'(W_l h_l(x))] with float64 accumulation. Returns (N,) float32."""
    N = model.config.N
    acc = torch.zeros(N, dtype=torch.float64, device=x.device)
    n = x.shape[0]
    model.eval()
    for start in range(0, n, batch_size):
        xb = x[start:min(start + batch_size, n)]
        h = model.embedding(xb)
        for i, blk in enumerate(model.blocks):
            if i == block_idx:
                break
            h = blk(h)
        with torch.enable_grad():
            dphi = _compute_dphi(model.blocks[block_idx], h)
        acc.add_(dphi.double().sum(0))
    return (acc / n).float()


def compute_constant_jacobian(
    model,
    x: torch.Tensor,
    block_idx: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (J0, J_branch) in float64 on CPU.

    J_branch = diag(mean_dphi) @ W  (N×N)
    J0 = I + J_branch  (residual) or J_branch (non-residual)
    """
    block = model.blocks[block_idx]
    mean_dphi = compute_mean_dphi(model, x, block_idx, batch_size).double().cpu()
    W = block.linear.weight.detach().double().cpu()  # (N, N)
    J_branch = mean_dphi.unsqueeze(1) * W            # (N, N)

    if block.use_skip_connection:
        N = W.shape[0]
        J0 = torch.eye(N, dtype=torch.float64) + J_branch
    else:
        J0 = J_branch

    return J0, J_branch


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_jacobian(model, x: torch.Tensor, block_idx: int, batch_size: int) -> None:
    """Compare analytic J_branch against autograd Jacobian on a small batch."""
    from torch.autograd.functional import jacobian as ag_jacobian

    block = model.blocks[block_idx]
    xb = x[:8]
    h = model.embedding(xb)
    for i in range(block_idx):
        h = model.blocks[i](h)
    h_in = h.detach()  # (8, N)

    # Analytic: average of diag(dphi_i) @ W over batch
    with torch.enable_grad():
        dphi = _compute_dphi(block, h_in)  # (8, N)
    W = block.linear.weight.detach().float()
    J_analytic_avg = (dphi.mean(0).unsqueeze(1) * W).cpu()

    # Also verify E[D(x)] W == E[D(x) W]  (linearity)
    J_per_sample = (dphi.unsqueeze(2) * W.unsqueeze(0)).mean(0).cpu()  # (N, N)
    diff = (J_analytic_avg - J_per_sample).abs().max().item()
    print(f"  [validate] linearity check: max |E[D]W - E[DW]| = {diff:.2e}")

    # Autograd Jacobian on one example
    h_one = h_in[:1].clone().requires_grad_(False)

    def fn(h):
        return block.activation(block.linear(h))

    with torch.no_grad():
        J_ag = ag_jacobian(fn, h_in[:1]).squeeze(0).squeeze(1).cpu()  # (N, N)

    dphi_one = _compute_dphi(block, h_in[:1]).squeeze(0).detach()
    J_analytic_one = (dphi_one.unsqueeze(1) * W).cpu()
    diff2 = (J_analytic_one - J_ag).abs().max().item()
    print(f"  [validate] single-example: max |analytic - autograd| = {diff2:.2e}  "
          f"(should be ~1e-6 or less)")


# ---------------------------------------------------------------------------
# SVD statistics
# ---------------------------------------------------------------------------

def svd_stats(J: torch.Tensor) -> dict[str, float]:
    sv = torch.linalg.svdvals(J.float()).cpu()
    log_sv = sv.clamp(min=1e-30).log()
    N = sv.shape[0]
    return {
        "sv_min":        sv.min().item(),
        "sv_max":        sv.max().item(),
        "sv_mean":       sv.mean().item(),
        "sv_median":     sv.median().item(),
        "sv_std_log":    log_sv.std().item(),
        "sv_condition":  (sv.max() / sv.clamp(min=1e-30).min()).item(),
        "sv_frac_05_2":  ((sv >= 0.5) & (sv <= 2.0)).float().mean().item(),
        "sv_frac_08_12": ((sv >= 0.8) & (sv <= 1.2)).float().mean().item(),
        "sv_p10":        sv.kthvalue(max(1, int(0.10 * N))).values.item(),
        "sv_p90":        sv.kthvalue(min(N, int(0.90 * N))).values.item(),
    }


def identity_distance(J: torch.Tensor) -> float:
    N = J.shape[0]
    I = torch.eye(N, dtype=J.dtype)
    return ((J - I).norm() / I.norm()).item()


# ---------------------------------------------------------------------------
# Walsh hhat directions (task-relevant vectors)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_hhat(
    model,
    x: torch.Tensor,
    layer_idx: int,
    indices: tuple[int, ...],
    batch_size: int,
) -> torch.Tensor:
    """E_x[h_layer(x) * chi_T(x)], shape (N,) float64."""
    N = model.config.N
    acc = torch.zeros(N, dtype=torch.float64, device=x.device)
    n = x.shape[0]
    model.eval()
    for start in range(0, n, batch_size):
        xb = x[start:min(start + batch_size, n)]
        _, acts = model(xb, return_activations=True)
        h = acts[layer_idx].detach().double()
        chi = compute_chi(xb, indices).double()
        acc.add_((chi.unsqueeze(1) * h).sum(0))
    return acc / n


# ---------------------------------------------------------------------------
# Mode preservation
# ---------------------------------------------------------------------------

def mode_preservation(J0: torch.Tensor, hhat: torch.Tensor) -> dict[str, float]:
    """P = ||J0^T v|| / ||v||  and  C = cos(v, J0^T v)  for v = hhat."""
    v = hhat.double().cpu()
    v_norm = v.norm().item()
    if v_norm < 1e-30:
        return {"norm_ratio": 0.0, "cosine": 0.0, "v_norm": 0.0}
    Jt_v = J0.double().T @ v
    transported_norm = Jt_v.norm().item()
    cos = F.cosine_similarity(v.unsqueeze(0), Jt_v.unsqueeze(0), dim=1, eps=1e-12).item()
    return {
        "norm_ratio": transported_norm / (v_norm + 1e-30),
        "cosine":     cos,
        "v_norm":     v_norm,
    }


# ---------------------------------------------------------------------------
# Cumulative backward transport
# ---------------------------------------------------------------------------

def cumulative_transport(
    J0_list: list[torch.Tensor],
    v: torch.Tensor,
) -> dict[str, float]:
    """T = J0[0]^T @ J0[1]^T @ ... @ J0[-1]^T, report ||Tv||/||v|| and cos(v, Tv)."""
    Tv = v.double().cpu()
    for J0 in reversed(J0_list):
        Tv = J0.double().T @ Tv
    v_cpu = v.double().cpu()
    v_norm = v_cpu.norm().item()
    Tv_norm = Tv.norm().item()
    cos = F.cosine_similarity(v_cpu.unsqueeze(0), Tv.unsqueeze(0), dim=1, eps=1e-12).item()
    return {
        "cum_norm_ratio": Tv_norm / (v_norm + 1e-30),
        "cum_cosine":     cos,
    }


# ---------------------------------------------------------------------------
# Detect modes from config
# ---------------------------------------------------------------------------

def detect_modes(run_dir: Path) -> tuple[tuple, tuple, tuple]:
    import yaml
    with (run_dir / "config.yaml").open() as f:
        raw = yaml.safe_load(f)
    task = raw.get("task", raw.get("model", {}))
    relevant_dim = int(task.get("relevant_dim", 16))
    exclude = task.get("exclude_targets", [])
    specs = tree_parity_specs(relevant_dim, exclude)
    d16 = [s for s in specs if s.degree == 16][0].indices
    d8  = [s for s in specs if s.degree == 8]
    A, B = d8[0].indices, d8[1].indices
    return d16, A, B


# ---------------------------------------------------------------------------
# Per-checkpoint analysis
# ---------------------------------------------------------------------------

def analyze_checkpoint(
    model,
    x: torch.Tensor,
    step: int,
    run_name: str,
    S: tuple, A: tuple, B: tuple,
    batch_size: int,
    validated: bool,
) -> list[dict]:
    L = len(model.blocks)
    is_residual = model.blocks[0].use_skip_connection
    records = []

    # Compute hhat for task modes at every residual-stream layer
    # hhat[l] -> dict: label -> (N,) tensor float64
    hhat_by_layer: list[dict[str, torch.Tensor]] = []
    for l in range(L + 1):
        hhats = {}
        for label, indices in [("A", A), ("B", B), ("S", S)]:
            hhats[label] = compute_hhat(model, x, l, indices, batch_size).cpu()
        hhat_by_layer.append(hhats)

    # Compute per-block constant Jacobians
    J0_list: list[torch.Tensor] = []
    J_branch_list: list[torch.Tensor] = []
    for l in range(L):
        J0, J_branch = compute_constant_jacobian(model, x, l, batch_size)
        J0_list.append(J0)
        J_branch_list.append(J_branch)

    # Validate on first call
    if not validated:
        print("  [validate] Running Jacobian validation on block 0 ...")
        validate_jacobian(model, x, 0, batch_size)

    # Per-block scalars
    N = model.config.N
    I = torch.eye(N, dtype=torch.float64)

    for l in range(L):
        J0 = J0_list[l]
        Jb = J_branch_list[l]
        base = dict(run=run_name, step=step, block=l, is_residual=int(is_residual))

        # Norms
        base["J0_frob"]     = J0.norm().item()
        base["Jb_frob"]     = Jb.norm().item()

        # Identity distance
        base["dist_I_J0"]   = identity_distance(J0)
        base["dist_I_Jb"]   = identity_distance(Jb)

        # SVD stats for J0
        sv_dict = svd_stats(J0)
        base.update(sv_dict)

        # Task-relevant mode preservation: use hhat at layer l (input to block l)
        for mode_label in ["A", "B", "S"]:
            hhat = hhat_by_layer[l][mode_label]
            mp = mode_preservation(J0, hhat)
            base[f"P_{mode_label}"]    = mp["norm_ratio"]
            base[f"C_{mode_label}"]    = mp["cosine"]
            base[f"vnorm_{mode_label}"] = mp["v_norm"]

        records.append(base)

    # Cumulative backward transport through all blocks
    # Use hhat_S at layer 0 (the "bottom" of the network) as reference direction
    v_S0 = hhat_by_layer[0]["S"]
    cum = cumulative_transport(J0_list, v_S0)
    records.append(dict(
        run=run_name, step=step, block=-1, is_residual=int(is_residual),
        **{k: float("nan") for k in records[0] if k not in ("run", "step", "block", "is_residual")},
        **cum,
    ))

    return records


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _steps_and(df: pd.DataFrame, run: str, block: int, col: str):
    sub = df[(df["run"] == run) & (df["block"] == block)].sort_values("step")
    return sub["step"].values, sub[col].values


BLOCK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot1_identity_distance(df: pd.DataFrame, runs: list[str], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, title in zip(
        axes,
        ["dist_I_J0", "dist_I_Jb"],
        ["||J0 - I||_F / ||I||_F  (full block jacobian)",
         "||J_branch - I||_F / ||I||_F  (nonlinear branch only)"],
    ):
        for run in runs:
            sub = df[(df["run"] == run) & (df["block"] >= 0)]
            L = int(sub["block"].max()) + 1
            for l in range(L):
                steps, vals = _steps_and(sub, run, l, col)
                ls = "--" if "no_skip" in run else "-"
                label = f"{run} bl{l}"
                ax.plot(steps, vals, ls=ls, color=BLOCK_COLORS[l % 4],
                        label=label, marker="o", markersize=3)
        ax.set_xlabel("step")
        ax.set_ylabel("distance from I")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot2_sv_evolution(df: pd.DataFrame, runs: list[str], out_path: str) -> None:
    L = int(df[df["block"] >= 0]["block"].max()) + 1
    n_runs = len(runs)
    fig, axes = plt.subplots(L, n_runs, figsize=(5 * n_runs, 3.5 * L), squeeze=False)
    for ri, run in enumerate(runs):
        for l in range(L):
            ax = axes[l][ri]
            sub = df[(df["run"] == run) & (df["block"] == l)].sort_values("step")
            steps = sub["step"].values
            med   = sub["sv_median"].values
            p10   = sub["sv_p10"].values
            p90   = sub["sv_p90"].values
            mn    = sub["sv_min"].values
            mx    = sub["sv_max"].values
            ax.fill_between(steps, mn, mx, alpha=0.12, color="steelblue", label="min-max")
            ax.fill_between(steps, p10, p90, alpha=0.30, color="steelblue", label="p10-p90")
            ax.plot(steps, med, color="steelblue", lw=1.5, marker="o", markersize=3, label="median")
            ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.set_title(f"{run}  block {l}", fontsize=8)
            ax.set_xlabel("step", fontsize=7)
            ax.set_ylabel("singular value", fontsize=7)
            if ri == 0 and l == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Singular value spectrum of J0 vs training step", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot3_mode_preservation(df: pd.DataFrame, runs: list[str], out_path: str) -> None:
    L = int(df[df["block"] >= 0]["block"].max()) + 1
    fig, axes = plt.subplots(2, L, figsize=(4.5 * L, 8), squeeze=False)
    mode_styles = {"A": ("b", "-"), "B": ("r", "-"), "S": ("g", "--")}
    for l in range(L):
        for ri, (metric, ylabel) in enumerate([("P", "||J0^T v|| / ||v||  (norm ratio)"),
                                               ("C", "cos(v, J0^T v)")]):
            ax = axes[ri][l]
            for run in runs:
                sub = df[(df["run"] == run) & (df["block"] == l)].sort_values("step")
                steps = sub["step"].values
                for mode_label, (color, ls) in mode_styles.items():
                    col = f"{metric}_{mode_label}"
                    if col not in sub.columns:
                        continue
                    run_short = run.replace("_N2048", "")
                    lbl = f"{run_short} {mode_label}"
                    lw = 1.5 if "no_skip" in run else 1.0
                    ax.plot(steps, sub[col].values, color=color, ls=ls,
                            lw=lw, marker="o", markersize=3, label=lbl)
            ax.set_title(f"Block {l}", fontsize=9)
            ax.set_xlabel("step", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            if ri == 1:
                ax.axhline(0, color="k", lw=0.5, alpha=0.4)
                ax.axhline(1, color="k", lw=0.5, ls="--", alpha=0.4)
            if l == 0:
                ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Task-relevant mode preservation by J0^T", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot4_cumulative_transport(df: pd.DataFrame, runs: list[str], out_path: str) -> None:
    cum = df[df["block"] == -1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ylabel in zip(
        axes,
        ["cum_norm_ratio", "cum_cosine"],
        ["||T v_S0|| / ||v_S0||", "cos(v_S0, T v_S0)"],
    ):
        for run in runs:
            sub = cum[cum["run"] == run].sort_values("step")
            ls = "--" if "no_skip" in run else "-"
            ax.plot(sub["step"].values, sub[col].values,
                    ls=ls, marker="o", markersize=4,
                    label=run.replace("_N2048", ""))
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[1].axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.4)
    axes[1].axhline(0.0, color="k", lw=0.4, alpha=0.3)
    fig.suptitle("Cumulative backward transport J0^T[all blocks] applied to hhat_S0", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot5_sv_frac(df: pd.DataFrame, runs: list[str], out_path: str) -> None:
    """Fraction of singular values near 1 vs step."""
    L = int(df[df["block"] >= 0]["block"].max()) + 1
    fig, axes = plt.subplots(1, L, figsize=(4.5 * L, 4), squeeze=False)
    for l in range(L):
        ax = axes[0][l]
        for run in runs:
            sub = df[(df["run"] == run) & (df["block"] == l)].sort_values("step")
            ls = "--" if "no_skip" in run else "-"
            ax.plot(sub["step"].values, sub["sv_frac_05_2"].values,
                    ls=ls, marker="o", markersize=3,
                    label=f"{run.replace('_N2048','')} [0.5,2]")
            ax.plot(sub["step"].values, sub["sv_frac_08_12"].values,
                    ls=ls, marker="s", markersize=3, alpha=0.6,
                    label=f"{run.replace('_N2048','')} [0.8,1.2]")
        ax.set_title(f"Block {l}", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("fraction of SVs")
        ax.set_ylim(0, 1.05)
        if l == 0:
            ax.legend(fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Fraction of J0 singular values near 1", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs-root", default="/root/parity/runs")
    parser.add_argument("--output-dir",
                        default="/root/parity/analysis_output/jacobian_transport")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_root = Path(args.runs_root)
    all_records: list[dict] = []
    validated = False

    for run_name in args.runs:
        run_dir = runs_root / run_name
        ckpt_dir = run_dir / "checkpoints"

        # Load dataset once
        dataset = load_dataset(run_dir / "test_data.pt", device, torch.float32)
        x = dataset.x

        S, A, B = detect_modes(run_dir)
        print(f"\n{'='*60}")
        print(f"Run: {run_name}  S={S[:4]}...  A={A[:4]}...  B={B[:4]}...")

        # Gather checkpoints: step_*.pt first, then final.pt as fallback
        step_ckpts = list_checkpoints(ckpt_dir)
        if args.max_checkpoints:
            step_ckpts = step_ckpts[:args.max_checkpoints]

        final_pt = ckpt_dir / "final.pt"
        if not step_ckpts and final_pt.exists():
            # Only final checkpoint available
            tmp_model, tmp_payload, _ = load_model_at_checkpoint(final_pt, device)
            step_ckpts = [(tmp_payload.get("step", -1), final_pt)]
            del tmp_model
        elif final_pt.exists():
            # Append final if not already the last step
            last_step = step_ckpts[-1][0] if step_ckpts else -2
            tmp_model, tmp_payload, _ = load_model_at_checkpoint(final_pt, device)
            final_step = tmp_payload.get("step", -1)
            del tmp_model
            if final_step > last_step:
                step_ckpts.append((final_step, final_pt))

        print(f"Checkpoints: {len(step_ckpts)}")

        for ck_idx, (step, ckpt_path) in enumerate(step_ckpts):
            print(f"\n[{ck_idx+1}/{len(step_ckpts)}] step={step}  {ckpt_path.name}")
            model, payload, dtype = load_model_at_checkpoint(ckpt_path, device)
            x_f = x.to(dtype=dtype)

            records = analyze_checkpoint(
                model, x_f, step, run_name, S, A, B,
                args.batch_size, validated,
            )
            validated = True
            all_records.extend(records)
            del model

    df = pd.DataFrame(all_records)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # Print final-step summary
    print("\n=== Final-step summary ===")
    final = df[df["block"] >= 0].groupby(["run", "block"]).last().reset_index()
    cols = ["run", "block", "is_residual", "dist_I_J0", "sv_median",
            "sv_frac_08_12", "P_A", "P_B", "P_S", "C_A", "C_B", "C_S"]
    print(final[[c for c in cols if c in final.columns]].to_string(index=False))

    # Plots
    runs = df["run"].unique().tolist()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        plot1_identity_distance(df, runs, str(plots_dir / "plot1_identity_distance.pdf"))
        plot2_sv_evolution(df, runs, str(plots_dir / "plot2_sv_evolution.pdf"))
        plot3_mode_preservation(df, runs, str(plots_dir / "plot3_mode_preservation.pdf"))
        plot4_cumulative_transport(df, runs, str(plots_dir / "plot4_cumulative_transport.pdf"))
        plot5_sv_frac(df, runs, str(plots_dir / "plot5_sv_frac.pdf"))
    except Exception as e:
        print(f"[WARN] plot error: {e}")

    # Upload
    remote = "gdrive:parity/analysis_output/jacobian_transport"
    print(f"\nUploading to {remote} ...")
    r = subprocess.run(["rclone", "copy", str(out_dir), remote, "--progress"],
                       capture_output=True, text=True)
    print("Upload complete." if r.returncode == 0 else f"[WARN] rclone: {r.stderr}")


if __name__ == "__main__":
    main()
