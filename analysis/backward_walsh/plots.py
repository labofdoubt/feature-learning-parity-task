"""Plotting functions for the backward Walsh-spectrum analysis.

All functions read from the results DataFrame with columns:
  run, step, layer, metric, mode_label, role, value
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _steps(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df["step"].unique())


def _get(df: pd.DataFrame, metric: str, layer: int, mode_label: str) -> np.ndarray:
    mask = (df["metric"] == metric) & (df["layer"] == layer) & (df["mode_label"] == mode_label)
    sub = df[mask].sort_values("step")
    return sub["value"].values


def _layer_mode_heatmap(
    df: pd.DataFrame,
    metric: str,
    role_filter: str,
    ax,
    title: str,
    log10: bool = True,
) -> None:
    steps = _steps(df)
    sub = df[(df["metric"] == metric) & (df["role"] == role_filter)]
    layers = sorted(sub["layer"].unique())
    mat = np.full((len(layers), len(steps)), np.nan)
    for i, lay in enumerate(layers):
        for j, step in enumerate(steps):
            vals = sub[(sub["layer"] == lay) & (sub["step"] == step)]["value"].values
            if len(vals):
                mat[i, j] = np.mean(vals)
    if log10:
        with np.errstate(divide="ignore"):
            mat = np.log10(np.where(mat > 0, mat, np.nan))
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[steps[0], steps[-1], layers[0] - 0.5, layers[-1] + 0.5],
                   interpolation="nearest")
    ax.set_xlabel("step")
    ax.set_ylabel("block")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Plot 1: Forward circuit + reverse credit assignment
# ---------------------------------------------------------------------------

def plot1_forward_and_reverse(
    df: pd.DataFrame,
    star_block: int,
    out_path: str,
) -> None:
    steps = _steps(df)

    H_A = _get(df, "H", star_block,     "A")
    H_B = _get(df, "H", star_block,     "B")
    H_S = _get(df, "H", star_block + 1, "S")

    M_A = _get(df, "M", star_block, "A")
    M_B = _get(df, "M", star_block, "B")

    ctrl_mask = (
        (df["metric"] == "M")
        & (df["layer"] == star_block)
        & (df["role"] == "ctrl_d8_partition")
    )
    ctrl_df = df[ctrl_mask].sort_values("step")
    ctrl_med = ctrl_df.groupby("step")["value"].median().reindex(steps).values
    ctrl_q25 = ctrl_df.groupby("step")["value"].quantile(0.25).reindex(steps).values
    ctrl_q75 = ctrl_df.groupby("step")["value"].quantile(0.75).reindex(steps).values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.semilogy(steps, np.maximum(H_A, 1e-12), label="H(A)", color="tab:blue")
    ax1.semilogy(steps, np.maximum(H_B, 1e-12), label="H(B)", color="tab:orange")
    ax1.semilogy(steps, np.maximum(H_S, 1e-12), label="H(S) [out]", color="tab:green")
    ax1.set_ylabel("forward Walsh power")
    ax1.legend(fontsize=8)
    ax1.set_title(f"Block {star_block}: forward circuit", fontsize=9)

    ax2.semilogy(steps, np.maximum(M_A, 1e-12), label="M(A←S)", color="tab:blue")
    ax2.semilogy(steps, np.maximum(M_B, 1e-12), label="M(B←S)", color="tab:orange")
    ax2.semilogy(steps, np.maximum(ctrl_med, 1e-12), "k--", label="ctrl median")
    ax2.fill_between(steps, np.maximum(ctrl_q25, 1e-12), np.maximum(ctrl_q75, 1e-12),
                     alpha=0.2, color="gray", label="ctrl IQR")
    ax2.set_ylabel("reverse transfer M")
    ax2.set_xlabel("training step")
    ax2.legend(fontsize=8)
    ax2.set_title("Reverse credit assignment", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Specificity ratio R_route
# ---------------------------------------------------------------------------

def plot2_specificity_ratio(
    df: pd.DataFrame,
    star_block: int,
    out_path: str,
    eps: float = 1e-30,
) -> None:
    steps = _steps(df)
    M_A = _get(df, "M", star_block, "A")
    M_B = _get(df, "M", star_block, "B")

    ctrl_mask = (
        (df["metric"] == "M")
        & (df["layer"] == star_block)
        & (df["role"] == "ctrl_d8_partition")
    )
    ctrl_df = df[ctrl_mask]
    ctrl_med_sq = ctrl_df.groupby("step")["value"].apply(
        lambda v: np.median(v.values ** 2)
    ).reindex(steps).values

    R_route = (M_A ** 2 + M_B ** 2) / (2 * ctrl_med_sq + eps)
    H_S = _get(df, "H", star_block + 1, "S")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax2 = ax.twinx()
    ax.semilogy(steps, np.maximum(R_route, 1e-3), color="tab:purple", label="R_route")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8)
    ax.set_ylabel("specificity ratio R_route", color="tab:purple")
    ax2.semilogy(steps, np.maximum(H_S, 1e-12), color="tab:green", alpha=0.5, label="H(S)")
    ax2.set_ylabel("H(S) [block out]", color="tab:green")
    ax.set_xlabel("training step")
    ax.set_title("Reverse routing specificity", fontsize=9)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Layer × step heatmap of M_l(A/B/ctrl←S)
# ---------------------------------------------------------------------------

def plot3_layer_step_heatmap_M(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _layer_mode_heatmap(df[df["mode_label"] == "A"], "M", "child_a",
                        axes[0], "log10 M(A←S)")
    _layer_mode_heatmap(df[df["mode_label"] == "B"], "M", "child_b",
                        axes[1], "log10 M(B←S)")
    ctrl_agg = (
        df[(df["metric"] == "M") & (df["role"] == "ctrl_d8_partition")]
        .groupby(["step", "layer"])["value"]
        .median()
        .reset_index()
        .rename(columns={"value": "value"})
    )
    ctrl_agg["metric"] = "M"
    ctrl_agg["mode_label"] = "ctrl_median"
    ctrl_agg["role"] = "ctrl_d8_partition"
    _layer_mode_heatmap(ctrl_agg, "M", "ctrl_d8_partition",
                        axes[2], "log10 M(ctrl median←S)")
    fig.suptitle("Reverse transfer heatmap", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Tangent accessibility Q over training
# ---------------------------------------------------------------------------

def plot4_q_over_training(
    df: pd.DataFrame,
    star_block: int,
    out_path: str,
) -> None:
    steps = _steps(df)
    H_A = _get(df, "H", star_block,     "A")
    H_B = _get(df, "H", star_block,     "B")
    H_S = _get(df, "H", star_block + 1, "S")
    Q_S = _get(df, "Q", star_block,     "S")

    ctrl_mask = (
        (df["metric"] == "Q")
        & (df["layer"] == star_block)
        & df["role"].str.startswith("ctrl_d16")
    )
    ctrl_df = df[ctrl_mask]
    ctrl_med = ctrl_df.groupby("step")["value"].median().reindex(steps).values
    ctrl_q25 = ctrl_df.groupby("step")["value"].quantile(0.25).reindex(steps).values
    ctrl_q75 = ctrl_df.groupby("step")["value"].quantile(0.75).reindex(steps).values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.semilogy(steps, np.maximum(H_A, 1e-12), label="H(A)", color="tab:blue")
    ax1.semilogy(steps, np.maximum(H_B, 1e-12), label="H(B)", color="tab:orange")
    ax1.semilogy(steps, np.maximum(H_S, 1e-12), label="H(S)", color="tab:green")
    ax1.set_ylabel("forward Walsh power")
    ax1.legend(fontsize=8)

    ax2.semilogy(steps, np.maximum(Q_S, 1e-20), label="Q(S)", color="tab:green", linewidth=2)
    ax2.semilogy(steps, np.maximum(ctrl_med, 1e-20), "k--", label="ctrl d16 median")
    ax2.fill_between(steps, np.maximum(ctrl_q25, 1e-20), np.maximum(ctrl_q75, 1e-20),
                     alpha=0.2, color="gray")
    ax2.set_ylabel("Q (tangent accessibility)")
    ax2.set_xlabel("training step")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Tangent accessibility  block {star_block}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Task-specific accessibility ratio R_Q
# ---------------------------------------------------------------------------

def plot5_accessibility_ratio(
    df: pd.DataFrame,
    star_block: int,
    out_path: str,
    eps: float = 1e-30,
) -> None:
    steps = _steps(df)
    Q_S = _get(df, "Q", star_block, "S")
    ctrl_mask = (
        (df["metric"] == "Q")
        & (df["layer"] == star_block)
        & df["role"].str.startswith("ctrl_d16")
    )
    ctrl_med = df[ctrl_mask].groupby("step")["value"].median().reindex(steps).values
    R_Q = Q_S / (ctrl_med + eps)

    H_S = _get(df, "H", star_block + 1, "S")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax2 = ax.twinx()
    ax.semilogy(steps, np.maximum(R_Q, 1e-3), color="tab:purple", label="R_Q(S)")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8)
    ax.set_ylabel("accessibility ratio R_Q", color="tab:purple")
    ax2.semilogy(steps, np.maximum(H_S, 1e-12), color="tab:green", alpha=0.5, label="H(S)")
    ax2.set_ylabel("H(S)", color="tab:green")
    ax.set_xlabel("training step")
    ax.set_title("Task-specific tangent accessibility", fontsize=9)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Gradient usability Lambda
# ---------------------------------------------------------------------------

def plot6_lambda(
    df: pd.DataFrame,
    star_block: int,
    out_path: str,
) -> None:
    steps = _steps(df)
    H_A = _get(df, "H", star_block,     "A")
    H_B = _get(df, "H", star_block,     "B")
    H_S = _get(df, "H", star_block + 1, "S")
    Lambda_S = _get(df, "Lambda", star_block, "S")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax1.semilogy(steps, np.maximum(H_A, 1e-12), label="H(A)", color="tab:blue")
    ax1.semilogy(steps, np.maximum(H_B, 1e-12), label="H(B)", color="tab:orange")
    ax1.semilogy(steps, np.maximum(H_S, 1e-12), label="H(S)", color="tab:green")
    ax1.legend(fontsize=8)
    ax1.set_ylabel("forward Walsh power")

    ax2.semilogy(steps, np.maximum(Lambda_S, 1e-20), color="tab:red", label="Lambda(S)")
    ax2.set_ylabel("Lambda (gradient usability)")
    ax2.set_xlabel("training step")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Gradient usability  block {star_block}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: Layer × step heatmap of Q_l(S)
# ---------------------------------------------------------------------------

def plot7_layer_step_heatmap_Q(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _layer_mode_heatmap(
        df[df["mode_label"] == "S"],
        "Q", "target", axes[0], "log10 Q(S) per block"
    )
    _layer_mode_heatmap(
        df[df["mode_label"] == "S"],
        "H", "target", axes[1], "log10 H(S) per layer"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary figure (4 panels, shared x-axis)
# ---------------------------------------------------------------------------

def plot_summary(
    df: pd.DataFrame,
    star_block: int,
    out_path: str,
    eps: float = 1e-30,
) -> None:
    steps = _steps(df)

    H_A = _get(df, "H", star_block,     "A")
    H_B = _get(df, "H", star_block,     "B")
    H_S = _get(df, "H", star_block + 1, "S")

    M_A = _get(df, "M", star_block, "A")
    M_B = _get(df, "M", star_block, "B")
    m_ctrl = df[
        (df["metric"] == "M") & (df["layer"] == star_block) & (df["role"] == "ctrl_d8_partition")
    ]
    ctrl_M_med = m_ctrl.groupby("step")["value"].median().reindex(steps).values
    ctrl_M_q25 = m_ctrl.groupby("step")["value"].quantile(0.25).reindex(steps).values
    ctrl_M_q75 = m_ctrl.groupby("step")["value"].quantile(0.75).reindex(steps).values

    Q_S = _get(df, "Q", star_block, "S")
    q_ctrl = df[
        (df["metric"] == "Q") & (df["layer"] == star_block) & df["role"].str.startswith("ctrl_d16")
    ]
    ctrl_Q_med = q_ctrl.groupby("step")["value"].median().reindex(steps).values
    R_Q = Q_S / (ctrl_Q_med + eps)

    Lambda_S = _get(df, "Lambda", star_block, "S")

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    ax.semilogy(steps, np.maximum(H_A, 1e-12), label="H(A)", color="tab:blue")
    ax.semilogy(steps, np.maximum(H_B, 1e-12), label="H(B)", color="tab:orange")
    ax.semilogy(steps, np.maximum(H_S, 1e-12), label="H(S)", color="tab:green")
    ax.set_ylabel("Walsh power")
    ax.legend(fontsize=8)
    ax.set_title("Panel 1: forward powers", fontsize=9)

    ax = axes[1]
    ax.semilogy(steps, np.maximum(M_A, 1e-12), label="M(A←S)", color="tab:blue")
    ax.semilogy(steps, np.maximum(M_B, 1e-12), label="M(B←S)", color="tab:orange")
    ax.semilogy(steps, np.maximum(ctrl_M_med, 1e-12), "k--", label="ctrl median")
    ax.fill_between(steps, np.maximum(ctrl_M_q25, 1e-12), np.maximum(ctrl_M_q75, 1e-12),
                    alpha=0.2, color="gray")
    ax.set_ylabel("M (reverse transfer)")
    ax.legend(fontsize=8)
    ax.set_title("Panel 2: reverse credit assignment S→A,B", fontsize=9)

    ax = axes[2]
    ax.semilogy(steps, np.maximum(R_Q, 1e-3), color="tab:purple", label="R_Q(S)=Q(S)/ctrl")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8)
    ax.set_ylabel("R_Q")
    ax.legend(fontsize=8)
    ax.set_title("Panel 3: tangent accessibility R_Q(S)", fontsize=9)

    ax = axes[3]
    ax.semilogy(steps, np.maximum(Lambda_S, 1e-20), color="tab:red", label="Lambda(S)")
    ax.set_ylabel("Lambda")
    ax.set_xlabel("training step")
    ax.legend(fontsize=8)
    ax.set_title("Panel 4: gradient usability Lambda(S)", fontsize=9)

    fig.suptitle(
        f"8+8→16 backward Walsh analysis  (block {star_block})", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
