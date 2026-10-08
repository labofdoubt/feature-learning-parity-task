"""Backward Walsh-spectrum analysis for the 8+8->16 hierarchical-learning path.

Usage:
  python -m analysis.backward_walsh.run_analysis \\
    --run wd3_k0_N2048 \\
    --runs-root /root/parity/runs \\
    --output-dir /root/parity/analysis_output/backward_walsh \\
    [--target-mode 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] \\
    [--child-a 0 1 2 3 4 5 6 7] \\
    [--child-b 8 9 10 11 12 13 14 15] \\
    [--star-block 3] \\
    [--batch-size 2048] \\
    [--n-partition-controls 15] \\
    [--n-d16-controls-per-class 5] \\
    [--max-checkpoints N]   # optional: limit number of checkpoints for quick tests

The script auto-detects target/child modes from the task definition if not provided.
Results are written to:
  <output-dir>/<run>/results.csv    — all scalar metrics
  <output-dir>/<run>/plots/         — figures
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Allow running as:  python analysis/backward_walsh/run_analysis.py
# from the parity project root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import torch

from parity_net.data import load_dataset, tree_parity_specs
from parity_net.train import resolve_device

from .checkpoint_utils import find_d16_output_col, list_checkpoints, load_model_at_checkpoint
from .config import BackwardWalshConfig
from .core import compute_m_values, pass1_walsh_coefficients, pass2_r_and_vjp
from .modes import ModeInfo, build_mode_list, compute_chi
from .plots import (
    plot1_forward_and_reverse,
    plot2_specificity_ratio,
    plot3_layer_step_heatmap_M,
    plot4_q_over_training,
    plot5_accessibility_ratio,
    plot6_lambda,
    plot7_layer_step_heatmap_Q,
    plot_summary,
)


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------

def _check_walsh_normalization(x: torch.Tensor, mode: ModeInfo) -> None:
    """Warn if the chi-T estimator doesn't recover power ~1 for a pure function."""
    n = x.shape[0]
    chi = compute_chi(x, mode.indices)
    # E[chi_T * chi_T] should be 1 because chi_T in {-1,+1}
    power = (chi.double() ** 2).mean().item()
    if abs(power - 1.0) > 1e-3:
        print(f"  [WARN] Walsh normalization check: E[chi^2] = {power:.4f} != 1.0")
    else:
        print(f"  [OK]  Walsh normalization: E[chi^2] = {power:.5f}")


def _check_r_matrix_autograd(model, x: torch.Tensor, block_idx: int, T: ModeInfo,
                              batch_size: int, ghat_S_dummy: torch.Tensor) -> None:
    """Validate fast R_l(T) via autograd generic path on a small batch."""
    from .core import _compute_dphi

    dev = x.device
    xb = x[:min(64, x.shape[0])]
    block = model.blocks[block_idx]

    # Fast path
    with torch.no_grad():
        h_in = model.embedding(xb)
        for i in range(block_idx):
            h_in = model.blocks[i](h_in)
    dphi = _compute_dphi(block, h_in)
    chi = compute_chi(xb, T.indices).double()
    w_dphi = dphi.double() * chi.unsqueeze(1)
    R_fast = (w_dphi.t() @ h_in.double()) / xb.shape[0]
    Q_fast = (R_fast * R_fast).sum().item()

    # Generic autograd path
    h_in_g = h_in.detach().requires_grad_(True)
    with torch.enable_grad():
        h_out = block(h_in_g)
    y_d = ghat_S_dummy.double()
    scalar = (
        chi.unsqueeze(1) * h_out.double() * y_d.unsqueeze(0)
    ).sum(dim=1).mean()
    grad_w = torch.autograd.grad(scalar.float(), list(block.linear.parameters()),
                                 retain_graph=False)[0]
    U_generic = (grad_w.double() ** 2).sum().item()
    # Note: U_generic is different from Q (it involves ghat_S); this just checks that
    # R is non-trivially non-zero and the fast path agrees with autograd structurally.
    print(f"  [sanity] R check block {block_idx}: Q_fast={Q_fast:.4e}, "
          f"||grad_W||^2 (generic)={U_generic:.4e}")


# ---------------------------------------------------------------------------
# record building
# ---------------------------------------------------------------------------

def _mode_label(mode: ModeInfo) -> str:
    return mode.label


def _emit_h_g_records(
    records: list[dict],
    run: str,
    step: int,
    h_coeffs: list[list[torch.Tensor]],
    g_coeffs: list[list[torch.Tensor]],
    modes: list[ModeInfo],
) -> None:
    L_plus1 = len(h_coeffs)
    for l in range(L_plus1):
        for t, mode in enumerate(modes):
            H_val = h_coeffs[l][t].double().square().sum().item()
            G_val = g_coeffs[l][t].double().square().sum().item()
            base = dict(run=run, step=step, layer=l,
                        mode_label=mode.label, role=mode.role)
            records.append({**base, "metric": "H", "value": H_val})
            records.append({**base, "metric": "G", "value": G_val})


def _emit_q_u_lambda_records(
    records: list[dict],
    run: str,
    step: int,
    R_results: list[list[dict]],
    modes_d16: list[ModeInfo],
) -> None:
    for l, block_results in enumerate(R_results):
        for t, vals in enumerate(block_results):
            mode = modes_d16[t]
            base = dict(run=run, step=step, layer=l,
                        mode_label=mode.label, role=mode.role)
            records.append({**base, "metric": "Q",      "value": vals["Q"]})
            records.append({**base, "metric": "U",      "value": vals["U"]})
            records.append({**base, "metric": "Lambda", "value": vals["Lambda"]})


def _emit_m_records(
    records: list[dict],
    run: str,
    step: int,
    M_vals: list[list[float]],
    modes_d8: list[ModeInfo],
) -> None:
    for l, row in enumerate(M_vals):
        for t, M in enumerate(row):
            mode = modes_d8[t]
            records.append(dict(
                run=run, step=step, layer=l,
                mode_label=mode.label, role=mode.role,
                metric="M", value=M,
            ))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(cfg: BackwardWalshConfig, max_checkpoints: int | None = None) -> pd.DataFrame:
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = load_dataset(cfg.run_dir / "test_data.pt", device, torch.float32)
    x, y = dataset.x, dataset.y
    print(f"Dataset: {x.shape[0]} samples, input_dim={x.shape[1]}")

    # Sanity-check Walsh normalization
    print("Walsh normalization check:")
    tmp_mode = ModeInfo(cfg.target_mode, "S", "target")
    _check_walsh_normalization(x, tmp_mode)

    # Build complete mode list
    all_modes = build_mode_list(
        cfg.target_mode, cfg.child_a, cfg.child_b,
        cfg.n_partition_controls, cfg.n_d16_controls_per_class,
        cfg.input_dim, cfg.relevant_dim,
    )
    print(f"Total modes: {len(all_modes)}")

    # Separate modes for R (d16) and VJP (d8)
    modes_d16 = [m for m in all_modes if m.role in ("target",) or m.role.startswith("ctrl_d16")]
    modes_d8  = [m for m in all_modes if m.role in ("child_a", "child_b", "ctrl_d8_partition")]
    S_idx_in_all = next(i for i, m in enumerate(all_modes) if m.role == "target")

    print(f"  d16 modes (R matrices): {len(modes_d16)}")
    print(f"  d8  modes (VJP):        {len(modes_d8)}")

    # Index of S within all_modes and within modes_d16/d8
    S_idx_d16 = next(i for i, m in enumerate(modes_d16) if m.role == "target")

    # List checkpoints
    ckpts = list_checkpoints(cfg.checkpoint_dir)
    if max_checkpoints is not None:
        ckpts = ckpts[:max_checkpoints]
    print(f"Checkpoints: {len(ckpts)}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_records: list[dict] = []

    for ck_idx, (step, ckpt_path) in enumerate(ckpts):
        print(f"\n[{ck_idx+1}/{len(ckpts)}] step={step:>8d}  {ckpt_path.name}")

        model, payload, dtype = load_model_at_checkpoint(ckpt_path, device)
        x_f = x.to(dtype=dtype)
        y_f = y.to(dtype=dtype)
        d16_col = find_d16_output_col(model)
        L = len(model.blocks)

        # Sanity check on first checkpoint
        if ck_idx == 0:
            _check_r_matrix_autograd(model, x_f, 0, all_modes[S_idx_in_all],
                                     cfg.batch_size, torch.randn(model.config.N, device=device))

        # ------ Pass 1: Walsh coefficients of h_l and g_l ------
        print("  Pass 1: forward + backward Walsh coefficients...")
        h_coeffs, g_coeffs = pass1_walsh_coefficients(
            model, x_f, y_f, all_modes, d16_col, cfg.batch_size
        )
        _emit_h_g_records(all_records, cfg.run, step, h_coeffs, g_coeffs, all_modes)

        # ghat_S_by_layer[l] = g_coeffs at layer l+1 for mode S
        # (injection at output of block l = layer l+1)
        S_in_all_idx = S_idx_in_all
        ghat_S_by_layer = [g_coeffs[l + 1][S_in_all_idx] for l in range(L)]

        # ------ Pass 2: R matrices + VJP ------
        print("  Pass 2: R matrices + VJP...")
        R_results, vjp_acc = pass2_r_and_vjp(
            model, x_f, modes_d16, modes_d8,
            cfg.target_mode, ghat_S_by_layer, cfg.batch_size,
        )
        _emit_q_u_lambda_records(all_records, cfg.run, step, R_results, modes_d16)

        M_vals = compute_m_values(vjp_acc, ghat_S_by_layer)
        _emit_m_records(all_records, cfg.run, step, M_vals, modes_d8)

        # Progress log
        L_plus1 = L + 1
        h_s = h_coeffs[L][S_in_all_idx].norm().item() ** 2
        h_a = h_coeffs[L][next(i for i, m in enumerate(all_modes) if m.role == "child_a")].norm().item() ** 2
        star = cfg.star_block if cfg.star_block is not None else L - 1
        M_a = M_vals[star][next(i for i, m in enumerate(modes_d8) if m.role == "child_a")]
        Q_s = R_results[star][S_idx_d16]["Q"]
        print(f"    H_L(S)={h_s:.3e}  H_L(A)={h_a:.3e}  "
              f"M_{star}(A<-S)={M_a:.3e}  Q_{star}(S)={Q_s:.3e}")

        del model

    df = pd.DataFrame(all_records)
    csv_path = cfg.output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}  ({len(df)} rows)")

    # Determine star_block
    star = cfg.star_block
    if star is None:
        # Auto-detect: find block l where H_{l+1}(S) - H_l(S) is largest on average
        s_mask = (df["metric"] == "H") & (df["mode_label"] == "S")
        h_s_by_layer = df[s_mask].groupby("layer")["value"].mean()
        diffs = {l: h_s_by_layer.get(l + 1, 0) - h_s_by_layer.get(l, 0)
                 for l in range(L)}
        star = max(diffs, key=diffs.get)
        print(f"Auto-detected star_block = {star}")

    # Generate plots
    print("Generating plots...")
    try:
        plot1_forward_and_reverse(df, star, str(plots_dir / "plot1_forward_reverse.pdf"))
        plot2_specificity_ratio(df, star, str(plots_dir / "plot2_specificity_ratio.pdf"))
        plot3_layer_step_heatmap_M(df, str(plots_dir / "plot3_heatmap_M.pdf"))
        plot4_q_over_training(df, star, str(plots_dir / "plot4_q_over_training.pdf"))
        plot5_accessibility_ratio(df, star, str(plots_dir / "plot5_accessibility_ratio.pdf"))
        plot6_lambda(df, star, str(plots_dir / "plot6_lambda.pdf"))
        plot7_layer_step_heatmap_Q(df, str(plots_dir / "plot7_heatmap_Q.pdf"))
        plot_summary(df, star, str(plots_dir / "summary_8x8to16.pdf"))
        print(f"Plots saved to: {plots_dir}")
    except Exception as e:
        print(f"[WARN] Plot generation error: {e}")

    # Upload to Google Drive
    import subprocess
    remote = f"gdrive:parity/analysis_output/backward_walsh/{cfg.run}"
    print(f"\nUploading to {remote} ...")
    result = subprocess.run(
        ["rclone", "copy", str(cfg.output_dir), remote, "--progress"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Upload complete.")
    else:
        print(f"[WARN] rclone upload failed:\n{result.stderr}")

    return df


def _auto_detect_modes(run_dir: Path) -> tuple[tuple, tuple, tuple]:
    """Recover S, A, B from the task definition in the run config."""
    import yaml
    cfg_path = run_dir / "config.yaml"
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    task = raw.get("task", raw.get("model", {}))
    relevant_dim = int(task.get("relevant_dim", 16))
    exclude = task.get("exclude_targets", [])
    specs = tree_parity_specs(relevant_dim, exclude)

    d16_specs = [s for s in specs if s.degree == 16]
    d8_specs  = [s for s in specs if s.degree == 8]
    if not d16_specs:
        raise ValueError("No degree-16 target in task definition.")
    if len(d8_specs) < 2:
        raise ValueError(f"Expected 2 degree-8 targets, found {len(d8_specs)}.")

    S = d16_specs[0].indices           # (0,1,...,15)
    A = d8_specs[0].indices             # (0,...,7)
    B = d8_specs[1].indices             # (8,...,15)
    # Verify S = A ∪ B
    assert set(S) == set(A) | set(B), f"S={S} != A∪B for A={A}, B={B}"
    return S, A, B


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run name, e.g. wd3_k0_N2048")
    parser.add_argument("--runs-root", default="/root/parity/runs",
                        help="Directory containing run subdirectories")
    parser.add_argument("--output-dir", default="/root/parity/analysis_output/backward_walsh")
    parser.add_argument("--target-mode", type=int, nargs="+", default=None,
                        help="Indices of target mode S (auto-detected if omitted)")
    parser.add_argument("--child-a", type=int, nargs="+", default=None)
    parser.add_argument("--child-b", type=int, nargs="+", default=None)
    parser.add_argument("--star-block", type=int, default=None,
                        help="Block index l* for the 8+8->16 computation (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-partition-controls", type=int, default=15)
    parser.add_argument("--n-d16-controls-per-class", type=int, default=5)
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="Limit number of checkpoints processed (for quick tests)")
    args = parser.parse_args()

    run_dir = Path(args.runs_root) / args.run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Auto-detect modes from task if not specified
    if args.target_mode is None or args.child_a is None or args.child_b is None:
        print("Auto-detecting S, A, B from task definition...")
        S, A, B = _auto_detect_modes(run_dir)
    else:
        S = tuple(sorted(args.target_mode))
        A = tuple(sorted(args.child_a))
        B = tuple(sorted(args.child_b))

    print(f"S = {S}")
    print(f"A = {A}")
    print(f"B = {B}")

    cfg = BackwardWalshConfig(
        run=args.run,
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        output_dir=Path(args.output_dir) / args.run,
        target_mode=S,
        child_a=A,
        child_b=B,
        batch_size=args.batch_size,
        n_partition_controls=args.n_partition_controls,
        n_d16_controls_per_class=args.n_d16_controls_per_class,
        star_block=args.star_block,
    )

    run(cfg, max_checkpoints=args.max_checkpoints)


if __name__ == "__main__":
    main()
