"""Backward Walsh analysis for the 4x-d4 walsh_clusters run (4 quads -> d16).

Adapts the 8+8->16 backward analysis to the four-cluster setup where the data
distribution induces gradients on the four degree-4 tree modes
C1..C4 = {0-3},{4-7},{8-11},{12-15} and the network composes them into
S = C1*C2*C3*C4 (skipping d8, per the forward decode).

Per checkpoint, all on UNIFORM data (eval_uniform.pt, fallback test_data.pt):

  Pass 1 (H/G): forward and backward Walsh coefficients of every tracked mode
    at every residual position l = 0..L:
      H_l(T) = ||E[h_l chi_T]||^2   (forward mode power)
      G_l(T) = ||E[g_l chi_T]||^2   (gradient/cotangent power; g = dL_d16/dh_l)

  Pass 2 per block:
      Q_l(T) = ||R_l(T)||_F^2, R_l(T) = E[chi_T phi'(z_l) h_in^T]
        -> tangent accessibility of degree-16 mode T at block l
      Lambda_l(T) = usable curvature given the current S-cotangent
      M_l(T <- S): inject a pure-S cotangent at block output, pull back through
        the block, measure the Walsh coefficient of the pulled-back gradient on
        mode T -> reverse credit assignment. Here the informative targets are:
          C_j   (d4):  needs d12 features in h_in  (chi_S*chi_Cj = chi_{S\\Cj})
          A_j = S\\C_j (d12): needs d4 features in h_in — the direction that
                       should light up once the induced d4s form
          d8 tree modes and a non-tree d4 as controls.

Usage:
  python -m analysis.backward_walsh.run_4xd4 --run walshc_N2048_rho09_4xd4_train500k \\
    [--runs-root /root/parity/runs] \\
    [--output-dir /root/parity/analysis_output/backward_walsh_4xd4] \\
    [--num-samples 65536] [--batch-size 2048] [--checkpoint-stride 1] \\
    [--max-checkpoints N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from parity_net.data import load_dataset
from parity_net.train import resolve_device

from .checkpoint_utils import find_d16_output_col, list_checkpoints, load_model_at_checkpoint
from .core import compute_m_values, pass1_walsh_coefficients, pass2_r_and_vjp
from .modes import ModeInfo


# ---------------------------------------------------------------------------
# Mode set for the 4x-d4 story (zero-based indices)
# ---------------------------------------------------------------------------

QUADS = [tuple(range(4 * j, 4 * j + 4)) for j in range(4)]           # C1..C4
S_MODE = tuple(range(16))
COMPL = [tuple(sorted(set(S_MODE) - set(q))) for q in QUADS]          # A_j = S \ C_j
D8S = [tuple(range(0, 8)), tuple(range(8, 16))]
CTRL_D4 = (0, 5, 10, 15)                                              # non-tree d4


def build_modes(input_dim: int) -> tuple[list[ModeInfo], list[ModeInfo], list[ModeInfo]]:
    """Returns (all_modes for pass1, modes for R/Q (deg-16), modes for VJP)."""
    modes: list[ModeInfo] = [ModeInfo(S_MODE, "S", "target")]
    for j, q in enumerate(QUADS):
        modes.append(ModeInfo(q, f"C{j+1}", "child_d4"))
    for j, a in enumerate(COMPL):
        modes.append(ModeInfo(a, f"A{j+1}", "compl_d12"))
    for j, d8 in enumerate(D8S):
        modes.append(ModeInfo(d8, f"d8_{j}", "tree_d8"))
    modes.append(ModeInfo(CTRL_D4, "ctrl_d4", "ctrl_d4"))
    # degree-16 controls: swap 1 / 4 relevant bits for irrelevant ones
    r1 = tuple(sorted(list(range(15)) + [16]))
    r4 = tuple(sorted(list(range(12)) + [16, 17, 18, 19]))
    modes.append(ModeInfo(r1, "d16_repl1", "ctrl_d16"))
    modes.append(ModeInfo(r4, "d16_repl4", "ctrl_d16"))

    modes_rq = [m for m in modes if m.role in ("target", "ctrl_d16")]
    modes_vjp = [m for m in modes if m.role in ("child_d4", "compl_d12", "tree_d8", "ctrl_d4")]
    return modes, modes_rq, modes_vjp


@torch.no_grad()
def pass_update_walsh(model, x, modes, batch_size):
    """Walsh coefficients of each block's UPDATE u_l = phi(V h_l) — i.e. right
    after the nonlinearity, BEFORE the residual addition. Returns
    coeffs[l][t] = E[u_l chi_T] as (N,) float64 for l = 0..L-1."""
    from .modes import compute_chi
    model.eval()
    L = len(model.blocks)
    N = model.config.N
    dev = x.device
    n_total = x.shape[0]
    acc = [[torch.zeros(N, dtype=torch.float64, device=dev) for _ in modes]
           for _ in range(L)]
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        xb = x[start:stop]
        chis = [compute_chi(xb, m.indices).double() for m in modes]
        h = model.embedding(xb)
        for l, block in enumerate(model.blocks):
            u = block.activation(block.linear(h))
            if getattr(block, "post_activation_linear", None) is not None:
                u = block.post_activation_linear(u)
            u_d = u.double()
            for t in range(len(modes)):
                acc[l][t].add_((chis[t].unsqueeze(1) * u_d).sum(0))
            h = block(h)
    for l in range(L):
        for t in range(len(modes)):
            acc[l][t].div_(n_total)
    return acc


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _series(df, metric, layer, label):
    d = df[(df.metric == metric) & (df.layer == layer) & (df.mode_label == label)]
    d = d.sort_values("step")
    return d["step"].values, d["value"].values


def _series_role_mean(df, metric, layer, role):
    d = df[(df.metric == metric) & (df.layer == layer) & (df.role == role)]
    d = d.groupby("step")["value"].mean().reset_index().sort_values("step")
    return d["step"].values, d["value"].values


def plot_gradient_usability(df: pd.DataFrame, L: int, path: str) -> None:
    """Q_l(S), Lambda_l(S) and cotangent power G_{l+1}(S) per block, over training."""
    fig, axes = plt.subplots(1, L, figsize=(4.6 * L, 3.6), sharex=True)
    for l in range(L):
        ax = axes[l]
        s, q = _series(df, "Q", l, "S")
        ax.semilogy(s, q, "-o", ms=3, color="tab:blue", label="Q_l(S) tangent access.")
        s, qc = _series_role_mean(df, "Q", l, "ctrl_d16")
        ax.semilogy(s, qc, "--", color="gray", label="Q_l(d16 ctrl)")
        s, lam = _series(df, "Lambda", l, "S")
        ax.semilogy(s, lam, ":", color="tab:purple", label="Lambda_l(S)")
        s, g = _series(df, "G", l + 1, "S")
        ax.semilogy(s, g, "-", color="tab:red", alpha=0.8, label="G_{l+1}(S) cotangent")
        ax.set_title(f"block {l}")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if l == 0:
            ax.set_ylabel("power (log)")
            ax.legend(fontsize=7)
    fig.suptitle("d16 gradient usability & tangent accessibility across training")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forward_backward_alignment(df: pd.DataFrame, L: int, path: str) -> None:
    """Control-subtracted, normalized shapes: H(d4), H(S) vs Q(S), G(S) per block.

    H and Q of degree-16 modes carry a finite-sample / network-roughness floor
    that is largest early in training and decays under weight decay; without
    subtracting the matched control that floor masquerades as an early peak.
    H(S) and Q(S) subtract the ctrl_d16 mean; H(d4) subtracts the non-tree
    ctrl_d4. All clipped at 0, then each curve is normalized by its max.
    """
    fig, axes = plt.subplots(1, L, figsize=(4.6 * L, 3.6), sharex=True, sharey=True)

    def norm(v):
        m = v.max() if len(v) and v.max() > 0 else 1.0
        return v / m

    def excess(series_pair):
        v, c = series_pair
        out = v - c
        out[out < 0] = 0.0
        return out

    for l in range(L):
        ax = axes[l]
        s, h4 = _series_role_mean(df, "H", l + 1, "child_d4")
        _, h4c = _series(df, "H", l + 1, "ctrl_d4")
        ax.plot(s, norm(excess((h4, h4c))), "-", color="tab:green",
                label="H(d4 mean) − ctl, fwd")
        s, hs = _series(df, "H", l + 1, "S")
        _, hsc = _series_role_mean(df, "H", l + 1, "ctrl_d16")
        ax.plot(s, norm(excess((hs, hsc))), "-", color="tab:blue",
                label="H(S) − ctl, fwd")
        s, q = _series(df, "Q", l, "S")
        _, qc = _series_role_mean(df, "Q", l, "ctrl_d16")
        ax.plot(s, norm(excess((q, qc))), "--", color="tab:orange",
                label="Q_l(S) − ctl, bwd access.")
        s, g = _series(df, "G", l + 1, "S")
        ax.plot(s, norm(g), ":", color="tab:red", label="G(S) cotangent")
        ax.set_title(f"block {l} (H at its output)")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if l == 0:
            ax.set_ylabel("normalized to max")
            ax.legend(fontsize=7)
    fig.suptitle("Forward mode formation vs backward accessibility "
                 "(control-subtracted, each curve / its max)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_magnitudes(df: pd.DataFrame, L: int, path: str, xlog: bool = False,
                             vertical: bool = False, title_suffix: str = "",
                             subtract_ctl: bool = True) -> None:
    """Companion to the alignment plot: ABSOLUTE magnitudes, one panel per
    quantity with one curve per block, shared log scale — so magnitudes are
    comparable across blocks (the normalized plot deliberately hides this).
    H and Q are control-subtracted like in the alignment plot; G is raw."""
    import numpy as np

    def excess(v, c):
        out = v - c
        out[out < 0] = 0.0
        return out

    sfx = " − ctl" if subtract_ctl else " (raw)"
    panels = []  # (title, list of (block_label, steps, values))
    for name in (f"H(d4 mean){sfx}", f"H(S){sfx}", f"Q_l(S){sfx}", "G_{l+1}(S)"):
        curves = []
        for l in range(L):
            if name.startswith("H(d4"):
                s, v = _series_role_mean(df, "H", l + 1, "child_d4")
                if subtract_ctl:
                    _, c = _series(df, "H", l + 1, "ctrl_d4")
                    v = excess(v, c)
            elif name.startswith("H(S"):
                s, v = _series(df, "H", l + 1, "S")
                if subtract_ctl:
                    _, c = _series_role_mean(df, "H", l + 1, "ctrl_d16")
                    v = excess(v, c)
            elif name.startswith("Q"):
                s, v = _series(df, "Q", l, "S")
                if subtract_ctl:
                    _, c = _series_role_mean(df, "Q", l, "ctrl_d16")
                    v = excess(v, c)
            else:
                s, v = _series(df, "G", l + 1, "S")
            curves.append((f"block {l}", s, v))
        panels.append((name, curves))

    colors = plt.cm.viridis(np.linspace(0.0, 0.85, L))
    if vertical:
        # Stacked panels sharing the x-axis, so onsets align visually.
        fig, axes = plt.subplots(4, 1, figsize=(8.5, 13), sharex=True)
    else:
        fig, axes = plt.subplots(1, 4, figsize=(19, 3.8), sharex=True)
    for ax, (title, curves) in zip(axes, panels):
        for (label, s, v), color in zip(curves, colors):
            ax.semilogy(s, np.maximum(v, 1e-12), "-", color=color, label=label)
        if xlog:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylabel("power (log)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7)
    if vertical:
        axes[-1].set_xlabel("step")
    else:
        for ax in axes:
            ax.set_xlabel("step")
    fig.suptitle("Absolute magnitudes across blocks (H at block outputs; Q per block)"
                 + title_suffix)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reverse_credit(df: pd.DataFrame, L: int, path: str) -> None:
    """M_l(T <- S) per block: does the S-cotangent create gradient on T?"""
    groups = [
        ("compl_d12", "tab:blue", "M(A_j=d12 <- S)  [needs d4 in h]"),
        ("child_d4", "tab:green", "M(C_j=d4 <- S)   [needs d12 in h]"),
        ("tree_d8", "tab:orange", "M(d8 <- S)       [needs d8 in h]"),
        ("ctrl_d4", "gray", "M(ctrl d4 <- S)"),
    ]
    fig, axes = plt.subplots(1, L, figsize=(4.6 * L, 3.6), sharex=True, sharey=True)
    for l in range(L):
        ax = axes[l]
        for role, color, label in groups:
            s, m = _series_role_mean(df, "M", l, role)
            ax.semilogy(s, m + 1e-12, "-", color=color, label=label)
        ax.set_title(f"block {l}")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if l == 0:
            ax.set_ylabel("M (log)")
            ax.legend(fontsize=7)
    fig.suptitle("Reverse credit assignment: Walsh content of the pulled-back pure-S cotangent")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, L: int, path: str) -> None:
    """Layer x step heatmaps of H(S), H(d4 mean), Q(S), G(S)."""
    specs = [
        ("H", "S", "H(S) forward", "layerp1"),
        ("H", None, "H(d4 mean) forward", "layerp1"),
        ("Q", "S", "Q(S) accessibility", "block"),
        ("G", "S", "G(S) cotangent", "layerp1"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.4))
    steps = sorted(df["step"].unique())
    for ax, (metric, label, title, kind) in zip(axes, specs):
        n_rows = L if kind == "block" else L + 1
        img = []
        for row in range(n_rows):
            if label is None:
                _, v = _series_role_mean(df, metric, row, "child_d4")
            else:
                _, v = _series(df, metric, row, label)
            img.append(v)
        import numpy as np
        img = np.log10(np.array(img) + 1e-12)
        im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis",
                       extent=[steps[0], steps[-1], -0.5, n_rows - 0.5])
        ax.set_title(title + "  (log10)")
        ax.set_xlabel("step")
        ax.set_ylabel("block" if kind == "block" else "residual layer")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-root", default="/root/parity/runs")
    parser.add_argument("--output-dir", default="/root/parity/analysis_output/backward_walsh_4xd4")
    parser.add_argument("--num-samples", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--update-h-only", action="store_true",
                        help="compute only the UPDATE-level H (post-nonlinearity, "
                             "pre-residual) over all checkpoints and emit plot2d, "
                             "reusing Q/G from an existing results.csv")
    parser.add_argument("--plot2c-only", action="store_true",
                        help="compute only what the absolute-magnitudes (log-x) plot "
                             "needs: no VJP/M pass, no d12/d8 modes; emit only plot2c")
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.runs_root) / args.run
    out_dir = Path(args.output_dir) / args.run
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # UNIFORM data only: prefer the exhaustive uniform eval set.
    data_path = run_dir / "eval_uniform.pt"
    if not data_path.exists():
        data_path = run_dir / "test_data.pt"
    print(f"Data: {data_path.name} (uniform)")
    ds = load_dataset(data_path, device, torch.float32)
    x, y = ds.x, ds.y
    if args.num_samples < x.shape[0]:
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(42))
        idx = perm[: args.num_samples]
        x, y = x[idx], y[idx]
    print(f"Samples: {x.shape[0]}, input_dim={x.shape[1]}")

    all_modes, modes_rq, modes_vjp = build_modes(x.shape[1])
    if args.plot2c_only:
        keep = {"target", "child_d4", "ctrl_d4", "ctrl_d16"}
        all_modes = [m for m in all_modes if m.role in keep]
        modes_vjp = []
    S_idx = next(i for i, m in enumerate(all_modes) if m.role == "target")
    print(f"Modes: {len(all_modes)} total, {len(modes_rq)} R/Q, {len(modes_vjp)} VJP")

    ckpts = list_checkpoints(run_dir / "checkpoints")
    ckpts = ckpts[:: args.checkpoint_stride]
    if args.max_checkpoints:
        ckpts = ckpts[: args.max_checkpoints]
    print(f"Checkpoints: {len(ckpts)}")

    if args.update_h_only:
        keep = {"target", "child_d4", "ctrl_d4", "ctrl_d16"}
        u_modes = [m for m in all_modes if m.role in keep]
        u_records: list[dict] = []
        for k, (step, ckpt_path) in enumerate(ckpts):
            print(f"[{k+1}/{len(ckpts)}] step={step} (update-H)")
            model, payload, dtype = load_model_at_checkpoint(ckpt_path, device)
            coeffs = pass_update_walsh(model, x.to(dtype=dtype), u_modes, args.batch_size)
            for l in range(len(model.blocks)):
                for t, mode in enumerate(u_modes):
                    u_records.append(dict(run=args.run, step=step, layer=l,
                                          mode_label=mode.label, role=mode.role,
                                          metric="Hu",
                                          value=coeffs[l][t].square().sum().item()))
            del model
        dfu = pd.DataFrame(u_records)
        dfu.to_csv(out_dir / "results_update_h.csv", index=False)
        main_df = pd.read_csv(out_dir / "results.csv")
        # Map update-H of block l onto the plot's "H at layer l+1" convention,
        # keep Q/G from the stream-level sweep.
        hrows = dfu.copy()
        hrows["metric"] = "H"
        hrows["layer"] = hrows["layer"] + 1
        comb = pd.concat([hrows, main_df[main_df.metric.isin(["Q", "G"])]],
                         ignore_index=True)
        L = int(comb[comb.metric == "Q"].layer.max()) + 1
        plot_absolute_magnitudes(
            comb, L, str(plots_dir / "plot2d_update_H_logx.pdf"),
            xlog=True, vertical=True,
            title_suffix="\nH panels: block UPDATES (post-nonlinearity, pre-residual)")
        import subprocess
        remote = f"gdrive:parity/analysis_output/backward_walsh_4xd4/{args.run}"
        r = subprocess.run(["rclone", "copy", str(out_dir), remote,
                            "--drive-chunk-size", "64M", "--transfers", "2", "-q"],
                           capture_output=True, text=True)
        print("Upload ok" if r.returncode == 0 else f"Upload failed: {r.stderr[:300]}")
        return

    records: list[dict] = []
    for k, (step, ckpt_path) in enumerate(ckpts):
        print(f"[{k+1}/{len(ckpts)}] step={step}")
        model, payload, dtype = load_model_at_checkpoint(ckpt_path, device)
        x_f, y_f = x.to(dtype=dtype), y.to(dtype=dtype)
        d16_col = find_d16_output_col(model)
        L = len(model.blocks)

        h_coeffs, g_coeffs = pass1_walsh_coefficients(
            model, x_f, y_f, all_modes, d16_col, args.batch_size)
        for l in range(L + 1):
            for t, mode in enumerate(all_modes):
                base = dict(run=args.run, step=step, layer=l,
                            mode_label=mode.label, role=mode.role)
                records.append({**base, "metric": "H",
                                "value": h_coeffs[l][t].square().sum().item()})
                records.append({**base, "metric": "G",
                                "value": g_coeffs[l][t].square().sum().item()})

        ghat_S_by_layer = [g_coeffs[l + 1][S_idx] for l in range(L)]
        R_results, vjp_acc = pass2_r_and_vjp(
            model, x_f, modes_rq, modes_vjp, S_MODE, ghat_S_by_layer, args.batch_size)
        for l in range(L):
            for t, mode in enumerate(modes_rq):
                base = dict(run=args.run, step=step, layer=l,
                            mode_label=mode.label, role=mode.role)
                for key in ("Q", "U", "Lambda"):
                    records.append({**base, "metric": key, "value": R_results[l][t][key]})
        M_vals = compute_m_values(vjp_acc, ghat_S_by_layer)
        for l in range(L):
            for t, mode in enumerate(modes_vjp):
                records.append(dict(run=args.run, step=step, layer=l,
                                    mode_label=mode.label, role=mode.role,
                                    metric="M", value=M_vals[l][t]))

        hs = h_coeffs[L][S_idx].square().sum().item()
        q1 = R_results[min(1, L - 1)][0]["Q"]
        print(f"    H_L(S)={hs:.3e}  Q_1(S)={q1:.3e}")
        del model

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"Saved: {out_dir/'results.csv'} ({len(df)} rows)")

    L = int(df[df.metric == "Q"].layer.max()) + 1
    if args.plot2c_only:
        plot_absolute_magnitudes(df, L, str(plots_dir / "plot2c_absolute_magnitudes_logx.pdf"),
                                 xlog=True, vertical=True)
    else:
        plot_gradient_usability(df, L, str(plots_dir / "plot1_gradient_usability.pdf"))
        plot_forward_backward_alignment(df, L, str(plots_dir / "plot2_forward_backward_alignment.pdf"))
        plot_absolute_magnitudes(df, L, str(plots_dir / "plot2b_absolute_magnitudes.pdf"))
        plot_absolute_magnitudes(df, L, str(plots_dir / "plot2c_absolute_magnitudes_logx.pdf"),
                                 xlog=True, vertical=True)
        plot_reverse_credit(df, L, str(plots_dir / "plot3_reverse_credit.pdf"))
        plot_heatmaps(df, L, str(plots_dir / "plot4_heatmaps.pdf"))
    print(f"Plots: {plots_dir}")

    # Direct upload: analysis_output/ is outside the watchers' coverage, and this
    # file set overlaps nothing they push, so a plain copy is single-writer safe.
    import subprocess
    remote = f"gdrive:parity/analysis_output/backward_walsh_4xd4/{args.run}"
    r = subprocess.run(["rclone", "copy", str(out_dir), remote,
                        "--drive-chunk-size", "64M", "--transfers", "2", "-q"],
                       capture_output=True, text=True)
    print("Upload ok" if r.returncode == 0 else f"Upload failed: {r.stderr[:300]}")


if __name__ == "__main__":
    main()
