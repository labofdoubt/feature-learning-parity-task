"""Merge the d4-adapter sweep and make figures for its report.

This script is intentionally local-only: it reads completed CSV slices, writes merged
CSVs and figures beside them, and performs no remote synchronization.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_output" / "accessibility_alpha_d4adapter_wd"
COMPARE = ROOT / "analysis_output" / "comparison_sources"
V2W4 = ROOT / "analysis_output" / "accessibility_alpha_v2w4wd" / "results_merged.csv"
RC = 1 / np.sqrt(8)
ALPHAS = [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0]
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def read_rows(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fields:
                raise ValueError(f"column mismatch in {path}")
            rows.extend(reader)
    if fields is None:
        raise FileNotFoundError("no input CSVs found")
    return fields, rows


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...], values: tuple[str, ...]):
    bins: dict[tuple[float, ...], dict[str, list[float]]] = defaultdict(
        lambda: {value: [] for value in values}
    )
    for row in rows:
        key = tuple(number(row, name) for name in keys)
        for value in values:
            bins[key][value].append(number(row, value))
    out = {}
    for key, samples in bins.items():
        out[key] = {
            value: (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            for value, vals in samples.items()
        }
    return out


def mean_curve(rows, lam, xkey, ykey):
    table = grouped(
        [row for row in rows if np.isclose(number(row, "lam"), lam)],
        ("alpha",),
        (xkey, ykey),
    )
    points = sorted((values[xkey][0], values[ykey][0], values[ykey][1]) for values in table.values())
    return tuple(np.array(part) for part in zip(*points))


result_paths = sorted(OUT.glob("results_d4adapter_g*.csv"))
trajectory_paths = sorted(OUT.glob("trajectories_d4adapter_g*.csv"))
result_fields, results = read_rows(result_paths)
trajectory_fields, trajectories = read_rows(trajectory_paths)

cells = {(int(number(r, "seed")), number(r, "alpha"), number(r, "lam")) for r in results}
expected = {(seed, alpha, lam) for seed in (0, 1, 2) for alpha in ALPHAS for lam in LAMBDAS}
if len(results) != 192 or cells != expected or len(cells) != len(results):
    raise ValueError(f"incomplete or duplicate result grid: rows={len(results)}, cells={len(cells)}")
if {r["wd_mode"] for r in results} != {"d4_adapter"}:
    raise ValueError("unexpected weight-decay mode")
trajectory_cells = {
    (int(number(r, "seed")), number(r, "alpha"), number(r, "lam")) for r in trajectories
}
if trajectory_cells != expected:
    raise ValueError(f"trajectory grid incomplete: {len(trajectory_cells)} cells")

write_rows(OUT / "results_merged.csv", result_fields, results)
write_rows(OUT / "trajectories_merged.csv", trajectory_fields, trajectories)

summary = grouped(
    results,
    ("alpha", "lam"),
    (
        "R0",
        "R_final",
        "frac22_w",
        "f4",
        "f4_base",
        "f4_adapter",
        "f4_block2",
        "V1_adapter_norm",
        "V2_norm",
        "W4_norm",
    ),
)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.2,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    }
)


def lam_y(lam: float) -> float:
    return lam if lam > 0 else 3e-5


# Phase diagrams using all three seeds.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
for ax, xname, xlabel in (
    (axes[0], "R0", r"initial accessibility $R_0$"),
    (axes[1], "R_final", r"final accessibility $R_{\rm final}$"),
):
    xs, ys, colors, failed = [], [], [], []
    for (alpha, lam), values in sorted(summary.items()):
        xs.append(values[xname][0])
        ys.append(lam_y(lam))
        colors.append(values["frac22_w"][0])
        failed.append(values["f4"][0] < 0.9)
    scatter = ax.scatter(
        xs,
        ys,
        c=colors,
        s=205,
        marker="s",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidths=0.35,
    )
    if any(failed):
        ax.scatter(
            np.array(xs)[failed],
            np.array(ys)[failed],
            marker="x",
            s=75,
            c="black",
            linewidths=1.0,
            label=r"mean $f_4<0.9$",
        )
    ax.axvline(RC, color="#b2182b", ls="--", lw=1.25, label=r"$R_c=1/\sqrt{8}$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.16, which="both")
axes[0].set_ylabel(r"matched decay $\lambda$ (bottom row is 0)")
axes[0].set_title("initial accessibility")
axes[1].set_title("after phase-2 training")
axes[0].legend(loc="upper left")
axes[1].legend(loc="upper left")
fig.suptitle("d4-adapter control: the compositional phase returns above the accessibility threshold")
fig.subplots_adjust(left=0.08, right=0.87, bottom=0.16, top=0.84, wspace=0.12)
cax = fig.add_axes((0.895, 0.16, 0.018, 0.68))
fig.colorbar(scatter, cax=cax, label=r"block-2 $2{+}2$ fraction")
fig.savefig(OUT / "adapter_phase_diagrams.pdf", bbox_inches="tight")
fig.savefig(OUT / "adapter_phase_diagrams.png", dpi=190, bbox_inches="tight")
plt.close(fig)


# Fixed-lambda cuts, protocol comparison, and route decomposition.
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
cut_lams = [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
palette = plt.cm.viridis(np.linspace(0.05, 0.9, len(cut_lams)))
for color, lam in zip(palette, cut_lams):
    x, y, sd = mean_curve(results, lam, "R0", "frac22_w")
    axes[0].plot(x, y, "o-", color=color, lw=1.4, ms=4, label=fr"$\lambda={lam:g}$")
axes[0].axvline(RC, color="#b2182b", ls="--", lw=1.0)
axes[0].set(xlabel=r"$R_0$", ylabel=r"block-2 $2{+}2$ fraction", ylim=(-0.03, 1.03))
axes[0].set_title("adapter cuts")
axes[0].legend(ncol=2, loc="upper left")

comparison_specs = [
    (results, "R0", "adapter", "#1b9e77", "o"),
]
for path, label, color, marker in (
    (COMPARE / "v2wd_results_merged.csv", "V2 decay", "#7570b3", "s"),
    (V2W4, "V2+W4 decay", "#d95f02", "^"),
    (COMPARE / "frozen_wd_results.csv", "frozen parents", "#333333", "D"),
):
    _, rows = read_rows([path])
    comparison_specs.append((rows, "R" if label == "frozen parents" else "R0", label, color, marker))
for rows, xkey, label, color, marker in comparison_specs:
    x, y, sd = mean_curve(rows, 1e-2, xkey, "frac22_w")
    axes[1].plot(x, y, marker=marker, color=color, lw=1.35, ms=4, label=label)
axes[1].axvline(RC, color="#b2182b", ls="--", lw=1.0)
axes[1].set(xlabel=r"initial or frozen $R$", ylabel=r"$2{+}2$ fraction", ylim=(-0.03, 1.03))
axes[1].set_title(r"protocol comparison at $\lambda=10^{-2}$")
axes[1].legend(loc="upper left")

route_values = ("f4_base", "f4_adapter", "f4_block2", "frac22_w")
route_summary = grouped(
    [r for r in results if np.isclose(number(r, "lam"), 1e-2)],
    ("alpha",),
    ("R0",) + route_values,
)
route_points = sorted(route_summary.values(), key=lambda v: v["R0"][0])
x = np.array([v["R0"][0] for v in route_points])
for key, label, color, marker in (
    ("f4_base", "frozen base route", "#e41a1c", "o"),
    ("f4_adapter", "trainable adapter route", "#377eb8", "s"),
    ("f4_block2", "block-2 route", "#4daf4a", "^"),
):
    y = np.array([v[key][0] for v in route_points])
    sd = np.array([v[key][1] for v in route_points])
    axes[2].errorbar(x, y, yerr=sd, color=color, marker=marker, lw=1.3, ms=4, capsize=2, label=label)
y = np.array([v["frac22_w"][0] for v in route_points])
axes[2].plot(x, y, "k--", lw=1.25, label="block-2 2+2 fraction")
axes[2].axvline(RC, color="#b2182b", ls="--", lw=1.0)
axes[2].set(xlabel=r"$R_0$", ylabel="signed d4 contribution / fraction", ylim=(-0.03, 1.03))
axes[2].set_title(r"route competition at $\lambda=10^{-2}$")
axes[2].legend(loc="upper left")
for ax in axes:
    ax.grid(alpha=0.17)
fig.tight_layout()
fig.savefig(OUT / "adapter_cuts_comparison_routes.pdf", bbox_inches="tight")
fig.savefig(OUT / "adapter_cuts_comparison_routes.png", dpi=190, bbox_inches="tight")
plt.close(fig)


# Representation drift and norm balance.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
drift_lams = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
palette = plt.cm.plasma(np.linspace(0.05, 0.9, len(drift_lams)))
for color, lam in zip(palette, drift_lams):
    table = grouped(
        [r for r in results if np.isclose(number(r, "lam"), lam)],
        ("alpha",),
        ("R0", "R_final"),
    )
    points = sorted((v["R0"][0], v["R_final"][0]) for v in table.values())
    x, y = (np.array(v) for v in zip(*points))
    axes[0].plot(x, y, "o-", color=color, lw=1.25, ms=4, label=fr"$\lambda={lam:g}$")
lim = 0.62
axes[0].plot([0, lim], [0, lim], "k:", lw=1.0, label="no drift")
axes[0].axvline(RC, color="#b2182b", ls="--", lw=0.9)
axes[0].axhline(RC, color="#b2182b", ls="--", lw=0.9)
axes[0].set(xlabel=r"$R_0$", ylabel=r"$R_{\rm final}$", xlim=(0, lim), ylim=(0, lim))
axes[0].set_title("accessibility drift")
axes[0].legend(ncol=2, loc="upper left")

norm_summary = grouped(
    [r for r in results if np.isclose(number(r, "lam"), 1e-2)],
    ("alpha",),
    ("R0", "V1_adapter_norm", "V2_norm", "W4_norm"),
)
norm_points = sorted(norm_summary.values(), key=lambda v: v["R0"][0])
x = np.array([v["R0"][0] for v in norm_points])
for key, label, color, marker in (
    ("V1_adapter_norm", r"$\|V'_1\|$", "#377eb8", "s"),
    ("V2_norm", r"$\|V_2\|$", "#4daf4a", "o"),
    ("W4_norm", r"$\|W_4\|$", "#984ea3", "^"),
):
    y = np.array([v[key][0] for v in norm_points])
    sd = np.array([v[key][1] for v in norm_points])
    axes[1].errorbar(x, y, yerr=sd, color=color, marker=marker, lw=1.35, ms=4, capsize=2, label=label)
axes[1].axvline(RC, color="#b2182b", ls="--", lw=0.9)
axes[1].set(xlabel=r"$R_0$", ylabel="Frobenius norm", ylim=(0, 2.05))
axes[1].set_title(r"matched-route norms at $\lambda=10^{-2}$")
axes[1].legend(loc="upper right")
for ax in axes:
    ax.grid(alpha=0.17)
fig.tight_layout()
fig.savefig(OUT / "adapter_drift_norms.pdf", bbox_inches="tight")
fig.savefig(OUT / "adapter_drift_norms.png", dpi=190, bbox_inches="tight")
plt.close(fig)


# Mean phase-plane trajectories across the three seeds.
fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.65), sharey=True)
alpha_colors = plt.cm.plasma(np.linspace(0.04, 0.94, len(ALPHAS)))
for ax, lam in zip(axes, (1e-3, 1e-2, 3e-2)):
    selected = [r for r in trajectories if np.isclose(number(r, "lam"), lam)]
    for alpha, color in zip(ALPHAS, alpha_colors):
        by_step = grouped(
            [r for r in selected if np.isclose(number(r, "alpha"), alpha)],
            ("step",),
            ("R", "frac22_w"),
        )
        points = sorted((step[0], v["R"][0], v["frac22_w"][0]) for step, v in by_step.items())
        _, x, y = (np.array(v) for v in zip(*points))
        ax.plot(x, y, color=color, lw=1.25, label=fr"$\alpha={alpha:g}$")
        ax.plot(x[0], y[0], "o", color=color, ms=3.5)
        ax.plot(x[-1], y[-1], "s", color=color, ms=3.5)
    ax.axvline(RC, color="#b2182b", ls="--", lw=1.0)
    ax.set_xlabel(r"mean $R(t)$")
    ax.set_title(fr"$\lambda={lam:g}$")
    ax.grid(alpha=0.16)
axes[0].set_ylabel(r"mean block-2 $2{+}2$ fraction")
axes[0].legend(ncol=2, loc="upper left", fontsize=6.7)
fig.suptitle("mean trajectories over three seeds (circles: onset, squares: final)")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "adapter_phase_plane.pdf", bbox_inches="tight")
fig.savefig(OUT / "adapter_phase_plane.png", dpi=190, bbox_inches="tight")
plt.close(fig)

max_sd_key, max_sd = max(
    ((key, values["frac22_w"][1]) for key, values in summary.items()),
    key=lambda item: item[1],
)
print(
    f"validated {len(results)} result cells and {len(trajectories)} trajectory rows "
    f"across {len(trajectory_cells)} cells"
)
print(f"largest seed SD in f22: {max_sd:.4f} at alpha={max_sd_key[0]:g}, lambda={max_sd_key[1]:g}")
print("merged CSVs and report figures written to", OUT)
