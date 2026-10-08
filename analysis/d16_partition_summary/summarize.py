"""Aggregate d16 partition decode results across seeds and format a summary.

Reads per-block CSVs (decode_d16_block{i}.csv) from each run directory and
produces a formatted text summary table saved to --output-dir.

Usage:
  python -m analysis.d16_partition_summary.summarize \
    --runs-root /root/parity/runs \
    --output-dir /root/parity/analysis_output/d16_partition_summary \
    --top-k 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

K0_RUNS = [
    "wd3_k0_N2048",
    "wd3_k0_s1_N2048",
    "wd3_k0_s2_N2048",
    "wd3_k0_s3_N2048",
    "wd3_k0_s4_N2048",
    "wd3_k0_s5_N2048",
]
K4_RUNS = [
    "wd3_k4_N2048",
    "wd3_k4_s1_N2048",
    "wd3_k4_s2_N2048",
    "wd3_k4_s3_N2048",
    "wd3_k4_s4_N2048",
    "wd3_k4_s5_N2048",
]

SEED_LABELS = ["s0 (orig)", "s1", "s2", "s3", "s4", "s5"]

# Predefined experiment groups: name -> (condition_label, run_list, seed_labels)
EXPERIMENT_GROUPS: dict[str, tuple[str, list[str], list[str]]] = {
    "wd3": ("wd3_k0 vs wd3_k4", K0_RUNS + K4_RUNS, SEED_LABELS * 2),
    "hwv3": (
        "hwv3",
        [
            "hwv3_k0_N2048",
            "hwv3_k0_s1_N2048", "hwv3_k0_s2_N2048", "hwv3_k0_s3_N2048",
            "hwv3_k0_s4_N2048", "hwv3_k0_s5_N2048",
            "hwv3_k6_N2048",
            "hwv3_k6_s1_N2048", "hwv3_k6_s2_N2048", "hwv3_k6_s3_N2048",
            "hwv3_k6_s4_N2048", "hwv3_k6_s5_N2048",
        ],
        ["k0_s0", "k0_s1", "k0_s2", "k0_s3", "k0_s4", "k0_s5",
         "k6_s0", "k6_s1", "k6_s2", "k6_s3", "k6_s4", "k6_s5"],
    ),
}


def _recover_orig_norm(df: pd.DataFrame) -> float:
    """orig_norm = norm_modified * cosine / score (same for all rows in a block)."""
    row = df.iloc[0]
    score = row["score"]
    if abs(score) < 1e-15:
        return 0.0
    return float(row["norm_modified"] * row["cosine"] / score)


def load_block_df(run_dir: Path, block: int) -> tuple[pd.DataFrame, float] | None:
    csv = run_dir / "analysis" / f"decode_d16_block{block}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    orig_norm = _recover_orig_norm(df)
    return df, orig_norm


def _fmt_partition(label: str, max_len: int = 54) -> str:
    if len(label) <= max_len:
        return label
    return label[:max_len - 3] + "..."


def _divider(char: str = "─", width: int = 82) -> str:
    return char * width


def format_summary(
    runs: list[str],
    runs_root: Path,
    seed_labels: list[str],
    condition_label: str,
    top_k: int,
) -> list[str]:
    lines: list[str] = []
    W = 82
    PART_W = 54
    NUM_W = 11

    lines.append("=" * W)
    lines.append(f"  {condition_label}  —  d16 Partition Decoding  (top {top_k} per block, sorted by score)")
    lines.append("=" * W)

    for run_idx, (run_name, seed_lbl) in enumerate(zip(runs, seed_labels)):
        run_dir = runs_root / run_name
        lines.append("")
        lines.append(_divider("─", W))
        lines.append(f"  Seed: {seed_lbl}   ({run_name})")
        lines.append(_divider("─", W))

        for block in range(4):
            result = load_block_df(run_dir, block)
            if result is None:
                lines.append(f"    Block {block}  [no data]")
                continue
            df, orig_norm = result
            top = df.head(top_k)

            lines.append(f"    Block {block}   orig_norm = {orig_norm:.5f}")

            # Table header
            hdr = (
                f"    {'Partition':<{PART_W}}  {'mod_norm':>{NUM_W}}  "
                f"{'cosine':>{NUM_W}}  {'score':>{NUM_W}}"
            )
            lines.append(hdr)
            lines.append("    " + _divider("·", PART_W + 3 * (NUM_W + 2)))

            for _, row in top.iterrows():
                part = _fmt_partition(row["partition"], PART_W)
                lines.append(
                    f"    {part:<{PART_W}}  {row['norm_modified']:>{NUM_W}.5f}  "
                    f"{row['cosine']:>{NUM_W}.5f}  {row['score']:>{NUM_W}.5f}"
                )
            lines.append("")

    return lines


def run_missing_decode(runs: list[str], runs_root: Path) -> None:
    """Run analyze_decode.py for any run missing d16 CSVs."""
    for run_name in runs:
        run_dir = runs_root / run_name
        csv = run_dir / "analysis" / "decode_d16_block0.csv"
        if csv.exists():
            continue
        print(f"Running d16 decode for {run_name} ...")
        r = subprocess.run(
            [
                sys.executable, "scripts/analyze_decode.py",
                "--run-dir", str(run_dir),
                "--degree", "16",
                "--num-samples", "65536",
                "--batch-size", "2048",
                "--max-partitions", "1000",
            ],
            cwd=str(_ROOT),
        )
        if r.returncode != 0:
            print(f"  [WARN] decode failed for {run_name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default="/root/parity/runs")
    parser.add_argument("--output-dir", default="/root/parity/analysis_output/d16_partition_summary")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-decode", action="store_true",
                        help="Skip running decode analysis (assume CSVs already exist)")
    parser.add_argument("--group", default=None,
                        help="Predefined experiment group to run (e.g. 'hwv3'). "
                             "If omitted, run the full wd3 group.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Explicit list of run names (overrides --group).")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Seed labels matching --runs (optional).")
    parser.add_argument("--condition-label", default=None,
                        help="Label for the condition block in the summary.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve run groups
    if args.runs:
        # Fully explicit
        groups = [(
            args.condition_label or "custom",
            args.runs,
            args.labels or [r.replace("_N2048", "") for r in args.runs],
        )]
        suffix = "custom"
    elif args.group and args.group in EXPERIMENT_GROUPS:
        cond_lbl, runs, lbls = EXPERIMENT_GROUPS[args.group]
        # Split into sub-groups for formatting (each unique prefix becomes a section)
        # For hwv3 treat as one group; for wd3 split into k0 / k4
        if args.group == "wd3":
            groups = [
                ("wd3_k0", K0_RUNS, SEED_LABELS),
                ("wd3_k4", K4_RUNS, SEED_LABELS),
            ]
        elif args.group == "hwv3":
            hwv3_k0_runs = [r for r in runs if "k0" in r]
            hwv3_k6_runs = [r for r in runs if "k6" in r]
            hwv3_k0_lbls = [l for l in lbls if "k0" in l]
            hwv3_k6_lbls = [l for l in lbls if "k6" in l]
            groups = [
                ("hwv3_k0", hwv3_k0_runs, hwv3_k0_lbls),
                ("hwv3_k6", hwv3_k6_runs, hwv3_k6_lbls),
            ]
        else:
            groups = [(cond_lbl, runs, lbls)]
        suffix = args.group
    else:
        # Default: full wd3
        groups = [
            ("wd3_k0", K0_RUNS, SEED_LABELS),
            ("wd3_k4", K4_RUNS, SEED_LABELS),
        ]
        suffix = "wd3"

    all_runs_flat = [r for _, runs, _ in groups for r in runs]
    if not args.skip_decode:
        run_missing_decode(all_runs_flat, runs_root)

    all_lines: list[str] = []
    for i, (cond_lbl, runs, lbls) in enumerate(groups):
        if i > 0:
            all_lines += ["", ""]
        all_lines.extend(format_summary(runs, runs_root, lbls, cond_lbl, args.top_k))

    summary_path = out_dir / f"summary_{suffix}.txt"
    summary_path.write_text("\n".join(all_lines) + "\n")
    print(f"\nSummary saved: {summary_path}")

    # Combined CSV
    rows = []
    for cond_lbl, runs, lbls in groups:
        for seed_lbl, run_name in zip(lbls, runs):
            run_dir = runs_root / run_name
            for block in range(4):
                result = load_block_df(run_dir, block)
                if result is None:
                    continue
                df, orig_norm = result
                top = df.head(args.top_k).copy()
                top.insert(0, "block", block)
                top.insert(0, "seed", seed_lbl)
                top.insert(0, "run", run_name)
                top.insert(0, "condition", cond_lbl)
                top.insert(0, "rank", range(1, len(top) + 1))
                top["norm_original"] = orig_norm
                rows.append(top[["condition", "run", "seed", "block", "rank",
                                  "partition", "norm_original", "norm_modified", "cosine", "score"]])

    combined = pd.concat(rows, ignore_index=True)
    csv_path = out_dir / f"top_partitions_{suffix}.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Combined CSV saved: {csv_path}")

    # Upload to Drive
    remote = "gdrive:parity/analysis_output/d16_partition_summary"
    print(f"\nUploading to {remote} ...")
    r = subprocess.run(["rclone", "copy", str(out_dir), remote, "--progress"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("Upload complete.")
    else:
        print(f"[WARN] rclone failed:\n{r.stderr}")

    # Print to stdout too
    print("\n" + "\n".join(all_lines))


if __name__ == "__main__":
    main()
