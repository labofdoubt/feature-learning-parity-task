"""Compare a re-run of two-block phase-diagram cells with the original sweep's results.

Matches rows by cell_id and writes one row per cell with the original and new
phase, objective, A4, C share and step count, plus their differences.

Usage:
  python -m analysis.accessibility.compare_two_block_reproduction \
      --new-dir runs/two_block_phase_pw_seed0/analysis/phase_grid \
      --reference analysis_output/two_block_phase_diagram/results_merged.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ("phase", "objective", "A4", "C_abs_share", "R", "steps")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-dir", required=True, type=Path,
                        help="directory with the re-run's results_g*.csv")
    parser.add_argument("--reference", required=True, type=Path,
                        help="original results CSV (e.g. results_merged.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <new-dir>/reproduction_vs_original.csv")
    args = parser.parse_args(argv)

    new = pd.concat([pd.read_csv(path) for path in sorted(args.new_dir.glob("results_g*.csv"))])
    old = pd.read_csv(args.reference)
    keep = ["cell_id", "lambda0", "ratio", *COLUMNS]
    merged = new[keep].merge(old[["cell_id", *COLUMNS]], on="cell_id",
                             suffixes=("_new", "_original"), validate="one_to_one")
    for column in ("objective", "A4", "C_abs_share", "R"):
        merged[f"{column}_abs_diff"] = (merged[f"{column}_new"]
                                        - merged[f"{column}_original"]).abs()
    merged["objective_rel_diff"] = (merged.objective_abs_diff
                                    / merged.objective_original.abs().clip(lower=1e-30))
    merged["phase_agrees"] = merged.phase_new == merged.phase_original
    merged = merged.sort_values(["lambda0", "ratio"])
    out = args.out or args.new_dir / "reproduction_vs_original.csv"
    merged.to_csv(out, index=False)
    print(f"{len(merged)} cells: phase agreement {merged.phase_agrees.sum()}/{len(merged)}, "
          f"median |rel objective diff| {merged.objective_rel_diff.median():.2e}, "
          f"max |A4 diff| {merged.A4_abs_diff.max():.3g}, "
          f"max |C share diff| {merged.C_abs_share_abs_diff.max():.3g} -> {out}")


if __name__ == "__main__":
    main()
