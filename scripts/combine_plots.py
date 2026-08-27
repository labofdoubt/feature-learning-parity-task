"""Combine all plots from a run into a single PDF.

Collects every PNG in <run-dir>/plots/, sorts them in a logical order,
and writes <run-dir>/results.pdf with one plot per page plus a title page.

Usage:
    python scripts/combine_plots.py --run-dir runs/my_exp/N_2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg


# Logical section order — files are sorted by the first matching prefix.
# Anything not matched goes at the end, alphabetically.
SECTION_ORDER = [
    "curves_total",
    "curves_by_degree",
    "embedding_gram",
    "pca_intervention",
    "parity_mode_gram",
    "parity_cross_block_alignment_d2",
    "parity_cross_block_alignment_d4",
    "parity_cross_block_alignment_d8",
    "parity_cross_block_alignment_d16",
    "decode_d4",
    "decode_d8",
    "decode_d16",
]


def sort_key(path: Path) -> tuple[int, str]:
    name = path.stem
    for i, prefix in enumerate(SECTION_ORDER):
        if name.startswith(prefix):
            return (i, name)
    return (len(SECTION_ORDER), name)


def run(run_dir: Path) -> None:
    plots_dir = run_dir / "plots"
    pngs = sorted(plots_dir.glob("*.png"), key=sort_key)
    if not pngs:
        print(f"No PNG files found in {plots_dir}")
        return

    out_path = run_dir / "results.pdf"
    with PdfPages(out_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, run_dir.name, ha="center", va="center", fontsize=24, fontweight="bold")
        fig.text(0.5, 0.45, str(run_dir.resolve()), ha="center", va="center", fontsize=9, color="gray")
        fig.text(0.5, 0.38, f"{len(pngs)} figures", ha="center", va="center", fontsize=12, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for png in pngs:
            img = mpimg.imread(str(png))
            h, w = img.shape[:2]
            # Scale to fit A4-landscape while preserving aspect ratio
            fig_w, fig_h = 11.0, 8.5
            scale = min(fig_w / (w / 100), fig_h / (h / 100))
            fig = plt.figure(figsize=(w / 100 * scale, h / 100 * scale))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.01, 0.01, png.name, fontsize=7, color="gray", transform=fig.transFigure)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  added: {png.name}")

    print(f"\nPDF saved: {out_path}  ({len(pngs)} figures + title page)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run(Path(args.run_dir))


if __name__ == "__main__":
    main()
