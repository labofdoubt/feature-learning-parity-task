"""Combine all plots from a run into a single PDF.

Collects every PDF in <run-dir>/plots/ (produced alongside PNGs by the analysis
scripts), sorts them in a logical order, and writes <run-dir>/results.pdf with
one plot per page plus a title page.  Requires pypdf (pip install pypdf).

Usage:
    python scripts/combine_plots.py --run-dir runs/my_exp/N_2048
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


def _make_title_page(run_dir: Path, n_figures: int) -> bytes:
    """Return a PDF title page as bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pp:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, run_dir.name, ha="center", va="center",
                 fontsize=24, fontweight="bold")
        fig.text(0.5, 0.45, str(run_dir.resolve()), ha="center", va="center",
                 fontsize=9, color="gray")
        fig.text(0.5, 0.38, f"{n_figures} figures", ha="center", va="center",
                 fontsize=12, color="gray")
        pp.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    buf.seek(0)
    return buf.read()


def run(run_dir: Path) -> None:
    import pypdf

    plots_dir = run_dir / "plots"
    pdfs = sorted(plots_dir.glob("*.pdf"), key=sort_key)
    if not pdfs:
        print(f"No PDF files found in {plots_dir}. Run analysis scripts first.")
        return

    out_path = run_dir / "results.pdf"
    writer = pypdf.PdfWriter()

    # Title page
    title_bytes = _make_title_page(run_dir, len(pdfs))
    writer.append(pypdf.PdfReader(io.BytesIO(title_bytes)))

    # Content pages
    for pdf in pdfs:
        writer.append(str(pdf))
        print(f"  added: {pdf.name}")

    with open(out_path, "wb") as f:
        writer.write(f)

    print(f"\nPDF saved: {out_path}  ({len(pdfs)} figures + title page)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run(Path(args.run_dir))


if __name__ == "__main__":
    main()
