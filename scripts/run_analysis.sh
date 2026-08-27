#!/usr/bin/env bash
# Run the full analysis pipeline for a trained model run.
#
# Usage:
#   ./scripts/run_analysis.sh <run-dir>
#   ./scripts/run_analysis.sh runs/my_exp/N_2048
#
# Optional env-var overrides:
#   PCA_SAMPLES=20000 ALIGN_INDICES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
#     ./scripts/run_analysis.sh runs/my_exp/N_2048

set -euo pipefail

RUN_DIR="${1:?Usage: $0 <run-dir>}"
PCA_SAMPLES="${PCA_SAMPLES:-20000}"
KEEP_PCS_MAX="${KEEP_PCS_MAX:-80}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_DECODE_SAMPLES="${NUM_DECODE_SAMPLES:-65536}"

echo "============================================================"
echo "Analysis pipeline for: $RUN_DIR"
echo "============================================================"

echo ""
echo "── 1/6  Train/test curves ──────────────────────────────────"
python3 scripts/analyze_curves.py \
    --run-dir "$RUN_DIR"

echo ""
echo "── 2/6  PCA interventions ──────────────────────────────────"
python3 scripts/analyze_pca.py \
    --run-dir "$RUN_DIR" \
    --pca-samples "$PCA_SAMPLES" \
    --keep-pcs-max "$KEEP_PCS_MAX" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "── 3/6  Embedding Gram matrix ──────────────────────────────"
python3 scripts/analyze_embedding_gram.py \
    --run-dir "$RUN_DIR"

echo ""
echo "── 4/6  Parity-mode Gram matrices + cross-layer alignment ──"
python3 scripts/analyze_parity_modes.py \
    --run-dir "$RUN_DIR" \
    --degrees 2 4 8 16 \
    --batch-size "$BATCH_SIZE"

echo ""
echo "── 5/7  Decode d4 ──────────────────────────────────────────"
python3 scripts/analyze_decode.py \
    --run-dir "$RUN_DIR" \
    --degree 4 \
    --num-samples "$NUM_DECODE_SAMPLES" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "── 6/7  Decode d8 ──────────────────────────────────────────"
python3 scripts/analyze_decode.py \
    --run-dir "$RUN_DIR" \
    --degree 8 \
    --num-samples "$NUM_DECODE_SAMPLES" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "── 7/7  Decode d16 ─────────────────────────────────────────"
python3 scripts/analyze_decode.py \
    --run-dir "$RUN_DIR" \
    --degree 16 \
    --num-samples "$NUM_DECODE_SAMPLES" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "── Combining all plots into results.pdf ────────────────────"
python3 scripts/combine_plots.py --run-dir "$RUN_DIR"

echo ""
echo "── Syncing to Google Drive ─────────────────────────────────"
REMOTE="gdrive:parity/runs/$(basename "$RUN_DIR")"
bash "$(dirname "$0")/backup_runs.sh" "$RUN_DIR" "$REMOTE" \
  || echo "[backup] Warning: Drive sync failed — results remain local"

echo ""
echo "Done."
echo "  Plots:   $RUN_DIR/plots/"
echo "  CSVs:    $RUN_DIR/analysis/"
echo "  Summary: $RUN_DIR/results.pdf"
