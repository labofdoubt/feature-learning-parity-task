#!/usr/bin/env bash
# Analysis for k=32 hierarchical_symmetric experiments.
#
# Runs:
#   1. Parity mode norms (d4, d8, d16, d32) across all residual blocks.
#      The d32 entry gives the single root-parity mode strength per block.
#   2. Partition decompositions for d4, d8, d16 (bits x1..x4 / x1..x8 / x1..x16).
#      d32 decomposition is skipped (intractable partition count).
#
# Usage:
#   bash scripts/analyze_k32.sh <run_dir> [num_samples]
#
# num_samples defaults to 65536.
set -euo pipefail

RUN_DIR=${1:?usage: analyze_k32.sh <run_dir> [num_samples]}
NUM_SAMPLES=${2:-65536}
PYTHON=/usr/bin/python3

echo "[analyze_k32] $(date): Starting analysis for $(basename "$RUN_DIR")"

echo "[analyze_k32] Parity mode norms (d4, d8, d16, d32) ..."
$PYTHON scripts/analyze_parity_modes.py \
    --run-dir "$RUN_DIR" \
    --degrees 4 8 16 32 \
    --num-samples "$NUM_SAMPLES" \
    --batch-size 2048 \
    2>&1 | tee "$RUN_DIR/analyze_modes_k32.log"

for degree in 4 8 16; do
    echo "[analyze_k32] Partition decomposition d${degree} ..."
    $PYTHON scripts/analyze_decode.py \
        --run-dir "$RUN_DIR" \
        --degree "$degree" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size 2048 \
        --max-partitions 1000 \
        2>&1 | tee "$RUN_DIR/analyze_decode_d${degree}.log"
done

echo "[analyze_k32] $(date): Done — $(basename "$RUN_DIR")"
