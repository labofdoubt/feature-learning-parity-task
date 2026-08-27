#!/usr/bin/env bash
# Wait for a training run to finish, run analysis, then start a follow-up training.
#
# Usage (run inside a tmux session):
#   bash scripts/run_sequence.sh <wait-run-dir> <next-config>
#
# Example:
#   tmux new -d -s seq "bash scripts/run_sequence.sh \
#       runs/mup_N2048  runs/no_skip_N2048/config.yaml"
set -euo pipefail

WAIT_DIR="${1:?usage: run_sequence.sh <wait-run-dir> <next-config>}"
NEXT_CONFIG="${2:?usage: run_sequence.sh <wait-run-dir> <next-config>}"
FINAL_PT="$WAIT_DIR/checkpoints/final.pt"

echo "[seq] Waiting for $FINAL_PT ..."
while [ ! -f "$FINAL_PT" ]; do
  sleep 60
done
echo "[seq] Training finished: $(date)"

echo "[seq] Running analysis on $WAIT_DIR ..."
cd "$(dirname "$0")/.."
bash scripts/run_analysis.sh "$WAIT_DIR"
echo "[seq] Analysis done: $(date)"

NEXT_DIR=$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('$NEXT_CONFIG').read_text())
print(cfg['training']['output_dir'])
")
mkdir -p "$NEXT_DIR"
echo "[seq] Starting next training: $NEXT_CONFIG -> $NEXT_DIR"
python3 -m parity_net.train --config "$NEXT_CONFIG" 2>&1 | tee "$NEXT_DIR/train.log"
echo "[seq] Next training done: $(date)"

echo "[seq] Running analysis on $NEXT_DIR ..."
bash scripts/run_analysis.sh "$NEXT_DIR"
echo "[seq] All done: $(date)"
