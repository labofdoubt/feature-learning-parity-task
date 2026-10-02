#!/bin/bash
# Launch the fully trainable two-block phase-diagram sweep on eight GPUs.
#
# Each GPU owns one row of lambda_1/lambda_2 values and runs the common
# lambda_0 grid, three seeds, and both initialization protocols sequentially.
# Existing completed cells are detected by cell_id and skipped, so relaunching
# the same command is the supported recovery path after interruption.
#
# Usage:
#   bash scripts/launch_two_block_phase_workers.sh
# Optional environment overrides:
#   SEEDS=0,1 PROTOCOLS=joint WIDTH=512 MAX_STEPS=1000 ...
#   SAVE_CHECKPOINTS=1 CHECKPOINT_DTYPE=float64   (save each cell's final weights)
set -euo pipefail

ROOT="${PARITY_ROOT:-/root/parity}"
PYTHON="${PYTHON:-/venv/main/bin/python3}"
OUT="${OUT:-$ROOT/analysis_output/two_block_phase_diagram}"
SESSION="${SESSION:-two_block_phase}"
SEEDS="${SEEDS:-0,1,2}"
PROTOCOLS="${PROTOCOLS:-joint,parent_warm}"
WIDTH="${WIDTH:-1024}"
MAX_STEPS="${MAX_STEPS:-50000}"
MIN_STEPS="${MIN_STEPS:-15000}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-500}"
LR="${LR:-0.01}"
READOUT_LR="${READOUT_LR:-}"
LAMBDA0S="${LAMBDA0S:-1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
CHECKPOINT_DTYPE="${CHECKPOINT_DTYPE:-float32}"
RATIOS=(0.05 0.1 0.25 0.5 1 2 5 20)

cd "$ROOT"
test -x "$PYTHON" || { echo "Python not executable: $PYTHON" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [ "$gpu_count" -lt 8 ]; then
  echo "Need 8 visible GPUs; found $gpu_count" >&2
  exit 1
fi
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists; inspect or remove it before relaunching" >&2
  exit 1
fi

mkdir -p "$OUT"
for gpu in 0 1 2 3 4 5 6 7; do
  ratio="${RATIOS[$gpu]}"
  tag="g${gpu}"
  log="$OUT/worker_${tag}.log"
  readout_arg=""
  if [ -n "$READOUT_LR" ]; then
    readout_arg="--readout-lr '$READOUT_LR'"
  fi
  checkpoint_arg=""
  if [ "$SAVE_CHECKPOINTS" = "1" ]; then
    checkpoint_arg="--save-checkpoints --checkpoint-dtype '$CHECKPOINT_DTYPE'"
  fi
  command="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' analysis/accessibility/run_two_block_phase_diagram.py --ratios '$ratio' --lambda0s '$LAMBDA0S' --seeds '$SEEDS' --protocols '$PROTOCOLS' --tag '$tag' --output-dir '$OUT' --width '$WIDTH' --lr '$LR' $readout_arg $checkpoint_arg --max-steps '$MAX_STEPS' --min-steps '$MIN_STEPS' --snapshot-every '$SNAPSHOT_EVERY' 2>&1 | tee -a '$log'"
  if [ "$gpu" -eq 0 ]; then
    tmux new-session -d -s "$SESSION" -n "$tag" "$command"
  else
    tmux new-window -t "$SESSION:" -n "$tag" "$command"
  fi
done

sleep 3
echo "Launched phase-diagram workers in tmux session '$SESSION'."
echo "Monitor: tmux attach -t $SESSION"
echo "After completion:"
echo "  $PYTHON analysis/accessibility/merge_two_block_phase_diagram.py --dir $OUT"
tmux list-windows -t "$SESSION" -F '#{window_name} dead=#{pane_dead} cmd=#{pane_current_command}'
