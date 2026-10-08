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
#   TAG_PREFIX=j   (worker tags j0..j7, so a second launch can share OUT)
#   READOUT_INIT=fan_in READOUT_LR=1e-3   (readout variance 1/N instead of muP's 1/N^2)
#   READOUT_INIT=base BASE_WIDTH=64   (d4/d8 trainer: rescale from base width N0 = 64, i.e.
#                                      readout var N0/N^2, readout lr lr*N0/N, embedding lr lr*N/N0)
#   EMBEDDING_LR=...   (d4/d8 trainer: explicit embedding rate)
#   PRETRAIN_MIN_STEPS=10000 PRETRAIN_MAX_STEPS=10000   (fixed-length parent pretraining)
#   TRAINER=analysis/accessibility/run_two_block_d4_d8_phase_diagram.py   (d4-parent/d8 task)
#   RATIO_LIST=0.1,0.3,1,2,5,10,30,100   (exactly eight lambda_1/lambda_2 values, one per GPU)
# Langevin annealing of the final phase (T0 = 0, the default, is the deterministic sweep):
#   TEMPERATURE_INITIAL=1e-7 TEMPERATURE_HOLD_STEPS=5000 TEMPERATURE_ANNEAL_STEPS=25000
#   TEMPERATURE_FINAL_RATIO=1e-3 NOISE_SEED=0
# With T0 > 0 the final phase uses MOMENTUM=0 (required), noise-free parent
# pretraining keeps PRETRAIN_MOMENTUM=0.9 (the deterministic sweep's warm start),
# and OUT / SESSION default to a per-T0 directory and session.
set -euo pipefail

ROOT="${PARITY_ROOT:-/root/parity}"
PYTHON="${PYTHON:-/venv/main/bin/python3}"
TEMPERATURE_INITIAL="${TEMPERATURE_INITIAL:-0}"
TEMPERATURE_HOLD_STEPS="${TEMPERATURE_HOLD_STEPS:-5000}"
TEMPERATURE_ANNEAL_STEPS="${TEMPERATURE_ANNEAL_STEPS:-25000}"
TEMPERATURE_FINAL_RATIO="${TEMPERATURE_FINAL_RATIO:-1e-3}"
NOISE_SEED="${NOISE_SEED:-0}"
NOISY=$(awk -v t="$TEMPERATURE_INITIAL" 'BEGIN { print (t + 0 > 0) ? 1 : 0 }')
if [ "$NOISY" = "1" ]; then
  T_LABEL="T${TEMPERATURE_INITIAL//./p}"
  OUT="${OUT:-$ROOT/analysis_output/two_block_phase_diagram_$T_LABEL}"
  SESSION="${SESSION:-two_block_phase_$T_LABEL}"
  MOMENTUM="${MOMENTUM:-0}"
  PRETRAIN_MOMENTUM="${PRETRAIN_MOMENTUM:-0.9}"
else
  OUT="${OUT:-$ROOT/analysis_output/two_block_phase_diagram}"
  SESSION="${SESSION:-two_block_phase}"
  MOMENTUM="${MOMENTUM:-}"
  PRETRAIN_MOMENTUM="${PRETRAIN_MOMENTUM:-}"
fi
TAG_PREFIX="${TAG_PREFIX:-g}"
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
READOUT_INIT="${READOUT_INIT:-}"
BASE_WIDTH="${BASE_WIDTH:-}"
EMBEDDING_LR="${EMBEDDING_LR:-}"
PRETRAIN_MIN_STEPS="${PRETRAIN_MIN_STEPS:-}"
PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-}"
TRAINER="${TRAINER:-analysis/accessibility/run_two_block_phase_diagram.py}"
IFS=',' read -r -a RATIOS <<< "${RATIO_LIST:-0.05,0.1,0.25,0.5,1,2,5,20}"
if [ "${#RATIOS[@]}" -ne 8 ]; then
  echo "RATIO_LIST must hold exactly eight ratios (one per GPU); got ${#RATIOS[@]}" >&2
  exit 1
fi

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
  tag="${TAG_PREFIX}${gpu}"
  log="$OUT/worker_${tag}.log"
  readout_arg=""
  if [ -n "$READOUT_LR" ]; then
    readout_arg="--readout-lr '$READOUT_LR'"
  fi
  checkpoint_arg=""
  if [ "$SAVE_CHECKPOINTS" = "1" ]; then
    checkpoint_arg="--save-checkpoints --checkpoint-dtype '$CHECKPOINT_DTYPE'"
  fi
  momentum_arg=""
  if [ -n "$MOMENTUM" ]; then
    momentum_arg="--momentum '$MOMENTUM'"
  fi
  if [ -n "$PRETRAIN_MOMENTUM" ]; then
    momentum_arg="$momentum_arg --pretrain-momentum '$PRETRAIN_MOMENTUM'"
  fi
  setup_arg=""
  if [ -n "$READOUT_INIT" ]; then
    setup_arg="--readout-init '$READOUT_INIT'"
  fi
  if [ -n "$BASE_WIDTH" ]; then
    setup_arg="$setup_arg --base-width '$BASE_WIDTH'"
  fi
  if [ -n "$EMBEDDING_LR" ]; then
    setup_arg="$setup_arg --embedding-lr '$EMBEDDING_LR'"
  fi
  if [ -n "$PRETRAIN_MIN_STEPS" ]; then
    setup_arg="$setup_arg --pretrain-min-steps '$PRETRAIN_MIN_STEPS'"
  fi
  if [ -n "$PRETRAIN_MAX_STEPS" ]; then
    setup_arg="$setup_arg --pretrain-max-steps '$PRETRAIN_MAX_STEPS'"
  fi
  noise_arg=""
  if [ "$NOISY" = "1" ]; then
    noise_arg="--temperature-initial '$TEMPERATURE_INITIAL' --temperature-hold-steps '$TEMPERATURE_HOLD_STEPS' --temperature-anneal-steps '$TEMPERATURE_ANNEAL_STEPS' --temperature-final-ratio '$TEMPERATURE_FINAL_RATIO' --noise-seed '$NOISE_SEED'"
  fi
  command="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' '$TRAINER' --ratios '$ratio' --lambda0s '$LAMBDA0S' --seeds '$SEEDS' --protocols '$PROTOCOLS' --tag '$tag' --output-dir '$OUT' --width '$WIDTH' --lr '$LR' $readout_arg $checkpoint_arg $momentum_arg $setup_arg $noise_arg --max-steps '$MAX_STEPS' --min-steps '$MIN_STEPS' --snapshot-every '$SNAPSHOT_EVERY' 2>&1 | tee -a '$log'"
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
