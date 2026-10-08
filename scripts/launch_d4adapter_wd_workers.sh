#!/bin/bash
# Launch 8 parallel workers for the matched d4-adapter WD control.
# One alpha is pinned to each GPU; seeds 0,1,2 run sequentially per worker.
# Usage: bash scripts/launch_d4adapter_wd_workers.sh
#
# E and the alpha-rescaled pretrained V1 are frozen. A parallel trainable
# block-1 adapter V1' is added, and matched phase-2 weight decay is applied to
# V1', V2, and only the d4 readout row W[2].
set -euo pipefail
P=/venv/main/bin/python3
OUT=/root/parity/analysis_output/accessibility_alpha_d4adapter_wd
CKPT=/root/parity/analysis_output/accessibility_ckpts
cd /root/parity

ALPHAS=(0.1 0.3 0.5 0.75 1.0 1.5 2.5 4.0)
SEEDS=0,1,2

# Workers share these immutable pretraining states. Never let them race to
# create the same checkpoint.
for seed in 0 1 2; do
  test -s "$CKPT/pretrain10k_seed${seed}.pt" || {
    echo "missing $CKPT/pretrain10k_seed${seed}.pt; create/restore all checkpoints first" >&2
    exit 1
  }
done

mkdir -p "$OUT"
for k in 0 1 2 3 4 5 6 7; do
  tag="d4adapter_g${k}"
  for artifact in "results_${tag}.csv" "trajectories_${tag}.csv" "pretrain_${tag}.json"; do
    test ! -e "$OUT/$artifact" || {
      echo "refusing to append duplicate data: $OUT/$artifact already exists" >&2
      exit 1
    }
  done
done

pkill -9 -f "run_alpha_unfrozen_experiment.py.*--wd-d4-adapter" 2>/dev/null || true
tmux kill-session -t d4adapter_workers 2>/dev/null || true
sleep 1

for k in 0 1 2 3 4 5 6 7; do
  alpha="${ALPHAS[$k]}"
  tag="d4adapter_g${k}"
  command="cd /root/parity && CUDA_VISIBLE_DEVICES=$k $P analysis/accessibility/run_alpha_unfrozen_experiment.py --alphas $alpha --seeds $SEEDS --tag $tag --wd-d4-adapter 2>&1 | tee /root/parity/${tag}.log"
  if [ "$k" -eq 0 ]; then
    tmux new-session -d -s d4adapter_workers -n "$tag" "$command"
  else
    tmux new-window -t d4adapter_workers: -n "$tag" "$command"
  fi
done

sleep 5
echo "$(date '+%H:%M:%S') launched 8 matched d4-adapter workers on GPUs 0-7"
tmux list-windows -t d4adapter_workers -F '#{window_name} dead=#{pane_dead} cmd=#{pane_current_command}'
