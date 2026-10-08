#!/bin/bash
# Launch 8 parallel workers for the matched V2 + d4-readout WD control.
# One alpha is pinned to each GPU; seeds 0,1,2 run sequentially per worker.
# Usage: bash scripts/launch_v2w4wd_workers.sh
#
# This is the accessibility_v2wd experiment with one intentional change:
# phase-2 weight decay is applied to V2 and the d4 readout row W[2]. E, V1,
# and the two d2 readout rows remain decay-free.
set -euo pipefail
P=/venv/main/bin/python
OUT=/root/parity/analysis_output/accessibility_alpha_v2w4wd
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
  tag="v2w4_g${k}"
  for artifact in "results_${tag}.csv" "trajectories_${tag}.csv" "pretrain_${tag}.json"; do
    test ! -e "$OUT/$artifact" || {
      echo "refusing to append duplicate data: $OUT/$artifact already exists" >&2
      exit 1
    }
  done
done

pkill -9 -f "run_alpha_unfrozen_experiment.py.*--wd-block2-readout" 2>/dev/null || true
tmux kill-session -t v2w4_workers 2>/dev/null || true
sleep 1

for k in 0 1 2 3 4 5 6 7; do
  alpha="${ALPHAS[$k]}"
  tag="v2w4_g${k}"
  command="cd /root/parity && CUDA_VISIBLE_DEVICES=$k $P analysis/accessibility/run_alpha_unfrozen_experiment.py --alphas $alpha --seeds $SEEDS --tag $tag --wd-block2-readout 2>&1 | tee /root/parity/${tag}.log"
  if [ "$k" -eq 0 ]; then
    tmux new-session -d -s v2w4_workers -n "$tag" "$command"
  else
    tmux new-window -t v2w4_workers: -n "$tag" "$command"
  fi
done

sleep 5
echo "$(date '+%H:%M:%S') launched 8 matched V2+W4 workers on GPUs 0-7"
tmux list-windows -t v2w4_workers -F '#{window_name} dead=#{pane_dead} cmd=#{pane_current_command}'
