#!/bin/bash
# Launch 4 parallel exp-6 workers on THIS machine, detached from any ssh.
# Usage: bash scripts/launch_v2wd_workers.sh <seed> <machine_tag>
# Idempotent: kills prior exp-6 processes/session first. Runs under tmux so
# workers survive; each worker appends every finished cell to its own CSV.
set -u
SEED="${1:?usage: launch_v2wd_workers.sh <seed> <tag>}"
TAG="${2:?usage: launch_v2wd_workers.sh <seed> <tag>}"
P=/venv/main/bin/python
cd /root/parity

pkill -9 -f run_alpha_unfrozen_experiment 2>/dev/null
tmux kill-session -t workers 2>/dev/null
sleep 1

PAIRS=("0.1,0.3" "0.5,0.75" "1.0,1.5" "2.5,4.0")
tmux new-session -d -s workers -n w0 \
  "cd /root/parity && $P analysis/accessibility/run_alpha_unfrozen_experiment.py --alphas ${PAIRS[0]} --seeds $SEED --tag ${TAG}_w0 --wd-block2-only 2>&1 | tee /root/parity/w0.log"
for k in 1 2 3; do
  tmux new-window -t workers: -n w$k \
    "cd /root/parity && $P analysis/accessibility/run_alpha_unfrozen_experiment.py --alphas ${PAIRS[$k]} --seeds $SEED --tag ${TAG}_w$k --wd-block2-only 2>&1 | tee /root/parity/w$k.log"
done
sleep 5
N=$(pgrep -fc run_alpha_unfrozen_experiment)
echo "$(date '+%H:%M:%S') launched $N workers (seed $SEED, tag $TAG)"
