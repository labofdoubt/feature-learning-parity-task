#!/bin/bash
# 6 parallel exp-7 workers on this machine: (3 seeds) x (2 target halves).
set -u
P=/venv/main/bin/python
H1="0.05,0.08,0.10,0.12,0.135,0.145,0.152"
H2="0.156,0.160,0.168,0.18,0.20,0.25,0.30"
pkill -9 -f run_d8_wd_experiment 2>/dev/null
tmux kill-session -t d8w 2>/dev/null
sleep 1
sleep 1
k=0
for spec in "0 $H1 s0a" "0 $H2 s0b" "1 $H1 s1a" "1 $H2 s1b" "2 $H1 s2a" "2 $H2 s2b"; do
  set -- $spec
  if [ $k -eq 0 ]; then
    tmux new-session -d -s d8w -n w$k \
      "cd /root/parity && $P analysis/accessibility/run_d8_wd_experiment.py --seeds $1 --targets $2 --tag d8_$3 2>&1 | tee /root/parity/d8_$3.log"
  else
    tmux new-window -t d8w: -n w$k \
      "cd /root/parity && $P analysis/accessibility/run_d8_wd_experiment.py --seeds $1 --targets $2 --tag d8_$3 2>&1 | tee /root/parity/d8_$3.log"
  fi
  k=$((k+1))
done
sleep 5
echo "$(date '+%H:%M:%S') d8wd workers: $(pgrep -fc run_d8_wd_experiment)"
