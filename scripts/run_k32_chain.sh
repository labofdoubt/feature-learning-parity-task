#!/usr/bin/env bash
# Chain script for k=32 hierarchical_symmetric experiments.
# Usage: bash scripts/run_k32_chain.sh <config.yaml> [<config.yaml> ...]
# After each run: launches k32 analysis (d4/d8/d16 decode + d4/d8/d16/d32 mode norms)
# and syncs to GD.
set -euo pipefail
cd /root/parity

analyze_and_sync() {
    local run_dir="$1"
    local run_name
    run_name=$(basename "$run_dir")
    echo "[chain] Analyzing $run_name"
    bash scripts/analyze_k32.sh "$run_dir" 65536
    echo "[chain] GD sync handled by standing watchers (tmux session 'backup')"
}

for cfg in "$@"; do
    run_dir=$(/usr/bin/python3 -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['training']['output_dir'])")
    run_name=$(basename "$run_dir")
    echo "================================================================"
    echo "[chain] $(date): Run: $run_name"
    echo "================================================================"
    if [ -f "$run_dir/checkpoints/final.pt" ]; then
        echo "[chain] Training already done, skipping."
    else
        /usr/bin/python3 -m parity_net.train --config "$cfg" 2>&1 | tee "$run_dir/train.log"
    fi
    analyze_and_sync "$run_dir"
done
echo "[chain] $(date): All k32 experiments complete."
