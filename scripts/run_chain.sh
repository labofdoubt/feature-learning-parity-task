#!/usr/bin/env bash
# Train and analyze a sequence of runs in order.
# Skips training if final.pt already exists; skips analysis if results.pdf exists.
# Usage: bash scripts/run_chain.sh config1.yaml config2.yaml ...
set -euo pipefail
cd /root/parity

for cfg in "$@"; do
    run_dir=$(python3 -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['training']['output_dir'])")
    echo ""
    echo "================================================================"
    echo "[chain] $(date)": Run: "$(basename $run_dir)"
    echo "================================================================"

    if [ -f "$run_dir/checkpoints/final.pt" ]; then
        echo "[chain] Training already done, skipping."
    else
        python3 -m parity_net.train --config "$cfg" 2>&1 | tee "$run_dir/train.log"
    fi

    if [ -f "$run_dir/results.pdf" ]; then
        echo "[chain] Analysis already done, skipping."
    else
        bash scripts/run_analysis.sh "$run_dir"
    fi
done
echo ""
echo "[chain] $(date): All experiments complete."
