#!/usr/bin/env bash
# Triggered when rho=0.5 training + decode finishes.
# 1. Copies all finished experiment checkpoints to GD.
# 2. Cleans intermediate step_*.pt files for finished experiments.
# 3. Waits for the reuse_star chain (rho=0.7) to complete.
# 4. Runs the hierarchical_symmetric chain.
set -uo pipefail
cd /root/parity

RHO05_LOG="runs/reuse_star_N2048_rho05/analyze_decode.log"
CHAIN_DONE_SENTINEL="runs/reuse_star_N2048_rho07/analyze_decode.log"

# ── Step 0: wait for rho=0.5 decode ─────────────────────────────────────────
echo "[post] $(date): Waiting for rho=0.5 decode to complete..."
until grep -q "Saved:" "$RHO05_LOG" 2>/dev/null; do sleep 15; done
echo "[post] $(date): rho=0.5 decode done."

# ── Step 1: sync all finished experiment checkpoints to GD ──────────────────
echo "[post] $(date): Syncing all finished checkpoints to GD..."
for run_dir in \
    runs/reuse_star_N2048_rho00 \
    runs/reuse_star_N2048_rho03 \
    runs/reuse_star_N2048_rho05; do
    if [ -f "$run_dir/checkpoints/final.pt" ]; then
        run_name=$(basename "$run_dir")
        echo "[post] Syncing $run_name..."
        bash scripts/backup_runs.sh \
            "$run_dir" \
            "gdrive:parity/runs/$run_name" \
            --all-checkpoints \
            2>&1 | tail -5
    fi
done

# ── Step 2: clean intermediate step_*.pt for finished experiments ────────────
echo "[post] $(date): Cleaning intermediate checkpoints..."
for run_dir in \
    runs/reuse_star_N2048_rho00 \
    runs/reuse_star_N2048_rho03 \
    runs/reuse_star_N2048_rho05; do
    if [ -f "$run_dir/checkpoints/final.pt" ]; then
        count=$(ls "$run_dir/checkpoints/step_"*.pt 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "[post] Removing $count intermediate checkpoints from $run_dir"
            rm "$run_dir/checkpoints/step_"*.pt
        fi
    fi
done

# ── Step 3: setup hsym configs ───────────────────────────────────────────────
echo "[post] $(date): Creating hsym configs..."
/usr/bin/python3 scripts/setup_hsym_configs.py

# ── Step 4: wait for reuse_star chain (rho=0.7) to finish ───────────────────
echo "[post] $(date): Waiting for reuse_star rho=0.7 to complete..."
until grep -q "Saved:" "$CHAIN_DONE_SENTINEL" 2>/dev/null; do sleep 30; done
echo "[post] $(date): rho=0.7 done. Cleaning its checkpoints + syncing..."

# Sync rho=0.7 too
bash scripts/backup_runs.sh \
    runs/reuse_star_N2048_rho07 \
    gdrive:parity/runs/reuse_star_N2048_rho07 \
    --all-checkpoints 2>&1 | tail -5
count=$(ls runs/reuse_star_N2048_rho07/checkpoints/step_*.pt 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    echo "[post] Removing $count intermediate checkpoints from rho=0.7"
    rm runs/reuse_star_N2048_rho07/checkpoints/step_*.pt
fi

# ── Step 5: run the hierarchical_symmetric chain ─────────────────────────────
echo "[post] $(date): Starting hierarchical_symmetric chain..."
bash scripts/run_hsym_chain.sh \
    runs/hsym_N2048_rho01/config.yaml \
    runs/hsym_N2048_rho03/config.yaml \
    runs/hsym_N512_rho01/config.yaml \
    runs/hsym_N512_rho03/config.yaml \
    2>&1 | tee runs/hsym_chain.log

echo "[post] $(date): All done."
