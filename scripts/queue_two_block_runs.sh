#!/bin/bash
# Run two-block phase-diagram configurations one after another on eight GPUs.
#
#   bash scripts/queue_two_block_runs.sh <queue_file> <combined_fig1_pdf>
#
# Each non-comment line of <queue_file> is "<run_name> [ENV=VALUE ...]"; the
# assignments go to scripts/launch_two_block_phase_workers.sh.  For every run the
# queue launches parent-warm (tags g0..g7) and joint (tags j0..j7) side by side
# into runs/<run_name>/analysis/phase_grid (seed 0, float64 final weights), waits
# for both, draws the figures with plot_two_block_langevin_phase.py (d4/d8 runs also get
# diagnose_two_block_d4_d8.py's Fig. 4 and cell_diagnostics.csv), and rewrites
# <combined_fig1_pdf> with the Fig. 1 of every finished run in queue order.
# Parent-warm cells reuse the warm starts in $WARM_STARTS unless the entry sets
# COPY_WARM_STARTS=0 (then each worker pretrains its own, e.g. for a different
# readout setup).  Runs whose figures already exist are skipped, so rerunning the
# same command resumes the queue.
# runs/<run_name>/config.yaml must exist so the backup watcher syncs the run.
set -euo pipefail

ROOT="${PARITY_ROOT:-/root/parity}"
PYTHON="${PYTHON:-/venv/main/bin/python3}"
WARM_STARTS="${WARM_STARTS:-$ROOT/runs/two_block_phase_langevin_T1e-7_seed0/analysis/phase_grid/pretrain}"
REFERENCE="${REFERENCE-$ROOT/analysis_output/two_block_phase_diagram/results_merged.csv}"
QUEUE="${1:?usage: queue_two_block_runs.sh <queue_file> <combined_fig1_pdf>}"
COMBINED="${2:?usage: queue_two_block_runs.sh <queue_file> <combined_fig1_pdf>}"
cd "$ROOT"

mapfile -t entries < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE")

combine() {   # Fig. 1 of every finished run, in queue order
  local pdfs=() entry fig
  for entry in "${entries[@]}"; do
    fig="$ROOT/runs/${entry%% *}/analysis/figures/fig1_theory_vs_annealed.pdf"
    if [ -f "$fig" ]; then pdfs+=("$fig"); fi
  done
  mkdir -p "$(dirname "$COMBINED")"
  "$PYTHON" - "$COMBINED" "${pdfs[@]}" <<'EOF'
import os
import sys

from pypdf import PdfWriter

target, figures = sys.argv[1], sys.argv[2:]
writer = PdfWriter()
for figure in figures:
    writer.append(figure)
with open(target + ".tmp", "wb") as handle:
    writer.write(handle)
os.replace(target + ".tmp", target)
EOF
  echo "[queue] $(date -u +%H:%M) combined ${#pdfs[@]} figure(s) -> $COMBINED"
}

for entry in "${entries[@]}"; do
  name="${entry%% *}"
  settings=""
  if [ "$entry" != "$name" ]; then settings="${entry#* }"; fi
  run="$ROOT/runs/$name"
  grid="$run/analysis/phase_grid"
  figures="$run/analysis/figures"
  test -f "$run/config.yaml" || { echo "[queue] $run/config.yaml missing" >&2; exit 1; }
  if [ -f "$figures/fig1_theory_vs_annealed.pdf" ]; then
    echo "[queue] $name already finished; skipping"
    continue
  fi
  mkdir -p "$grid/pretrain"
  if [[ " $settings " != *" COPY_WARM_STARTS=0 "* ]]; then
    for warm in "$WARM_STARTS"/parent_warm_g*_N1024_seed0.pt; do
      if [ ! -e "$grid/pretrain/$(basename "$warm")" ]; then cp "$warm" "$grid/pretrain/"; fi
    done
  fi
  echo "[queue] $(date -u +%H:%M) start $name: $settings"
  for pair in parent_warm:g joint:j; do
    # shellcheck disable=SC2086  # settings is a list of ENV=VALUE words
    env $settings OUT="$grid" SEEDS=0 SAVE_CHECKPOINTS=1 CHECKPOINT_DTYPE=float64 \
      PROTOCOLS="${pair%%:*}" TAG_PREFIX="${pair##*:}" SESSION="${name}_${pair##*:}" \
      bash scripts/launch_two_block_phase_workers.sh > /dev/null
  done
  while tmux has-session -t "=${name}_g" 2>/dev/null || tmux has-session -t "=${name}_j" 2>/dev/null; do
    sleep 60
  done
  count=$(find "$grid/checkpoints" -name '*.pt' | wc -l)
  echo "[queue] $(date -u +%H:%M) $name trained: $count/144 checkpoints"
  if [ "$count" -ne 144 ]; then
    echo "[queue] $name is incomplete; stopping the queue" >&2
    exit 1
  fi
  reference_arg=()   # REFERENCE= (empty) skips the comparison, e.g. for the d4/d8 task
  if [ -n "$REFERENCE" ]; then reference_arg=(--reference "$REFERENCE"); fi
  "$PYTHON" -m analysis.accessibility.plot_two_block_langevin_phase --grid-dir "$grid" \
    --out-dir "$figures" "${reference_arg[@]}" > "$run/analysis/figures.log" 2>&1
  if [[ " $settings " == *"run_two_block_d4_d8_phase_diagram.py"* ]]; then   # Fig. 4 + diagnostics
    "$PYTHON" -m analysis.accessibility.diagnose_two_block_d4_d8 --grid-dir "$grid" \
      > "$run/analysis/diagnostics.log" 2>&1
  fi
  combine
done
combine   # also covers a resumed queue whose runs had all finished before
echo "[queue] $(date -u +%H:%M) all runs finished"
