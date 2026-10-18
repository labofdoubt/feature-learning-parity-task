#!/usr/bin/env bash
# Rebuild the TensorBoard + Streamlit state of a previous box from Google Drive.
#
# The previous box saved a manifest with scripts/save_tb_streamlit_state.py:
#   <REMOTE_ROOT>/_state/tb_streamlit_state.json   (per-run: tb_logs?, neuron_walsh?,
#   hidden marker?, ...) and <REMOTE_ROOT>/_state/streamlit_order.txt.
# This script downloads, for every run in the manifest, exactly the light artifacts
# the two apps read (config, metrics, tb_logs only where the manifest says they were
# shown, analysis/ incl. neuron_walsh artifacts, plots, results.pdf), recreates the
# .hide_from_streamlit markers and runs/.streamlit_order, and optionally launches
# TensorBoard (port 8081) and Streamlit (port 8501) in the orchestrator tmux session.
# Checkpoints are NOT downloaded unless --final-checkpoints / --all-checkpoints is given.
#
# Usage:
#   bash scripts/restore_tb_streamlit_state.sh gdrive:parity/runs_germany [runs_root] [--final-checkpoints|--all-checkpoints] [--launch]
set -euo pipefail
REMOTE_ROOT=${1:?usage: restore_tb_streamlit_state.sh <REMOTE_ROOT> [runs_root] [--final-checkpoints|--all-checkpoints] [--launch]}
shift
RUNS_ROOT=/root/parity/runs; CKPT=none; LAUNCH=0
for a in "$@"; do case "$a" in
  --final-checkpoints) CKPT=final;; --all-checkpoints) CKPT=all;; --launch) LAUNCH=1;; *) RUNS_ROOT=$a;; esac; done
mkdir -p "$RUNS_ROOT"
STATE=$(mktemp -d)
rclone copy "$REMOTE_ROOT/_state/" "$STATE/"
MANIFEST="$STATE/tb_streamlit_state.json"
[ -f "$MANIFEST" ] || { echo "no manifest at $REMOTE_ROOT/_state/tb_streamlit_state.json" >&2; exit 1; }
python3 - "$MANIFEST" > "$STATE/runs.tsv" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for name, r in m["runs"].items():
    print(name, int(r["tb_logs"]), int(r["hide_from_streamlit"]), sep="\t")
PY
echo "[restore] manifest from $(python3 -c "import json,sys;m=json.load(open('$MANIFEST'));print(m['created'], m['host'])"): $(wc -l < "$STATE/runs.tsv") runs"
while IFS=$'\t' read -r name tb hidden; do
  dest="$RUNS_ROOT/$name"; mkdir -p "$dest"
  FILTERS=( --filter "+ config.yaml" --filter "+ metrics.csv" --filter "+ analysis/**" --filter "+ plots/**" --filter "+ results.pdf" )
  [ "$tb" = 1 ] && FILTERS+=( --filter "+ tb_logs/**" )
  [ "$CKPT" = final ] && FILTERS+=( --filter "+ checkpoints/final.pt" )
  [ "$CKPT" = all ] && FILTERS+=( --filter "+ checkpoints/**" )
  FILTERS+=( --filter "- **" )
  rclone copy "$REMOTE_ROOT/$name" "$dest" "${FILTERS[@]}" --transfers 8 --checkers 16 --drive-chunk-size 64M
  # TensorBoard must not see event files for runs that were hidden from it
  [ "$tb" = 1 ] || rm -rf "$dest/tb_logs"
  if [ "$hidden" = 1 ]; then touch "$dest/.hide_from_streamlit"; else rm -f "$dest/.hide_from_streamlit"; fi
  echo "[restore] $name  (tb_logs=$tb hidden=$hidden)"
done < "$STATE/runs.tsv"
[ -f "$STATE/streamlit_order.txt" ] && cp "$STATE/streamlit_order.txt" "$RUNS_ROOT/.streamlit_order"
cp "$MANIFEST" "$RUNS_ROOT/../state_restored_manifest.json" 2>/dev/null || true
echo "[restore] done: $(ls "$RUNS_ROOT" | wc -l) run dirs under $RUNS_ROOT"
if [ "$LAUNCH" = 1 ]; then
  REPO=$(cd "$RUNS_ROOT/.." && pwd)
  tmux has-session -t orchestrator 2>/dev/null || tmux new-session -d -s orchestrator -n shell
  tmux kill-window -t orchestrator:tb 2>/dev/null || true
  tmux new-window -t orchestrator: -n tb "export PATH=/venv/main/bin:\$PATH; cd $REPO && tensorboard --logdir runs --port 8081 --host 127.0.0.1 2>&1 | tee /tmp/tb.log"
  tmux kill-window -t orchestrator:streamlit 2>/dev/null || true
  tmux new-window -t orchestrator: -n streamlit "export PATH=/venv/main/bin:\$PATH; cd $REPO && streamlit run analysis/neuron_walsh/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true -- --runs-root $RUNS_ROOT 2>&1 | tee /tmp/streamlit.log"
  echo "[restore] launched TensorBoard (8081) and Streamlit (8501); tunnel: ssh -L 8080:localhost:8081 -L 8501:localhost:8501"
fi
