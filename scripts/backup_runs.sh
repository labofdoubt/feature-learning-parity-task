#!/usr/bin/env bash
# Back up training results to an rclone remote (e.g. Google Drive).
#
#   scripts/backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints|--all-checkpoints]
#
# Light tier (default): metrics CSV, configs, test data and TensorBoard events.
# A few MB per run – cheap enough to run on a timer while training.
#
# --with-checkpoints  also copies each run's final `final.pt`.
# --all-checkpoints   also copies every `step_*.pt` intermediate checkpoint.
#
# Example:
#   scripts/backup_runs.sh runs/my_exp gdrive:parity/runs
#   scripts/backup_runs.sh runs/my_exp gdrive:parity/runs --with-checkpoints
set -uo pipefail

SRC=${1:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
DST=${2:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
shift 2

WITH_CKPT=0
ALL_CKPT=0
for arg in "$@"; do
  case "$arg" in
    --with-checkpoints) WITH_CKPT=1 ;;
    --all-checkpoints)  WITH_CKPT=1; ALL_CKPT=1 ;;
  esac
done

# rclone filter rules: first match wins; trailing "- **" makes the include list exhaustive.
# Note: paths are relative to SRC root, so no leading **/ needed.
FILTERS=( --filter "- *.tmp" )
[ "$ALL_CKPT" = 1 ] && FILTERS+=( --filter "+ checkpoints/step_*.pt" ) \
                    || FILTERS+=( --filter "- checkpoints/step_*.pt" )
FILTERS+=(
  --filter "+ metrics.csv"
  --filter "+ config.yaml"
  --filter "+ tb_logs/**"
  --filter "+ plots/**"
  --filter "+ analysis/**"
  --filter "+ results.pdf"
  --filter "+ test_data.pt"
  --filter "+ train_data.pt"
)
[ "$WITH_CKPT" = 1 ] && FILTERS+=( --filter "+ checkpoints/final.pt" )
FILTERS+=( --filter "- **" )

tier=light; [ "$WITH_CKPT" = 1 ] && tier=final-checkpoint; [ "$ALL_CKPT" = 1 ] && tier=all-checkpoints
echo "[backup] $SRC -> $DST  (tier: $tier)"
rclone copy "$SRC" "$DST" "${FILTERS[@]}" \
    --drive-chunk-size 128M --drive-pacer-min-sleep 10ms \
    --transfers 4 --checkers 16 --retries 3 --low-level-retries 10 \
    --stats 30s --stats-one-line -v
rc=$?
echo "[backup] rclone exit=$rc  $(date -Is)"
exit $rc
