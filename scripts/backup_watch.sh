#!/usr/bin/env bash
# Periodic backup loop – run in its own tmux pane while training.
#
#   tmux new -d -s backup "bash scripts/backup_watch.sh \
#       runs/my_exp gdrive:parity/runs 600 --all-checkpoints"
#
# Default interval: 600 s (10 min).  With --all-checkpoints the interval must
# stay well under checkpoint_every × keep_last_checkpoints so that no
# intermediate checkpoint is pruned locally before it reaches the remote.
set -uo pipefail

SRC=${1:?usage: backup_watch.sh <runs_dir> <remote:path> [interval_s] [extra flags]}
DST=${2:?usage: backup_watch.sh <runs_dir> <remote:path> [interval_s] [extra flags]}
INTERVAL=${3:-600}
shift $(( $# < 3 ? $# : 3 ))
EXTRA=( "$@" )
HERE=$(cd "$(dirname "$0")" && pwd)

echo "[backup-watch] every ${INTERVAL}s: $SRC -> $DST ${EXTRA[*]:-}"
while true; do
  bash "$HERE/backup_runs.sh" "$SRC" "$DST" "${EXTRA[@]:-}" \
    || echo "[backup-watch] rclone failed, retrying next cycle"
  sleep "$INTERVAL"
done
