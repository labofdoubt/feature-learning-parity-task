#!/usr/bin/env bash
# Standing sync: push EVERY run under a runs root to Drive on a timer.
#
#   bash scripts/backup_all_watch.sh <interval_s> [tier flags]
#
# Two watchers per machine (run each in its own tmux window):
#   backup_tb   : bash scripts/backup_all_watch.sh 60                  (light tier)
#   backup_ckpt : bash scripts/backup_all_watch.sh 600 --all-checkpoints
#
# Env overrides: RUNS_ROOT (default /root/parity/runs),
#                REMOTE_ROOT (default gdrive:parity/runs).
# Each machine must push to its own remote prefix — mixing prefixes across
# machines makes provenance unrecoverable. Set REMOTE_ROOT accordingly when a
# second box comes up (e.g. gdrive:parity/runs_<name>).
#
# Cycles set BACKUP_MIN_AGE=2m so files mid-write land next cycle rather than
# uploading truncated. All pushes serialize on backup_runs.sh's flock, so these
# watchers, chain-end pushes and manual pushes never race on the same file.
#
# With --all-checkpoints the interval must stay well under
# checkpoint_every x keep_last_checkpoints (if any pruning is active) so no
# intermediate checkpoint dies locally before reaching the remote.
set -uo pipefail

INTERVAL=${1:?usage: backup_all_watch.sh <interval_s> [tier flags]}
shift
EXTRA=( "$@" )
RUNS_ROOT=${RUNS_ROOT:-/root/parity/runs}
REMOTE_ROOT=${REMOTE_ROOT:-gdrive:parity/runs}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "[backup-all] every ${INTERVAL}s: $RUNS_ROOT/* -> $REMOTE_ROOT/*  ${EXTRA[*]:-'(light)'}"
while true; do
  for run_dir in "$RUNS_ROOT"/*/; do
    [ -f "$run_dir/config.yaml" ] || continue
    run_name=$(basename "$run_dir")
    BACKUP_MIN_AGE=2m bash "$HERE/backup_runs.sh" \
      "$run_dir" "$REMOTE_ROOT/$run_name" "${EXTRA[@]:-}" \
      | grep -E '\[backup\]' \
      || echo "[backup-all] $run_name: push failed, retrying next cycle"
  done
  sleep "$INTERVAL"
done
