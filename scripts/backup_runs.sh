#!/usr/bin/env bash
# Back up training results to an rclone remote (e.g. Google Drive).
#
#   scripts/backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints|--all-checkpoints|--checkpoints-only]
#
# Light tier (default): metrics CSV, configs, plots, analysis CSVs, results.pdf,
# test data and TensorBoard events. A few MB per run – cheap enough to run on a
# timer while training.
#
# --with-checkpoints  also copies each run's final `final.pt`.
# --all-checkpoints   also copies every `step_*.pt` intermediate checkpoint.
# --checkpoints-only  copies ONLY checkpoints (step_*.pt + final.pt), nothing else.
#                     Uses its own lock so multi-GB checkpoint transfers never
#                     block the light tier (analysis/plots/results.pdf).
#
# Env knobs:
#   BACKUP_MIN_AGE=2m   skip files younger than this (watchers set it so a file
#                       mid-write lands next cycle instead of uploading truncated)
#   BACKUP_VERIFY=1     force the post-copy `rclone check` even for the light tier
#                       (checkpoint tiers always verify)
#
# Single-writer discipline: every invocation serializes on one flock, so a
# watcher cycle, a chain's final push and a manual push can never upload the
# same file concurrently (concurrent uploads make Drive objects mutate under
# the reader — "corrupted on transfer", md5 differs between retries).
#
# Always `rclone copy`, never `sync`: local deletions do not propagate upward
# (that is what makes checkpoint cleanup safe). Garbage on Drive must be
# `rclone purge`d explicitly.
#
# Example:
#   scripts/backup_runs.sh runs/my_exp gdrive:parity/runs/my_exp
#   scripts/backup_runs.sh runs/my_exp gdrive:parity/runs/my_exp --all-checkpoints
set -uo pipefail

SRC=${1:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
DST=${2:?usage: backup_runs.sh <runs_dir> <remote:path> [--with-checkpoints]}
shift 2

WITH_CKPT=0
ALL_CKPT=0
CKPT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --with-checkpoints) WITH_CKPT=1 ;;
    --all-checkpoints)  WITH_CKPT=1; ALL_CKPT=1 ;;
    --checkpoints-only) WITH_CKPT=1; ALL_CKPT=1; CKPT_ONLY=1 ;;
  esac
done

# rclone filter rules: first match wins; trailing "- **" makes the include list exhaustive.
# Note: paths are relative to SRC root, so no leading **/ needed.
FILTERS=( --filter "- *.tmp" )
if [ "$CKPT_ONLY" = 1 ]; then
  FILTERS+=(
    --filter "+ checkpoints/step_*.pt"
    --filter "+ checkpoints/final.pt"
  )
else
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
fi
FILTERS+=( --filter "- **" )

# Files still being written land next cycle instead of uploading truncated.
[ -n "${BACKUP_MIN_AGE:-}" ] && FILTERS+=( --min-age "$BACKUP_MIN_AGE" )

# Throughput: 2 transfers x 64M chunks. One 128M-chunked transfer is fast, but
# several parallel 128M uploads through the shared client_id wedge Drive into
# failing every chunk at commit; 2x64M with socket timeouts is the sweet spot.
RCLONE_OPTS=(
  --drive-chunk-size 64M --transfers 2 --checkers 8
  --drive-pacer-min-sleep 10ms
  --timeout 30s --contimeout 15s
  --retries 3 --low-level-retries 10
  --stats 30s --stats-one-line -v
)

tier=light; [ "$WITH_CKPT" = 1 ] && tier=final-checkpoint; [ "$ALL_CKPT" = 1 ] && tier=all-checkpoints; [ "$CKPT_ONLY" = 1 ] && tier=checkpoints-only
echo "[backup] $SRC -> $DST  (tier: $tier)"

# Two locks: the checkpoints-only tier touches a disjoint file set from every
# other tier, so it gets its own lock — bulk checkpoint transfers must never
# starve the light tier. All non-ckpt-only tiers still share one lock.
LOCK=/tmp/parity_backup.lock
[ "$CKPT_ONLY" = 1 ] && LOCK=/tmp/parity_backup_ckpt.lock
exec 9>"$LOCK"
flock 9   # serialize all pushes machine-wide (single-writer discipline)

rclone copy "$SRC" "$DST" "${FILTERS[@]}" "${RCLONE_OPTS[@]}"
rc=$?
echo "[backup] rclone exit=$rc  $(date -Is)"
if [ $rc -ne 0 ]; then
  exit $rc
fi

# Verify, don't trust: "it looked like it finished" and "it failed" are
# indistinguishable without a check. Size-only one-way check; exit code is
# the signal. Checkpoint tiers always verify; light tier only on request.
if [ "$WITH_CKPT" = 1 ] || [ "${BACKUP_VERIFY:-0}" = 1 ]; then
  rclone check "$SRC" "$DST" --one-way --size-only "${FILTERS[@]}" -q
  vrc=$?
  if [ $vrc -eq 0 ]; then
    echo "[backup] verify ok  $(date -Is)"
  else
    echo "[backup] VERIFY FAILED exit=$vrc  $(date -Is)"
  fi
  exit $vrc
fi
exit 0
