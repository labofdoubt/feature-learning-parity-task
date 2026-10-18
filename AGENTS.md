# AGENTS.md

This repository trains and analyzes parity networks, usually on short-lived vast.ai GPU servers. Read `AGENT_GUIDE.md` before remote-server, experiment, analysis, TensorBoard, Streamlit, Google Drive, or checkpoint-cleanup work; it is the detailed source of truth. Keep that guide and this summary consistent when operational behavior changes.

## Fresh or switched SSH server: standard bootstrap

Treat every new SSH endpoint as an unconfigured machine. Unless the user narrows the request, make the normal reusable setup complete rather than only launching the immediate experiment.

1. Use `/workspace/parity` for the real repo and ensure `/root/parity` is a symlink to it. Configs and scripts hardcode `/root/parity`, while `/workspace` is the persistent disk and the supervisor TensorBoard log root.
2. Copy the local checkout to `/workspace/parity/` with `rsync`, excluding `.git`, caches, `.DS_Store`, egg-info, and notebook checkpoints. Do not treat a previous server's filesystem as current. Keep new/edited experiment configs in the local repo as well as on the server.
3. Use vast.ai's `/venv/main` environment. Interactive shells commonly activate it; `ssh host "cmd"` does not. In remote one-liners use `/venv/main/bin/python3` and `/venv/main/bin/pip`, or export `PATH=/venv/main/bin:$PATH` before scripts that call bare `python`.
4. Install the repo editable plus the routine dependencies up front: `pandas matplotlib tensorboard pypdf pyyaml`, then verify `import parity_net, torch` and run a real matmul on the GPU — `torch.cuda.is_available()` alone passes on V100 boxes whose torch build lacks sm_70 kernels (fix in `AGENT_GUIDE.md` §1). TensorBoard must exist before training or no event files are produced; missing `pypdf` breaks the final analysis merge only after earlier analysis steps finish.
5. Confirm the normal TensorBoard service. With the repo under `/workspace`, vast.ai's supervisor should expose logdir `/workspace` on remote `127.0.0.1:16006`; use an SSH tunnel to view it. Do not use remote port 8080 for a manual service because vast.ai owns it for Jupyter. If manual TensorBoard is genuinely needed, use a free remote port such as 8081 in tmux.
6. Set up Google Drive authentication and the two standing backup watchers described below. Authentication is the user's one-time action on each machine; agents must never print, copy into chat, or otherwise expose `rclone.conf` or its tokens.
7. Streamlit is opt-in, not part of the default bootstrap. Set it up only when the user specifically requests it.

Remote setup should be idempotent: inspect existing sessions/services first, reuse healthy ones, and repair missing pieces. Run long-lived training, watchers, TensorBoard, or Streamlit in named tmux sessions/windows so they survive SSH disconnects.

## Google Drive synchronization

- Training chains must never block on Drive uploads. Do not add `backup_runs.sh` to chain scripts; standing watchers own synchronization.
- Each machine needs its own `REMOTE_ROOT` prefix, such as `gdrive:parity/runs_<machine>`. Never let two servers write same-named runs into one prefix; provenance and TensorBoard series become ambiguous.
- Start a `backup` tmux session with two windows, targeting `backup:` (including the colon): a 60-second light watcher and a 600-second `--checkpoints-only` watcher. The exact commands are in `AGENT_GUIDE.md`. Check that both windows remain alive and inspect their logs for successful cycles.
- The user supplies `~/.config/rclone/rclone.conf` from their own machine. Verify Drive with a write/read/delete round trip, not merely `ls`. A private, non-empty `client_id` and `client_secret` are required; do not reveal their values. If credentials need repair, the user fixes/reconnects them locally and re-copies the config.
- Use only `scripts/backup_runs.sh` / `scripts/backup_all_watch.sh`; do not improvise raw `rclone copy` commands. The scripts enforce filters, minimum age, locks, throughput settings, and verification. Preserve the two disjoint locks and `copy`-not-`sync` behavior.
- The light allowlist includes configs, metrics, TensorBoard logs, plots, `analysis/`, `results.pdf`, and saved train/test data. New artifact types can be silently omitted; update the filter deliberately if a new important output is introduced. `eval_uniform.pt` is intentionally regenerated rather than backed up.
- Before deleting intermediate checkpoints, run an all-checkpoints backup and require successful verification. Never clean a run that lacks `checkpoints/final.pt`. Because uploads use `copy`, deleting locally does not delete Drive copies.
- Deploy edits to a live sync script by copying to a temporary remote filename and moving it over the target. Never overwrite an executing bash script in place.
- Before stopping a box, snapshot the TensorBoard/Streamlit view with `scripts/save_tb_streamlit_state.py` and push it to `<REMOTE_ROOT>/_state/`; rebuild it elsewhere with `scripts/restore_tb_streamlit_state.sh` (details in `AGENT_GUIDE.md`).

## Running experiments and analysis

- Standard k=16 queue: `scripts/run_chain.sh`; hierarchical-symmetric series: `scripts/run_hsym_chain.sh`; k=32/d=64: `scripts/run_k32_chain.sh`. Use one sequential chain per GPU unless the user explicitly chooses otherwise.
- Chains are resumable. Training skips when `checkpoints/final.pt` exists; the standard chain skips analysis when `results.pdf` exists. Re-run the appropriate chain after a crash instead of manually reconstructing completed steps.
- A new experiment needs `runs/<name>/config.yaml`, with `output_dir` matching that run. Copy the config back into the local checkout so server loss does not erase experiment provenance.
- Experiments outside the parity_net trainer (e.g. the two-block phase-diagram sweep) still get `runs/<name>/config.yaml` for provenance, with all outputs under `runs/<name>/analysis/` so the standard light watcher syncs them. `scripts/launch_two_block_phase_workers.sh` saves per-cell weights only with `SAVE_CHECKPOINTS=1`.
- Queue follow-on chains from another tmux window by waiting for the current chain's distinctive completion line. Do not make the current chain wait for backup.
- “Forward composition analysis” usually means the full per-run pipeline `bash scripts/run_analysis.sh runs/<name>`, producing `runs/<name>/results.pdf`. Run it with `/venv/main/bin` on `PATH` and the repo installed editable. The similarly named `analysis/forward_composition/run_analysis.py` is a narrower Walsh-composition probe; when ambiguous, produce the `results.pdf` pipeline first.
- Streamlit's neuron-Walsh browser requires artifacts from `python -m analysis.neuron_walsh.compute`. Bind Streamlit to `127.0.0.1` and access it through an SSH tunnel. If regenerated artifacts look stale, restart the app to clear its in-process cache.

## Scientific and codebase invariants

- Non-uniform training runs are analyzed on `eval_uniform.pt`, not their training-distribution sample. Analysis scripts should fall back to `test_data.pt` only when the uniform set is absent.
- Before truncating an eval set to `--num-samples`, shuffle it with a fixed seed (the existing scripts use 42). Prefix slicing an ordered exhaustive parity set creates severe spurious Walsh/degree artifacts; do not reintroduce that historical bug.
- For `relevant_dim <= 20`, uniform eval enumerates relevant-bit configurations with repeated irrelevant-bit draws; above 20 it uses Monte Carlo. At `input_dim > 62`, packed int64 exclusion keys are unavailable, but collisions at d=64 are negligible.
- Balanced-tree target names are `d2_i`, `d4_i`, ..., `d{k}_0`. `exclude_targets` removes outputs and loss terms; `num_reuse_targets` has no effect unless `task_type: reuse_star`.
- Preserve muP scaling when changing width: hidden variance `1/N`, readout variance `1/N^2`, and embedding variance `1/input_dim`.
- `train_samples` is a fixed pool reused across epochs; `test_samples` is for training-time monitoring; scientific analysis uses the uniform eval set.
- Prefer adding focused tests under `tests/` for data-distribution or analysis changes. Run the relevant tests before launching expensive remote experiments.

## Safety and hygiene

- Never expose SSH keys, OAuth tokens, `rclone.conf`, private client secrets, or service credentials in chat, logs, commits, or patches.
- Inspect remote paths and active tmux processes before destructive operations. Do not delete runs or checkpoints merely to free space without verified Drive copies.
- Preserve user results and unrelated working-tree changes. Avoid copying `.git` to compute boxes or committing generated runs, checkpoints, logs, and large analysis artifacts unless explicitly requested.
