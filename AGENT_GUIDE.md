# Agent guide: server setup & operations

How to set up this repo on a fresh GPU server (vast.ai), run experiments and analysis, browse results (TensorBoard, Streamlit), and keep everything synced to Google Drive. Assumes SSH access like `ssh -p <PORT> root@<IP>`. (Formerly `SERVER_GUIDE.md`.)

## 1. Repo setup on a vast.ai instance

Layout convention: the repo lives at `/workspace/parity`, and `/root/parity` is a **symlink** to it. All configs (`output_dir: /root/parity/runs/...`) and chain scripts (`cd /root/parity`) hardcode `/root/parity`, while `/workspace` is both the large persistent disk and the default TensorBoard logdir — the symlink satisfies both.

```bash
ssh -p <PORT> root@<IP> "mkdir -p /workspace/parity && ln -sfn /workspace/parity /root/parity"

# from the local machine (repo root):
rsync -az -e "ssh -p <PORT>" \
  --exclude '.git' --exclude '__pycache__' --exclude '.DS_Store' \
  --exclude '*.egg-info' --exclude '.ipynb_checkpoints' \
  ./ root@<IP>:/workspace/parity/
```

Python: vast.ai images ship torch+CUDA in `/venv/main`, which auto-activates in **interactive** shells (tmux panes) but **not** in `ssh host "cmd"` one-liners — there `python3` is the bare system python. Run everything inside tmux, or call `/venv/main/bin/python3` explicitly.

```bash
/venv/main/bin/pip install pandas matplotlib tensorboard pypdf pyyaml
/venv/main/bin/pip install -e /workspace/parity
/venv/main/bin/python3 -c "import parity_net, torch; print(torch.cuda.is_available())"
```

Caveats:
- **Verify CUDA with a real kernel, not `torch.cuda.is_available()`** — on Volta boxes (Tesla V100, compute capability 7.0) the image's `torch 2.11.0+cu128` has no sm_70 kernels: `is_available()` is True but every GPU op fails with "no kernel image is available". Fix: `/venv/main/bin/pip install "torch==2.11.0+cu126" "torchvision==0.26.0+cu126" "torchaudio==2.11.0+cu126" --index-url https://download.pytorch.org/whl/cu126` — pin the `+cu126` local version, or pip treats `torch==2.11.0` as already satisfied by the cu128 build. Check `"sm_70" in torch.cuda.get_arch_list()` and run a float64 matmul on each GPU.
- **tensorboard must be installed before training starts** — if missing, training runs fine but silently writes no TB events (no error, no `tb_logs/`).
- **pypdf** is only needed by `combine_plots.py`, i.e. the analysis pipeline fails at its *last* step without it. Install it up front.

## 2. Running experiments

One GPU → runs go sequentially through chain scripts, launched in a tmux session (e.g. `tmux new -s orchestrator`):

| Script | Analysis it runs | Use for |
|---|---|---|
| `scripts/run_chain.sh cfg1.yaml cfg2.yaml ...` | full pipeline via `run_analysis.sh`: curves, PCA, embedding Gram, parity-mode Grams + cross-block alignment, d4/d8/d16 decode → merges 12 figures into `results.pdf` | standard k=16 runs |
| `scripts/run_hsym_chain.sh cfg...` | d16 partition decode only | hierarchical_symmetric series |
| `scripts/run_k32_chain.sh cfg...` | `analyze_k32.sh`: mode norms d4/8/16/32 + d4/d8/d16 decode (no d32 decode — partition count intractable) | k=32 / d=64 runs |

**Terminology — "forward composition analysis"**: when the user asks for the *forward composition analysis* (or "the script that produces `results.pdf`"), they mean the full per-run pipeline `bash scripts/run_analysis.sh runs/<name>` — curves, PCA, embedding Gram, parity-mode Grams + cross-block alignment, d4/d8/d16 partition decode, merged by `combine_plots.py` into **`runs/<name>/results.pdf`**. It needs the repo importable (`/venv/main/bin/pip install -e /workspace/parity`, or `PYTHONPATH=/root/parity`), `pypdf` for the final merge, and `/venv/main/bin` on `PATH` — its internal steps call bare `python`, so from a script or `ssh` one-liner run it as `export PATH=/venv/main/bin:$PATH && bash scripts/run_analysis.sh runs/<name>` (both failure modes have bitten: missing pypdf kills step 6/6, missing PATH kills step 1/6). The similarly-named module `analysis/forward_composition/run_analysis.py --runs <r1> <r2> ...` is a *different, narrower* probe: layer-wise Walsh power H_l for the d16 target's canonical d8 children vs. alternative 8+8 partitions, writing `results.csv` + `h_by_layer.pdf` + `composition_comparison.pdf` under `analysis_output/forward_composition/` (it uploads itself to Drive at the end). Both run in ~minutes on a final checkpoint — when the request is ambiguous, produce the `results.pdf` pipeline first.

Chains do **not** sync to Drive — the standing watchers (§5) handle that in the background, so a chain's completion line prints right after analysis and the next queued run takes the GPU immediately. Never add a blocking `backup_runs.sh` call back into a chain.

All chains are **resumable**: training is skipped if `checkpoints/final.pt` exists; `run_chain.sh` skips analysis if `results.pdf` exists. Re-running after a crash is safe.

Each chain prints a distinctive completion line (`All hsym experiments complete.` / `All k32 experiments complete.` / `All experiments complete.`). To queue work behind a running chain, run a watcher loop in another tmux window that greps the chain's log for that line, then launches the next chain.

New experiment = new dir `runs/<name>/config.yaml` (copy a sibling config, edit `output_dir` to match the dir name). Keep a copy of the config in the local repo too.

Experiments that don't use the parity_net trainer (e.g. the two-block phase-diagram sweep, `analysis/accessibility/run_two_block_phase_diagram.py`) also get a run folder: `runs/<name>/config.yaml` records the sweep and where its inputs came from (header: "NOT a parity_net training config"), and every output goes under `runs/<name>/analysis/`. The light watcher's `analysis/**` rule then syncs all of it (including weights) with no extra script — see `runs/two_block_phase_pw_seed0/`. The two-block launcher keeps per-cell final weights only with `SAVE_CHECKPOINTS=1` (add `CHECKPOINT_DTYPE=float64` for exact weights); its default saves none. `TEMPERATURE_INITIAL=<T0>` (plus `TEMPERATURE_HOLD_STEPS`, `TEMPERATURE_ANNEAL_STEPS`, `TEMPERATURE_FINAL_RATIO`, `NOISE_SEED`) turns the final phase into Langevin SGD with annealing: momentum 0, exactly `MAX_STEPS` updates, noisy cell ids suffixed `_T<T0>_n<seed>`, and parent pretraining still at momentum 0.9. `TAG_PREFIX=j` lets a second launch (e.g. `PROTOCOLS=joint`) share the same `OUT` and GPUs; see `runs/two_block_phase_langevin_T1e-7_seed0/` and `plot_two_block_langevin_phase.py` for the figures. `scripts/queue_two_block_runs.sh <queue_file> <combined_fig1.pdf>` runs several such grids back to back (parent-warm and joint side by side on the eight GPUs, figures via `plot_two_block_langevin_phase.py`, one combined Fig.-1 PDF, resumable); each queue line is `<run_name> ENV=VALUE ...` with launcher variables such as `MOMENTUM`, `LR`, `READOUT_INIT=fan_in`, `READOUT_LR`, `MAX_STEPS`=`MIN_STEPS` (no early stopping) and `COPY_WARM_STARTS=0`. `TRAINER=analysis/accessibility/run_two_block_d4_d8_phase_diagram.py` with an 8-value `RATIO_LIST` switches to the d4-parent/d8-target task (set `REFERENCE=` empty for the queue); its width parametrization is `READOUT_INIT=fan_in` (default: readouts Var 1/N, every rate `LR`), `READOUT_INIT=mup` (readouts Var 1/N², readout rate `LR`/N, as in the d2/d4 runs) or `READOUT_INIT=base BASE_WIDTH=<N0>` (readouts Var N0/N², readout rate `LR`·N0/N, embedding rate `LR`·N/N0: muP relative to a base width N0), with `READOUT_LR` / `EMBEDDING_LR` as explicit overrides; `python -m analysis.accessibility.diagnose_two_block_d4_d8 --grid-dir <grid>` then adds the block-split, route-read and theory-free-energy diagnostics (`runs/queue_two_block_d4d8.txt`, `reports/two_block_d4_d8/`).

## 3. TensorBoard

vast.ai already runs TB via supervisor: `tensorboard --port 16006 --logdir ${TENSORBOARD_LOG_DIR:-/workspace}`, bound to `127.0.0.1:16006` (caddy proxies it on 6006; port 8080 is Jupyter). Because the repo is under `/workspace`, every run's `tb_logs/` is picked up automatically — nothing to configure.

View locally: add `-L 16006:localhost:16006` to the ssh command, open `http://localhost:16006`.

If the repo is **not** under `/workspace` (so the supervisor TB never sees it), run TB manually in a tmux window instead — and pick a free port: **8080 is vast's supervisor-managed Jupyter** (killing it just respawns it), so use e.g. 8081:
```bash
tmux new-window -t orchestrator: -n tb 'export PATH=/venv/main/bin:$PATH; cd /root/parity && tensorboard --logdir runs --port 8081 --host 127.0.0.1 2>&1 | tee /tmp/tb.log'
# locally: ssh ... -L 8080:localhost:8081  → open http://localhost:8080
```

## 4. Streamlit app (neuron-Walsh browser)

`analysis/neuron_walsh/app.py` is the interactive browser over per-checkpoint **neuron-Walsh artifacts**: overview heatmap of per-neuron Walsh coefficients (sortable by purity/coverage/dominant mode, low-variance filter), mode view (top neurons + cumulative power per mode), and per-neuron training dynamics across checkpoints.

**Artifacts first** — the app only lists runs that have `runs/<name>/analysis/neuron_walsh/step_*.pt`; generate them with:
```bash
/venv/main/bin/python -m analysis.neuron_walsh.compute --run <name> --checkpoints all
```
(~1 min for 50 checkpoints of an N=2048/L=4 run on a 16 GB GPU; safe to run alongside training — it needs <1 GB. `--checkpoints final` for just the last one.)

**Setup & launch** (deps: `pip install streamlit plotly` into `/venv/main`):
```bash
tmux new-window -t orchestrator: -n streamlit 'export PATH=/venv/main/bin:$PATH; cd /root/parity && streamlit run analysis/neuron_walsh/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true -- --runs-root /root/parity/runs 2>&1 | tee /tmp/streamlit.log'
# locally: ssh ... -L 8501:localhost:8501  → open http://localhost:8501
```
Bound to `127.0.0.1` deliberately — access is through the SSH tunnel only. The run list refreshes on a 60 s TTL, so a freshly analyzed run appears on browser refresh without restarting the app. Per-checkpoint artifacts are cached in-process (`max_entries=8`); if you *regenerate* artifacts for a run someone already has open, restart the app (kill the tmux window, relaunch) to drop stale caches. If the page stops loading locally, the app is usually fine — it's the SSH tunnel that died; re-establish it (a self-reconnecting loop `while true; do ssh -N -L 8501:127.0.0.1:8501 ...; sleep 5; done` survives flaky links).

### Preserving the TensorBoard / Streamlit view across boxes

The apps read only light artifacts (`tb_logs/`, `analysis/` incl. `neuron_walsh/`,
`metrics.csv`, `config.yaml`), which the light watcher already copies to Drive. What
the watcher does NOT carry is the *view state*: the `.hide_from_streamlit` markers,
`runs/.streamlit_order`, and which runs had their event logs deleted. Before stopping
a box, snapshot it and push the snapshot next to the runs:

```bash
python3 scripts/save_tb_streamlit_state.py runs state/tb_streamlit_state.json
cp runs/.streamlit_order state/streamlit_order.txt
rclone copy state gdrive:parity/runs_<machine>/_state/
```

On a new box (after the normal bootstrap), rebuild exactly that view from Drive:

```bash
bash scripts/restore_tb_streamlit_state.sh gdrive:parity/runs_<machine> /root/parity/runs --launch
```

It downloads the light artifacts per run (event logs only for runs that were visible),
recreates markers and the order file, and launches TensorBoard on 8081 and Streamlit on
8501. Add `--final-checkpoints` / `--all-checkpoints` if analyses need to be re-run.
Streamlit also needs `pip install streamlit plotly`.

## 5. Google Drive sync

Two layers: auth (once, by the user) and standing watchers (per machine). Chains do **not** push — the watchers own all syncing, so the GPU hands off to the next queued run the moment analysis finishes instead of blocking on a multi-GB upload. Everything funnels through `scripts/backup_runs.sh`, which wraps `rclone copy` with an allowlist filter. Tiers:
- default (light): `metrics.csv`, `config.yaml`, `tb_logs/`, `plots/`, `analysis/`, `results.pdf`, `test_data.pt`, `train_data.pt`
- `--with-checkpoints`: + `checkpoints/final.pt`
- `--all-checkpoints`: + every `checkpoints/step_*.pt`
- `--checkpoints-only`: ONLY `checkpoints/*` — the bulk tier the ckpt watcher uses; disjoint from the light tier by construction

**Auth (user does this, not an agent)** — `rclone.conf` holds a live OAuth refresh token; it must never pass through a conversation. The user scps it from their own machine: `scp -P <PORT> ~/.config/rclone/rclone.conf root@<IP>:/root/.config/rclone/`. Then verify with a **write round-trip**, not a listing (read-works-write-doesn't is a real failure mode): `rclone copy` a test file up, `rclone cat` it back, `rclone delete` + `rmdir`. The remote should carry the user's **private `client_id`/`client_secret`** (set up 2026-09-15; check that the fields have **values** — present-but-empty means rclone's shared client, which throttles under load and is slated for retirement during 2026). If a box shows them empty, the fix happens on the user's machine (Cloud Console OAuth client + `rclone config update` + `rclone config reconnect`, see weight-sparsity's `docs/vastai-agent-guide.md` §2), then re-scp the conf and restart the watchers. Private client removes API-quota stalls; raw bandwidth is box-dependent — for bulk downloads use parallelism (`--transfers 8 --checkers 16` measured ~54 MB/s vs ~10 MB/s single-stream).

**Standing watchers (per machine)** — the whole sync layer is two tmux windows in a `backup` session (note the `backup:` target form — bare `-t backup` can silently fail to create the window):
```bash
tmux new-session -d -s backup -n backup_tb   "bash scripts/backup_all_watch.sh 60  2>&1 | tee -a backup_tb.log"
tmux new-window  -t backup:   -n backup_ckpt "bash scripts/backup_all_watch.sh 600 --checkpoints-only 2>&1 | tee -a backup_ckpt.log"
```
`backup_all_watch.sh` loops over every `runs/*/` with a `config.yaml` and pushes each to `$REMOTE_ROOT/<run_name>`. Expected latency to Drive: light artifacts (plots, analysis, `results.pdf`, metrics, TB events) ≤ ~1 min; new checkpoints ≤ 10 min + transfer time — and since checkpoints upload *while training runs*, most of a run's data is already on Drive by the time it finishes. Pure bash+rclone, so the windows don't need the venv. This is the **only** sanctioned way to sync: don't run ad-hoc `rclone copy` by hand (bypasses the filters, min-age and verify); if you need a file on Drive *now*, call `backup_runs.sh` directly — the flock makes that safe against the watchers.

No separate analysis watcher is needed in **this** repo because `analysis/`, `plots/` and `results.pdf` are already in the light tier — but coverage is by filter, and filters lie in wait: any future large artifact that doesn't match (e.g. `.npy` probe tensors) silently gets no backup until a filter or watcher is added. Each machine pushes to its **own remote prefix** (`REMOTE_ROOT`, default `gdrive:parity/runs`); when a second box comes up, give it `runs_<name>` — mixing prefixes makes provenance unrecoverable, and same-named TB runs from two machines would silently merge into one interleaved series.

**Rules encoded in backup_runs.sh** (don't undo them):
- **Single-writer, two locks**: pushes serialize on a machine-wide `flock` — but there are two, because the file sets are disjoint: `/tmp/parity_backup.lock` for every tier that touches light files, and `/tmp/parity_backup_ckpt.lock` for `--checkpoints-only`. A single lock for everything failed operationally: a multi-GB checkpoint transfer held it for hours and **starved the light tier**, so fresh `results.pdf`s never reached Drive. Never give two tiers overlapping filters unless they share a lock — two concurrent uploads of the same file make Drive objects mutate under a reader ("corrupted on transfer", md5s differing between retries).
- **`copy`, never `sync`**: local deletions don't propagate upward (that's what makes checkpoint cleanup safe) — but it cuts both ways: garbage on Drive must be `rclone purge`d explicitly or it haunts every listing and restore.
- **Throughput**: 2 transfers × 64M chunks with socket timeouts. A single 128M-chunked transfer is faster than the 1.4 MB/s default, but several parallel 128M uploads through the shared client_id wedge Drive into failing every chunk at commit.
- **Mid-write files**: watcher cycles set `BACKUP_MIN_AGE=2m` so a file still being written lands next cycle instead of uploading truncated. Manual one-shot pushes of finished files can skip it.
- **Verify, don't trust**: checkpoint tiers end with `rclone check --one-way --size-only` and the exit code is the signal (`BACKUP_VERIFY=1` forces it for the light tier). "It looked like it finished" and "it failed" are otherwise indistinguishable.
- **Editing sync scripts on a live box**: deploy with `scp` to a temp name + `mv` over the target (new inode). Overwriting a script in place while a bash process is executing it makes that bash re-read the new bytes at its old file offset — this produced garbage execution twice before becoming a rule.

**Other quirks**: Drive permits duplicate directory names — count recursively (`rclone lsf -R`) and use `rclone dedupe --dedupe-mode list <path>` before concluding a run is missing. `eval_uniform.pt` is deliberately not backed up — it's regenerated deterministically from the config seed. With `--all-checkpoints`, the watcher interval must stay well under `checkpoint_every × keep_last_checkpoints` if any local pruning is active, or intermediates die before reaching Drive.

Disk cleanup pattern (intermediate checkpoints are ~130MB each, 50 per run): first `backup_runs.sh ... --all-checkpoints` (which now verifies), then `rm runs/<name>/checkpoints/step_*.pt`. Never clean a run without `final.pt` (still training).

## 6. Codebase notes & caveats

**Data distributions** — `task.data_distribution` selects the training distribution: `uniform` (default), `hierarchical`, `hierarchical_degree2`, `hierarchical_symmetric`, `hierarchical_symmetric_degree2`, `walsh_single_mode` (`p(x) = 2^-d (1 + rho·chi_A)`, mode A from `data_walsh_mode`, zero-based indices, rho ∈ [−1,1]), `walsh_clusters` (latent mixture of single-mode clusters from `data_walsh_clusters`, per-cluster rho; scalar `data_rho` is ignored). `data_rho` sets correlation strength for the hierarchical variants (ignored for uniform); they require power-of-2 `relevant_dim`, the walsh variants don't.

**Two eval datasets for non-uniform runs** — when `data_rho > 0` and distribution ≠ uniform, training saves both:
- `eval_uniform.pt` — uniform-distribution eval set, the default input for all analysis scripts (they fall back to `test_data.pt` if absent). Measuring Walsh/parity statistics on *uniform* data is what makes mode norms interpretable.
- `eval_nonuniform.pt` — same size, sampled from the training distribution with the same rho, for on-distribution metrics.

**eval_uniform.pt structure** — for `relevant_dim ≤ 20`: all `2^k` relevant-bit configs × `eval_noise_repeats` fresh irrelevant-bit draws, **shuffled at generation** (`randperm`). Historical bug to not reintroduce: the set was once laid out in config order, so any prefix subset (`x[:N]`) covered only low-index configs (e.g. all with bit 15 = −1), producing spurious degree-2 artifacts in decode analysis. Defense in depth: `analyze_decode.py`, `analyze_parity_modes.py`, `analyze_pca.py` each shuffle with a fixed seed (42) before truncating to `--num-samples`. Keep that pattern in any new analysis script. For `relevant_dim > 20` the set falls back to Monte Carlo (`eval_noise_repeats · 2^17` uniform samples) — statistically fine, not exhaustive.

**input_dim > 62** — inputs can't be packed into int64 keys, so `make_dataset` falls back to random (possibly duplicate) sampling and `exclusion_keys` returns empty (no train/test dedup). Harmless at d=64: collision probability is negligible.

**Targets** — `tree_parity_specs` builds the balanced-tree parities over bits 0..k−1: names `d2_0..d2_{k/2−1}`, `d4_i`, `d8_i`, ..., `d{k}_0`. `task.exclude_targets` removes targets from the outputs/loss by name (`"d8_1"` = bits 8–15) or whole degree (`"d8"`). Model `output_dim` adjusts automatically.

**Sample-size semantics** — `train_samples` is a *fixed training pool* drawn once (from the configured distribution) and reshuffled each epoch; `test_samples` only feeds in-training monitoring. The analysis set is always `eval_uniform.pt`. 100k for both is fine at any k since learning is from the distribution, not enumeration.

**muP init convention** — `hidden_weight_variance = 1/N`, `readout_weight_variance = 1/N²`, `embedding_weight_variance = 1/input_dim`. Changing N means changing all three consistently (e.g. N=512 → 1/512, 1/512², 0.03125 stays if input_dim=32).

**num_reuse_targets** — only read when `task_type: reuse_star`; a leftover value in a `tree_parity` config is a no-op.
