# parity-net

Minimal PyTorch library for training residual networks on the binary-tree
k-parity staircase described in `MOTIVATION.md`.

By default the task uses 32-dimensional `{-1,+1}` inputs. The first 16
coordinates define 15 outputs: eight degree-2 parities, four degree-4 parities,
two degree-8 parities, and one degree-16 parity. The remaining coordinates are
noise. The task config can change the input length, the number of leading
relevant positions, and which binary-tree parity targets are included.

## Install

```bash
pip install -e .
```

## Train

```bash
parity-train --config parity_net/default_config.yaml
```

The default run samples a fresh random training batch at every optimizer step
and keeps one fixed held-out test set for evaluation. It writes:

- `runs/parity/config.yaml`
- `runs/parity/metrics.csv`
- `runs/parity/checkpoints/step_*.pt`
- `runs/parity/checkpoints/final.pt`

## Analyze

```bash
parity-analyze \
  --checkpoint runs/parity/checkpoints/final.pt \
  --output-dir runs/parity/analysis \
  --intervention-layer 2 \
  --keep-pcs 50
```

This writes weight variances, baseline per-degree MSE, PCA ranks needed for
90% and 99% variance at each layer, and per-degree MSE after the PCA
intervention.

## Config Notes

`TaskConfig` controls the parity task. `input_dim` is the full sequence length,
and `relevant_dim` is the number of leading coordinates used in the binary-tree
parity targets. Both must be even. Included degree-`k` targets require
`relevant_dim` to be divisible by `k`; therefore a d2-only task works for any
even `relevant_dim`, while the full binary-tree task still needs a compatible
power-of-two `relevant_dim`. Use `exclude_targets` to remove targets from both
the readout and the MSE, for example `["d8", "d16"]`, `["d8_*", "d16_*"]`, or
exact names such as `["d4_2"]`. For backward compatibility, old configs with
`input_dim` and `relevant_dim` only under `model` still load.

`ModelConfig` controls the network shape: width `N`, depth
`L`, readout barrier toggle, embedding scale, residual-block form,
hidden-layer initialization variance, and readout initialization variance.
Supported activations are `relu`, `gelu`, `tanh`, `silu`, `half-tanh`, and
`square`; `half-tanh` is `relu(tanh(x))` and `square` is `x^2`. `square` is the
only unbounded one, which makes it prone to diverging at learning rates the
others tolerate — lower `lr` or `embedding_weight_variance` if it produces NaNs.
It is a natural fit for this task because a product of two bits is exactly
representable by one layer, via `ab = ((a+b)^2 - (a-b)^2) / 4`.
`activation_scale` multiplies the activation output by a fixed constant `c`, so
the block computes `x + W (c phi(Vx))`. It is not a learned parameter and adds
nothing to the state dict, so checkpoints load across different values. At the
default `1.0` the bare activation module is used and the module tree is
identical to before. It applies to whichever activation is selected, not only
`half-tanh`.
Set `use_post_activation_linear` to `true` to use residual blocks of the form
`x + W phi(Vx)`; otherwise blocks use `x + phi(Vx)`.
Set `use_skip_connections` to `false` to drop every skip connection, so a block
computes `W phi(Vx)` rather than `x + W phi(Vx)` and the stack becomes a plain
deep MLP. Under the attention architecture it removes the residual around the
sequence-mixing sub-layer as well, leaving `mlp(mixing(x))`. Note this also
removes the only path that carries the input forward when a block's weights are
small, so deep stacks are much harder to train without it.
The initialization variance fields are literal per-entry variances:
`embedding_weight_variance` rescales the frozen orthonormal embedding to have
approximately that per-entry variance; omit it or set it to `null` to keep the
unscaled QR embedding. Set `freeze_embedding` to `false` to train the embedding
weights. `hidden_weight_variance` initializes hidden weights with
`std = sqrt(hidden_weight_variance)`, and `readout_weight_variance`
initializes readout weights with `std = sqrt(readout_weight_variance)`.
`post_activation_linear_variance` overrides the initialization variance of the
post-activation linear `W` in `x + W phi(Vx)`; leave it `null` and that layer
reuses `hidden_weight_variance`. Only `W` is affected — the pre-activation `V`
always follows `hidden_weight_variance` — and the field is ignored entirely when
`use_post_activation_linear` is `false`. It applies to both architectures, since
the attention model's blocks reuse the same MLP. Note that `W` is a hidden weight
for muP purposes, so a width-independent value here breaks the `1/fan_in` rule.

`TrainingConfig` controls `num_steps`, fresh-batch size, fixed held-out test
set size, optimizer, checkpointing, and the readout barrier parameters.
`validate_every` runs a full test evaluation and appends a row to `metrics.csv`;
`checkpoint_every` writes a checkpoint. When both fall on the same step the
evaluation runs once and is shared, so coinciding schedules cost nothing extra.
`progress_every` prints a cheap heartbeat line with the current batch loss,
broken down by individual parity target, and runs no evaluation at all; it
defaults to `0`, which disables it, and it never adds a row to `metrics.csv`. `validate_every` was previously called `log_every`; configs
and checkpoints written under the old name still load. The
optimizer supports optional per-group learning rates: `lr_embedding`,
`lr_hidden`, and `lr_readout`; any omitted value falls back to `lr`. It also
supports optional per-group weight decays: `wd_embedding`, `wd_hidden`, and
`wd_readout`; any omitted value falls back to `weight_decay`. Embedding-group
optimizer settings are ignored when `freeze_embedding` leaves the embedding frozen.
barrier coefficient `c` lives in the training config because it is a loss
regularizer. If `barrier_c` is omitted, training uses `7 / N`, matching the
mean-field-scale box from `MOTIVATION.md`.
Training saves the exact held-out test set to `test_data.pt` in the run
directory and rejects any fresh training batch samples that match that saved
test set.

`teacher_forcing_ratio` (default `1.0`) is the probability, drawn per step, that the
answer positions are filled with the true parities. Below `1.0` the model instead
generates the sequence autoregressively first, under `no_grad`, and trains on that
self-produced context; the loss targets remain the true parities either way, so this
is scheduled sampling rather than self-distillation. `0.0` never teacher-forces,
which removes the shortcut whereby every target is a product of two exact values
already in the context. It applies only to the attention architecture; the residual
MLP emits all targets in one pass and warns that the setting is ignored. Each
non-teacher-forced step costs one extra forward pass for the rollout.

`curriculum` (default `false`) gates the loss by target degree. With it on, only the
lowest-degree targets are in the loss at first; the next degree unlocks once the
current highest one's training MSE on the batch falls below
`curriculum_mse_threshold` (default `0.01`), and unlocks are permanent. So degree-4
targets start training only after degree-2 is below the threshold, then degree-8
after degree-4. Because targets are ordered by degree, the active set is always a
prefix of the columns, and locked degrees contribute no gradient at all - with
`use_layerwise_readouts`, their readouts stay at initialization until unlocked. Under
a curriculum `train_mse` in `metrics.csv` is the loss actually optimized, i.e. the
unlocked degrees only, and a `curriculum_max_degree` column records which those are;
the `test_*` columns always cover every target. The unlock test uses a single batch,
so a lucky batch can unlock slightly early.

`train_samples` bounds the training data. Left `null` (the default), every step
draws a fresh batch, so training never repeats an input. Set to an integer, it
draws that many distinct inputs once, none of them in the test set, saves them to
`train_data.pt`, and trains only on that pool for all `num_steps` — shuffled
epochs, so each input is seen equally often. If the pool is smaller than
`batch_size`, every step uses the whole pool and a warning says so; if the input
space cannot supply that many inputs outside the test set, the pool is truncated
with a warning. When a pool is in use, `metrics.csv` gains `train_set_*` columns
holding the same metrics over the whole pool, so the gap against the `test_*`
columns measures memorization. (`train_mse` remains the current batch's loss.)

`matmul_precision` selects whether float32 matmuls may use tensor cores, via
`torch.set_float32_matmul_precision`. It defaults to `highest`, which is true
float32 and matches PyTorch's own default. `high` enables TF32: matmul inputs
are rounded to a 10-bit mantissa while accumulation stays in float32, giving
roughly 1e-3 relative precision per matmul instead of 1e-7, and is typically
several times faster on Ampere and later. `medium` additionally permits
bfloat16 inputs. Only matmuls are affected; the optimizer, the loss, and
elementwise ops stay in float32. Because it is recorded in the run config, the
precision each run used stays attached to its checkpoints. Note that it sets
global process state, so it also applies to anything run after training in the
same session.

## Attention architecture

Set `use_attention: true` in the model config to train a causal transformer on
the same task instead of the residual MLP. The task becomes next-position
prediction: the sequence is the `input_dim` input bits followed by the target
parities in binary-tree order (all degree-2 targets, then degree-4, degree-8,
degree-16, minus anything `exclude_targets` removes). The last input position
predicts the first degree-2 parity, the following position predicts the second
target, and so on, so the sequence is `input_dim + num_targets - 1` positions
long — the final target is never fed back and needs no input position.

Every position owns a learnable embedding vector, input bits and intermediate
answer positions alike, and the `{-1,+1}` value at that position scales it. This
is the same `W x` structure the residual net uses, extended to the answer
positions, so no separate positional encoding is needed. Each of the `L` blocks
is causal multi-head self-attention with a residual connection, followed by the
same MLP block the residual net uses (also residual, with
`use_post_activation_linear` optional). There is no layer normalization. The
unembedding is a single position-independent `N -> 1` map applied at every
prediction position, and the loss is MSE against the target parities.

Training is teacher-forced on the true parities. Evaluation is autoregressive:
only the input bits are given and each prediction is fed back into the next
position, so the `test_mse` columns in `metrics.csv` measure end-to-end
generation error, including error compounded through the fed-back predictions.
Generation uses a per-layer key/value cache, so it runs the input bits once and
then costs one single-position forward pass per target instead of recomputing
the whole prefix each step. Set `use_kv_cache: false` in the model config (or
pass `use_cache=False` to `generate`) for the recompute path; it produces the
same numbers and is what interventions fall back to, since an intervention may
be a function of the whole prefix.

Attention-specific model config:

- `sequence_mixing` chooses how positions exchange information inside each block.
  `attention` (default) is learned causal self-attention. `uniform` is
  parameter-free: each position simply adds the mean of the residual stream over its
  causal prefix, with no Q, K, V or O at all. `uniform_vo` keeps the V and O
  projections and takes the uniform prefix mean of the value vectors, so it ablates
  query-key selectivity alone. Both uniform modes are controls for a task whose
  operands always live at fixed, input-independent positions; because the position
  embeddings are orthogonal, a uniform sum still carries every earlier value. `none`
  removes the sub-layer entirely, so each position sees only its own value and cannot
  compute a parity of others; it warns on construction.
- `num_heads` (default `1`) must divide `N`. It has no effect outside `attention`.
- `attention_logit_scale` is `1/sqrt(d)` (standard) or `1/d` (muP, which keeps
  query-key logits `Theta(1)` as `head_dim` grows with width). Attention
  projections are initialized with `hidden_weight_variance` and belong to the
  `hidden` optimizer group, so the muP hidden-weight rules apply to them.
- `autoregressive_feedback` is `raw` (feed the model's own scalar output back,
  the default) or `sign` (feed `+/-1`, matching the teacher-forced input
  distribution).

`use_layerwise_readouts` is not supported together with `use_attention`.
