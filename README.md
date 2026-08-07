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
Supported activations are `relu`, `gelu`, `tanh`, `silu`, and `half-tanh`;
`half-tanh` is `relu(tanh(x))`.
Set `use_post_activation_linear` to `true` to use residual blocks of the form
`x + W phi(Vx)`; otherwise blocks use `x + phi(Vx)`.
The initialization variance fields are literal per-entry variances:
`embedding_weight_variance` rescales the frozen orthonormal embedding to have
approximately that per-entry variance; omit it or set it to `null` to keep the
unscaled QR embedding. Set `freeze_embedding` to `false` to train the embedding
weights. `hidden_weight_variance` initializes hidden weights with
`std = sqrt(hidden_weight_variance)`, and `readout_weight_variance`
initializes readout weights with `std = sqrt(readout_weight_variance)`.

`TrainingConfig` controls `num_steps`, fresh-batch size, fixed held-out test
set size, optimizer, checkpointing, and the readout barrier parameters. The
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

- `num_heads` (default `1`) must divide `N`.
- `attention_logit_scale` is `1/sqrt(d)` (standard) or `1/d` (muP, which keeps
  query-key logits `Theta(1)` as `head_dim` grows with width). Attention
  projections are initialized with `hidden_weight_variance` and belong to the
  `hidden` optimizer group, so the muP hidden-weight rules apply to them.
- `autoregressive_feedback` is `raw` (feed the model's own scalar output back,
  the default) or `sign` (feed `+/-1`, matching the teacher-forced input
  distribution).

`use_layerwise_readouts` is not supported together with `use_attention`.
