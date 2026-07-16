# parity-net

Minimal PyTorch library for training residual networks on the binary-tree
k-parity staircase described in `MOTIVATION.md`.

The task uses 32-dimensional `{-1,+1}` inputs. The first 16 coordinates define
15 outputs: eight degree-2 parities, four degree-4 parities, two degree-8
parities, and one degree-16 parity. The remaining 16 coordinates are noise.

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

`ModelConfig` controls the network shape: width `N`, depth
`L`, readout barrier toggle, embedding scale, residual-block form,
hidden-layer initialization variance, and readout initialization variance.
Set `use_post_activation_linear` to `true` to use residual blocks of the form
`x + W phi(Vx)`; otherwise blocks use `x + phi(Vx)`.
The initialization variance fields are literal per-entry variances:
`embedding_weight_variance` rescales the frozen orthonormal embedding to have
approximately that per-entry variance; omit it or set it to `null` to keep the
unscaled QR embedding. `hidden_weight_variance` initializes hidden weights with
`std = sqrt(hidden_weight_variance)`, and `readout_weight_variance`
initializes readout weights with `std = sqrt(readout_weight_variance)`.

`TrainingConfig` controls `num_steps`, fresh-batch size, fixed held-out test
set size, optimizer, checkpointing, and the readout barrier parameters. The
barrier coefficient `c` lives in the training config because it is a loss
regularizer. If `barrier_c` is omitted, training uses `7 / N`, matching the
mean-field-scale box from `MOTIVATION.md`.
Training saves the exact held-out test set to `test_data.pt` in the run
directory and rejects any fresh training batch samples that match that saved
test set.
