# Supplementary experiments and seed dependence

This note records diagnostic experiments and matched-seed results that are useful for
interpreting the phase diagram but were omitted from the concise main report. The main sweep
data are in `analysis_output/two_block_phase_diagram/results_merged.csv`.

## 1. What “joint” and “parent-warm” mean

The full sweep used three independent initialization seeds: 0, 1, and 2. For every seed and
every one of the 72 decay cells, two training protocols started from the same randomly
initialized base model:

- **Joint:** train both degree-2 parent targets and the degree-4 target simultaneously, with
  every model parameter trainable.
- **Parent-warm:** first train the embedding, first block, and degree-2 readout to produce the
  two degree-2 parents in `h1`; then unfreeze the entire model and train exactly the same final
  objective as in the joint protocol.

Thus the two protocols differ only in training history when the measured phase is recorded.
“Protocol disagreement” means that the matched joint and parent-warm runs have different
phase labels for the same `(seed, lambda0, lambda1/lambda2)` triple. It does not compare
different random seeds or different final losses.

## 2. Seed-by-seed protocol dependence

There are 216 matched comparisons: 3 seeds times 72 decay cells. The two protocols selected
different phases in 128/216 comparisons, or **59.3%**.

| Seed | Joint: H_C / H_D / suppressed | Parent-warm: H_C / H_D / suppressed | Disagreement |
|---:|---:|---:|---:|
| 0 | 3 / 52 / 17 | 46 / 11 / 15 | 44/72 (61.1%) |
| 1 | 3 / 52 / 17 | 44 / 13 / 15 | 42/72 (58.3%) |
| 2 | 3 / 52 / 17 | 44 / 13 / 15 | 42/72 (58.3%) |
| **All** | **9 / 156 / 51** | **134 / 37 / 45** | **128/216 (59.3%)** |

The disagreement has a highly directional structure:

| Joint phase -> parent-warm phase | Matched cells |
|---|---:|
| H_D -> H_C | 122 |
| suppressed -> H_C | 3 |
| suppressed -> H_D | 3 |
| all other phase changes | 0 |

Consequently, the sentence “joint and parent-warm training can reach different circuits under
the identical final objective” means something precise: in 122 cells the common random
initialization trained jointly remained in a direct-circuit basin, whereas degree-2
pretraining led the same seed and final objective to a compositional basin. The near-identical
counts across seeds show that this is systematic rather than a single-seed accident.

Parent warming is not universally a lower-objective solution. It had the lower final objective
in 90/216 matched cells (41.7%), although its mean objective difference was lower by 0.0488.
This is why the main report calls the lower-objective protocol the **best-observed** proxy and
does not identify either training protocol with the equilibrium solution.

## 3. Degree-4-only objective difference

The supplementary PDF also compares the matched degree-4 terms directly. It defines

- `L4 = 0.5 * gamma4 * mse4`, the degree-4 data term; and
- `J4_specific = L4 + reg_w4`, which adds the only decay term uniquely attributable to the
  degree-4 target.

Shared hidden-weight penalties are deliberately excluded because they cannot be assigned
uniquely to the parent or degree-4 tasks. The heatmaps show parent-warm minus joint, averaged
over the three matched seeds, with a signed-log color normalization. Negative (blue) values
favor parent-warm. Most active cells have very small positive differences: both protocols fit
the degree-4 target almost perfectly, with joint usually having a slightly smaller degree-4
data loss. The isolated order-one negative cells occur when joint training suppresses the
degree-4 task but parent warming keeps it active. Parent-warm has smaller `L4` in only 8/216
matched runs and smaller `J4_specific` in 70/216. The mean differences are nevertheless
negative because the few order-one joint-suppression failures outweigh many positive
differences of only `1e-9` to `1e-3`.

## 4. Failed naïve learning-rate launch

The first launch used width 1024, float64, full-batch SGD with momentum 0.9, and learning rate
0.01 for every parameter group. This is inconsistent with the model’s muP initialization:
the readouts begin at scale `1/N`, while their individual gradients are order one. The
parent-pretraining states became non-finite, and the main objective raised a non-finite error
at step 8 on every worker. No valid phase-diagram cell was produced.

The failed artifacts were preserved under
`analysis_output/two_block_phase_diagram_failed_sgd_20261001/`. This failure is numerical
evidence for using a readout learning rate scaled as `1/N`; it is not evidence for or against
either circuit phase.

## 5. Corrected width-1024 smoke test

Before restarting the full sweep, one cell was tested with the corrected learning-rate groups:

- width 1024, seed 0, `lambda0 = 0.01`, `lambda1/lambda2 = 1`;
- hidden/embedding learning rate `0.01`;
- readout learning rate `0.01/1024`;
- 1,000 parent-pretraining steps and 1,000 final-objective steps.

| Protocol | Phase | Objective | A4 | R | Compositional share | Gradient norm |
|---|---:|---:|---:|---:|---:|---:|
| Joint | H_D | 2.29699 | 0.9453 | 0.4454 | 0.1131 | 1.0490 |
| Parent-warm | H_C | 2.28398 | 0.9445 | 0.5879 | 0.5844 | 0.9698 |

Both runs remained finite and the parent-warm run already selected the compositional route.
Because the gradient norms were still order one and the run lasted only 1,000 steps, these
numbers are a stability/basin diagnostic, not stationary phase-diagram measurements. The raw
smoke-test files are archived as `analysis_output/two_block_phase_smoke_mup_20261001/`.
