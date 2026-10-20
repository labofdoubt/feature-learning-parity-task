# Rank-K bottleneck experiment (experiments/rank1_bottleneck)

`q_sweep_d4_K4_wd<wd>.png` / `.csv` (wd 1e-3, 1e-2, 1e-4): d = 4, inputs x1..x4 and x1x2, x3x4, target x1x2x3x4,
K = 4 random bottleneck, d_hid 1024, all init 1/d_hid except the singleton embedding rows scaled
by Q (= Q_1/Q_2), SGD lr 1e-3, 2000 steps, seed 0. Runs `rank1_d4_sp_K4_wd<wd>_Q<q>` (and `rank1_d4_sp_K2_*` for the K = 2 series)
on swish (Drive prefix `runs_swish`). Top: degree-2 share of the code power (final, and at
500 / 1000 steps); bottom: final test MSE. Figure by `experiments/rank1_bottleneck/plot_q_sweep.py`.

`q_sweep_d4_nobn_wd1e-2_preact.png` / `.csv`: same inputs and target, NO bottleneck (the ReLU layer
reads h(x) directly), wd 1e-2, 2000 steps. The share is the degree-2 fraction of the exact pre-activation
Walsh power summed over hidden units (`preact/fraction_degree2`). Runs `rank1_d4_sp_nobn_wd1e-2_Q<q>`.

`neuron_decomp_d4_K2_wd1e-3_Q4.5.csv`: per-neuron decomposition of the K = 2, wd 1e-3, Q = 4.5 run
(singleton regime). Columns: neuron index, target power (square of the unit's contribution
r_j * <relu(z_j), x1x2x3x4> to the output target coefficient, exact over all 16 inputs), that signed
contribution, the exact coefficients of the unit's pre-activation on x1, x2, x3, x4, x1x2, x3x4, and the
readout weight. Sorted by target power. Made by `experiments/rank1_bottleneck/neuron_decomposition.py`.

`q_sweep_d8_s44_nobn_wd3e-2_preact.png` / `.csv`: d = 8, inputs x1..x8 and the two degree-4 modes
x1x2x3x4, x5x6x7x8, target x1...x8, NO bottleneck, wd 3e-2, d_hid 1024, SGD lr 1e-3, 10000 steps, seed 0;
singleton embedding rows scaled by Q = Q_1/Q_4 from 3 to 9 step 0.5. Share = degree-4 fraction of the exact
pre-activation Walsh power (`preact/fraction_degree4`). Runs `rank1_d8_s44_nobn_wd3e-2_Q<q>` on swish.
Step count chosen from the probe `rank1_d8_sing_nobn_wd3e-2_probe` (singletons only, same wd): the
singleton route reaches accuracy 1 by 1.5k steps and MSE 0.08 by 4k, floor ~0.056. NOTE: at 10k steps the
degree-4 takeover is still in progress for Q = 5.5-7 (share rising at the last steps), so the "final" curve
there is a snapshot, not an equilibrium.

`q_sweep_d8_s44_nobn_wd9e-2_50k_preact.png` / `.csv`: same d = 8 setup (x1..x8 + x1x2x3x4, x5x6x7x8, no
bottleneck) at wd 9e-2, 50000 steps, Q = Q_1/Q_4 from 3 to 9 step 0.2. Runs
`rank1_d8_s44_nobn_wd9e-2_Q<q>_50k` on swish. Three regimes: Q <= 5.8 degree-4 route (share 1, MSE 0.014);
Q = 6.0-7.0 a stable mixed plateau (share 0.74: both degree-4 modes at power 1.14 each plus every singleton
at 0.10, MSE 0.12, accuracy 1) reached by 5-10k steps and essentially frozen afterwards; Q >= 7.2 total
collapse (all embedding power decays to 0, MSE 1). Plot: `plot_q_sweep.py --suffix _50k`.

`full_loss_d8_s44_nobn_wd9e-2_50k.png` / `.csv`: for the same wd 9e-2 sweep, the final checkpoints' exact
test MSE, the weight-decay term (wd/2) * sum ||theta||^2 over embedding, linear and readout, and their sum
(the objective SGD descends). Made by `experiments/rank1_bottleneck/full_loss.py`. The three layers end
with equal squared norms (weight-decay balancing of a homogeneous net).

`q_sweep_d8_s44_nobn_wd1e-2_50k_preact.png` / `.csv`: the wd 9e-2 d = 8 sweep repeated at wd 1e-2
(50k steps, Q step 0.2; runs `rank1_d8_s44_nobn_wd1e-2_Q<q>_50k` on swish). Lower panel: final test MSE
and full loss (MSE + (wd/2) sum ||theta||^2 from final.pt, `plot_q_sweep.py --full-loss`). Regimes:
Q <= 5.8 degree-4 (MSE 1.6e-4, full 0.038); Q = 6.0-7.2 mixed plateau (share 0.70, MSE 1e-3, full 0.079);
Q >= 7.4 singleton route survives at this wd (share 0, MSE 5e-3, full 0.20) instead of collapsing.

`q_sweep_d4_nobn_wd1e-1_rowd1e-2_preact.png`, `q_sweep_d4_nobn_wd1e-1_rowd1e-3_preact.png` (+ `.csv`):
the d = 4 no-bottleneck wd 0.1 sweep (`q_sweep_d4_nobn_wd1e-1_preact.png`) repeated with a SMALLER weight
decay on the readout only (`weight_decay_readout`: 1e-2 resp. 1e-3; embedding and linear keep 0.1),
2000 steps, Q = 1..5 step 0.1. Runs `rank1_d4_sp_nobn_wd1e-1_rowd{1e-2,1e-3}_Q<q>` on the box
85.10.218.46:35675 (Drive prefix `runs_germany2`). Lower panel shows final test MSE and full loss.

`q_sweep_d4_nobn_wd1e-1_noise_compare.png`: the d = 4 no-bottleneck wd 0.1 sweep repeated with the two
pair inputs x1x2, x3x4 replaced by independent random +-1 bits (`noise_modes: true`; same embedding rows
and init scale, so Q still scales the singleton rows relative to the noise rows). Runs
`rank1_d4_spnoise_nobn_wd1e-1_Q<q>` on swish, 2000 steps, Q = 1..5 step 0.1; per-run figure/CSV
`q_sweep_d4_spnoise_nobn_wd1e-1_preact.*` (its "degree-2 share" is the share of the noise rows). Top: final
test MSE of the original sweep and of the noise sweep; bottom: full loss (MSE + (wd/2) sum ||theta||^2
from final.pt) for both. The noise network reproduces the original's singleton regime (MSE ~0.13, full
loss ~1.75) at every Q, so with only singletons the pair route's values (MSE 0.017, full loss 1.28) are
never reached. Made by `experiments/rank1_bottleneck/plot_q_compare.py`.

`q_sweep_d8_s44_nobn_wd1e-2_noise_compare.png`: the d = 8 wd 1e-2 50k-step sweep repeated with the two
degree-4 inputs x1x2x3x4, x5x6x7x8 replaced by independent random +-1 bits (`noise_modes: true`), same
steps, evaluation (full cube, tiled so the noise averages) and 31 Q points. Runs
`rank1_d8_s44noise_nobn_wd1e-2_Q<q>_50k` (even Q indices on swish, odd on the 85.10.218.46:35675 box,
final checkpoints gathered on swish); per-run figure/CSV `q_sweep_d8_s44noise_nobn_wd1e-2_50k_preact.*`.
Top: final test MSE, original vs noise; bottom: full loss, original vs noise. The noise network gives
MSE 5.0e-3 and full loss 0.202 at every Q, identical to the original's singleton regime (Q >= 7.4), i.e.
the singleton route is the only route and the degree-4 route (MSE 1.6e-4, full 0.038) is 31x / 5.3x better.
