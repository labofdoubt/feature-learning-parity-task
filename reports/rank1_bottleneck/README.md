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

`q_sweep_d8_s44_nobn_wd1e-2_noise1_compare.png`: the converse of the previous figure, for Q_1/Q_4 > 6
only (6.2..9.0 step 0.2): the eight SINGLETON inputs are replaced by independent random +-1 bits
(`noise_modes: [1]`) while the two degree-4 modes stay, same steps, evaluation and wd 1e-2. Runs
`rank1_d8_s44noise1_nobn_wd1e-2_Q<q>_50k` (split across swish and the 85.10.218.46:35675 box, final
checkpoints gathered on swish); per-run figure/CSV `q_sweep_d8_s44noise1_nobn_wd1e-2_50k_preact.*`. Top:
final test MSE, original vs singletons-noised; bottom: full loss. With only the degree-4 modes available
the network reaches MSE 1.6e-4 and full loss 0.038 at every Q, exactly the original's degree-4 regime,
while the original sits on the mixed plateau (Q 6.2-7.2) or the singleton route (Q >= 7.4) there.

`neuron_decomp_d8_s44_nobn_wd1e-2_Q4.0_50k.csv`, `..._Q8.0_50k.csv`: per-neuron Walsh decomposition of
the final models of the d = 8 wd 1e-2 sweep at Q_1/Q_4 = 4 (degree-4 regime) and 8 (singleton regime).
Columns: neuron, target power ((r_j <relu(z_j), x1..x8>)^2, exact on all 256 inputs), the signed target
contribution, the exact pre-activation coefficients on x1..x8, x1x2x3x4, x5x6x7x8, and the readout
weight; sorted by target power. Q = 4: every active unit is a(p +- q) with p = x1x2x3x4, q = x5x6x7x8,
|b/a| = 1 to 4 digits, singleton coefficients < 4e-5; p+q units all have readout > 0, p-q units all
readout < 0. Q = 8: every active unit is a * sum_i s_i x_i with |s_i| = 1 (a sign pattern), 128/128
patterns up to overall sign present, readout sign = -parity(s) for all 514 active units, degree-4
coefficients < 2e-4 (a template construction of the parity). Script `neuron_decomposition.py`.

`q_sweep_d8_s44_nobn_wd1e-2_frozenemb_50k_preact.png` / `.csv`: the d = 8 wd 1e-2 50k-step sweep with the
EMBEDDING FROZEN at its Q-scaled init (`freeze_embedding: true`: not trained, not decayed, so the full
loss covers only the linear layer and the readout). Runs `rank1_d8_s44_nobn_wd1e-2_frozenemb_Q<q>_50k`
(split across swish and the 85.10.218.46:35675 box, final checkpoints gathered on swish). No regimes:
the degree-4 share falls smoothly from 0.94 at Q = 3 to ~0 by Q = 5, and the share curves at 5k/10k/25k
are still rising everywhere, i.e. the degree-4 route keeps taking over slowly. MSE and full loss are
nearly flat (MSE 2.1e-4 -> 1.1e-4, full loss 0.029 -> 0.020 from Q = 3 to 9) and DEcrease with Q, because
the frozen singleton rows supply input scale that costs no penalty.

`q_sweep_d8_s44_nobn_wd1e-2_frozenorth_50k_preact.png` / `.csv`: as the frozen-embedding sweep above but
with the embedding rows initialized mutually ORTHOGONAL (`embedding_orthogonal: true`: QR of a Gaussian
draw, each row at the random-draw norm 1, singleton rows then scaled by Q) and frozen. Runs
`rank1_d8_s44_nobn_wd1e-2_frozenorth_Q<q>_50k` (split across swish and the 85.10.218.46:35675 box). The
result is indistinguishable from the random frozen embedding: smooth decline of the degree-4 share
(0.94 at Q = 3, ~0 by Q = 5), MSE 2.0e-4 -> 1.0e-4 and full loss 0.028 -> 0.020 from Q = 3 to 9.

`q_sweep_d8_s44_nobn_wd5e-2_frozenorth_100k_preact.png` / `.csv`: the frozen-orthogonal-embedding sweep
at wd 5e-2 and 100k steps (Q = 3..9 step 0.2; runs `rank1_d8_s44_nobn_wd5e-2_frozenorth_Q<q>_100k`, split
across swish and the 85.10.218.46:35675 box). The heavier decay restores a sharp transition at
Q ~ 5.7: Q <= 5.4 pure degree-4 route (share 1), Q >= 6.2 pure singleton route (share < 1e-3), reached
through a slow takeover (at Q = 5.2 the share is 0.03 at 25k, 0.79 at 50k, 1.0 at 100k). MSE and full loss
are nearly flat across the transition (5.0e-3 / 0.136 on the degree-4 side, 5.5e-3 / 0.14 at Q = 6) and
then decrease slowly with Q on the singleton side (2.5e-3 / 0.098 at Q = 9).

## Experiment 1: planting a degree-4 circuit into the trained singleton solution

`plantd4_Q8_summary.png`, `plantd4_Q8_traj.png`, `plantd4_Q8_summary.csv`; driver
`experiments/rank1_bottleneck/plant_degree4.py`, figures `plot_plant_degree4.py`, tests
`tests/test_plant_degree4.py`. Source: `runs/rank1_d8_s44_nobn_wd1e-2_Q8.0_50k/checkpoints/final.pt`
(sha256 41849d5c..., step 50000; singleton solution: MSE 5.0e-3, L 0.2023, degree-4 share 1e-7).
Construction: M = 4 silent units J = [195, 404, 471, 872] (output RMS 1.4e-8), two directions U orthogonal
to all embedding rows, prepared baseline W_base = W0 (I - U U^T) with rows J and readout entries J zeroed
(predictions preserved to 8e-16, penalty change -1.3e-6), then V_p += rho/sqrt2 u1, V_q += rho/sqrt2 u2,
W_J = rho/sqrt(2M) S U^T, r_J = rho/sqrt(M) s1 s2. Validated (float64 / float32): y_rho - y_base =
rho^3/4 chi_8 to 1e-15 / 5e-7, no other Walsh coefficients, norm increase 3 rho^2, objective change
-2(1-a_base) rho^3/4 + (rho^3/4)^2 + 3 lambda rho^2 / 2 to 1e-16 / 3e-12; projected regularized gradient
onto the three templates = 2 e kappa_2 rho^2 - lambda rho to 1e-17. Every arm: fresh SGD (zero momentum
buffers, momentum 0.9, lr 1e-3, wd 1e-2 on all three trained matrices, unused W_enc/W_dec excluded),
batch 512, identical minibatch sequence (data seed 777; repeats 778/779), 50k post-intervention steps,
exact full-cube diagnostics every 5 steps to 500, every 100 to 5k, then every 250. Run dirs
`runs/plantd4_Q8_<arm>_s<seed>` (`analysis/`: intervention.json with all metadata + validation,
trajectory.csv, final.json with the final singleton-only / degree-4-only ablations, templates.pt;
`checkpoints/`: intervened.pt, final.pt) split across swish and the 85.10.218.46:35675 box; 150k-step
continuations under `plantd4_Q8L_*`. Local copies of every `analysis/` folder in `analysis_output/plantd4_Q8/`.
Result: finite seed barrier between rho = 0.28 (decay, 3 minibatch seeds) and 0.285 (rescue at 150k; 0.45
share at 50k), prediction 0.2831; rescued arms reach L 0.0376-0.040 (prediction 0.03764), decayed arms
0.2020 (prediction 0.20196). Along coherent trajectories b, w, r stay equal to 0.3% and a_planted = b w r / 4
to < 1e-4 until the takeover (symmetric-circuit approximation holds); the barrier is a basin boundary
(rho = 0.30 shrinks the rest first and grows only slowly for 20k steps). Target-orthogonal controls: decay at
0.28 and 0.34 (planted units overtaken by singleton leakage), persistent mixed solution at 0.40 (share
0.70, L 0.078 unchanged from 50k to 150k, planted coefficient 0.64 above the symmetric formula).
Coarse grid added afterwards (rho = 0.05, 0.10, 0.20, 0.50, 0.60, 0.80, 1.0 and orthogonal 0.60, 1.0): every
rho <= 0.28 decays, every rho >= 0.285 is rescued to the same L = 0.0377; orthogonal controls rescue at 0.60
and 1.0 with other units carrying 0.25 of the target. `plantd4_Q8_summary_zoom.png` zooms on rho in [0.25,
0.33]. Brief write-up: `reports/plantd4_rescue_report.pdf` (`.tex`, compiled with tectonic).
