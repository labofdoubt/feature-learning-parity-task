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
