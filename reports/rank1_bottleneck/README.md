# Rank-K bottleneck experiment (experiments/rank1_bottleneck)

`q_sweep_d4_K4_wd1e-3.png` / `.csv`: d = 4, inputs x1..x4 and x1x2, x3x4, target x1x2x3x4,
K = 4 random bottleneck, d_hid 1024, all init 1/d_hid except the singleton embedding rows scaled
by Q (= Q_1/Q_2), SGD lr 1e-3, wd 1e-3, 2000 steps, seed 0. Runs `rank1_d4_sp_K4_wd1e-3_Q<q>`
on swish (Drive prefix `runs_swish`). Top: degree-2 share of the code power (final, and at
500 / 1000 steps); bottom: final test MSE. Figure by `experiments/rank1_bottleneck/plot_q_sweep.py`.
