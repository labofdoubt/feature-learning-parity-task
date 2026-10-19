# One-block d32 hyperparameter sweep (wd, lr) — validation curves

Base: `runs/L1cap_1xd32_in32_lremb1e-3_wdro1e-3_seed1` (one residual block, N = 2048,
32 inputs all relevant, single target d32_0, seed 1, fresh batches of 512, 500k steps).
One lr and one wd applied identically to embedding, hidden and readout
(lr ∈ {1e-3, 3e-4, 1e-4}, wd ∈ {1e-2, 1e-3, 1e-4}); hidden init 1/N throughout.
Each panel: test (blue) and train (orange) MSE vs step, log scale; rows = wd, cols = lr.

| file | activation | embedding init | readout init |
|---|---|---|---|
| halftanh_A_all_init_1N.png | half-tanh | 1/N | 1/N |
| halftanh_B_readout_1N2.png | half-tanh | 1/N | 1/N² |
| halftanh_C_readout_1N2_embedding_1d.png | half-tanh | 1/32 | 1/N² |
| relu_*.png | relu | same three families | |

Run names on germany / Drive (`gdrive:parity/runs_germany`): `hp_d32_<family>_lr<lr>_wd<wd>`
and `hp_d32relu_<family>_lr<lr>_wd<wd>` with family ∈ {var1N, ro1N2, ro1N2_emb1d}.
Figures produced by `scripts/plot_hp_grid.py`.

## 8xd4 + d32 trained simultaneously (lr 1e-3, half-tanh, seed 1, 500k steps)

`cap_8xd4_d32_L{1,2}_{var1N,ro1N2_emb1d}.png`: upper panel d4 test MSE, lower panel d32 test MSE,
for wd 1e-3 and 1e-4; one figure per (number of blocks, init family). Runs: `cap_8xd4_d32_L<L>_<family>_wd<wd>`.
Figures by `scripts/plot_d4_d32_curves.py`.

`cap_8xd4_d32_L2_var1N_twostage.png`: same setup (L=2, all init 1/N, lr 1e-3, wd 1e-4) trained in two
stages, block 1 frozen (`training.freeze_blocks: [1]`) for 200k steps, then unfrozen and resumed to 700k,
against the one-stage run. Run: `cap_8xd4_d32_L2_var1N_wd1e-4_twostage` (configs `config_stage1.yaml`,
`config_stage2.yaml`).
