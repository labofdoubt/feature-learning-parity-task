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
