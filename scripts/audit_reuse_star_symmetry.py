"""
Systematic symmetry audit for the reuse-star implementation.

Checks:
  1. Target identities from actual x tensors (exact, every sample)
  2. Loss mask and reduction
  3. Per-target gradient norms at initialization (3 seeds)
  4. Statistical symmetry of B, C, D blocks (large sample)
  5. Positional-swap diagnostic: swap B<->C, check if target_0 (contiguous bits 0-15) always wins
  6. Isolated single-target training: S1-only vs S2-only vs S3-only
  7. Multiple-seed joint training: does S1 always win, or does the winner rotate?

Run:
  /usr/bin/python3 scripts/audit_reuse_star_symmetry.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parity_net.data import sample_reuse_star_inputs, REUSE_STAR_TARGET_NAMES
from parity_net.config import ModelConfig
from parity_net.model import build_model

CPU = torch.device("cpu")
try:
    import torch.backends
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = CPU
except Exception:
    DEVICE = CPU

print(f"Device: {DEVICE}")

RHO   = 0.3
DIST  = "hierarchical"
BATCH = 512
N_MODEL = 256   # small model for fast audit; scale up if needed


def make_model(seed: int = 0, N: int = N_MODEL) -> "object":
    torch.manual_seed(seed)
    cfg = ModelConfig(
        input_dim=32, relevant_dim=32, N=N, L=4,
        activation="half-tanh", activation_scale=1.0,
        use_skip_connections=True,
        hidden_weight_variance=1.0 / N,
        readout_weight_variance=1.0 / (N * N),
        freeze_embedding=True,
        embedding_weight_variance=None,
        use_readout_barrier=False,
    )
    return build_model(cfg, output_dim=3, target_names_=REUSE_STAR_TARGET_NAMES).to(DEVICE)


def sample(n: int, rho: float = RHO, dist: str = DIST, seed: int | None = None):
    gen = None
    if seed is not None:
        gen = torch.Generator(DEVICE)
        gen.manual_seed(seed)
    return sample_reuse_star_inputs(n, rho, dist, DEVICE, torch.float32, gen)


# ════════════════════════════════════════════════════════════════════════════
# 1. TARGET IDENTITY VERIFICATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("1. TARGET IDENTITY VERIFICATION (50 000 samples, rho=0.3)")
print("=" * 65)

x, targets = sample(50_000, seed=42)

A_x = x[:, :8].prod(dim=1)
B_x = x[:, 8:16].prod(dim=1)
C_x = x[:, 16:24].prod(dim=1)
D_x = x[:, 24:32].prod(dim=1)

checks = [
    ("S1 == prod(x[0:16])",           targets[:, 0], x[:, :16].prod(dim=1)),
    ("S2 == prod(x[0:8])*prod(x[16:24])", targets[:, 1], A_x * C_x),
    ("S3 == prod(x[0:8])*prod(x[24:32])", targets[:, 2], A_x * D_x),
    ("target[0] == A_x*B_x",          targets[:, 0], A_x * B_x),
    ("target[1] == A_x*C_x",          targets[:, 1], A_x * C_x),
    ("target[2] == A_x*D_x",          targets[:, 2], A_x * D_x),
]
all_ok = True
for name, a, b in checks:
    err = (a - b).abs().max().item()
    ok = err < 1e-5
    all_ok = all_ok and ok
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  max_err={err:.2e}")

if not all_ok:
    print("  *** IDENTITY FAILURE: targets do not match x ***")

# ════════════════════════════════════════════════════════════════════════════
# 2. LOSS MASK / REDUCTION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("2. LOSS MASK & REDUCTION (m=3)")
print("=" * 65)

m_model = make_model(seed=0)
x_b, y_b = sample(BATCH, seed=1)
m_model.eval()
with torch.no_grad():
    pred = m_model(x_b)

per_t = F.mse_loss(pred, y_b, reduction="none").mean(dim=0)
active_mask = torch.ones(3, device=DEVICE)
mse = (per_t * active_mask).sum() / active_mask.sum()
match = abs(mse.item() - per_t.mean().item()) < 1e-6

print(f"  pred.shape        : {tuple(pred.shape)}")
print(f"  target.shape      : {tuple(y_b.shape)}")
print(f"  per_target_mse    : {per_t.tolist()}")
print(f"  active_mask       : {active_mask.tolist()}")
print(f"  final_loss        : {mse.item():.5f}")
print(f"  mean(per_target)  : {per_t.mean().item():.5f}")
print(f"  [{'OK' if match else 'FAIL'}] loss == mean(L1,L2,L3)")

# ════════════════════════════════════════════════════════════════════════════
# 3. PER-TARGET GRADIENT NORMS AT INIT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("3. PER-TARGET GRADIENT NORMS AT INIT (5 seeds × 5 batches)")
print("=" * 65)

GRAD_SEEDS = [0, 1, 2, 3, 4]
body_norms_all = {j: [] for j in range(3)}
readout_norms_all = {j: [] for j in range(3)}

for seed in GRAD_SEEDS:
    model_g = make_model(seed=seed)
    for batch_seed in range(5):
        x_g, y_g = sample(BATCH, seed=seed * 100 + batch_seed)
        for j in range(3):
            model_g.zero_grad()
            pred_g = model_g(x_g)
            loss_j = F.mse_loss(pred_g[:, j], y_g[:, j])
            loss_j.backward()
            body = model_g.blocks[-1].linear.weight.grad
            bn = body.norm().item() if body is not None else 0.0
            rn = model_g.readout.weight.grad[j].norm().item() \
                 if model_g.readout.weight.grad is not None else 0.0
            body_norms_all[j].append(bn)
            readout_norms_all[j].append(rn)

print("  Mean body-gradient norm (last block) per target:")
for j in range(3):
    vals = body_norms_all[j]
    print(f"    S{j+1}: mean={sum(vals)/len(vals):.5f}  "
          f"min={min(vals):.5f}  max={max(vals):.5f}")

print("  Mean readout-row gradient norm per target:")
for j in range(3):
    vals = readout_norms_all[j]
    print(f"    S{j+1}: mean={sum(vals)/len(vals):.5f}  "
          f"min={min(vals):.5f}  max={max(vals):.5f}")

# ════════════════════════════════════════════════════════════════════════════
# 4. STATISTICAL SYMMETRY OF B, C, D BLOCKS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("4. BLOCK STATISTICS  (200 000 samples, rho=0.3, hierarchical)")
print("=" * 65)

x, tgt = sample(200_000, seed=0)
A_v = x[:, :8].prod(dim=1)
B_v = x[:, 8:16].prod(dim=1)
C_v = x[:, 16:24].prod(dim=1)
D_v = x[:, 24:32].prod(dim=1)

print(f"  E[A] = {A_v.mean():.4f}  (expected ~{RHO})")
print(f"  E[B] = {B_v.mean():.4f}  E[C] = {C_v.mean():.4f}  E[D] = {D_v.mean():.4f}  (all expected ~0)")
print(f"  E[S1]= {tgt[:,0].mean():.4f}  E[S2]= {tgt[:,1].mean():.4f}  E[S3]= {tgt[:,2].mean():.4f}")

print("\n  Per-bit means of each private block  (all expected ~0):")
for bname, sl in [("B(8:16)", slice(8,16)), ("C(16:24)", slice(16,24)), ("D(24:32)", slice(24,32))]:
    means = x[:, sl].mean(dim=0).tolist()
    print(f"    {bname}: " + "  ".join(f"{m:.4f}" for m in means))

print("\n  E[x_i * root] for each private block  (expected ~rho * tree_coeff):")
for bname, sl, root in [("B", slice(8,16), B_v), ("C", slice(16,24), C_v), ("D", slice(24,32), D_v)]:
    corrs = [(x[:, sl.start+i] * root).mean().item() for i in range(8)]
    print(f"    {bname}: " + "  ".join(f"{c:.4f}" for c in corrs))

print("\n  E[S_j * x_k] for each private block (A-block bits 0-7):")
for j, (tname, sl) in enumerate([("S1","B"), ("S2","C"), ("S3","D")]):
    priv_sl = [slice(8,16), slice(16,24), slice(24,32)][j]
    corrs = [(tgt[:, j] * x[:, priv_sl.start + i]).mean().item() for i in range(8)]
    print(f"    {tname}-priv: " + "  ".join(f"{c:.4f}" for c in corrs))

# ════════════════════════════════════════════════════════════════════════════
# 5. POSITIONAL-SWAP DIAGNOSTIC: swap B and C blocks in x
#    If the target using bits 0-15 (contiguous) always wins -> positional bias
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("5. POSITIONAL-SWAP DIAGNOSTIC (2000 steps)")
print("   Layout: [A|B|C|D] vs [A|C|B|D]")
print("   In swapped layout, target_0 = A*C (bits 0-15 contiguous)")
print("   If target_0 wins -> bits 0-15 have structural advantage")
print("=" * 65)


def run_swap_trial(swap_bc: bool, seed: int = 0, n_steps: int = 2000):
    torch.manual_seed(seed)
    model = make_model(seed=seed)
    # same lr schedule for all trials
    opt = torch.optim.SGD(
        [{"params": model.blocks.parameters(), "lr": 1e-3},
         {"params": model.readout.parameters(), "lr": 1e-3}],
        lr=1e-3, momentum=0.9,
    )
    active_mask = torch.ones(3, device=DEVICE)

    def make_batch(seed_b):
        x_raw, _ = sample_reuse_star_inputs(BATCH, RHO, DIST, DEVICE, torch.float32,
                                             torch.Generator(DEVICE).manual_seed(seed_b))
        if swap_bc:
            x_used = torch.cat([x_raw[:, :8], x_raw[:, 16:24],
                                 x_raw[:, 8:16], x_raw[:, 24:32]], dim=1)
        else:
            x_used = x_raw
        A_ = x_used[:, :8].prod(dim=1)
        y_ = torch.stack([A_ * x_used[:, 8:16].prod(dim=1),
                          A_ * x_used[:, 16:24].prod(dim=1),
                          A_ * x_used[:, 24:32].prod(dim=1)], dim=1)
        return x_used, y_

    # Fixed eval set
    x_eval_raw, _ = sample_reuse_star_inputs(10_000, RHO, DIST, DEVICE, torch.float32,
                                              torch.Generator(DEVICE).manual_seed(9999))
    if swap_bc:
        x_eval = torch.cat([x_eval_raw[:, :8], x_eval_raw[:, 16:24],
                             x_eval_raw[:, 8:16], x_eval_raw[:, 24:32]], dim=1)
    else:
        x_eval = x_eval_raw
    A_e = x_eval[:, :8].prod(dim=1)
    y_eval = torch.stack([A_e * x_eval[:, 8:16].prod(dim=1),
                          A_e * x_eval[:, 16:24].prod(dim=1),
                          A_e * x_eval[:, 24:32].prod(dim=1)], dim=1)

    log: dict = {}
    for step in range(1, n_steps + 1):
        model.train()
        x_b, y_b = make_batch(step)
        opt.zero_grad()
        pred = model(x_b)
        per_t = F.mse_loss(pred, y_b, reduction="none").mean(dim=0)
        loss = (per_t * active_mask).sum() / active_mask.sum()
        loss.backward()
        opt.step()

        if step in (100, 500, 1000, 2000):
            model.eval()
            with torch.no_grad():
                pe = F.mse_loss(model(x_eval), y_eval, reduction="none").mean(dim=0)
            log[step] = pe.tolist()
            model.train()

    return log


for swap, label in [(False, "ORIGINAL [A|B|C|D]"), (True, "SWAPPED  [A|C|B|D]")]:
    print(f"\n  {label}")
    if not swap:
        print("  target_0=S1(bits 0-15 contig), target_1=S2, target_2=S3")
    else:
        print("  target_0=A*C (bits 0-15 contig), target_1=A*B, target_2=S3")
    res = run_swap_trial(swap_bc=swap, seed=0)
    for step, losses in sorted(res.items()):
        print(f"    step {step:4d}: T0={losses[0]:.4f}  T1={losses[1]:.4f}  T2={losses[2]:.4f}"
              f"  winner={'T0' if losses[0] < losses[1] and losses[0] < losses[2] else 'other'}")

# ════════════════════════════════════════════════════════════════════════════
# 6. ISOLATED SINGLE-TARGET TRAINING
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("6. ISOLATED SINGLE-TARGET TRAINING (2000 steps, 3 seeds each)")
print("=" * 65)

for tgt_idx in range(3):
    losses_final = []
    accs_final = []
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        model = make_model(seed=seed)
        opt = torch.optim.SGD(
            [{"params": model.blocks.parameters(), "lr": 1e-3},
             {"params": model.readout.parameters(), "lr": 1e-3}],
            lr=1e-3, momentum=0.9,
        )
        active_mask = torch.zeros(3, device=DEVICE)
        active_mask[tgt_idx] = 1.0

        for step in range(1, 2001):
            model.train()
            x_b, y_b = sample_reuse_star_inputs(BATCH, RHO, DIST, DEVICE, torch.float32)
            opt.zero_grad()
            pred = model(x_b)
            per_t = F.mse_loss(pred, y_b, reduction="none").mean(dim=0)
            loss = (per_t * active_mask).sum() / active_mask.sum()
            loss.backward()
            opt.step()

        x_e, y_e = sample(10_000, seed=9999)
        model.eval()
        with torch.no_grad():
            pe = model(x_e)
            l_j = F.mse_loss(pe[:, tgt_idx], y_e[:, tgt_idx]).item()
            a_j = ((pe[:, tgt_idx] * y_e[:, tgt_idx]) > 0).float().mean().item()
        losses_final.append(l_j)
        accs_final.append(a_j)

    avg_l = sum(losses_final) / len(losses_final)
    avg_a = sum(accs_final) / len(accs_final)
    target_bits = ["bits 0-15", "bits 0-7 + 16-23", "bits 0-7 + 24-31"][tgt_idx]
    print(f"  S{tgt_idx+1} only ({target_bits}): "
          f"avg_loss={avg_l:.4f}  avg_acc={avg_a:.4f}  "
          f"per_seed_loss={[f'{l:.4f}' for l in losses_final]}")

# ════════════════════════════════════════════════════════════════════════════
# 7. JOINT TRAINING ACROSS MULTIPLE SEEDS: does S1 always win?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("7. JOINT TRAINING (m=3), 5 SEEDS — which target learns fastest?")
print("=" * 65)

for seed in range(5):
    torch.manual_seed(seed)
    model = make_model(seed=seed)
    opt = torch.optim.SGD(
        [{"params": model.blocks.parameters(), "lr": 1e-3},
         {"params": model.readout.parameters(), "lr": 1e-3}],
        lr=1e-3, momentum=0.9,
    )
    active_mask = torch.ones(3, device=DEVICE)

    for step in range(1, 2001):
        model.train()
        x_b, y_b = sample_reuse_star_inputs(BATCH, RHO, DIST, DEVICE, torch.float32)
        opt.zero_grad()
        pred = model(x_b)
        per_t = F.mse_loss(pred, y_b, reduction="none").mean(dim=0)
        loss = (per_t * active_mask).sum() / active_mask.sum()
        loss.backward()
        opt.step()

    x_e, y_e = sample(10_000, seed=9999)
    model.eval()
    with torch.no_grad():
        pe = model(x_e)
        ls = [F.mse_loss(pe[:, j], y_e[:, j]).item() for j in range(3)]
        accs = [((pe[:, j] * y_e[:, j]) > 0).float().mean().item() for j in range(3)]
    winner = ["S1", "S2", "S3"][ls.index(min(ls))]
    print(f"  seed={seed}: S1_loss={ls[0]:.4f}  S2_loss={ls[1]:.4f}  S3_loss={ls[2]:.4f}"
          f"  winner={winner}  (acc: S1={accs[0]:.3f} S2={accs[1]:.3f} S3={accs[2]:.3f})")

print("\nAudit complete.")
