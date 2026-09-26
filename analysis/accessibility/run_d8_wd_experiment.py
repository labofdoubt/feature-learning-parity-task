"""Experiment 7: frozen accessibility x weight decay, lifted from d4 to d8.

8 relevant bits, full 2^8 cube, pure-ReLU residual blocks, N=1024, init var 1/N.
Stage 1 trains embedding + block 1 on the two d4 parents (x1x2x3x4, x5x6x7x8),
lambda=0, to convergence; frozen thereafter (checkpoint saved). ReLU
homogeneity gives h1(alpha) = h0 + alpha*ReLU(V1 h0). A per-seed PILOT maps
alpha -> measured Q4/Q1 and inverts it so conditions land on the requested
target ratios, densely bracketing the theory boundary Q4/Q1 = 5/32 = 0.15625
(1^8 vs 4+4 tie; kappa_8 = 5/32 wants R_18 = 5 Q1/(32 sqrt 8) vs
R_44 = Q4/(2 sqrt 2)). Q2 is tracked for the 2^4 control route
(R_24 = Q2/8; 4+4 beats 2^4 iff Q4/Q2 > 1/sqrt 8).

Stage 2: fresh block 2 + readout trained on the full d8 parity, weight decay
on BOTH (exp-2 convention), identical stage-2 init across all (ratio, lambda)
within a seed. Classification is by pre-activation Walsh shares over the full
255-mode basis: eight singletons (1^8), four relevant pairs (2^4), two
relevant quads (4+4), else mixed; contribution-weighted over neurons carrying
95% of |w*c8|.

Numerics: training in float32 (float64 at 256x1024 is ~64x slower on consumer
GPUs; exp 1-6 were 16-point-cube cheap), ALL Walsh measurement/classification
in float64. Convergence: min 15k / max 50k steps, plateau tol 1e-4 over a 5k
window, or objective < 1e-9 (fp32 floor). Every finished cell is appended to
CSV immediately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="0.05,0.08,0.10,0.12,0.135,0.145,0.152,"
                                     "0.156,0.160,0.168,0.18,0.20,0.25,0.30")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--tag", required=True)
ARGS = ap.parse_args()
TARGETS = [float(t) for t in ARGS.targets.split(",")]
SEEDS = [int(s) for s in ARGS.seeds.split(",")]

OUT = Path("/root/parity/analysis_output/accessibility_d8_wd")
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("/root/parity/analysis_output/accessibility_ckpts")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 1024
D = 8
LAMBDAS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LR, MOM = 0.01, 0.9
MAX1, MAX2, MIN2 = 50_000, 50_000, 15_000
SNAP_EVERY = 1000
RC = 5 / 32  # Q4/Q1 tie point for 1^8 vs 4+4

# full cube [256, 8] and full character matrix [256 inputs, 256 modes] (f64)
X64 = torch.tensor([[1.0 if (m >> k) & 1 == 0 else -1.0 for k in range(D)]
                    for m in range(2 ** D)], device=DEV, dtype=torch.float64)
CHIM = torch.ones(2 ** D, 2 ** D, device=DEV, dtype=torch.float64)
for mode in range(2 ** D):
    for k in range(D):
        if (mode >> k) & 1:
            CHIM[:, mode] *= X64[:, k]
X32 = X64.float()

def midx(*bits):
    m = 0
    for b in bits:
        m |= 1 << b
    return m

SINGLES = [midx(k) for k in range(8)]
PAIRS_REL = [midx(0, 1), midx(2, 3), midx(4, 5), midx(6, 7)]
QUADS_REL = [midx(0, 1, 2, 3), midx(4, 5, 6, 7)]
FULL = 255
POP = np.array([bin(m).count("1") for m in range(256)])
Y2 = torch.stack([CHIM[:, QUADS_REL[0]], CHIM[:, QUADS_REL[1]]], 1).float()  # stage-1 targets
Y8_64 = CHIM[:, FULL]
Y8 = Y8_64.float().unsqueeze(1)


def init_lin(rows, cols, gen):
    return (torch.randn(rows, cols, generator=gen, device=DEV, dtype=torch.float64)
            / np.sqrt(N)).float()


def qstats(h1_64):
    """h1 [256, N] float64 -> mode-norm stats."""
    qc = CHIM.T @ h1_64 / 256.0          # [256 modes, N]
    qn = qc.norm(dim=1)                   # [256]
    q1 = float(qn[SINGLES].mean())
    q2 = float(qn[PAIRS_REL].mean())
    q4 = float(qn[QUADS_REL].mean())
    return dict(Q1=q1, Q2=q2, Q4=q4, Q8=float(qn[FULL]),
                r41=q4 / max(q1, 1e-30), r21=q2 / max(q1, 1e-30),
                r42=q4 / max(q2, 1e-30))


def classify(h1_64, V2, w_out):
    """Pre-activation circuit shares for d8-relevant block-2 neurons."""
    z = h1_64 @ V2.double().T             # [256, N] f64
    a = torch.relu(z)
    c8 = (a * Y8_64.unsqueeze(1)).mean(0)  # [N]
    contrib = (w_out.double().squeeze(0) * c8).abs()
    total = float(contrib.sum())
    base = dict(n_rel=0, f18_w=np.nan, f24_w=np.nan, f44_w=np.nan, fmix_w=np.nan,
                n_18=0, n_24=0, n_44=0, n_mix=0)
    if total < 1e-12:
        return base
    order = torch.argsort(contrib, descending=True)
    csum = torch.cumsum(contrib[order], 0)
    rel = order[: int((csum < 0.95 * total).sum()) + 1]
    zc = CHIM.T @ z / 256.0               # [256, N]
    var = (zc[1:] ** 2).sum(0)            # exclude constant
    s1 = (zc[SINGLES] ** 2).sum(0) / (var + 1e-30)
    s2 = (zc[PAIRS_REL] ** 2).sum(0) / (var + 1e-30)
    s4 = (zc[QUADS_REL] ** 2).sum(0) / (var + 1e-30)
    wgt = contrib[rel] / contrib[rel].sum()
    f18 = float((wgt * s1[rel]).sum())
    f24 = float((wgt * s2[rel]).sum())
    f44 = float((wgt * s4[rel]).sum())
    n18 = int((s1[rel] >= 0.6).sum())
    n24 = int((s2[rel] >= 0.6).sum())
    n44 = int((s4[rel] >= 0.6).sum())
    return dict(n_rel=len(rel), f18_w=f18, f24_w=f24, f44_w=f44,
                fmix_w=max(0.0, 1 - f18 - f24 - f44),
                n_18=n18, n_24=n24, n_44=n44,
                n_mix=len(rel) - n18 - n24 - n44)


rows, traj_rows, pre_info = [], [], {}

for seed in SEEDS:
    # ── stage 1: load or train the d4 representation ─────────────────────────
    ckpt_path = CKPT_DIR / f"d8pre_seed{seed}.pt"
    gen = torch.Generator(device=DEV).manual_seed(seed)
    E0 = init_lin(N, D, gen)
    V10 = init_lin(N, N, gen)
    V20 = init_lin(N, N, gen)   # stage-2 init, fixed per seed
    w0 = init_lin(1, N, gen)
    head0 = init_lin(2, N, gen)
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=DEV, weights_only=True)
        E, V1 = saved["E"].to(DEV).float(), saved["V1"].to(DEV).float()
        print(f"[seed {seed}] loaded {ckpt_path.name}", flush=True)
    else:
        E = E0.clone().requires_grad_(True)
        V1 = V10.clone().requires_grad_(True)
        head = head0.clone().requires_grad_(True)
        opt = torch.optim.SGD([E, V1, head], lr=LR, momentum=MOM)
        hist = []
        for t in range(MAX1):
            opt.zero_grad(set_to_none=True)
            h0 = X32 @ E.T
            h1 = h0 + torch.relu(h0 @ V1.T)
            loss = ((h1 @ head.T - Y2) ** 2).mean()
            loss.backward()
            opt.step()
            hist.append(float(loss))
            if t + 1 >= 10_000 and (hist[-1] < 1e-9 or
                                    (hist[-5000] - hist[-1]) / max(hist[-5000], 1e-12) < 1e-4):
                break
        E, V1 = E.detach(), V1.detach()
        torch.save({"E": E.cpu(), "V1": V1.cpu(), "seed": seed,
                    "steps": t + 1, "loss": hist[-1]}, ckpt_path)
        print(f"[seed {seed}] stage1 {t+1} steps loss {hist[-1]:.2e} -> {ckpt_path.name}",
              flush=True)

    with torch.no_grad():
        h0_64 = (X32 @ E.T).double()
        relu1_64 = torch.relu((X32 @ E.T) @ V1.T).double()

    # ── pilot: map alpha -> Q4/Q1, invert to hit the target ratios ───────────
    agrid = np.geomspace(0.02, 64, 48)
    pil = []
    with torch.no_grad():
        for a in agrid:
            st = qstats(h0_64 + a * relu1_64)
            pil.append((a, st["r41"], st["r42"], st["r21"]))
    r41s = np.array([p[1] for p in pil])
    print(f"[seed {seed}] pilot: r41 range [{r41s.min():.4f}, {r41s.max():.4f}], "
          f"r42 range [{min(p[2] for p in pil):.3f}, {max(p[2] for p in pil):.3f}]",
          flush=True)
    pre_info[f"seed{seed}"] = dict(pilot=[dict(alpha=float(a), r41=float(r), r42=float(r2),
                                               r21=float(r3)) for a, r, r2, r3 in pil])
    alphas = []
    for tgt in TARGETS:
        if tgt > r41s.max():
            print(f"[seed {seed}] target {tgt} unreachable (max {r41s.max():.4f}) -- skipped",
                  flush=True)
            continue
        i = int(np.argmin(np.abs(r41s - tgt)))
        # local log-interpolation refinement
        lo, hi = max(i - 1, 0), min(i + 1, len(agrid) - 1)
        aa = np.geomspace(agrid[lo], agrid[hi], 12)
        with torch.no_grad():
            rr = [qstats(h0_64 + a * relu1_64)["r41"] for a in aa]
        alphas.append(float(aa[int(np.argmin(np.abs(np.array(rr) - tgt)))]))

    # ── stage 2 sweep ─────────────────────────────────────────────────────────
    for alpha in alphas:
        with torch.no_grad():
            h1_64 = h0_64 + alpha * relu1_64
            st0 = qstats(h1_64)
            h1c = h1_64.float()
        for lam in LAMBDAS:
            V2 = V20.clone().requires_grad_(True)
            w = w0.clone().requires_grad_(True)
            opt = torch.optim.SGD([{"params": [V2, w], "weight_decay": lam}],
                                  lr=LR, momentum=MOM)
            hist2 = []
            for t2 in range(MAX2):
                opt.zero_grad(set_to_none=True)
                h2 = h1c + torch.relu(h1c @ V2.T)
                loss = ((h2 @ w.T - Y8) ** 2).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    sq = float((V2 ** 2).sum() + (w ** 2).sum())
                hist2.append(float(loss) + lam / 2 * sq)
                if t2 % SNAP_EVERY == 0 or t2 == MAX2 - 1:
                    with torch.no_grad():
                        h2 = h1c + torch.relu(h1c @ V2.T)
                        pred = (h2 @ w.T).double().squeeze(1)
                        f8 = float((pred * Y8_64).mean())
                        cls = classify(h1_64, V2.detach(), w.detach())
                    traj_rows.append(dict(seed=seed, alpha=alpha, r41=st0["r41"],
                                          lam=lam, step=t2, loss=float(loss),
                                          f8=f8, e8=1 - f8, sqnorm=sq,
                                          **{k: cls[k] for k in
                                             ("f18_w", "f24_w", "f44_w", "n_rel")}))
                if t2 >= MIN2 and (hist2[-1] < 1e-9 or
                                   (hist2[-5000] - hist2[-1]) / max(abs(hist2[-5000]), 1e-12) < 1e-4):
                    break
            with torch.no_grad():
                h2 = h1c + torch.relu(h1c @ V2.T)
                pred = (h2 @ w.T).double().squeeze(1)
                f8 = float((pred * Y8_64).mean())
                mse = float(((pred - Y8_64) ** 2).mean())
                cls = classify(h1_64, V2.detach(), w.detach())
                vN, wN = float(V2.norm()), float(w.norm())
                sq = float((V2 ** 2).sum() + (w ** 2).sum())
            rows.append(dict(seed=seed, alpha=alpha, lam=lam, steps=t2 + 1,
                             Q1=st0["Q1"], Q2=st0["Q2"], Q4=st0["Q4"], Q8_frozen=st0["Q8"],
                             r41=st0["r41"], r21=st0["r21"], r42=st0["r42"],
                             mse=mse, f8=f8, e8=1 - f8,
                             V2_norm=vN, w_norm=wN, sqnorm=sq,
                             wd_penalty=lam / 2 * sq,
                             **cls))
            print(f"[s{seed}] r41={st0['r41']:.4f} lam={lam:<7} steps={t2+1} "
                  f"f8={f8:.3f} f18={cls['f18_w'] if cls['f18_w']==cls['f18_w'] else -1:.3f} "
                  f"f44={cls['f44_w'] if cls['f44_w']==cls['f44_w'] else -1:.3f} "
                  f"f24={cls['f24_w'] if cls['f24_w']==cls['f24_w'] else -1:.3f}",
                  flush=True)
            rp = OUT / f"results_{ARGS.tag}.csv"
            pd.DataFrame(rows[-1:]).to_csv(rp, mode="a", header=not rp.exists(), index=False)
            tp_ = OUT / f"trajectories_{ARGS.tag}.csv"
            new_tr = [r for r in traj_rows if r["seed"] == seed
                      and r["alpha"] == alpha and r["lam"] == lam]
            pd.DataFrame(new_tr).to_csv(tp_, mode="a", header=not tp_.exists(), index=False)

(OUT / f"pilot_{ARGS.tag}.json").write_text(json.dumps(pre_info, indent=2))
print("saved", flush=True)

import subprocess
subprocess.run(["rclone", "copy", str(CKPT_DIR),
                "gdrive:parity/analysis_output/accessibility_ckpts/", "-q"])
r = subprocess.run(["rclone", "copy", str(OUT),
                    f"gdrive:parity/analysis_output/{OUT.name}/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
