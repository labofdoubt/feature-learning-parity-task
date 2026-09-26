"""Unfrozen interpolation: d2 pretraining duration x weight decay -> d4 circuit.

Interpolates between the frozen-d2 experiments and fully simultaneous d2+d4
training. Phase 1 trains the WHOLE net (embedding, both blocks, 3-output
readout with the d4 head masked) on the two d2 parents, zero wd, saving
checkpoints (params + SGD momentum buffers). Per seed, 7 checkpoints are
selected spanning R0 = Q2/Q1 at h1 (input to block 2) from t=0 to converged.
Phase 2 resumes from each checkpoint with d4 supervision ON, d2 losses kept,
ALL layers trainable, sweeping global weight decay (same lambda on embedding,
V1, V2, readout); identical checkpoint + optimizer state across lambda.

Recorded densely during phase 2: Q1(t), Q2(t), R(t) (per-mode norms too),
f4_hat(t) = E[pred_d4 * chi_1234], e4 = 1 - f4_hat, all three losses,
preactivation-based circuit fractions (contribution-weighted and count-based
with an explicit mixed class), and layer norms ||V1||, ||V2||, ||W||.

Fixed choices: N=1024, 3 seeds, SGD(lr=0.01, momentum=0.9), float64,
full 16-point cube. Theory boundary R_c = 1/sqrt(8).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/root/parity/analysis_output/accessibility_unfrozen")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64

N = 1024
SEEDS = [0, 1, 2]
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_CKPT = 7
LR, MOM = 0.01, 0.9
MAX1, MAX2, MIN2 = 50_000, 50_000, 15_000
SNAP_EVERY = 500
RC = 1 / np.sqrt(8)

SINGLES = [(0,), (1,), (2,), (3,)]
PAIRS_REL = [(0, 1), (2, 3)]
QUAD = (0, 1, 2, 3)
ALL_SUBSETS = [s for r in range(1, 5) for s in itertools.combinations(range(4), r)]

X = torch.tensor([[1.0 if (m >> k) & 1 == 0 else -1.0 for k in range(4)]
                  for m in range(16)], device=DEV, dtype=DT)
CHI = {s: X[:, list(s)].prod(dim=1) for s in ALL_SUBSETS}
Y2 = torch.stack([CHI[PAIRS_REL[0]], CHI[PAIRS_REL[1]]], dim=1)
Y4 = CHI[QUAD].unsqueeze(1)


def walsh(v):
    return {s: (v * CHI[s].unsqueeze(-1) if v.dim() == 2 else v * CHI[s]).mean(dim=0)
            for s in ALL_SUBSETS}


def init_lin(rows, cols, gen):
    return torch.randn(rows, cols, generator=gen, device=DEV, dtype=DT) / np.sqrt(N)


def forward(E, V1, V2, W, x):
    h0 = x @ E.T
    h1 = h0 + torch.relu(h0 @ V1.T)
    h2 = h1 + torch.relu(h1 @ V2.T)
    return h1, h2 @ W.T          # readout: [16,3] -> d2a, d2b, d4


def qstats(h1):
    q = {s: float((h1 * CHI[s].unsqueeze(1)).mean(dim=0).norm()) for s in ALL_SUBSETS}
    q1 = np.array([q[s] for s in SINGLES])
    q2 = np.array([q[s] for s in PAIRS_REL])
    return dict(Q1=q1.mean(), Q1_min=q1.min(), Q1_max=q1.max(),
                Q2a=q2[0], Q2b=q2[1], Q2=q2.mean(),
                R=q2.mean() / max(q1.mean(), 1e-30), Q4=q[QUAD])


def classify(h1, V2, w4):
    z = h1 @ V2.T
    a = torch.relu(z)
    c4 = (a * CHI[QUAD].unsqueeze(1)).mean(dim=0)
    contrib = (w4 * c4).abs()
    total = float(contrib.sum())
    if total < 1e-12:
        return dict(n_rel=0, frac22_w=np.nan, frac1111_w=np.nan,
                    f22_n=np.nan, f1111_n=np.nan, fmix_n=np.nan)
    order = torch.argsort(contrib, descending=True)
    csum = torch.cumsum(contrib[order], 0)
    rel = order[: int((csum < 0.95 * total).sum()) + 1]
    zc = walsh(z)
    var_z = sum(zc[s] ** 2 for s in ALL_SUBSETS)
    singl = sum(zc[s] ** 2 for s in SINGLES) / (var_z + 1e-30)
    pair = sum(zc[s] ** 2 for s in PAIRS_REL) / (var_z + 1e-30)
    wgt = contrib[rel] / contrib[rel].sum()
    n = len(rel)
    n22 = int((pair[rel] >= 0.6).sum())
    n14 = int((singl[rel] >= 0.6).sum())
    return dict(n_rel=n,
                frac22_w=float((wgt * pair[rel]).sum()),
                frac1111_w=float((wgt * singl[rel]).sum()),
                f22_n=n22 / n, f1111_n=n14 / n, fmix_n=(n - n22 - n14) / n)


rows, traj_rows, pretrain_info = [], [], {}

for seed in SEEDS:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    E0 = init_lin(N, 4, gen)
    V10 = init_lin(N, N, gen)
    V20 = init_lin(N, N, gen)
    W0 = init_lin(3, N, gen)

    # ── phase 1: pretrain everything on the two d2 targets, zero wd ──────────
    params = [p.clone().requires_grad_(True) for p in (E0, V10, V20, W0)]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOM)
    ckpts, hist = [], []

    def save_ckpt(step):
        with torch.no_grad():
            h1, _ = forward(*params, X)
            st = qstats(h1)
        bufs = [opt.state.get(p, {}).get("momentum_buffer") for p in params]
        ckpts.append(dict(step=step, R0=st["R"], qs=st,
                          params=[p.detach().clone() for p in params],
                          bufs=[None if b is None else b.clone() for b in bufs]))

    save_ckpt(0)
    for t in range(MAX1):
        opt.zero_grad(set_to_none=True)
        _, out = forward(*params, X)
        loss = ((out[:, :2] - Y2) ** 2).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss))
        s = t + 1
        if (s <= 200 and s % 5 == 0) or (s <= 1000 and s % 50 == 0) \
                or (s <= 5000 and s % 250 == 0) or s % 1000 == 0:
            save_ckpt(s)
        if s >= 10_000 and (hist[-1] < 1e-10 or
                            (hist[-5000] - hist[-1]) / max(hist[-5000], 1e-12) < 1e-4):
            break
    save_ckpt(t + 1)

    # select checkpoints by PRETRAINING DURATION (log-spaced): unfrozen
    # training saturates R0 in a narrow band (random init already gives
    # R0 ~ 0.19; corner units grow Q1 alongside Q2, capping R0 ~ 0.33), so
    # duration is the controlled axis and R0 the measured covariate.
    steps_saved = np.array([c["step"] for c in ckpts])
    want = [0, 5, 15, 40, 100, 400, int(steps_saved.max())]
    sel_idx = sorted({int(np.abs(steps_saved - wt).argmin()) for wt in want})
    sel = [ckpts[i] for i in sel_idx]
    pretrain_info[f"seed{seed}"] = dict(
        pretrain_steps=t + 1, final_loss=hist[-1],
        selected=[dict(step=c["step"], R0=c["R0"],
                       Q1=c["qs"]["Q1"], Q1_spread=[c["qs"]["Q1_min"], c["qs"]["Q1_max"]],
                       Q2a=c["qs"]["Q2a"], Q2b=c["qs"]["Q2b"], Q4=c["qs"]["Q4"])
                  for c in sel])
    print(f"[seed {seed}] pretrain {t+1} steps loss {hist[-1]:.2e}; "
          f"selected R0: {[round(c['R0'], 3) for c in sel]}", flush=True)

    # ── phase 2: from each checkpoint, d2+d4 with global wd sweep ────────────
    for ck in sel:
        for lam in LAMBDAS:
            prm = [p.clone().requires_grad_(True) for p in ck["params"]]
            opt = torch.optim.SGD([{"params": prm, "weight_decay": lam}],
                                  lr=LR, momentum=MOM)
            for p, b in zip(prm, ck["bufs"]):
                if b is not None:
                    opt.state[p] = {"momentum_buffer": b.clone()}
            E, V1, V2, W = prm
            hist2 = []
            for t2 in range(MAX2):
                opt.zero_grad(set_to_none=True)
                _, out = forward(E, V1, V2, W, X)
                l2a = ((out[:, 0] - Y2[:, 0]) ** 2).mean()
                l2b = ((out[:, 1] - Y2[:, 1]) ** 2).mean()
                l4 = ((out[:, 2] - Y4[:, 0]) ** 2).mean()
                loss = l2a + l2b + l4
                loss.backward()
                opt.step()
                with torch.no_grad():
                    sq = float(sum((p ** 2).sum() for p in prm))
                hist2.append(float(loss) + lam / 2 * sq)
                if t2 % SNAP_EVERY == 0 or t2 == MAX2 - 1:
                    with torch.no_grad():
                        h1, out = forward(E, V1, V2, W, X)
                        st = qstats(h1)
                        cls = classify(h1, V2, W[2])
                        f4 = float((out[:, 2] * Y4[:, 0]).mean())
                    traj_rows.append(dict(
                        seed=seed, pre_step=ck["step"], R0=ck["R0"], lam=lam,
                        step=t2, R=st["R"], Q1=st["Q1"], Q2=st["Q2"],
                        Q2a=st["Q2a"], Q2b=st["Q2b"], f4=f4, e4=1 - f4,
                        l4=float(l4), l2=float(l2a + l2b),
                        V1_norm=float(V1.norm()), V2_norm=float(V2.norm()),
                        W_norm=float(W.norm()), **{k: cls[k] for k in
                        ("frac22_w", "frac1111_w", "f22_n", "f1111_n", "fmix_n", "n_rel")}))
                if t2 >= MIN2 and (hist2[-1] < 1e-10 or
                                   (hist2[-5000] - hist2[-1]) / max(abs(hist2[-5000]), 1e-12) < 1e-4):
                    break
            with torch.no_grad():
                h1, out = forward(E, V1, V2, W, X)
                st = qstats(h1)
                cls = classify(h1, V2, W[2])
                f4 = float((out[:, 2] * Y4[:, 0]).mean())
                acc = float((torch.sign(out[:, 2:3]) == Y4).float().mean())
            rows.append(dict(seed=seed, pre_step=ck["step"], R0=ck["R0"], lam=lam,
                             steps=t2 + 1, R_final=st["R"], Q1_f=st["Q1"], Q2_f=st["Q2"],
                             f4=f4, e4=1 - f4, acc=acc,
                             l4=float(((out[:, 2] - Y4[:, 0]) ** 2).mean()),
                             l2=float(((out[:, :2] - Y2) ** 2).mean()),
                             V1_norm=float(V1.norm()), V2_norm=float(V2.norm()),
                             W_norm=float(W.norm()),
                             **{k: cls[k] for k in ("frac22_w", "frac1111_w",
                                "f22_n", "f1111_n", "fmix_n", "n_rel")}))
            print(f"[s{seed}] pre={ck['step']:<5} R0={ck['R0']:.3f} lam={lam:<7} "
                  f"steps={t2+1} f4={f4:.3f} Rf={st['R']:.3f} "
                  f"f22={cls['frac22_w']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "results.csv", index=False)
pd.DataFrame(traj_rows).to_csv(OUT / "trajectories.csv", index=False)
(OUT / "pretrain.json").write_text(json.dumps(pretrain_info, indent=2))

# ── plots ─────────────────────────────────────────────────────────────────────
lam_y = lambda l: l if l > 0 else 1e-5 / 3


def phase_plot(xcol, fname, xlabel):
    m = df.groupby(["pre_step", "lam"]).agg(
        x=(xcol, "mean"), f=("frac22_w", "mean"), f4=("f4", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(m.x, [lam_y(l) for l in m.lam], c=m.f, s=340, marker="s",
                    cmap="coolwarm", vmin=0, vmax=1, edgecolors="k", linewidths=0.4)
    fail = m[m.f4.abs() < 0.5]
    ax.scatter(fail.x, [lam_y(l) for l in fail.lam], marker="x", s=120, c="k",
               label="failed (|f4|<0.5)")
    ax.axvline(RC, color="r", ls="--", lw=1.2, label=r"$R_c=1/\sqrt{8}$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"global weight decay $\lambda$  (bottom row = 0)")
    ax.set_title(f"unfrozen d2+d4: mean 2+2 fraction vs ({xlabel}, λ), 3 seeds",
                 fontsize=10)
    fig.colorbar(sc, label="fraction of 2+2→4 computation")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=170, bbox_inches="tight")


phase_plot("R0", "phase_R0", "$R_0$ (at d4 onset)")
phase_plot("R_final", "phase_Rfinal", "$R_{final}$")

# trajectories for representative conditions (seed 0)
tr = pd.read_csv(OUT / "trajectories.csv")
pres = sorted(tr[tr.seed == 0].pre_step.unique())
pre_sel = [pres[0], pres[len(pres) // 2], pres[-1]]
lam_sel = [0.0, 3e-3, 3e-2]
fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=False)
for i, pre in enumerate(pre_sel):
    for j, lam in enumerate(lam_sel):
        ax = axes[i, j]
        g = tr[(tr.seed == 0) & (tr.pre_step == pre) & (tr.lam == lam)].sort_values("step")
        ax.plot(g.step, g.frac22_w, color="tab:blue", label="f 2+2")
        ax.plot(g.step, g.frac1111_w, color="tab:orange", label="f 1⁴")
        ax.plot(g.step, g.e4.abs(), color="tab:green", ls=":", label="|e4|")
        ax2 = ax.twinx()
        ax2.plot(g.step, g.R, color="k", lw=1, alpha=0.6)
        ax2.axhline(RC, color="r", ls="--", lw=0.7)
        ax2.set_ylabel("R(t)", fontsize=7)
        ax.set_title(f"pre={pre} (R0={g.R0.iloc[0]:.2f}), λ={lam:g}", fontsize=8)
        if i == 0 and j == 0:
            ax.legend(fontsize=6)
fig.suptitle("phase-2 dynamics (seed 0): circuit fractions, |e4|, and R(t)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "trajectories_panels.pdf", bbox_inches="tight")
fig.savefig(OUT / "trajectories_panels.png", dpi=150, bbox_inches="tight")

# phase-plane R(t) vs f22(t)
fig, ax = plt.subplots(figsize=(6.6, 4.6))
cm = plt.cm.plasma
for i, pre in enumerate(pre_sel):
    for j, lam in enumerate(lam_sel):
        g = tr[(tr.seed == 0) & (tr.pre_step == pre) & (tr.lam == lam)].sort_values("step")
        c = cm((i * 3 + j) / 8)
        ax.plot(g.R, g.frac22_w, "-", color=c, lw=1.2,
                label=f"pre={pre}, λ={lam:g}")
        ax.plot(g.R.iloc[:1], g.frac22_w.iloc[:1], "o", color=c, ms=6)
        ax.plot(g.R.iloc[-1:], g.frac22_w.iloc[-1:], "s", color=c, ms=6)
ax.axvline(RC, color="r", ls="--", label=r"$R_c$")
ax.set_xlabel("R(t)")
ax.set_ylabel("f 2+2(t)")
ax.set_title("phase plane: does R cross $R_c$ before the circuit flips? (o start, ■ end)",
             fontsize=9)
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "phase_plane.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase_plane.png", dpi=170, bbox_inches="tight")
print("saved to", OUT, flush=True)

import subprocess
r = subprocess.run(["rclone", "copy", str(OUT),
                    "gdrive:parity/analysis_output/accessibility_unfrozen/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
