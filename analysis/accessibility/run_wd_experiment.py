"""Weight decay vs accessibility: which d4 circuit does block 2 learn?

Builds on run_experiment.py (frozen d2 representation, measured R = Q2/Q1).
Stage 1 (embedding + block 1) trains with ZERO weight decay and is frozen, so
lambda can never touch the parent modes; stage 2 (fresh block 2 + readout)
sweeps lambda over a broad range including 0, with the exact same stage-2
init for every (R, lambda) within a seed.

Main variant decays block-2 hidden AND readout weights (the minimum-norm
theory setting); --hidden-only decays only V2. Theory:
  eta_direct = Q1/8, eta_comp = Q2/(2*sqrt(2)); tie at R = 1/sqrt(8).
  min-norm penalty for amplitude A is lambda*|A|/eta -> larger eta wins,
  boundary should NOT move with lambda; suppression at lambda_c ~ e*eta.

Fixed choices: N=1024, 3 seeds, SGD(lr=0.01, momentum=0.9), float64,
full 16-point cube (exact Walsh), alpha grid bracketing R_c densely.
"""
from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--hidden-only", action="store_true",
                help="decay only block-2 hidden weights (control variant)")
ARGS = ap.parse_args()

TAG = "hiddenonly" if ARGS.hidden_only else "both"
OUT = Path(f"/root/parity/analysis_output/accessibility_wd/{TAG}")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64

N = 1024
SEEDS = [0, 1, 2]
ALPHAS = [0.2, 0.4, 0.6, 0.75, 0.9, 1.1, 1.5, 2.0]
LAMBDAS = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LR, MOM = 0.01, 0.9
MAX1, MAX2, MIN2 = 50_000, 50_000, 15_000
SNAP_EVERY = 1000
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


def q_norms(h):
    return {s: float((h * CHI[s].unsqueeze(1)).mean(dim=0).norm()) for s in ALL_SUBSETS}


def init_lin(rows, cols, gen):
    return torch.randn(rows, cols, generator=gen, device=DEV, dtype=DT) / np.sqrt(N)


def classify(h1, V2, w_out):
    z = h1 @ V2.T
    a = torch.relu(z)
    c4 = (a * CHI[QUAD].unsqueeze(1)).mean(dim=0)
    contrib = (w_out.squeeze(0) * c4).abs()
    total = float(contrib.sum())
    base = dict(n_rel=0, frac22_w=np.nan, frac1111_w=np.nan,
                n_22=0, n_1111=0, n_mixed=0, neurons=None)
    if total < 1e-12:
        return base
    order = torch.argsort(contrib, descending=True)
    csum = torch.cumsum(contrib[order], 0)
    rel = order[: int((csum < 0.95 * total).sum()) + 1]
    zc = walsh(z)
    var_z = sum(zc[s] ** 2 for s in ALL_SUBSETS)
    singl = sum(zc[s] ** 2 for s in SINGLES) / (var_z + 1e-30)
    pair = sum(zc[s] ** 2 for s in PAIRS_REL) / (var_z + 1e-30)
    wgt = contrib[rel] / contrib[rel].sum()
    neurons = pd.DataFrame(dict(
        neuron=rel.cpu().numpy(), contrib=contrib[rel].cpu().numpy(),
        singleton_frac=singl[rel].cpu().numpy(), pair_frac=pair[rel].cpu().numpy(),
        c4=c4[rel].cpu().numpy(), v_norm=V2[rel].norm(dim=1).cpu().numpy(),
        w_out=w_out.squeeze(0)[rel].cpu().numpy(),
    ))
    return dict(n_rel=len(rel),
                frac22_w=float((wgt * pair[rel]).sum()),
                frac1111_w=float((wgt * singl[rel]).sum()),
                n_22=int((pair[rel] >= 0.6).sum()),
                n_1111=int((singl[rel] >= 0.6).sum()),
                n_mixed=int(((pair[rel] < 0.6) & (singl[rel] < 0.6)).sum()),
                neurons=neurons)


rows, traj_rows, neuron_tables, stage1_info = [], [], [], {}

for seed in SEEDS:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    E = init_lin(N, 4, gen).requires_grad_(True)
    V1 = init_lin(N, N, gen).requires_grad_(True)
    head = init_lin(2, N, gen).T.clone().requires_grad_(True)

    opt = torch.optim.SGD([E, V1, head], lr=LR, momentum=MOM)  # zero wd, stage 1
    hist = []
    for t in range(MAX1):
        opt.zero_grad(set_to_none=True)
        h0 = X @ E.T
        loss = (((h0 + torch.relu(h0 @ V1.T)) @ head - Y2) ** 2).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if t >= 10_000 and (hist[t] < 1e-10 or
                            (hist[t - 5000] - hist[t]) / max(hist[t - 5000], 1e-12) < 1e-4):
            break
    with torch.no_grad():
        h0 = X @ E.detach().T
        relu1 = torch.relu(h0 @ V1.detach().T)
        # block-1 mechanism check
        zc1 = walsh(h0 @ V1.detach().T)
        var1 = sum(zc1[s] ** 2 for s in ALL_SUBSETS)
        s_share = sum(zc1[s] ** 2 for s in SINGLES) / (var1 + 1e-30)
        shares = []
        for P in PAIRS_REL:
            cP = (relu1 * CHI[P].unsqueeze(1)).mean(dim=0).abs()
            order = torch.argsort(cP, descending=True)
            top = order[: int((torch.cumsum(cP[order], 0) < 0.95 * cP.sum()).sum()) + 1]
            shares.append(float((cP[top] / cP[top].sum() * s_share[top]).sum()))
        qn = q_norms(h0 + relu1)
    stage1_info[f"seed{seed}"] = dict(steps=t + 1, loss=hist[-1],
                                      corner_shares=shares, Q4_alpha1=qn[QUAD])
    print(f"[seed {seed}] stage1 {t+1} steps loss {hist[-1]:.2e} "
          f"corners {shares[0]:.3f}/{shares[1]:.3f}", flush=True)

    V2_0 = init_lin(N, N, gen)
    w_0 = init_lin(1, N, gen)

    for alpha in ALPHAS:
        with torch.no_grad():
            h1 = (h0 + alpha * relu1).detach()
            qn = q_norms(h1)
            Q1 = float(np.mean([qn[s] for s in SINGLES]))
            Q2 = float(np.mean([qn[s] for s in PAIRS_REL]))
            R = Q2 / Q1
        for lam in LAMBDAS:
            V2 = deepcopy(V2_0).requires_grad_(True)
            w = deepcopy(w_0).requires_grad_(True)
            groups = ([{"params": [V2], "weight_decay": lam},
                       {"params": [w], "weight_decay": 0.0}] if ARGS.hidden_only
                      else [{"params": [V2, w], "weight_decay": lam}])
            opt = torch.optim.SGD(groups, lr=LR, momentum=MOM)
            hist = []
            for t in range(MAX2):
                opt.zero_grad(set_to_none=True)
                mse = (((h1 + torch.relu(h1 @ V2.T)) @ w.T - Y4) ** 2).mean()
                mse.backward()
                opt.step()
                with torch.no_grad():
                    sq = float((V2 ** 2).sum() + (w ** 2).sum())
                obj = float(mse) + lam / 2 * sq
                hist.append(obj)
                if t % SNAP_EVERY == 0 or t == MAX2 - 1:
                    with torch.no_grad():
                        cls = classify(h1, V2.detach(), w.detach())
                        pred = (h1 + torch.relu(h1 @ V2.T)) @ w.T
                        A = float((pred * Y4).mean())
                        acc = float((torch.sign(pred) == Y4).float().mean())
                    traj_rows.append(dict(seed=seed, alpha=alpha, R=R, lam=lam,
                                          step=t, mse=float(mse), A=A, acc=acc,
                                          sqnorm=sq, frac22_w=cls["frac22_w"],
                                          frac1111_w=cls["frac1111_w"]))
                if t >= max(MIN2, 5000) and (
                        hist[t] < 1e-10 or
                        (hist[t - 5000] - hist[t]) / max(abs(hist[t - 5000]), 1e-12) < 1e-4):
                    break
            with torch.no_grad():
                cls = classify(h1, V2.detach(), w.detach())
                pred = (h1 + torch.relu(h1 @ V2.T)) @ w.T
                A = float((pred * Y4).mean())
                acc = float((torch.sign(pred) == Y4).float().mean())
                mse_f = float(((pred - Y4) ** 2).mean())
                sq = float((V2 ** 2).sum() + (w ** 2).sum())
                vN = float(V2.norm())
                wN = float(w.norm())
            if cls["neurons"] is not None:
                nt = cls["neurons"]
                nt.insert(0, "lam", lam)
                nt.insert(0, "alpha", alpha)
                nt.insert(0, "seed", seed)
                neuron_tables.append(nt)
            rows.append(dict(seed=seed, alpha=alpha, Q1=Q1, Q2=Q2, R=R, lam=lam,
                             steps=t + 1, mse=mse_f, A=A, acc=acc,
                             frac22_w=cls["frac22_w"], frac1111_w=cls["frac1111_w"],
                             n_rel=cls["n_rel"], n_22=cls["n_22"],
                             n_1111=cls["n_1111"], n_mixed=cls["n_mixed"],
                             V2_norm=vN, w_norm=wN, sqnorm=sq,
                             wd_penalty=lam / 2 * sq))
            print(f"[s{seed}] a={alpha:<4} R={R:.3f} lam={lam:<7} steps={t+1} "
                  f"mse={mse_f:.1e} A={A:.3f} f22={cls['frac22_w']:.3f} "
                  f"nrel={cls['n_rel']}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "results.csv", index=False)
pd.DataFrame(traj_rows).to_csv(OUT / "trajectories.csv", index=False)
if neuron_tables:
    pd.concat(neuron_tables).to_csv(OUT / "block2_neurons.csv", index=False)
(OUT / "stage1.json").write_text(json.dumps(stage1_info, indent=2))

# ── plots ─────────────────────────────────────────────────────────────────────
LAM_PLOT = {0.0: 1e-6}  # place lambda=0 on the log axis one decade below min>0
lam_y = lambda l: LAM_PLOT.get(l, l)

# 1. phase diagram: mean frac22_w on the (R, lambda) grid; X = failed (|A|<0.5)
m = df.groupby(["alpha", "lam"]).agg(R=("R", "mean"), f=("frac22_w", "mean"),
                                     A=("A", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(7.2, 4.6))
sc = ax.scatter(m.R, [lam_y(l) for l in m.lam], c=m.f, s=340, marker="s",
                cmap="coolwarm", vmin=0, vmax=1, edgecolors="k", linewidths=0.4)
fail = m[m.A.abs() < 0.5]
ax.scatter(fail.R, [lam_y(l) for l in fail.lam], marker="x", s=120, c="k",
           label="failed (|A|<0.5)")
ax.axvline(RC, color="r", ls="--", lw=1.2, label=r"$R_c=1/\sqrt{8}$")
ax.set_yscale("log")
ax.set_xlabel("measured $R=Q_2/Q_1$")
ax.set_ylabel(r"stage-2 weight decay $\lambda$  ($10^{-6}$ row = $\lambda=0$)")
ax.set_title(f"phase diagram: mean 2+2 fraction ({TAG} decay), 3 seeds", fontsize=10)
fig.colorbar(sc, label="fraction of 2+2→4 computation")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "phase_diagram.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase_diagram.png", dpi=170, bbox_inches="tight")

# 2. frac22 vs R at fixed lambdas
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for lam in [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    g = m[m.lam == lam].sort_values("R")
    ax.plot(g.R, g.f, "o-", label=f"λ={lam:g}")
ax.axvline(RC, color="r", ls="--")
ax.set_xlabel("measured $R=Q_2/Q_1$")
ax.set_ylabel("fraction of 2+2→4 computation")
ax.set_title("does the transition stay at $R_c$ as λ grows?", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "cuts_fixed_lambda.pdf", bbox_inches="tight")
fig.savefig(OUT / "cuts_fixed_lambda.png", dpi=170, bbox_inches="tight")

# 3. suppression: d4 amplitude vs lambda per R
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cm = plt.cm.viridis
Rs = sorted(m.alpha.unique())
for i, a in enumerate(Rs):
    g = m[m.alpha == a].sort_values("lam")
    ax.plot([lam_y(l) for l in g.lam], g.A.abs(), "o-",
            color=cm(i / max(len(Rs) - 1, 1)), label=f"R={g.R.iloc[0]:.2f}")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$  ($10^{-6}$ = 0)")
ax.set_ylabel("|d4 output amplitude A|")
ax.set_title("suppression of the d4 solution by weight decay", fontsize=10)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "suppression.pdf", bbox_inches="tight")
fig.savefig(OUT / "suppression.png", dpi=170, bbox_inches="tight")
print("saved to", OUT, flush=True)

import subprocess
r = subprocess.run(["rclone", "copy", str(OUT),
                    f"gdrive:parity/analysis_output/accessibility_wd/{TAG}/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
