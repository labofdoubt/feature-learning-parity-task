"""Accessibility-controlled circuit selection (4-bit, two ReLU blocks).

Does the measured accessibility ratio R = Q2/Q1 of the frozen block-1
representation determine whether a fresh block 2 learns x1x2x3x4 as
1+1+1+1->4 (from singletons) or 2+2->4 (from the d2 parents)?
Theory: balanced p-parent ReLU gates give |c4|/|v| = kappa_p*|q|/sqrt(p),
so the compositional route wins iff Q2/Q1 > 1/sqrt(8) ~ 0.354.

Design (per seed):
  stage 1: train embedding+block1(+temp head) full-batch on [x1x2, x3x4]
           until converged; freeze; verify block-1 pairs are 1+1->2 corners.
  rescale: h1_alpha(x) = h0 + alpha*ReLU(V1 h0)  (exact for ReLU, alpha>0);
           measure Q1(alpha), Q2(alpha) directly from h1_alpha.
  stage 2: for each alpha, train an IDENTICALLY-initialized fresh block2 +
           readout (same tensors copied per alpha) on x1x2x3x4, full batch.
  classify every d4-contributing block-2 neuron by the Walsh decomposition
  of its PREACTIVATION z_j = v_j . h1_alpha: singleton-share vs parent-share.

Everything uses the full 16-point cube: all expectations are exact.
Fixed choices (not in the spec): N=1024, 4 seeds, SGD(lr=0.01, momentum=0.9),
alpha grid spanning R below/above R_c. No weight decay anywhere.
"""
from __future__ import annotations

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

OUT = Path("/root/parity/analysis_output/accessibility_4bit")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64  # 16 inputs: exactness is free

N = 1024
SEEDS = [0, 1, 2, 3]
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
LR = 0.01
MOM = 0.9
MAX1, MAX2 = 50_000, 50_000
MIN2 = 15_000
SNAP_EVERY = 1000
RC = 1 / np.sqrt(8)

SINGLES = [(0,), (1,), (2,), (3,)]
PAIRS_REL = [(0, 1), (2, 3)]
QUAD = (0, 1, 2, 3)
ALL_SUBSETS = [s for r in range(1, 5) for s in itertools.combinations(range(4), r)]

# full cube and characters
X = torch.tensor([[1.0 if (m >> k) & 1 == 0 else -1.0 for k in range(4)]
                  for m in range(16)], device=DEV, dtype=DT)
CHI = {s: X[:, list(s)].prod(dim=1) for s in ALL_SUBSETS}
Y2 = torch.stack([CHI[PAIRS_REL[0]], CHI[PAIRS_REL[1]]], dim=1)
Y4 = CHI[QUAD].unsqueeze(1)


def walsh(v: torch.Tensor) -> dict:
    """v: [16] or [16,K] -> {subset: coeff} exact Walsh coefficients."""
    return {s: (v * CHI[s].unsqueeze(-1) if v.dim() == 2 else v * CHI[s]).mean(dim=0)
            for s in ALL_SUBSETS}


def q_norms(h: torch.Tensor) -> dict:
    """h: [16,N] residual stream -> {subset: ||q_S||}."""
    return {s: float((h * CHI[s].unsqueeze(1)).mean(dim=0).norm()) for s in ALL_SUBSETS}


def init_lin(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(rows, cols, generator=gen, device=DEV, dtype=DT) / np.sqrt(N)


def train_full_batch(params: list, closure, max_steps: int, min_steps: int,
                     snap=None, tol: float = 1e-4, window: int = 5000):
    opt = torch.optim.SGD(params, lr=LR, momentum=MOM)
    hist = []
    for t in range(max_steps):
        opt.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()
        opt.step()
        lv = float(loss)
        hist.append(lv)
        if snap is not None and (t % SNAP_EVERY == 0 or t == max_steps - 1):
            snap(t, lv)
        if t >= max(min_steps, window):
            prev = hist[t - window]
            if lv < 1e-10 or (prev - lv) / max(prev, 1e-12) < tol:
                if snap is not None:
                    snap(t, lv)
                return t + 1, lv
    return max_steps, hist[-1]


def block1_mechanism(h0: torch.Tensor, V1: torch.Tensor) -> dict:
    """Are the pair modes made by 1+1->2 corner units? (spec: inspect block 1)"""
    z = h0 @ V1.T                      # [16,N]
    a = torch.relu(z)
    zc = walsh(z)
    var_z = sum(zc[s] ** 2 for s in ALL_SUBSETS)          # [N]
    singl_share = sum(zc[s] ** 2 for s in SINGLES) / (var_z + 1e-30)
    out = {}
    for P in PAIRS_REL:
        cP = (a * CHI[P].unsqueeze(1)).mean(dim=0).abs()  # [N]
        order = torch.argsort(cP, descending=True)
        csum = torch.cumsum(cP[order], 0)
        top = order[: int((csum < 0.95 * cP.sum()).sum()) + 1]
        w = cP[top] / cP[top].sum()
        out[str(P)] = dict(
            n_top=len(top),
            weighted_singleton_share=float((w * singl_share[top]).sum()),
        )
    return out


def classify_block2(h1: torch.Tensor, V2: torch.Tensor, w_out: torch.Tensor) -> dict:
    """Preactivation-based circuit decomposition of the d4 computation."""
    z = h1 @ V2.T
    a = torch.relu(z)
    c4 = (a * CHI[QUAD].unsqueeze(1)).mean(dim=0)          # [N]
    contrib = (w_out.squeeze(0) * c4).abs()
    total = float(contrib.sum())
    if total < 1e-12:
        return dict(n_rel=0, frac22_w=np.nan, frac1111_w=np.nan,
                    n_22=0, n_1111=0, n_mixed=0, neurons=None)
    order = torch.argsort(contrib, descending=True)
    csum = torch.cumsum(contrib[order], 0)
    rel = order[: int((csum < 0.95 * total).sum()) + 1]

    zc = walsh(z)
    var_z = sum(zc[s] ** 2 for s in ALL_SUBSETS)
    singl = sum(zc[s] ** 2 for s in SINGLES) / (var_z + 1e-30)
    pair = sum(zc[s] ** 2 for s in PAIRS_REL) / (var_z + 1e-30)

    wgt = contrib[rel] / contrib[rel].sum()
    frac22_w = float((wgt * pair[rel]).sum())
    frac1111_w = float((wgt * singl[rel]).sum())
    n22 = int((pair[rel] >= 0.6).sum())
    n1111 = int((singl[rel] >= 0.6).sum())
    neurons = pd.DataFrame(dict(
        neuron=rel.cpu().numpy(),
        contrib=contrib[rel].cpu().numpy(),
        singleton_frac=singl[rel].cpu().numpy(),
        pair_frac=pair[rel].cpu().numpy(),
        c4=c4[rel].cpu().numpy(),
        v_norm=V2[rel].norm(dim=1).cpu().numpy(),
        w_out=w_out.squeeze(0)[rel].cpu().numpy(),
    ))
    return dict(n_rel=len(rel), frac22_w=frac22_w, frac1111_w=frac1111_w,
                n_22=n22, n_1111=n1111, n_mixed=len(rel) - n22 - n1111,
                neurons=neurons)


rows, traj_rows, neuron_tables, stage1_info = [], [], [], {}

for seed in SEEDS:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    E = init_lin(N, 4, gen).requires_grad_(True)
    V1 = init_lin(N, N, gen).requires_grad_(True)
    head = init_lin(2, N, gen).T.clone().requires_grad_(True)   # [N,2]

    def closure1():
        h0 = X @ E.T
        h1 = h0 + torch.relu(h0 @ V1.T)
        return ((h1 @ head - Y2) ** 2).mean()

    steps1, loss1 = train_full_batch([E, V1, head], closure1, MAX1, 10_000)

    with torch.no_grad():
        E_f, V1_f = E.detach().clone(), V1.detach().clone()
        h0 = X @ E_f.T
        relu1 = torch.relu(h0 @ V1_f.T)
        mech = block1_mechanism(h0, V1_f)
        qn1 = q_norms(h0 + relu1)
    stage1_info[f"seed{seed}"] = dict(
        steps=steps1, final_loss=loss1, block1_mechanism=mech,
        Q1_alpha1=[qn1[s] for s in SINGLES],
        Q2_alpha1=[qn1[s] for s in PAIRS_REL],
        Q4_alpha1=qn1[QUAD],
    )
    print(f"[seed {seed}] stage1: {steps1} steps, loss {loss1:.3e}, "
          f"corner-share {mech[str(PAIRS_REL[0])]['weighted_singleton_share']:.3f}/"
          f"{mech[str(PAIRS_REL[1])]['weighted_singleton_share']:.3f}", flush=True)

    # identical fresh block-2/readout initial tensors for every alpha
    V2_0 = init_lin(N, N, gen)
    w_0 = init_lin(1, N, gen)

    for alpha in ALPHAS:
        with torch.no_grad():
            h1 = (h0 + alpha * relu1)
            qn = q_norms(h1)
            Q1s = np.array([qn[s] for s in SINGLES])
            Q2s = np.array([qn[s] for s in PAIRS_REL])
            R = Q2s.mean() / Q1s.mean()
        V2 = deepcopy(V2_0).requires_grad_(True)
        w = deepcopy(w_0).requires_grad_(True)
        h1c = h1.detach()

        def closure2():
            h2 = h1c + torch.relu(h1c @ V2.T)
            return ((h2 @ w.T - Y4) ** 2).mean()

        def snap(t, lv):
            with torch.no_grad():
                cls = classify_block2(h1c, V2.detach(), w.detach())
                pred = (h1c + torch.relu(h1c @ V2.T)) @ w.T
                acc = float((torch.sign(pred) == Y4).float().mean())
            traj_rows.append(dict(seed=seed, alpha=alpha, R=R, step=t, loss=lv,
                                  acc=acc, frac22_w=cls["frac22_w"],
                                  frac1111_w=cls["frac1111_w"]))

        steps2, loss2 = train_full_batch([V2, w], closure2, MAX2, MIN2, snap=snap)

        with torch.no_grad():
            cls = classify_block2(h1c, V2.detach(), w.detach())
            pred = (h1c + torch.relu(h1c @ V2.T)) @ w.T
            acc = float((torch.sign(pred) == Y4).float().mean())
        if cls["neurons"] is not None:
            nt = cls["neurons"]
            nt.insert(0, "alpha", alpha)
            nt.insert(0, "seed", seed)
            neuron_tables.append(nt)
        rows.append(dict(
            seed=seed, alpha=alpha,
            Q1_mean=Q1s.mean(), Q1_min=Q1s.min(), Q1_max=Q1s.max(),
            Q2_a=Q2s[0], Q2_b=Q2s[1], Q2_mean=Q2s.mean(),
            R=R, Q4_frozen=qn[QUAD],
            steps=steps2, loss=loss2, acc=acc,
            n_rel=cls["n_rel"], frac22_w=cls["frac22_w"],
            frac1111_w=cls["frac1111_w"], n_22=cls["n_22"],
            n_1111=cls["n_1111"], n_mixed=cls["n_mixed"],
        ))
        print(f"[seed {seed}] a={alpha:<5} R={R:.3f} steps={steps2} "
              f"loss={loss2:.2e} acc={acc:.3f} f22={cls['frac22_w']:.3f} "
              f"f1111={cls['frac1111_w']:.3f} nrel={cls['n_rel']}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "results.csv", index=False)
pd.DataFrame(traj_rows).to_csv(OUT / "trajectories.csv", index=False)
pd.concat(neuron_tables).to_csv(OUT / "block2_neurons.csv", index=False)
(OUT / "stage1.json").write_text(json.dumps(stage1_info, indent=2))

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for seed, g in df.groupby("seed"):
    g = g.sort_values("R")
    ax.plot(g.R, g.frac22_w, "o-", alpha=0.5, label=f"seed {seed}")
m = df.groupby("alpha").agg(R=("R", "mean"), f=("frac22_w", "mean")).sort_values("R")
ax.plot(m.R, m.f, "k-", lw=2.5, label="mean")
ax.axvline(RC, color="r", ls="--", label=r"$R_c=1/\sqrt{8}$")
ax.set_xlabel("measured  $R=Q_2/Q_1$"); ax.set_ylabel("fraction of 2+2→4 computation")
ax.set_xscale("log"); ax.legend(fontsize=8)
ax.set_title("contribution-weighted parent share of block-2 preactivations", fontsize=9)
ax = axes[1]
for seed, g in df.groupby("seed"):
    g = g.sort_values("R")
    ax.plot(g.R, g.frac1111_w, "o-", alpha=0.5)
m2 = df.groupby("alpha").agg(R=("R", "mean"), f=("frac1111_w", "mean")).sort_values("R")
ax.plot(m2.R, m2.f, "k-", lw=2.5)
ax.axvline(RC, color="r", ls="--")
ax.set_xlabel("measured  $R=Q_2/Q_1$"); ax.set_ylabel("fraction of 1+1+1+1→4 computation")
ax.set_xscale("log")
ax.set_title("singleton share of the same neurons", fontsize=9)
fig.suptitle("Accessibility transition: which circuit does block 2 learn?", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "main_transition.pdf", bbox_inches="tight")
fig.savefig(OUT / "main_transition.png", dpi=170, bbox_inches="tight")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for i, (col, lab) in enumerate([("Q1_mean", "$Q_1(\\alpha)$"),
                                ("Q2_mean", "$Q_2(\\alpha)$"), ("R", "$Q_2/Q_1$")]):
    ax = axes[i]
    for seed, g in df.groupby("seed"):
        g = g.sort_values("alpha")
        ax.plot(g.alpha, g[col], "o-", alpha=0.6)
    if col == "R":
        ax.axhline(RC, color="r", ls="--")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(lab)
    ax.set_xscale("symlog", linthresh=0.05)
fig.suptitle("rescaling diagnostics: measured mode norms vs alpha", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "alpha_diagnostics.pdf", bbox_inches="tight")
fig.savefig(OUT / "alpha_diagnostics.png", dpi=170, bbox_inches="tight")
print("saved to", OUT, flush=True)

import subprocess
r = subprocess.run(["rclone", "copy", str(OUT),
                    "gdrive:parity/analysis_output/accessibility_4bit/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED (rclone.conf missing?)")
