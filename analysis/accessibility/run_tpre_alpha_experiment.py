"""Experiment 5: pretraining duration x alpha x weight decay (unfrozen).

Extends experiment 4 (run_alpha_unfrozen_experiment.py) with t_pre as a third
axis: the checkpoint at which V1 -> alpha*V1 is applied and d4 supervision
begins is captured after t_pre in {0, 40, 150, 1000, ...} steps of d2-only
pretraining (t_pre=0: the raw init -- fully simultaneous d2+d4 training).
Exp-4's converged runs provide the t_pre=10000 column at merge time.

Everything else matches exp 4 exactly so the columns are comparable: whole net
trainable in phase 2, d2 losses kept, global weight decay, FRESH optimizer at
d4 onset (exp-4 convention), identical checkpoint across lambda within
(seed, t_pre, alpha), R0 measured after rescaling, snapshots every 500 steps.

Split across machines: --tpres "0,40" --tag norway / --tpres "150,1000" --tag
texas_tpre. Merge script combines all tags plus the exp-4 CSVs.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tpres", required=True, help="comma list, e.g. 0,40,150,1000")
ap.add_argument("--alphas", default="0.1,0.3,0.5,0.75,1.0,1.5,2.5,4.0")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--tag", required=True)
ARGS = ap.parse_args()
TPRES = [int(t) for t in ARGS.tpres.split(",")]
ALPHAS = [float(a) for a in ARGS.alphas.split(",")]
SEEDS = [int(s) for s in ARGS.seeds.split(",")]

OUT = Path("/root/parity/analysis_output/accessibility_tpre_alpha")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64

N = 1024
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
LR, MOM = 0.01, 0.9
MAX2, MIN2 = 50_000, 15_000
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
    return h1, h2 @ W.T


def qstats(h1):
    q = {s: float((h1 * CHI[s].unsqueeze(1)).mean(dim=0).norm()) for s in ALL_SUBSETS}
    q1 = np.array([q[s] for s in SINGLES])
    q2 = np.array([q[s] for s in PAIRS_REL])
    return dict(Q1=q1.mean(), Q2=q2.mean(), Q2a=q2[0], Q2b=q2[1],
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


rows, traj_rows, pre_info = [], [], {}

for seed in SEEDS:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    params = [init_lin(N, 4, gen).requires_grad_(True),
              init_lin(N, N, gen).requires_grad_(True),
              init_lin(N, N, gen).requires_grad_(True),
              init_lin(3, N, gen).T.clone().T.contiguous().requires_grad_(True)]
    ckpts = {}
    if 0 in TPRES:
        ckpts[0] = [p.detach().clone() for p in params]
    need = max(TPRES)
    if need > 0:
        opt = torch.optim.SGD(params, lr=LR, momentum=MOM)
        for t in range(need):
            opt.zero_grad(set_to_none=True)
            _, out = forward(*params, X)
            loss = ((out[:, :2] - Y2) ** 2).mean()
            loss.backward()
            opt.step()
            if (t + 1) in TPRES:
                ckpts[t + 1] = [p.detach().clone() for p in params]
                with torch.no_grad():
                    h1, _ = forward(*[p.detach() for p in params], X)
                pre_info[f"seed{seed}_t{t+1}"] = dict(
                    loss=float(loss), **{k: v for k, v in qstats(h1).items()})
    print(f"[seed {seed}] checkpoints captured: {sorted(ckpts)}", flush=True)

    for tpre in TPRES:
        ck = ckpts[tpre]
        for alpha in ALPHAS:
            for lam in LAMBDAS:
                E = ck[0].clone().requires_grad_(True)
                V1 = (alpha * ck[1]).clone().requires_grad_(True)
                V2 = ck[2].clone().requires_grad_(True)
                W = ck[3].clone().requires_grad_(True)
                with torch.no_grad():
                    h1, _ = forward(E, V1, V2, W, X)
                    R0 = qstats(h1)["R"]
                opt = torch.optim.SGD([{"params": [E, V1, V2, W],
                                        "weight_decay": lam}], lr=LR, momentum=MOM)
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
                        sq = float(sum((p ** 2).sum() for p in (E, V1, V2, W)))
                    hist2.append(float(loss) + lam / 2 * sq)
                    if t2 % SNAP_EVERY == 0 or t2 == MAX2 - 1:
                        with torch.no_grad():
                            h1, out = forward(E, V1, V2, W, X)
                            st = qstats(h1)
                            cls = classify(h1, V2, W[2])
                            f4 = float((out[:, 2] * Y4[:, 0]).mean())
                        traj_rows.append(dict(
                            seed=seed, tpre=tpre, alpha=alpha, R0=R0, lam=lam,
                            step=t2, R=st["R"], Q1=st["Q1"], Q2=st["Q2"],
                            f4=f4, e4=1 - f4, l4=float(l4), l2=float(l2a + l2b),
                            V1_norm=float(V1.norm()), V2_norm=float(V2.norm()),
                            W_norm=float(W.norm()),
                            **{k: cls[k] for k in ("frac22_w", "frac1111_w",
                               "f22_n", "f1111_n", "fmix_n", "n_rel")}))
                    if t2 >= MIN2 and (hist2[-1] < 1e-10 or
                                       (hist2[-5000] - hist2[-1]) / max(abs(hist2[-5000]), 1e-12) < 1e-4):
                        break
                with torch.no_grad():
                    h1, out = forward(E, V1, V2, W, X)
                    st = qstats(h1)
                    cls = classify(h1, V2, W[2])
                    f4 = float((out[:, 2] * Y4[:, 0]).mean())
                    acc = float((torch.sign(out[:, 2:3]) == Y4).float().mean())
                rows.append(dict(seed=seed, tpre=tpre, alpha=alpha, R0=R0, lam=lam,
                                 steps=t2 + 1, R_final=st["R"], Q1_f=st["Q1"],
                                 Q2_f=st["Q2"], f4=f4, e4=1 - f4, acc=acc,
                                 l4=float(((out[:, 2] - Y4[:, 0]) ** 2).mean()),
                                 l2=float(((out[:, :2] - Y2) ** 2).mean()),
                                 V1_norm=float(V1.norm()), V2_norm=float(V2.norm()),
                                 W_norm=float(W.norm()),
                                 **{k: cls[k] for k in ("frac22_w", "frac1111_w",
                                    "f22_n", "f1111_n", "fmix_n", "n_rel")}))
                print(f"[s{seed}] tp={tpre:<5} a={alpha:<4} R0={R0:.3f} lam={lam:<7} "
                      f"steps={t2+1} f4={f4:.3f} Rf={st['R']:.3f} "
                      f"f22={cls['frac22_w']:.3f}", flush=True)
                # incremental persistence: append this cell so a killed run loses nothing
                rp = OUT / f"results_{ARGS.tag}.csv"
                pd.DataFrame(rows[-1:]).to_csv(rp, mode="a", header=not rp.exists(),
                                               index=False)
                tp_ = OUT / f"trajectories_{ARGS.tag}.csv"
                new_tr = [r for r in traj_rows if r["seed"] == seed and r["tpre"] == tpre
                          and r["alpha"] == alpha and r["lam"] == lam]
                pd.DataFrame(new_tr).to_csv(tp_, mode="a", header=not tp_.exists(),
                                            index=False)

(OUT / f"pretrain_{ARGS.tag}.json").write_text(json.dumps(pre_info, indent=2))
print("saved", flush=True)

import subprocess
r = subprocess.run(["rclone", "copy", str(OUT),
                    "gdrive:parity/analysis_output/accessibility_tpre_alpha/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
