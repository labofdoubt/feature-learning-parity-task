"""Experiment 4: alpha-controlled initial accessibility, unfrozen dynamics.

Phase 1 pretrains the whole net (embedding, V1, V2, 3-output readout with the
d4 head masked) on the two d2 parents to convergence, zero wd (single large
t_pre -- no duration sweep). At d4 onset, block-1 weights are rescaled
V1 -> alpha*V1 (ReLU homogeneity: rescales all block-1 features), which sets
the measured initial accessibility R0 = Q2/Q1 on either side of
R_c = 1/sqrt(8). Phase 2 then trains EVERYTHING (d2 losses kept, d4 on) with
a global weight decay lambda; identical initial state across lambda within
(seed, alpha); fresh optimizer at onset (pretraining momentum ~ 0 at
convergence). R(t), circuit fractions, f4, losses, norms recorded every 500
steps -- the drift of R from its alpha-controlled initial value is a primary
observable.

Three selective-decay controls are available. ``--wd-block2-only`` reproduces
exp. 6 by decaying V2 alone. ``--wd-block2-readout`` decays V2 and the d4
readout row W[2] with the same lambda, while leaving E, V1, and the two d2
readout rows decay-free. ``--wd-d4-adapter`` freezes E and the rescaled,
pretrained V1, adds a small independently initialized parallel block-1
adapter V1_adapter, and applies matched decay to V1_adapter, V2, and W[2].
This last mode taxes newly formed d4 routes in either block without decaying
the pretrained parent-producing base.

Split across machines by alpha: --alphas "0.1,0.5,1.0,2.5" --tag texas.
Outputs results_<tag>.csv / trajectories_<tag>.csv / pretrain_<tag>.json in a
mode-specific analysis_output/accessibility_alpha_*/ directory, synced to
Drive; a separate merge script concatenates the halves and draws the figures.

Fixed choices: N=1024, seeds 0-2, SGD(lr=0.01, momentum=0.9), float64,
full 16-point cube, MIN 15k / MAX 50k steps with plateau stopping.
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
ap.add_argument("--alphas", required=True, help="comma list, e.g. 0.1,0.5,1.0,2.5")
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", default="0,1,2", help="comma list of seeds")
wd = ap.add_mutually_exclusive_group()
wd.add_argument("--wd-block2-only", action="store_true",
                help="apply phase-2 weight decay only to V2 (exp 6)")
wd.add_argument("--wd-block2-readout", action="store_true",
                help="apply matched phase-2 weight decay to V2 and only the "
                     "d4 readout row W[2]")
wd.add_argument("--wd-d4-adapter", action="store_true",
                help="freeze E and pretrained V1, add a parallel trainable "
                     "block-1 adapter, and decay the adapter, V2, and W[2]")
ap.add_argument("--adapter-init-scale", type=float, default=1e-3,
                help="scale of the independently initialized block-1 adapter "
                     "relative to standard initialization (default: 1e-3)")
ARGS = ap.parse_args()
if ARGS.wd_d4_adapter and ARGS.adapter_init_scale <= 0:
    ap.error("--adapter-init-scale must be positive in --wd-d4-adapter mode")
ALPHAS = [float(a) for a in ARGS.alphas.split(",")]

WD_MODE = ("d4_adapter" if ARGS.wd_d4_adapter else
           "v2_w4" if ARGS.wd_block2_readout else
           "v2_only" if ARGS.wd_block2_only else "global")
OUT_NAME = {"global": "accessibility_alpha_unfrozen",
            "v2_only": "accessibility_alpha_v2wd",
            "v2_w4": "accessibility_alpha_v2w4wd",
            "d4_adapter": "accessibility_alpha_d4adapter_wd"}[WD_MODE]
OUT = Path("/root/parity/analysis_output") / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("/root/parity/analysis_output/accessibility_ckpts")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64

N = 1024
SEEDS = [int(s) for s in ARGS.seeds.split(",")]
LAMBDAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
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


def forward(E, V1, V2, W, x, V1_adapter=None, return_parts=False):
    h0 = x @ E.T
    h1_base = h0 + torch.relu(h0 @ V1.T)
    h1_adapter = (torch.zeros_like(h1_base) if V1_adapter is None else
                  torch.relu(h0 @ V1_adapter.T))
    h1 = h1_base + h1_adapter
    h2_residual = torch.relu(h1 @ V2.T)
    out = (h1 + h2_residual) @ W.T
    if return_parts:
        return h1, out, (h1_base, h1_adapter, h2_residual)
    return h1, out


def d4_route_contributions(parts, w4):
    """Signed target coefficients from base, adapter, and block-2 residual."""
    h1_base, h1_adapter, h2_residual = parts
    target = CHI[QUAD]
    q4_adapter = (h1_adapter * target.unsqueeze(1)).mean(dim=0)
    q4_block2 = (h2_residual * target.unsqueeze(1)).mean(dim=0)
    return dict(
        f4_base=float(((h1_base @ w4) * target).mean()),
        f4_adapter=float(((h1_adapter @ w4) * target).mean()),
        f4_block2=float(((h2_residual @ w4) * target).mean()),
        Q4_adapter=float(q4_adapter.norm()),
        Q4_block2=float(q4_block2.norm()),
    )


def qstats(h1):
    q = {s: float((h1 * CHI[s].unsqueeze(1)).mean(dim=0).norm()) for s in ALL_SUBSETS}
    q1 = np.array([q[s] for s in SINGLES])
    q2 = np.array([q[s] for s in PAIRS_REL])
    return dict(Q1=q1.mean(), Q2=q2.mean(), Q2a=q2[0], Q2b=q2[1],
                R=q2.mean() / max(q1.mean(), 1e-30), Q4=q[QUAD])


def classify_preactivation(z, w4):
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


def classify(h1, V2, w4):
    return classify_preactivation(h1 @ V2.T, w4)


rows, traj_rows, pre_info = [], [], {}

for seed in SEEDS:
    ckpt_path = CKPT_DIR / f"pretrain10k_seed{seed}.pt"
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=DEV, weights_only=True)
        ck = [saved[k].to(DEV, DT) for k in ("E", "V1", "V2", "W")]
        print(f"[seed {seed}] loaded pretrain checkpoint {ckpt_path.name}", flush=True)
        hist = [float("nan")]
        t = -1
    else:
        gen = torch.Generator(device=DEV).manual_seed(seed)
        params = [init_lin(N, 4, gen).requires_grad_(True),
                  init_lin(N, N, gen).requires_grad_(True),
                  init_lin(N, N, gen).requires_grad_(True),
                  init_lin(3, N, gen).T.clone().T.contiguous().requires_grad_(True)]
        opt = torch.optim.SGD(params, lr=LR, momentum=MOM)
        hist = []
        for t in range(MAX1):
            opt.zero_grad(set_to_none=True)
            _, out = forward(*params, X)
            loss = ((out[:, :2] - Y2) ** 2).mean()
            loss.backward()
            opt.step()
            hist.append(float(loss))
            if t + 1 >= 10_000 and (hist[-1] < 1e-10 or
                                    (hist[-5000] - hist[-1]) / max(hist[-5000], 1e-12) < 1e-4):
                break
        ck = [p.detach().clone() for p in params]
        torch.save({"E": ck[0].cpu(), "V1": ck[1].cpu(), "V2": ck[2].cpu(),
                    "W": ck[3].cpu(), "seed": seed, "steps": t + 1,
                    "loss": hist[-1]}, ckpt_path)
        print(f"[seed {seed}] saved pretrain checkpoint -> {ckpt_path.name}", flush=True)
    with torch.no_grad():
        h1, _ = forward(*ck, X)
        st1 = qstats(h1)
    pre_info[f"seed{seed}"] = dict(steps=t + 1, loss=hist[-1], R_alpha1=st1["R"],
                                   Q1=st1["Q1"], Q2a=st1["Q2a"], Q2b=st1["Q2b"])
    print(f"[seed {seed}] pretrain {t+1} steps loss {hist[-1]:.2e} R(a=1)={st1['R']:.3f}",
          flush=True)

    # One seed-specific initialization is cloned into every (alpha, lambda)
    # cell, so lambda is the only intervention within each alpha column.  A
    # small nonzero scale preserves the pretrained representation at onset
    # without the dead ReLU gradients produced by exact zero initialization.
    if WD_MODE == "d4_adapter":
        adapter_gen = torch.Generator(device=DEV).manual_seed(100_000 + seed)
        adapter0 = ARGS.adapter_init_scale * init_lin(N, N, adapter_gen)
    else:
        adapter0 = None

    for alpha in ALPHAS:
        for lam in LAMBDAS:
            freeze_base = WD_MODE == "d4_adapter"
            E = ck[0].clone().requires_grad_(not freeze_base)
            V1 = (alpha * ck[1]).clone().requires_grad_(not freeze_base)
            V1_adapter = (adapter0.clone().requires_grad_(True)
                          if adapter0 is not None else None)
            V2 = ck[2].clone().requires_grad_(True)
            W = ck[3].clone().requires_grad_(True)
            with torch.no_grad():
                h1, _, parts = forward(E, V1, V2, W, X, V1_adapter,
                                       return_parts=True)
                R0 = qstats(h1)["R"]
                R0_base = qstats(parts[0])["R"]
            groups = ([{"params": [V1_adapter, V2], "weight_decay": lam},
                       {"params": [W], "weight_decay": 0.0}]
                      if WD_MODE == "d4_adapter" else
                      [{"params": [V2], "weight_decay": lam},
                       {"params": [E, V1, W], "weight_decay": 0.0}]
                      if WD_MODE in ("v2_only", "v2_w4") else
                      [{"params": [E, V1, V2, W], "weight_decay": lam}])
            opt = torch.optim.SGD(groups, lr=LR, momentum=MOM)
            hist2 = []
            for t2 in range(MAX2):
                opt.zero_grad(set_to_none=True)
                _, out = forward(E, V1, V2, W, X, V1_adapter)
                l2a = ((out[:, 0] - Y2[:, 0]) ** 2).mean()
                l2b = ((out[:, 1] - Y2[:, 1]) ** 2).mean()
                l4 = ((out[:, 2] - Y4[:, 0]) ** 2).mean()
                data_loss = l2a + l2b + l4
                # W is a single 3 x N parameter, so optimizer weight decay
                # would also penalize the two d2 heads. Penalize only W[2]
                # explicitly in controls with a decayed d4 readout.
                w4_penalty = (lam / 2 * W[2].square().sum()
                              if WD_MODE in ("v2_w4", "d4_adapter") else 0.0)
                objective = data_loss + w4_penalty
                objective.backward()
                opt.step()
                with torch.no_grad():
                    if WD_MODE == "global":
                        reg_sq = float(sum(p.square().sum() for p in (E, V1, V2, W)))
                    elif WD_MODE == "v2_w4":
                        reg_sq = float(V2.square().sum() + W[2].square().sum())
                    elif WD_MODE == "d4_adapter":
                        reg_sq = float(V1_adapter.square().sum() +
                                       V2.square().sum() + W[2].square().sum())
                    else:
                        reg_sq = float(V2.square().sum())
                hist2.append(float(data_loss) + lam / 2 * reg_sq)
                if t2 % SNAP_EVERY == 0 or t2 == MAX2 - 1:
                    with torch.no_grad():
                        h1, out, parts = forward(E, V1, V2, W, X, V1_adapter,
                                                 return_parts=True)
                        st = qstats(h1)
                        st_base = qstats(parts[0])
                        cls = classify(h1, V2, W[2])
                        routes = d4_route_contributions(parts, W[2])
                        if V1_adapter is not None:
                            z_adapter = (X @ E.T) @ V1_adapter.T
                            adapter_cls = classify_preactivation(z_adapter, W[2])
                        else:
                            adapter_cls = dict(n_rel=0, frac22_w=np.nan,
                                               frac1111_w=np.nan)
                        f4 = float((out[:, 2] * Y4[:, 0]).mean())
                        traj_rows.append(dict(
                            wd_mode=WD_MODE,
                            seed=seed, alpha=alpha, R0=R0, R0_base=R0_base,
                            lam=lam, step=t2,
                            R=st["R"], Q1=st["Q1"], Q2=st["Q2"], f4=f4, e4=1 - f4,
                            R_base=st_base["R"], Q4_base=st_base["Q4"],
                            l4=float(l4), l2=float(l2a + l2b),
                            V1_norm=float(V1.norm()), V2_norm=float(V2.norm()),
                            W_norm=float(W.norm()), W4_norm=float(W[2].norm()),
                            V1_adapter_norm=(float(V1_adapter.norm())
                                             if V1_adapter is not None else 0.0),
                            adapter_n_rel=adapter_cls["n_rel"],
                            adapter_frac22_w=adapter_cls["frac22_w"],
                            adapter_frac1111_w=adapter_cls["frac1111_w"],
                            **routes,
                            **{k: cls[k] for k in ("frac22_w", "frac1111_w",
                               "f22_n", "f1111_n", "fmix_n", "n_rel")}))
                if t2 >= MIN2 and (hist2[-1] < 1e-10 or
                                   (hist2[-5000] - hist2[-1]) / max(abs(hist2[-5000]), 1e-12) < 1e-4):
                    break
            with torch.no_grad():
                h1, out, parts = forward(E, V1, V2, W, X, V1_adapter,
                                         return_parts=True)
                st = qstats(h1)
                st_base = qstats(parts[0])
                cls = classify(h1, V2, W[2])
                routes = d4_route_contributions(parts, W[2])
                if V1_adapter is not None:
                    z_adapter = (X @ E.T) @ V1_adapter.T
                    adapter_cls = classify_preactivation(z_adapter, W[2])
                else:
                    adapter_cls = dict(n_rel=0, frac22_w=np.nan,
                                       frac1111_w=np.nan)
                f4 = float((out[:, 2] * Y4[:, 0]).mean())
                acc = float((torch.sign(out[:, 2:3]) == Y4).float().mean())
            rows.append(dict(wd_mode=WD_MODE,
                             seed=seed, alpha=alpha, R0=R0, R0_base=R0_base,
                             lam=lam, steps=t2 + 1,
                             R_final=st["R"], Q1_f=st["Q1"], Q2_f=st["Q2"],
                             R_base_f=st_base["R"], Q4_base_f=st_base["Q4"],
                             f4=f4, e4=1 - f4, acc=acc,
                             l4=float(((out[:, 2] - Y4[:, 0]) ** 2).mean()),
                             l2=float(((out[:, :2] - Y2) ** 2).mean()),
                             V1_norm=float(V1.norm()), V2_norm=float(V2.norm()),
                             W_norm=float(W.norm()), W4_norm=float(W[2].norm()),
                             V1_adapter_norm=(float(V1_adapter.norm())
                                              if V1_adapter is not None else 0.0),
                             adapter_n_rel=adapter_cls["n_rel"],
                             adapter_frac22_w=adapter_cls["frac22_w"],
                             adapter_frac1111_w=adapter_cls["frac1111_w"],
                             **routes,
                             **{k: cls[k] for k in ("frac22_w", "frac1111_w",
                                "f22_n", "f1111_n", "fmix_n", "n_rel")}))
            print(f"[s{seed}] a={alpha:<4} R0={R0:.3f} lam={lam:<7} steps={t2+1} "
                  f"f4={f4:.3f} Rf={st['R']:.3f} f22={cls['frac22_w']:.3f}", flush=True)
            # incremental persistence: append this cell so a killed run loses nothing
            rp = OUT / f"results_{ARGS.tag}.csv"
            pd.DataFrame(rows[-1:]).to_csv(rp, mode="a", header=not rp.exists(), index=False)
            tp_ = OUT / f"trajectories_{ARGS.tag}.csv"
            new_tr = [r for r in traj_rows if r["seed"] == seed
                      and r["alpha"] == alpha and r["lam"] == lam]
            pd.DataFrame(new_tr).to_csv(tp_, mode="a", header=not tp_.exists(), index=False)

(OUT / f"pretrain_{ARGS.tag}.json").write_text(json.dumps(pre_info, indent=2))
print("saved", flush=True)

import subprocess
subprocess.run(["rclone", "copy", str(CKPT_DIR),
                "gdrive:parity/analysis_output/accessibility_ckpts/", "-q"])
r = subprocess.run(["rclone", "copy", str(OUT),
                    f"gdrive:parity/analysis_output/{OUT.name}/", "-q"])
print("drive sync:", "ok" if r.returncode == 0 else "FAILED")
