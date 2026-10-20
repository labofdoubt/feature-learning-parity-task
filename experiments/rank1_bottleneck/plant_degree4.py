"""Experiment 1: rescue a trained singleton solution by planting a controlled degree-4 circuit.

Starting from a trained no-bottleneck rank-1 checkpoint whose inputs are the d singletons plus the
two degree-4 modes p = x1x2x3x4, q = x5x6x7x8 (target chi_8 = p q), we

  1. evaluate the source model exactly on the full 2^d cube,
  2. pick M = 4m least-contributing hidden units J and two orthonormal directions U = [u1 u2]
     orthogonal to every existing embedding row,
  3. prepare a baseline W_base = W0 (I - U U^T), W_base[J] = 0, r_base[J] = 0 (predictions change only
     by the removed units' negligible output),
  4. plant, for a seed strength rho, V_p += rho/sqrt2 u1, V_q += rho/sqrt2 u2,
     W_j = rho/sqrt(2M) (s_j1 u1 + s_j2 u2), r_j = rho/sqrt(M) s_j1 s_j2 for j in J with S the balanced
     +-1 pattern matrix, which gives exactly y_planted = rho^3/4 chi_8 and Delta P = 3 lambda/2 rho^2,
  5. validate the algebra (float64, then the training dtype) and the projected regularized gradient,
  6. continue training each arm with a fresh SGD optimizer (momentum 0.9, same lr/wd/batch) on identical
     minibatch sequences, logging exact full-cube diagnostics along the way,
  7. at the end, ablate singleton vs degree-4 input columns of C = W V^T to see which inputs support the
     computation.

Arms: "orig" (unmodified checkpoint), "prep" (prepared baseline = rho 0), "rho<r>" (coherent seed),
"orth<r>" (same V/W seed, readout template replaced by a random unit vector orthogonal to t_r: same
norm, zero injected target coefficient).

Usage (prepare + validate only):
  python -m experiments.rank1_bottleneck.plant_degree4 --source runs/<run>/checkpoints/final.pt \
      --out-root runs --tag plantd4_Q8 --rhos 0.15 0.28 --validate-only
Training one arm (arms are independent so they can be spread over windows / machines):
  python -m experiments.rank1_bottleneck.plant_degree4 --source ... --tag plantd4_Q8 --arm rho0.28 --steps 50000
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .config import Rank1Config, to_dict
from .model import Rank1BottleneckNet, features

KAPPA2 = 0.25          # target coefficient per unit of rho^2 . rho in the planted circuit (y = rho^3 / 4)


# ----------------------------------------------------------------------------------------------
# exact full-cube evaluation from raw tensors (float64)
# ----------------------------------------------------------------------------------------------
def full_cube(d: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product([-1.0, 1.0], repeat=d)), dtype=torch.float64)


def forward_parts(V, W, r, phi):
    """phi: (B, M) mode features. Returns (z, act, y) with z = W (phi V), y = act r."""
    h = phi @ V
    z = h @ W.t()
    act = torch.relu(z)
    return z, act, act @ r


def penalty(V, W, r, lam, lam_r):
    return 0.5 * (lam * (float((V ** 2).sum()) + float((W ** 2).sum())) + lam_r * float((r ** 2).sum()))


def walsh_spectrum(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """All 2^d Walsh coefficients of y on the full cube x (rows = inputs); index = bitmask of the support."""
    d = x.shape[1]
    out = torch.empty(2 ** d, dtype=torch.float64)
    for mask in range(2 ** d):
        chi = torch.ones(x.shape[0], dtype=torch.float64)
        for i in range(d):
            if mask >> i & 1:
                chi = chi * x[:, i]
        out[mask] = (y * chi).mean()
    return out


class Setup:
    """Everything shared across arms: source tensors, cube, selected units, directions, templates."""

    def __init__(self, source: Path, M: int, intervention_seed: int):
        self.source = Path(source)
        payload = torch.load(self.source, map_location="cpu", weights_only=False)
        self.source_sha256 = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.source_step = int(payload["step"])
        self.cfg = Rank1Config(**payload["config"])
        cfg = self.cfg
        assert not cfg.use_bottleneck and not cfg.residual_after_bottleneck, "expects a plain (no-bottleneck, no-skip) model"
        assert not cfg.noise_modes and not cfg.freeze_embedding, "expects clean trainable inputs"
        self.model = Rank1BottleneckNet(cfg)
        self.model.load_state_dict(payload["model_state"])
        self.dtype = self.model.embedding.dtype
        self.supports = self.model.supports
        self.d = cfg.d
        half = self.d // 2
        lo, hi = tuple(range(half)), tuple(range(half, self.d))
        assert lo in self.supports and hi in self.supports, "needs the two degree-d/2 modes among the inputs"
        self.ip, self.iq = self.supports.index(lo), self.supports.index(hi)
        self.singles = [i for i, m in enumerate(self.supports) if len(m) == 1]
        assert len(self.singles) == self.d and len(self.supports) == self.d + 2
        self.lam, self.lam_r = cfg.weight_decay, cfg.readout_wd()
        # float64 copies
        self.V0 = self.model.embedding.detach().double().clone()
        self.W0 = self.model.linear.weight.detach().double().clone()
        self.r0 = self.model.readout.weight[0].detach().double().clone()
        self.N = self.V0.shape[1]
        self.x = full_cube(self.d)
        self.phi = features(self.x, self.supports).double()
        self.chi8 = self.x.prod(dim=1)
        self.M = M
        assert M % 4 == 0 and M >= 4
        self.intervention_seed = intervention_seed
        self._select_units()
        self._build_directions()
        self._prepare_baseline()
        self._templates()

    # ---- measurements ------------------------------------------------------------------------
    def measure(self, V, W, r, J=None) -> dict:
        z, act, y = forward_parts(V, W, r, self.phi)
        a = float((y * self.chi8).mean())
        mse = float(((y - self.chi8) ** 2).mean())
        P = penalty(V, W, r, self.lam, self.lam_r)
        C = W @ V.t()                                         # (N_hidden, n_modes) exact pre-activation coefficients
        Cd4 = C[:, [self.ip, self.iq]]
        out = dict(mse=mse, P=P, L=mse + P, a=a, nontarget_power=float(((y - a * self.chi8) ** 2).mean()),
                   share_d4=float((Cd4 ** 2).sum() / ((C ** 2).sum() + 1e-30)),
                   V_fro=float(V.norm()), W_fro=float(W.norm()), r_norm=float(r.norm()),
                   V_single_rownorm_mean=float(V[self.singles].norm(dim=1).mean()),
                   V_p_rownorm=float(V[self.ip].norm()), V_q_rownorm=float(V[self.iq].norm()))
        if J is not None:
            yJ = act[:, J] @ r[J]
            out["a_planted"] = float((yJ * self.chi8).mean())
            out["a_rest"] = a - out["a_planted"]
            CJ = C[J]
            out["planted_single_leak"] = float((CJ[:, self.singles] ** 2).sum() / ((CJ ** 2).sum() + 1e-30))
            out["b_par"] = float((V * self.TV).sum())
            out["w_par"] = float((W[J] * self.TW).sum())
            out["r_par"] = float((r[J] * self.tr).sum())
            out["a_planted_pred"] = KAPPA2 * out["b_par"] * out["w_par"] * out["r_par"]
            out["a_planted_minus_pred"] = out["a_planted"] - out["a_planted_pred"]
        return out

    def ablate(self, V, W, r) -> dict:
        """Evaluate with the singleton input rows zeroed (degree-4 only) and with p,q zeroed (singletons only)."""
        res = {}
        for name, rows in (("only_degree4", self.singles), ("only_singletons", [self.ip, self.iq])):
            Vm = V.clone(); Vm[rows] = 0
            _, _, y = forward_parts(Vm, W, r, self.phi)
            res[name] = dict(mse=float(((y - self.chi8) ** 2).mean()), a=float((y * self.chi8).mean()))
        return res

    # ---- construction ------------------------------------------------------------------------
    def _select_units(self):
        z, act, y = forward_parts(self.V0, self.W0, self.r0, self.phi)
        contrib = act * self.r0                                   # (B, N) per-unit output contributions
        d_j = contrib.pow(2).mean(0).sqrt()
        pnorm = (self.W0 ** 2).sum(1) + self.r0 ** 2
        order = sorted(range(self.N), key=lambda j: (float(d_j[j]), float(pnorm[j])))
        self.J = sorted(order[: self.M])
        yJ = contrib[:, self.J].sum(1)
        self.selection = dict(J=self.J, d_j=[float(d_j[j]) for j in self.J], param_norm_sq=[float(pnorm[j]) for j in self.J],
                              yJ_rms=float(yJ.pow(2).mean().sqrt()), yJ_max=float(yJ.abs().max()),
                              n_units_silent=int((d_j < 1e-6).sum()))

    def _build_directions(self):
        g = torch.Generator().manual_seed(self.intervention_seed)
        B = torch.linalg.qr(self.V0.t(), mode="reduced").Q           # (N, n_modes) orthonormal basis of span(V0 rows)
        G = torch.randn(self.N, 2, dtype=torch.float64, generator=g)
        G = G - B @ (B.t() @ G)
        self.U = torch.linalg.qr(G, mode="reduced").Q               # (N, 2)
        self.gen_after_U = g                                       # continue the same stream for the orth control
        self.ortho_checks = dict(VU_max=float((self.V0 @ self.U).abs().max()),
                                 UtU_err=float((self.U.t() @ self.U - torch.eye(2, dtype=torch.float64)).abs().max()))

    def _prepare_baseline(self):
        U, J = self.U, self.J
        self.Wbase = self.W0 - (self.W0 @ U) @ U.t()
        self.Wbase[J] = 0.0
        self.rbase = self.r0.clone(); self.rbase[J] = 0.0
        # original vs prepared
        z0, act0, y0 = forward_parts(self.V0, self.W0, self.r0, self.phi)
        zb, actb, yb = forward_parts(self.V0, self.Wbase, self.rbase, self.phi)
        yJ = act0[:, J] @ self.r0[J]
        keep = [j for j in range(self.N) if j not in set(J)]
        self.prep_checks = dict(preact_change_nonplanted_max=float((zb[:, keep] - z0[:, keep]).abs().max()),
                                output_identity_err=float((yb - (y0 - yJ)).abs().max()),
                                P_original=penalty(self.V0, self.W0, self.r0, self.lam, self.lam_r),
                                P_prepared=penalty(self.V0, self.Wbase, self.rbase, self.lam, self.lam_r))
        self.prep_checks["P_change"] = self.prep_checks["P_prepared"] - self.prep_checks["P_original"]

    def _templates(self):
        M, U = self.M, self.U
        S = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], dtype=torch.float64).repeat_interleave(M // 4, dim=0)
        self.S = S
        TV = torch.zeros_like(self.V0)
        TV[self.ip] = U[:, 0] / math.sqrt(2)
        TV[self.iq] = U[:, 1] / math.sqrt(2)
        self.TV = TV
        self.TW = (S @ U.t()) / math.sqrt(2 * M)                     # (M, N)
        self.tr = S.prod(dim=1) / math.sqrt(M)                        # (M,)
        # target-orthogonal readout template: random unit vector orthogonal to tr (same RNG stream)
        g = self.gen_after_U
        v = torch.randn(M, dtype=torch.float64, generator=g)
        v = v - (v @ self.tr) * self.tr
        self.tr_orth = v / v.norm()

    def plant(self, rho: float, orth: bool = False):
        V = self.V0 + rho * self.TV
        W = self.Wbase.clone(); W[self.J] = rho * self.TW
        r = self.rbase.clone(); r[self.J] = rho * (self.tr_orth if orth else self.tr)
        return V, W, r

    def arm_tensors(self, arm: str):
        if arm == "orig":
            return self.V0.clone(), self.W0.clone(), self.r0.clone(), None
        if arm == "prep":
            return self.plant(0.0) + (0.0,)
        if arm.startswith("rho"):
            rho = float(arm[3:]); return self.plant(rho) + (rho,)
        if arm.startswith("orth"):
            rho = float(arm[4:]); return self.plant(rho, orth=True) + (rho,)
        raise ValueError(arm)

    # ---- validation ----------------------------------------------------------------------------
    def validate(self, rho: float) -> dict:
        J, M = self.J, self.M
        V, W, r = self.plant(rho)
        res = dict(rho=rho, **self.ortho_checks)
        zb, actb, yb = forward_parts(self.V0, self.Wbase, self.rbase, self.phi)
        for tag, dt in (("f64", torch.float64), ("train_dtype", self.dtype)):
            Vd, Wd, rd = V.to(dt), W.to(dt), r.to(dt)
            z, act, y = forward_parts(Vd, Wd, rd, self.phi.to(dt))
            z, y = z.double(), y.double()
            keep = [j for j in range(self.N) if j not in set(J)]
            p, q = self.phi[:, self.ip], self.phi[:, self.iq]
            z_pred = (rho ** 2 / (2 * math.sqrt(M))) * (self.S[:, 0][None, :] * p[:, None] + self.S[:, 1][None, :] * q[:, None])
            dy = y - yb
            spec = walsh_spectrum(dy, self.x)
            full_mask = 2 ** self.d - 1
            spec_other = spec.clone(); spec_other[full_mask] = 0
            sq_total = float((Vd.double() ** 2).sum() + (Wd.double() ** 2).sum() + (rd.double() ** 2).sum())
            sq_base = float((self.V0 ** 2).sum() + (self.Wbase ** 2).sum() + (self.rbase ** 2).sum())
            Lb = self.measure(self.V0, self.Wbase, self.rbase)["L"]
            Lr = self.measure(Vd.double(), Wd.double(), rd.double())["L"]
            a_base = float((yb * self.chi8).mean())
            dL_pred = -2 * (1 - a_base) * rho ** 3 / 4 + (rho ** 3 / 4) ** 2 + 1.5 * self.lam * rho ** 2
            res[tag] = dict(nonplanted_preact_err=float((z[:, keep] - zb[:, keep]).abs().max()),
                            planted_preact_err=float((z[:, J] - z_pred).abs().max()),
                            output_err=float((dy - rho ** 3 / 4 * self.chi8).abs().max()),
                            target_coef_of_planted=float(spec[full_mask]), target_coef_expected=rho ** 3 / 4,
                            other_walsh_max=float(spec_other.abs().max()),
                            sqnorm_increase=sq_total - sq_base, sqnorm_increase_expected=3 * rho ** 2,
                            dL=Lr - Lb, dL_expected=dL_pred, dL_err=abs(Lr - Lb - dL_pred))
        # target-orthogonal control: injected target coefficient must vanish
        Vo, Wo, ro = self.plant(rho, orth=True)
        _, _, yo = forward_parts(Vo, Wo, ro, self.phi)
        res["orth_control"] = dict(injected_target_coef=float(((yo - yb) * self.chi8).mean()),
                                   readout_norm_J=float(ro[J].norm()), coherent_readout_norm_J=float(r[J].norm()))
        res["gradient_check"] = self.gradient_check(rho)
        return res

    def gradient_check(self, rho: float) -> dict:
        """Full-cube regularized gradient (no update); projections of -grad onto the three templates."""
        V, W, r = self.plant(rho)
        V, W, r = V.clone().requires_grad_(True), W.clone().requires_grad_(True), r.clone().requires_grad_(True)
        _, _, y = forward_parts(V, W, r, self.phi)
        loss = ((y - self.chi8) ** 2).mean() + 0.5 * (self.lam * ((V ** 2).sum() + (W ** 2).sum()) + self.lam_r * (r ** 2).sum())
        loss.backward()
        e = 1 - float((y.detach() * self.chi8).mean())
        expected = 2 * e * KAPPA2 * rho ** 2 - self.lam * rho
        expected_r = 2 * e * KAPPA2 * rho ** 2 - self.lam_r * rho
        pv = float((-V.grad * self.TV).sum()); pw = float((-W.grad[self.J] * self.TW).sum()); pr = float((-r.grad[self.J] * self.tr).sum())
        return dict(e=e, proj_V=pv, proj_W=pw, proj_r=pr, expected=expected, expected_r=expected_r,
                    err_V=abs(pv - expected), err_W=abs(pw - expected), err_r=abs(pr - expected_r))

    def metadata(self) -> dict:
        return dict(source=str(self.source), source_sha256=self.source_sha256, source_step=self.source_step,
                    config=to_dict(self.cfg), d=self.d, ip=self.ip, iq=self.iq, singles=self.singles, M=self.M,
                    intervention_seed=self.intervention_seed, J=self.J, selection=self.selection,
                    ortho_checks=self.ortho_checks, prep_checks=self.prep_checks,
                    S=self.S.tolist(), tr=self.tr.tolist(), tr_orth=self.tr_orth.tolist(),
                    original=self.measure(self.V0, self.W0, self.r0), prepared=self.measure(self.V0, self.Wbase, self.rbase, self.J))


# ----------------------------------------------------------------------------------------------
# training of one arm
# ----------------------------------------------------------------------------------------------
def eval_schedule(step: int) -> bool:
    if step <= 500:
        return step % 5 == 0
    if step <= 5000:
        return step % 100 == 0
    return step % 250 == 0


def train_arm(setup: Setup, arm: str, out_dir: Path, steps: int, data_seed: int, device: torch.device,
              save_templates: bool = True) -> None:
    cfg = setup.cfg
    V, W, r, rho = setup.arm_tensors(arm)
    J = None if arm == "orig" else setup.J
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis").mkdir(exist_ok=True); (out_dir / "checkpoints").mkdir(exist_ok=True)
    run_cfg = to_dict(cfg); run_cfg["output_dir"] = str(out_dir); run_cfg["num_steps"] = setup.source_step + steps
    (out_dir / "config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    meta = dict(arm=arm, rho=rho, steps=steps, data_seed=data_seed,
                optimizer="fresh SGD, zero momentum buffers, momentum 0.9, lr/wd/batch from config; unused W_enc/W_dec excluded",
                **setup.metadata())
    if J is not None:
        meta["intervened"] = setup.measure(V, W, r, J)
        if arm.startswith("rho") or arm.startswith("orth"):
            meta["validation"] = setup.validate(rho) if arm.startswith("rho") else None
            if arm.startswith("orth"):
                _, _, yb = forward_parts(setup.V0, setup.Wbase, setup.rbase, setup.phi)
                _, _, yo = forward_parts(V, W, r, setup.phi)
                meta["orth_injected_target_coef"] = float(((yo - yb) * setup.chi8).mean())
    (out_dir / "analysis" / "intervention.json").write_text(json.dumps(meta, indent=1))
    if save_templates:
        torch.save(dict(U=setup.U, TV=setup.TV, TW=setup.TW, tr=setup.tr, tr_orth=setup.tr_orth, J=setup.J, S=setup.S),
                   out_dir / "analysis" / "templates.pt")

    model = Rank1BottleneckNet(cfg)
    model.load_state_dict(setup.model.state_dict())
    with torch.no_grad():
        model.embedding.copy_(V.to(setup.dtype)); model.linear.weight.copy_(W.to(setup.dtype)); model.readout.weight[0].copy_(r.to(setup.dtype))
    model.to(device)
    torch.save(dict(step=setup.source_step, post_step=0, model_state=model.state_dict(), config=cfg.__dict__, arm=arm, rho=rho),
               out_dir / "checkpoints" / "intervened.pt")
    trained = [model.embedding, model.linear.weight]
    opt = torch.optim.SGD([dict(params=trained, weight_decay=cfg.weight_decay),
                           dict(params=list(model.readout.parameters()), weight_decay=cfg.readout_wd())],
                          lr=cfg.lr, momentum=cfg.momentum)
    gen = torch.Generator(device=device); gen.manual_seed(data_seed)
    hist = []

    def snapshot(post_step: int, train_mse: float | None):
        Vn = model.embedding.detach().double().cpu(); Wn = model.linear.weight.detach().double().cpu(); rn = model.readout.weight[0].detach().double().cpu()
        row = dict(post_step=post_step, step=setup.source_step + post_step, train_mse=train_mse, **setup.measure(Vn, Wn, rn, J))
        hist.append(row)

    snapshot(0, None)
    t0 = time.perf_counter()
    for s in range(1, steps + 1):
        x = torch.randint(0, 2, (cfg.batch_size, cfg.d), device=device, generator=gen).float() * 2 - 1
        loss = F.mse_loss(model(x), x.prod(dim=1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if eval_schedule(s) or s == steps:
            snapshot(s, loss.item())
            if s % 5000 == 0 or s == steps:
                pd.DataFrame(hist).to_csv(out_dir / "analysis" / "trajectory.csv", index=False)
    pd.DataFrame(hist).to_csv(out_dir / "analysis" / "trajectory.csv", index=False)
    torch.save(dict(step=setup.source_step + steps, post_step=steps, model_state=model.state_dict(), optimizer_state=opt.state_dict(),
                    config=cfg.__dict__, arm=arm, rho=rho), out_dir / "checkpoints" / "final.pt")
    Vn = model.embedding.detach().double().cpu(); Wn = model.linear.weight.detach().double().cpu(); rn = model.readout.weight[0].detach().double().cpu()
    final = dict(final=setup.measure(Vn, Wn, rn, J), ablation=setup.ablate(Vn, Wn, rn), seconds=time.perf_counter() - t0)
    (out_dir / "analysis" / "final.json").write_text(json.dumps(final, indent=1))
    print(f"[{arm}] done {steps} steps in {final['seconds']:.0f}s: L={final['final']['L']:.4f} mse={final['final']['mse']:.3g} "
          f"share_d4={final['final']['share_d4']:.3f} a={final['final']['a']:.4f} ablation={json.dumps(final['ablation'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--tag", default="plantd4")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--intervention-seed", type=int, default=1234)
    ap.add_argument("--data-seed", type=int, default=777)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--rhos", type=float, nargs="*", default=[0.15, 0.23, 0.26, 0.28, 0.30, 0.34, 0.40])
    ap.add_argument("--arm", default=None, help="train exactly this arm (orig | prep | rho<r> | orth<r>)")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    from parity_net.train import resolve_device
    device = resolve_device(args.device)
    setup = Setup(Path(args.source), args.M, args.intervention_seed)
    print("source:", setup.source, setup.source_sha256[:16], "step", setup.source_step)
    print("original:", json.dumps(setup.metadata()["original"]))
    print("selection:", json.dumps(setup.selection))
    print("ortho:", setup.ortho_checks, "prep:", setup.prep_checks)
    if args.validate_only or args.arm is None:
        for rho in args.rhos:
            v = setup.validate(rho)
            print(f"--- rho={rho}")
            print(" f64:", json.dumps(v["f64"]))
            print(" train_dtype:", json.dumps(v["train_dtype"]))
            print(" orth:", json.dumps(v["orth_control"]))
            print(" grad:", json.dumps(v["gradient_check"]))
        if args.validate_only:
            return
    arms = [args.arm] if args.arm else ["orig", "prep"] + [f"rho{r:g}" for r in args.rhos]
    for arm in arms:
        out_dir = Path(args.out_root) / f"{args.tag}_{arm}_s{args.data_seed}"
        train_arm(setup, arm, out_dir, args.steps, args.data_seed, device)


if __name__ == "__main__":
    main()
