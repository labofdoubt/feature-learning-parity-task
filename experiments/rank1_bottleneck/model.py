"""y(x) = r . relu( W  w_dec (w_enc . h(x)) ),   h(x) = sum_m phi_m(x) v_m,
(with residual_after_bottleneck: y = r . ( b + relu(W b) ),  b = W_dec W_enc h)

with phi(x) = (x_1, ..., x_d, prod_i x_i) in {-1,+1}^{d+1} the d singleton modes plus the
product mode, and V = [v_1 ... v_d, v_prod] the (d+1) x d_hid embedding. Everything is
trainable. Because the bottleneck code is linear in phi, its Walsh coefficients are exact:
s(x) = w_enc . h(x) = sum_m (w_enc . v_m) phi_m(x), so `bottleneck_coefficients` returns
(w_enc . v_m)_m with no sampling.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import Rank1Config


def features(x: torch.Tensor, supports=None) -> torch.Tensor:
    """(batch, d) +-1 inputs -> (batch, M) Walsh modes chi_S(x) for the given supports
    (default: the d singletons followed by their product)."""
    if supports is None:
        return torch.cat([x, x.prod(dim=1, keepdim=True)], dim=1)
    if supports is False:                                   # legacy: singletons only
        return x
    return torch.stack([x[:, list(m)].prod(dim=1) for m in supports], dim=1)


class Rank1BottleneckNet(nn.Module):
    def __init__(self, cfg: Rank1Config) -> None:
        super().__init__()
        self.cfg = cfg
        d, N, K = cfg.d, cfg.d_hid, cfg.bottleneck_k
        self.supports = cfg.mode_supports()
        self.n_modes = len(self.supports)
        self.product_idx = cfg.product_mode_index()
        self.embedding = nn.Parameter(torch.randn(self.n_modes, N) * math.sqrt(cfg.emb_var))
        if cfg.embedding_scale_by_degree:
            with torch.no_grad():
                for i, m in enumerate(self.supports):
                    self.embedding[i] *= float(cfg.embedding_scale_by_degree.get(len(m), 1.0))
        # rank-K bottleneck: K encoder directions (rows of W_enc) and K decoder columns
        self.W_enc = nn.Parameter(torch.randn(K, N) * math.sqrt(cfg.bn_var))
        self.W_dec = nn.Parameter(torch.randn(N, K) * math.sqrt(cfg.bn_var))
        self.linear = nn.Linear(N, N, bias=False)
        self.readout = nn.Linear(N, 1, bias=False)
        nn.init.normal_(self.linear.weight, std=math.sqrt(cfg.hid_var))
        nn.init.normal_(self.readout.weight, std=math.sqrt(cfg.ro_var))
        with torch.no_grad():
            self.W_enc[0].copy_(self.initial_w_enc())            # alignment modes act on direction 0

    # the first encoder direction, kept under the old name for K = 1 code and analyses
    @property
    def w_enc(self) -> torch.Tensor:
        return self.W_enc[0]

    @property
    def w_dec(self) -> torch.Tensor:
        return self.W_dec[:, 0]

    @torch.no_grad()
    def initial_w_enc(self) -> torch.Tensor:
        cfg, N = self.cfg, self.cfg.d_hid
        norm = math.sqrt(N * cfg.bn_var)                         # the norm a random draw has in expectation
        random = torch.randn(N)
        if cfg.enc_init == "random":
            return random * math.sqrt(cfg.bn_var)
        if cfg.enc_init in ("product", "random_no_product") and self.product_idx is None:
            raise ValueError(f"enc_init={cfg.enc_init!r} needs the product mode among input_modes")
        if cfg.enc_init == "random_no_product":
            vp = self.embedding[self.product_idx]
            u = vp / vp.norm().clamp_min(1e-12)
            return (random - (random @ u) * u) * math.sqrt(cfg.bn_var)   # exactly orthogonal to v_prod
        singles = [i for i, m in enumerate(self.supports) if len(m) == 1]
        target = self.embedding[singles].sum(0) if cfg.enc_init == "linear_sum" else self.embedding[self.product_idx]
        u = target / target.norm().clamp_min(1e-12)
        perp = random - (random @ u) * u
        perp = perp / perp.norm().clamp_min(1e-12)
        c = cfg.enc_cosine
        return norm * (c * u + math.sqrt(max(0.0, 1.0 - c * c)) * perp)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return features(x, self.supports) @ self.embedding        # (batch, N)

    def forward(self, x: torch.Tensor, *, return_parts: bool = False):
        h = self.embed(x)
        s = h @ self.W_enc.t()                                    # (batch, K) bottleneck code
        b = s @ self.W_dec.t()                                    # (batch, N)
        a = F.relu(self.linear(b))
        if self.cfg.residual_after_bottleneck:
            a = b + a
        y = self.readout(a).squeeze(1)
        if return_parts:
            return y, dict(h=h, code=s, bottleneck_out=b, hidden=a)
        return y

    # ---- analysis helpers (exact, no sampling) ------------------------------------------
    @torch.no_grad()
    def code_coefficients(self) -> torch.Tensor:
        """Walsh coefficients of every code direction on every input mode: (K, M),
        C[k, m] = W_enc[k] . v_m (the code is linear in the mode features)."""
        return self.W_enc @ self.embedding.t()

    @torch.no_grad()
    def bottleneck_coefficients(self) -> torch.Tensor:
        """K = 1 convenience: coefficients of the single code direction, (M,)."""
        return self.code_coefficients()[0]

    @torch.no_grad()
    def mode_power(self) -> torch.Tensor:
        """Power each input mode carries through the bottleneck, summed over the K
        directions: p_m = sum_k C[k, m]^2, shape (M,)."""
        return (self.code_coefficients() ** 2).sum(0)

    @torch.no_grad()
    def enc_cosines(self) -> torch.Tensor:
        """cos(W_enc[0], v_m) for every mode m."""
        V = self.embedding
        return (V @ self.w_enc) / (V.norm(dim=1) * self.w_enc.norm() + 1e-12)

    @torch.no_grad()
    def enc_cosine_to(self, which: str) -> float:
        V = self.embedding
        if which == "product":
            if self.product_idx is None:
                return float("nan")
            target = V[self.product_idx]
        else:
            target = V[[i for i, m in enumerate(self.supports) if len(m) == 1]].sum(0)
        return float(target @ self.w_enc / (target.norm() * self.w_enc.norm() + 1e-12))
