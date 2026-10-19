"""y(x) = r . relu( W  w_dec (w_enc . h(x)) ),   h(x) = sum_m phi_m(x) v_m,

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


def features(x: torch.Tensor) -> torch.Tensor:
    """(batch, d) +-1 inputs -> (batch, d+1) modes: singletons then their product."""
    return torch.cat([x, x.prod(dim=1, keepdim=True)], dim=1)


class Rank1BottleneckNet(nn.Module):
    def __init__(self, cfg: Rank1Config) -> None:
        super().__init__()
        self.cfg = cfg
        d, N = cfg.d, cfg.d_hid
        self.embedding = nn.Parameter(torch.randn(d + 1, N) * math.sqrt(cfg.emb_var))
        self.w_enc = nn.Parameter(torch.empty(N))
        self.w_dec = nn.Parameter(torch.randn(N) * math.sqrt(cfg.bn_var))
        self.linear = nn.Linear(N, N, bias=False)
        self.readout = nn.Linear(N, 1, bias=False)
        nn.init.normal_(self.linear.weight, std=math.sqrt(cfg.hid_var))
        nn.init.normal_(self.readout.weight, std=math.sqrt(cfg.ro_var))
        with torch.no_grad():
            self.w_enc.copy_(self.initial_w_enc())

    @torch.no_grad()
    def initial_w_enc(self) -> torch.Tensor:
        cfg, N = self.cfg, self.cfg.d_hid
        norm = math.sqrt(N * cfg.bn_var)                         # the norm a random draw has in expectation
        random = torch.randn(N)
        if cfg.enc_init == "random":
            return random * math.sqrt(cfg.bn_var)
        if cfg.enc_init == "random_no_product":
            vp = self.embedding[cfg.d]
            u = vp / vp.norm().clamp_min(1e-12)
            return (random - (random @ u) * u) * math.sqrt(cfg.bn_var)   # exactly orthogonal to v_prod
        target = self.embedding[: cfg.d].sum(0) if cfg.enc_init == "linear_sum" else self.embedding[cfg.d]
        u = target / target.norm().clamp_min(1e-12)
        perp = random - (random @ u) * u
        perp = perp / perp.norm().clamp_min(1e-12)
        c = cfg.enc_cosine
        return norm * (c * u + math.sqrt(max(0.0, 1.0 - c * c)) * perp)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return features(x) @ self.embedding                       # (batch, N)

    def forward(self, x: torch.Tensor, *, return_parts: bool = False):
        h = self.embed(x)
        s = h @ self.w_enc                                        # (batch,) bottleneck code
        b = s.unsqueeze(1) * self.w_dec                           # (batch, N)
        a = F.relu(self.linear(b))
        y = self.readout(a).squeeze(1)
        if return_parts:
            return y, dict(h=h, code=s, bottleneck_out=b, hidden=a)
        return y

    # ---- analysis helpers (exact, no sampling) ------------------------------------------
    @torch.no_grad()
    def bottleneck_coefficients(self) -> torch.Tensor:
        """Walsh coefficients of the code s(x) on (x_1..x_d, product): (w_enc . v_m)_m."""
        return self.embedding @ self.w_enc

    @torch.no_grad()
    def enc_cosines(self) -> torch.Tensor:
        """cos(w_enc, v_m) for every mode m."""
        V = self.embedding
        return (V @ self.w_enc) / (V.norm(dim=1) * self.w_enc.norm() + 1e-12)

    @torch.no_grad()
    def enc_cosine_to(self, which: str) -> float:
        V = self.embedding
        target = V[: self.cfg.d].sum(0) if which == "linear_sum" else V[self.cfg.d]
        return float(target @ self.w_enc / (target.norm() * self.w_enc.norm() + 1e-12))
