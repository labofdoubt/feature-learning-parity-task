"""Tests for the rank-1 bottleneck experiment (experiments/rank1_bottleneck)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rank1_bottleneck.config import Rank1Config, load_config, save_config
from experiments.rank1_bottleneck.model import Rank1BottleneckNet, features
from experiments.rank1_bottleneck.train import make_test_set


def test_features_and_forward_shapes():
    torch.manual_seed(0)
    cfg = Rank1Config(d=6, d_hid=64)
    m = Rank1BottleneckNet(cfg)
    x = torch.randint(0, 2, (10, 6)).float() * 2 - 1
    f = features(x)
    assert f.shape == (10, 7) and torch.allclose(f[:, 6], x.prod(1))
    y, parts = m(x, return_parts=True)
    assert y.shape == (10,) and parts["code"].shape == (10, 1) and parts["bottleneck_out"].shape == (10, 64)
    # bottleneck output is rank one: every row is a multiple of w_dec
    assert torch.allclose(parts["bottleneck_out"], parts["code"][:, :1] * m.w_dec)


def test_bottleneck_coefficients_are_exact_walsh_coefficients():
    torch.manual_seed(1)
    cfg = Rank1Config(d=8, d_hid=32)
    m = Rank1BottleneckNet(cfg)
    x = make_test_set(cfg, torch.device("cpu"))                   # all 256 inputs
    code = m(x, return_parts=True)[1]["code"][:, 0]
    direct = features(x).t() @ code / x.shape[0]                  # E[s(x) phi_m(x)]
    assert torch.allclose(direct, m.bottleneck_coefficients(), atol=1e-5)


@pytest.mark.parametrize("mode,which", [("linear_sum", "linear_sum"), ("product", "product")])
def test_enc_init_cosine(mode, which):
    for c in (1.0, 0.5, 0.0):
        torch.manual_seed(2)
        cfg = Rank1Config(d=8, d_hid=256, enc_init=mode, enc_cosine=c)
        m = Rank1BottleneckNet(cfg)
        assert m.enc_cosine_to(which) == pytest.approx(c, abs=1e-5)
        assert m.w_enc.norm().item() == pytest.approx(math.sqrt(256 * cfg.bn_var), rel=1e-5)
    torch.manual_seed(3)
    r = Rank1BottleneckNet(Rank1Config(d=8, d_hid=256, enc_init="random"))
    assert abs(r.enc_cosine_to("product")) < 0.3
    torch.manual_seed(3)
    q = Rank1BottleneckNet(Rank1Config(d=8, d_hid=256, enc_init="random_no_product"))
    assert abs(q.enc_cosine_to("product")) < 1e-6 and abs(q.bottleneck_coefficients()[8]) < 1e-6


def test_init_variances_and_config_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = Rank1Config(d=4, d_hid=4096)
    m = Rank1BottleneckNet(cfg)
    assert m.embedding.var().item() == pytest.approx(1 / 4096, rel=0.1)
    assert m.linear.weight.var().item() == pytest.approx(1 / 4096, rel=0.05)
    assert m.readout.weight.var().item() == pytest.approx(1 / 4096, rel=0.2)
    save_config(Rank1Config(d=12, enc_init="product", enc_cosine=0.3, weight_decay=1e-4), tmp_path / "c.yaml")
    c = load_config(tmp_path / "c.yaml")
    assert (c.d, c.enc_init, c.enc_cosine, c.weight_decay) == (12, "product", 0.3, 1e-4)
    with pytest.raises(ValueError):
        Rank1Config(enc_init="nope")


def test_no_product_input():
    torch.manual_seed(0)
    cfg = Rank1Config(d=8, d_hid=64, include_product_input=False, enc_init="linear_sum", enc_cosine=0.7)
    m = Rank1BottleneckNet(cfg)
    assert m.embedding.shape == (8, 64) and features(torch.ones(3, 8), m.supports).shape == (3, 8)
    assert m.enc_cosine_to("linear_sum") == pytest.approx(0.7, abs=1e-5)
    assert m(torch.ones(3, 8)).shape == (3,)
    with pytest.raises(ValueError):
        Rank1BottleneckNet(Rank1Config(d=8, d_hid=64, include_product_input=False, enc_init="product"))


def test_rank_k_bottleneck_and_pairs_modes():
    torch.manual_seed(0)
    cfg = Rank1Config(d=8, d_hid=64, input_modes="singletons+pairs", bottleneck_k=8)
    assert cfg.mode_names() == [f"x{i}" for i in range(1, 9)] + ["x1x2", "x3x4", "x5x6", "x7x8"]
    assert cfg.product_mode_index() is None
    m = Rank1BottleneckNet(cfg)
    assert m.embedding.shape == (12, 64) and m.W_enc.shape == (8, 64) and m.W_dec.shape == (64, 8)
    x = make_test_set(cfg, torch.device("cpu"))
    y, parts = m(x, return_parts=True)
    assert parts["code"].shape == (256, 8) and y.shape == (256,)
    # exact Walsh coefficients of every code direction
    direct = parts["code"].t() @ features(x, m.supports) / 256
    assert torch.allclose(direct, m.code_coefficients(), atol=1e-5)
    assert m.mode_power().shape == (12,)
    with pytest.raises(ValueError):
        Rank1BottleneckNet(Rank1Config(d=8, d_hid=64, input_modes="singletons+pairs", enc_init="product"))
    explicit = Rank1Config(d=4, d_hid=16, input_modes=[[0], [1], [0, 1, 2, 3]])
    assert explicit.mode_names() == ["x1", "x2", "product"] and explicit.product_mode_index() == 2
    with pytest.raises(ValueError):
        Rank1Config(d=4, input_modes=[[0, 9]])


def test_residual_after_bottleneck():
    torch.manual_seed(0)
    cfg = Rank1Config(d=6, d_hid=32, bottleneck_k=2, residual_after_bottleneck=True)
    m = Rank1BottleneckNet(cfg)
    x = torch.randint(0, 2, (5, 6)).float() * 2 - 1
    y, parts = m(x, return_parts=True)
    expected = parts["bottleneck_out"] + torch.relu(m.linear(parts["bottleneck_out"]))
    assert torch.allclose(parts["hidden"], expected) and torch.allclose(y, m.readout(expected).squeeze(1))


def test_test_set_enumerates_all_inputs():
    x = make_test_set(Rank1Config(d=5), torch.device("cpu"))
    assert x.shape == (32, 5) and torch.unique(x, dim=0).shape[0] == 32
