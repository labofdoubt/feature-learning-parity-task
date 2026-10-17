"""Tests for the optional TopK bottleneck on the residual stream.

Run with:  python -m pytest tests/test_bottleneck.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from parity_net.config import ExperimentConfig, ModelConfig, OptimizerConfig, load_config, save_config
from parity_net.model import TopKBottleneck, build_model
from parity_net.train import build_optimizer


def _config(**overrides) -> ModelConfig:
    base = dict(
        input_dim=8,
        relevant_dim=4,
        N=32,
        L=3,
        hidden_weight_variance=1 / 32,
        readout_weight_variance=1 / 32**2,
        use_bottleneck=True,
        bottleneck_dim=16,
        bottleneck_k=4,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_topk_sparsity_and_abs_semantics():
    torch.manual_seed(0)
    b = TopKBottleneck(width=10, dim=6, k=2, abs_topk=True, encoder_variance=None, decoder_variance=None, bias=False)
    x = torch.randn(5, 10)
    codes = b.encode(x)
    assert codes.shape == (5, 6)
    assert torch.all((codes != 0).sum(dim=-1) == 2)
    # Abs: the kept entries are the two largest in magnitude.
    z = b.encoder(x)
    kept = codes != 0
    for row in range(5):
        kept_min = z[row][kept[row]].abs().min()
        dropped_max = z[row][~kept[row]].abs().max()
        assert kept_min >= dropped_max
    # Signed: the kept entries are the two largest values, negatives allowed.
    b.abs_topk = False
    codes = b.encode(x)
    for row in range(5):
        kept = codes[row] != 0
        assert z[row][kept].min() >= z[row][~kept].max()
    assert b(x).shape == (5, 10)


def test_default_init_variances_and_bias():
    torch.manual_seed(0)
    b = TopKBottleneck(width=4096, dim=2048, k=8, abs_topk=True, encoder_variance=None, decoder_variance=None, bias=True)
    assert b.encoder.weight.var().item() == pytest.approx(1 / 4096, rel=0.05)
    assert b.decoder.weight.var().item() == pytest.approx(1 / 8, rel=0.05)
    assert b.encoder.bias is not None and b.decoder.bias is not None
    b2 = TopKBottleneck(width=4096, dim=2048, k=8, abs_topk=True, encoder_variance=0.01, decoder_variance=0.02, bias=False)
    assert b2.encoder.weight.var().item() == pytest.approx(0.01, rel=0.05)
    assert b2.decoder.weight.var().item() == pytest.approx(0.02, rel=0.05)


def test_residual_net_has_one_bottleneck_per_block_with_per_block_k():
    config = _config(bottleneck_k=[1, 2, 3])
    model = build_model(config, output_dim=3)
    assert [b.bottleneck.k for b in model.blocks] == [1, 2, 3]
    x = torch.randint(0, 2, (7, 8)).float() * 2 - 1
    codes = model.bottleneck_codes(x)
    assert len(codes) == 3
    for k, c in zip([1, 2, 3], codes):
        assert c.shape == (7, 16)
        assert torch.all((c != 0).sum(dim=-1) == k)
    # The recorded residual stream after each block is the post-bottleneck stream.
    y, activations = model(x, return_activations=True)
    assert y.shape == (7, 3) and len(activations) == 4
    # No bottleneck when disabled; old-style configs are unaffected.
    plain = build_model(_config(use_bottleneck=False), output_dim=3)
    assert all(b.bottleneck is None for b in plain.blocks)
    assert plain.bottleneck_modules() == []


def test_gradients_flow_to_all_bottleneck_params():
    torch.manual_seed(0)
    model = build_model(_config(), output_dim=3)
    x = torch.randint(0, 2, (16, 8)).float() * 2 - 1
    model(x).square().sum().backward()
    for p in model.bottleneck_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_optimizer_groups_separate_bottleneck_from_hidden():
    model = build_model(_config(freeze_embedding=False), output_dim=3)
    opt_cfg = OptimizerConfig(lr=1e-3, weight_decay=0.1, wd_hidden=0.2, wd_bottleneck=0.3, lr_bottleneck=5e-4)
    opt = build_optimizer(model, opt_cfg)
    groups = {g["name"]: g for g in opt.param_groups}
    assert set(groups) == {"embedding", "hidden", "bottleneck", "readout"}
    bottleneck_ids = {id(p) for p in model.bottleneck_parameters()}
    assert {id(p) for p in groups["bottleneck"]["params"]} == bottleneck_ids
    assert bottleneck_ids.isdisjoint({id(p) for p in groups["hidden"]["params"]})
    assert groups["bottleneck"]["weight_decay"] == 0.3
    assert groups["bottleneck"]["lr"] == 5e-4
    assert groups["hidden"]["weight_decay"] == 0.2
    # Every parameter is in exactly one group.
    all_ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids)) == sum(1 for _ in model.parameters())
    # Without a bottleneck the group simply does not appear.
    plain = build_model(_config(use_bottleneck=False), output_dim=3)
    assert "bottleneck" not in {g["name"] for g in build_optimizer(plain, opt_cfg).param_groups}


def test_weight_variances_report_bottleneck_layers():
    model = build_model(_config(), output_dim=3)
    keys = model.weight_variances()
    assert "blocks.0.bottleneck.encoder.weight" in keys
    assert "blocks.2.bottleneck.decoder.weight" in keys


def test_transformer_bottleneck_after_mlp_only():
    config = _config(use_attention=True, bottleneck_k=2)
    model = build_model(config, output_dim=3)
    assert all(b.bottleneck is not None and b.mlp.bottleneck is b.bottleneck for b in model.blocks)
    x = torch.randint(0, 2, (4, 8)).float() * 2 - 1
    targets = torch.randint(0, 2, (4, 3)).float() * 2 - 1
    assert model(x, targets=targets).shape == (4, 3)
    # Cached and uncached generation agree with the bottleneck in place.
    cached = model.generate(x, use_cache=True)
    uncached = model.generate(x, use_cache=False)
    assert torch.allclose(cached, uncached, atol=1e-5)
    codes = model.bottleneck_codes(x, targets=targets)
    assert len(codes) == 3 and codes[0].shape == (4, 8 + 3 - 1, 16)
    assert torch.all((codes[0] != 0).sum(dim=-1) == 2)


def test_config_validation_and_yaml_round_trip(tmp_path):
    with pytest.raises(ValueError):
        _config(bottleneck_dim=None).bottleneck_ks()
    with pytest.raises(ValueError):
        _config(bottleneck_k=None).bottleneck_ks()
    with pytest.raises(ValueError):
        _config(bottleneck_k=[1, 2]).bottleneck_ks()
    with pytest.raises(ValueError):
        _config(bottleneck_k=17).bottleneck_ks()
    with pytest.raises(ValueError):
        _config(bottleneck_decoder_weight_variance=-1.0).bottleneck_ks()
    assert _config(use_bottleneck=False, bottleneck_dim=None).bottleneck_ks() == []

    config = ExperimentConfig(model=_config(bottleneck_k=[4, 2, 1], bottleneck_abs_topk=False))
    config.training.optimizer.wd_bottleneck = 0.05
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.model.bottleneck_k == [4, 2, 1]
    assert loaded.model.bottleneck_abs_topk is False
    assert loaded.model.bottleneck_dim == 16
    assert loaded.training.optimizer.wd_bottleneck == 0.05
    assert loaded.model.bottleneck_ks() == [4, 2, 1]


def test_state_dict_round_trip():
    torch.manual_seed(0)
    config = _config(bottleneck_k=[1, 2, 3])
    a = build_model(config, output_dim=3)
    b = build_model(config, output_dim=3)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 2, (5, 8)).float() * 2 - 1
    assert torch.allclose(a(x), b(x))


def test_readout_before_bottleneck():
    torch.manual_seed(0)
    config = _config(bottleneck_k=[2, 2, 2], readout_before_bottleneck=True, use_layerwise_readouts=True)
    names = ["d2_0", "d2_1", "d4_0"]
    model = build_model(config, output_dim=3, target_names_=names)
    x = torch.randint(0, 2, (6, 8)).float() * 2 - 1
    y, acts = model(x, return_activations=True)
    assert y.shape == (6, 3)
    # Recorded streams are pre-bottleneck: block 0's output without its bottleneck.
    h1_pre = model.blocks[0](acts[0], apply_bottleneck=False)
    assert torch.allclose(acts[1], h1_pre)
    # Block 1 receives the post-bottleneck stream.
    h2_pre = model.blocks[1](model.blocks[0].bottleneck(h1_pre), apply_bottleneck=False)
    assert torch.allclose(acts[2], h2_pre)
    # Layerwise readouts read the pre-bottleneck stream.
    expected = torch.cat([model.layerwise_readouts["0"](acts[1]), model.layerwise_readouts["1"](acts[2])], dim=1)
    assert torch.allclose(y, expected)
    # The last bottleneck gets no gradient; the others do.
    y.square().sum().backward()
    assert model.blocks[2].bottleneck.encoder.weight.grad is None
    assert model.blocks[0].bottleneck.encoder.weight.grad.abs().sum() > 0
    # Same with a single final readout: it reads before the last bottleneck.
    single = build_model(_config(bottleneck_k=[2, 2, 2], readout_before_bottleneck=True), output_dim=3)
    y_single, acts_single = single(x, return_activations=True)
    assert torch.allclose(y_single, single.readout(acts_single[-1]))
    assert torch.allclose(acts_single[-1], single.blocks[2](single.blocks[1].bottleneck(acts_single[2]), apply_bottleneck=False))
    with pytest.raises(ValueError):
        build_model(_config(use_attention=True, readout_before_bottleneck=True), output_dim=3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
