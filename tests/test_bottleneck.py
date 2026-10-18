"""Tests for the optional TopK bottleneck on the residual stream.

Run with:  python -m pytest tests/test_bottleneck.py -v
"""
from __future__ import annotations

import math
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


def test_md_decouple_init_forward_and_gradient_split():
    torch.manual_seed(0)
    from parity_net.model import MDLinear, RAW_GAIN_ONE
    lin = MDLinear(in_features=32, out_features=16, width=32, bias=False)
    # init: direction on the sphere, gains exactly 1, fused == direction
    assert lin.direction.norm().item() == pytest.approx(math.sqrt(16 * 32 / 32), rel=1e-6)
    assert torch.allclose(lin.gain_row, torch.ones(16)) and torch.allclose(lin.gain_col, torch.ones(32))
    assert torch.allclose(lin.weight, lin.direction)
    # perturb the gains, check autograd reproduces the paper's gradient split
    with torch.no_grad():
        lin.raw_gain_row.copy_(torch.randn(16)); lin.raw_gain_col.copy_(torch.randn(32))
    x = torch.randn(8, 32)
    G_target = torch.randn(8, 16)
    (lin(x) * G_target).sum().backward()
    G = G_target.t() @ x                      # dL/dW for the fused matrix
    gr, gc = lin.gain_row.detach(), lin.gain_col.detach()
    W_hat = lin.direction.detach()
    assert torch.allclose(lin.direction.grad, gr[:, None] * G * gc[None, :], atol=1e-5)
    g_row = ((W_hat * G) * gc[None, :]).sum(1) * torch.sigmoid(lin.raw_gain_row.detach())
    g_col = (gr[:, None] * (W_hat * G)).sum(0) * torch.sigmoid(lin.raw_gain_col.detach())
    assert torch.allclose(lin.raw_gain_row.grad, g_row, atol=1e-5)
    assert torch.allclose(lin.raw_gain_col.grad, g_col, atol=1e-5)
    # a step off the sphere is undone by project_
    with torch.no_grad():
        lin.direction.add_(0.1 * torch.randn_like(lin.direction))
    assert lin.direction.norm().item() != pytest.approx(float(lin.c_f), rel=1e-3)
    lin.project_()
    assert lin.direction.norm().item() == pytest.approx(float(lin.c_f), rel=1e-6)


def test_md_decouple_model_optimizer_and_training_step():
    torch.manual_seed(0)
    from parity_net.model import RAW_GAIN_ONE
    config = _config(bottleneck_md_decouple=True, bottleneck_k=[2, 2, 2])
    model = build_model(config, output_dim=3)
    for b in model.blocks:
        assert b.bottleneck.md_decouple
        assert b.bottleneck.encoder.c_f.item() == pytest.approx(math.sqrt(16))   # sqrt(d_b * N / N)
        assert b.bottleneck.decoder.c_f.item() == pytest.approx(math.sqrt(16))
    # bottleneck group: direction + gains, no weight decay; wd_bottleneck rejected
    opt = build_optimizer(model, OptimizerConfig(name="sgd", lr=1e-2, momentum=0.9, weight_decay=0.1))
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["bottleneck"]["weight_decay"] == 0.0 and groups["hidden"]["weight_decay"] == 0.1
    n_bn = sum(p.numel() for p in groups["bottleneck"]["params"])
    assert n_bn == 3 * 2 * (16 * 32 + 16 + 32)
    with pytest.raises(ValueError):
        build_optimizer(model, OptimizerConfig(wd_bottleneck=0.01))
    # one SGD step + projection keeps every direction on its sphere, gains move
    x = torch.randint(0, 2, (16, 8)).float() * 2 - 1
    model(x).square().sum().backward()
    opt.step(); model.md_project_()
    for b in model.blocks:
        for lin in (b.bottleneck.encoder, b.bottleneck.decoder):
            assert lin.direction.norm().item() == pytest.approx(float(lin.c_f), rel=1e-6)
            # gains received gradient and moved (by a tiny amount at this init scale)
            assert lin.raw_gain_row.grad is not None and lin.raw_gain_row.grad.abs().sum() > 0
            assert not torch.equal(lin.raw_gain_row, torch.full_like(lin.raw_gain_row, RAW_GAIN_ONE))
    # variances report the fused weight; config rejects explicit variances under MD
    assert "blocks.0.bottleneck.encoder.weight" in model.weight_variances()
    with pytest.raises(ValueError):
        _config(bottleneck_md_decouple=True, bottleneck_encoder_weight_variance=0.1).bottleneck_ks()
    # state dict round trip and config round trip
    other = build_model(config, output_dim=3)
    other.load_state_dict(model.state_dict())
    assert torch.allclose(model(x), other(x))
    assert not build_model(_config(), output_dim=3).blocks[0].bottleneck.md_decouple


def test_score_selection_hard_is_a_pure_filter():
    torch.manual_seed(0)
    b = TopKBottleneck(width=10, dim=6, k=2, abs_topk=True, encoder_variance=None, decoder_variance=None,
                       bias=False, selection="scores", score_grad="hard")
    assert b.scores.shape == (6,)
    m = b.score_mask()
    assert int((m != 0).sum()) == 2
    kept = (m != 0).nonzero().flatten()
    assert torch.allclose(m[kept], b.scores[kept])
    x = torch.randn(5, 10)
    codes = b.encode(x)
    # same channels on every input, linear in the encoder output
    assert torch.equal((codes != 0).float(), (m != 0).float().expand(5, -1))
    assert torch.allclose(codes, b.encoder(x) * m)
    # hard: only the selected scores get gradient
    (b(x) ** 2).sum().backward()
    g = b.scores.grad
    assert torch.all(g[kept] != 0) and torch.all(g[(m == 0)] == 0)
    # abs ranking keeps the largest |s|
    with torch.no_grad():
        b.scores.copy_(torch.tensor([0.1, -3.0, 0.2, 2.0, -0.5, 0.0]))
    assert set(b.score_mask().nonzero().flatten().tolist()) == {1, 3}


def test_score_selection_laplace_ste_gradient():
    torch.manual_seed(0)
    tau = 0.05
    b = TopKBottleneck(width=4, dim=5, k=2, abs_topk=False, encoder_variance=None, decoder_variance=None,
                       bias=False, selection="scores", score_grad="laplace_ste", score_temperature=tau)
    with torch.no_grad():
        b.scores.copy_(torch.tensor([1.0, 0.9, 0.7, 0.2, -1.0]))
    m = b.score_mask()
    assert set(m.nonzero().flatten().tolist()) == {0, 1}          # forward is hard, exact K
    assert torch.allclose(m[:2], b.scores[:2]) and torch.all(m[2:] == 0)
    v = torch.randn(5)
    (m * v).sum().backward()
    # d m_j / d s_j = hard_j + s_j * laplace_pdf(s_j - theta), theta = (0.9 + 0.7) / 2
    theta = 0.8
    s = b.scores.detach()
    pdf = torch.exp(-(s - theta).abs() / tau) / (2 * tau)
    hard = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    expected = v * (hard + s * pdf)
    assert torch.allclose(b.scores.grad, expected, atol=1e-5)
    assert b.scores.grad[2] != 0 and abs(b.scores.grad[4]) < 1e-6   # near-threshold learns, far does not
    # K == dim: no threshold, plain hard mask
    full = TopKBottleneck(width=4, dim=3, k=3, abs_topk=False, encoder_variance=None, decoder_variance=None,
                          bias=False, selection="scores", score_grad="laplace_ste", score_temperature=tau)
    assert torch.allclose(full.score_mask(), full.scores)


def test_score_selection_gumbel():
    torch.manual_seed(0)
    b = TopKBottleneck(width=4, dim=50, k=5, abs_topk=True, encoder_variance=None, decoder_variance=None,
                       bias=False, selection="scores", score_grad="gumbel", score_temperature=1.0)
    b.train()
    masks = [(b.score_mask() != 0).nonzero().flatten().tolist() for _ in range(20)]
    assert all(len(m) == 5 for m in masks)                 # always exactly K
    assert len({tuple(m) for m in masks}) > 1              # noise changes the selection
    x = torch.randn(3, 4)
    codes = b.encode(x)
    assert torch.all((codes != 0).sum(-1) == 5)            # per-forward mask, shared across the batch
    assert torch.equal((codes != 0)[0], (codes != 0)[1])
    (b.score_mask() * torch.randn(50)).sum().backward()
    assert int((b.scores.grad != 0).sum()) == 5            # exact gradient, kept scores only
    b.eval()
    det = set(b.scores.abs().topk(5).indices.tolist())
    assert all(set((b.score_mask() != 0).nonzero().flatten().tolist()) == det for _ in range(5))
    # tiny temperature in training mode reproduces the deterministic choice
    b.score_temperature = 1e-6; b.train()
    assert set((b.score_mask() != 0).nonzero().flatten().tolist()) == det
    with pytest.raises(ValueError):
        _config(bottleneck_selection="scores", bottleneck_score_grad="gumbel",
                bottleneck_score_temperature=0.0).bottleneck_ks()


def test_score_selection_gumbel_reinforce():
    torch.manual_seed(0)
    tau = 0.5
    b = TopKBottleneck(width=4, dim=6, k=2, abs_topk=False, encoder_variance=None, decoder_variance=None,
                       bias=False, selection="scores", score_grad="gumbel_reinforce", score_temperature=tau)
    b.train()
    with torch.no_grad():
        b.scores.copy_(torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]))
    m = b.score_mask()
    assert b.last_logp is not None and int((m != 0).sum()) == 2
    idx = (m != 0).nonzero().flatten()
    # the stored log p matches the Plackett-Luce formula for the drawn order
    r = b.scores.detach(); logits = r / tau
    order = idx[torch.argsort(-r[idx])]      # descending clean rank == draw order for this mask
    manual = 0.0; remaining = torch.ones(6, dtype=torch.bool)
    for i in order:
        manual = manual + logits[i] - torch.logsumexp(logits.masked_fill(~remaining, float("-inf")), 0)
        remaining[i] = False
    # (the drawn order is the perturbed-rank order; recompute with it directly)
    assert torch.allclose(b._plackett_luce_logp(r, order), manual, atol=1e-6)
    # surrogate: first call uses the loss itself as baseline (zero term), then EMA
    loss = torch.tensor(0.7)
    s0 = b.reinforce_surrogate(loss, scale=1.0)
    assert s0.item() == 0.0 and b.reinforce_baseline.item() == pytest.approx(0.7) and b.last_logp is None
    b.score_mask()
    s1 = b.reinforce_surrogate(torch.tensor(0.5), scale=2.0)
    assert s1.requires_grad
    s1.backward()
    assert torch.all(b.scores.grad != 0)                  # every score gets the policy gradient
    assert b.reinforce_baseline.item() == pytest.approx(0.99 * 0.7 + 0.01 * 0.5)
    # eval: deterministic, no log p, surrogate is zero
    b.eval(); b.score_mask()
    assert b.last_logp is None and b.reinforce_surrogate(torch.tensor(1.0), 1.0).item() == 0.0
    # model-level: the surrogate is zero for other regimes and only depends on scores
    model = build_model(_config(bottleneck_selection="scores", bottleneck_score_grad="gumbel_reinforce",
                                bottleneck_score_temperature=0.05), output_dim=3)
    model.train()
    x = torch.randint(0, 2, (8, 8)).float() * 2 - 1
    loss = model(x).square().mean()
    model.reinforce_surrogate(loss)          # first step: baseline init
    loss = model(x).square().mean()
    sur = model.reinforce_surrogate(loss)
    sur.backward()
    assert all(b.bottleneck.scores.grad is not None for b in model.blocks)
    assert all(b.linear.weight.grad is None for b in model.blocks)
    plain = build_model(_config(bottleneck_selection="scores", bottleneck_score_grad="gumbel"), output_dim=3)
    plain.train(); l = plain(x).square().mean()
    assert plain.reinforce_surrogate(l).item() == 0.0


def test_score_selection_model_config_and_training_step():
    torch.manual_seed(0)
    config = _config(bottleneck_k=[2, 2, 2], bottleneck_selection="scores",
                     bottleneck_score_grad="laplace_ste", bottleneck_score_temperature=0.02)
    model = build_model(config, output_dim=3)
    for b in model.blocks:
        assert b.bottleneck.selection == "scores" and b.bottleneck.scores.shape == (16,)
        assert (b.bottleneck.scores - 1).abs().max() < 0.1
    opt = build_optimizer(model, OptimizerConfig(name="sgd", lr=1e-2, momentum=0.9, wd_bottleneck=0.01))
    groups = {g["name"]: g for g in opt.param_groups}
    assert any(p is model.blocks[0].bottleneck.scores for p in groups["bottleneck"]["params"])
    x = torch.randint(0, 2, (16, 8)).float() * 2 - 1
    codes_before = model.bottleneck_codes(x)
    assert all(torch.all((c != 0).sum(-1) == 2) for c in codes_before)
    model(x).square().sum().backward()
    assert model.blocks[0].bottleneck.scores.grad is not None
    opt.step()
    # config validation and yaml round trip
    with pytest.raises(ValueError):
        _config(bottleneck_selection="nope").bottleneck_ks()
    with pytest.raises(ValueError):
        _config(bottleneck_score_grad="laplace_ste", bottleneck_score_temperature=0.0).bottleneck_ks()
    plain = build_model(_config(), output_dim=3)
    assert plain.blocks[0].bottleneck.scores is None
    # works together with MD decoupling
    md = build_model(_config(bottleneck_selection="scores", bottleneck_md_decouple=True), output_dim=3)
    assert md.blocks[0].bottleneck.md_decouple and md.blocks[0].bottleneck.scores is not None
    assert md(x).shape == (16, 3)


def test_checkpoint_without_reinforce_buffers_loads():
    from parity_net.checkpoint import load_model_state
    config = _config(bottleneck_selection="scores")
    model = build_model(config, output_dim=3)
    old_state = {k: v for k, v in model.state_dict().items()
                 if not k.endswith((".reinforce_baseline", ".reinforce_steps"))}
    assert len(old_state) < len(model.state_dict())
    fresh = build_model(config, output_dim=3)
    load_model_state(fresh, old_state)               # must not raise
    x = torch.randint(0, 2, (4, 8)).float() * 2 - 1
    assert torch.allclose(fresh(x), model(x))
    with pytest.raises(RuntimeError):
        fresh.load_state_dict({k: v for k, v in old_state.items() if "scores" not in k})


def test_reinit_readouts_only_touches_selected_readouts():
    from parity_net.train import reinit_readouts
    torch.manual_seed(0)
    config = _config(use_layerwise_readouts=True, bottleneck_k=[2, 2, 2])
    names = ["d2_0", "d2_1", "d4_0"]
    a = build_model(config, output_dim=3, target_names_=names)
    torch.manual_seed(0)
    b = build_model(config, output_dim=3, target_names_=names)
    before = {k: v.clone() for k, v in b.state_dict().items()}
    rng_state = torch.random.get_rng_state()
    done = reinit_readouts(b, seed=5, keys=["1"])
    assert done == ["layerwise_readouts[1]"]
    assert torch.equal(torch.random.get_rng_state(), rng_state)        # global stream untouched
    after = b.state_dict()
    changed = [k for k in after if not torch.equal(after[k], before[k])]
    assert changed == ["layerwise_readouts.1.weight"]
    assert torch.equal(a.layerwise_readouts["0"].weight, b.layerwise_readouts["0"].weight)
    assert after["layerwise_readouts.1.weight"].std().item() == pytest.approx(math.sqrt(config.readout_weight_variance), rel=0.5)
    # same seed -> same draw; all readouts when keys is None
    c = build_model(config, output_dim=3, target_names_=names); reinit_readouts(c, seed=5, keys=["1"])
    assert torch.equal(c.layerwise_readouts["1"].weight, b.layerwise_readouts["1"].weight)
    assert len(reinit_readouts(c, seed=6)) == 2
    with pytest.raises(ValueError):
        reinit_readouts(c, seed=1, keys=["7"])


def test_reinit_bottleneck_weights_only_touches_encoder_decoder():
    from parity_net.train import reinit_bottleneck_weights
    for md in (False, True):
        torch.manual_seed(0)
        config = _config(bottleneck_selection="scores", bottleneck_md_decouple=md, bottleneck_k=[2, 2, 2])
        m = build_model(config, output_dim=3)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        state = torch.random.get_rng_state()
        assert reinit_bottleneck_weights(m, seed=3) == 3
        assert torch.equal(torch.random.get_rng_state(), state)
        after = m.state_dict()
        changed = sorted(k for k in after if not torch.equal(after[k], before[k]))
        want = "direction" if md else "weight"
        assert changed == sorted(f"blocks.{b}.bottleneck.{e}.{want}" for b in range(3) for e in ("encoder", "decoder"))
        if md:
            for b in m.blocks:
                assert b.bottleneck.encoder.direction.norm().item() == pytest.approx(float(b.bottleneck.encoder.c_f), rel=1e-6)
        m2 = build_model(config, output_dim=3); reinit_bottleneck_weights(m2, seed=3)
        assert torch.equal(m2.blocks[0].bottleneck.encoder.weight, m.blocks[0].bottleneck.encoder.weight)


def test_non_power_of_two_relevant_dim_gets_full_parity_root():
    from parity_net.data import tree_parity_specs, labels_from_inputs
    specs = tree_parity_specs(34, exclude_targets=["d2", "d4", "d8", "d16", "d32"])
    assert [s.name for s in specs] == ["d34_0"] and specs[0].indices == tuple(range(34))
    assert [s.name for s in tree_parity_specs(34, exclude_targets=["d4", "d8", "d16", "d32"])][-1] == "d34_0"
    assert [s.name for s in tree_parity_specs(16)][-1] == "d16_0"           # power of two: unchanged
    assert "d34_0" not in [s.name for s in tree_parity_specs(34, exclude_targets=["d4", "d8", "d16", "d32", "d34"])]
    x = torch.randint(0, 2, (50, 34)).float() * 2 - 1
    y = labels_from_inputs(x, 34, ["d2", "d4", "d8", "d16", "d32"])
    assert torch.allclose(y[:, 0], x.prod(dim=1))


def test_freeze_blocks_excludes_block_from_optimizer():
    config = _config(bottleneck_k=[2, 2, 2])
    model = build_model(config, output_dim=3)
    for p in model.blocks[1].parameters():
        p.requires_grad_(False)
    opt = build_optimizer(model, OptimizerConfig(lr=1e-3))
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) not in ids for p in model.blocks[1].parameters())
    assert all(id(p) in ids for p in model.blocks[0].parameters())
    assert all(id(p) in ids for p in model.blocks[2].bottleneck.parameters())
    x = torch.randint(0, 2, (8, 8)).float() * 2 - 1
    model(x).square().sum().backward()
    assert model.blocks[1].linear.weight.grad is None and model.blocks[0].linear.weight.grad is not None
    # optimizer state from the frozen stage cannot be loaded into the unfrozen optimizer
    full = build_optimizer(build_model(config, output_dim=3), OptimizerConfig(lr=1e-3))
    with pytest.raises(ValueError):
        full.load_state_dict(opt.state_dict())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
