"""The d4-parent / d8-target two-block trainer and its figure-side analysis."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.accessibility.diagnose_two_block_d4_d8 import READ_GROUPS, network_diagnostics
from analysis.accessibility.plot_two_block_d4_d8_layerwise_phase import (
    F_ZERO,
    free_energy_surfaces,
    solve_cell,
)
from analysis.accessibility.plot_two_block_langevin_phase import neuron_route_shares_d4_d8
from analysis.accessibility.run_two_block_d4_d8_phase_diagram import (
    PARENT_MASKS,
    TARGET_MASK,
    build_cube8,
    loss_terms8,
    main,
    measure8,
)
from analysis.accessibility.run_two_block_phase_diagram import (
    TwoBlockPhaseNet,
    build_sgd,
    decay_coefficients,
)

CPU = torch.device("cpu")


def test_cube8_holds_the_two_d4_parents_and_the_d8_target():
    data = build_cube8(CPU, torch.float64)
    assert data.x.shape == (256, 8) and set(data.x.unique().tolist()) == {-1.0, 1.0}
    assert torch.equal(data.y4[:, 0], data.x[:, :4].prod(dim=1))
    assert torch.equal(data.y4[:, 1], data.x[:, 4:].prod(dim=1))
    assert torch.equal(data.y8[:, 0], data.x.prod(dim=1))
    assert torch.allclose(data.chi.T @ data.chi / 256, torch.eye(256, dtype=torch.float64))
    assert PARENT_MASKS[0] | PARENT_MASKS[1] == TARGET_MASK


def test_zero_network_is_suppressed_and_routes_reconstruct_the_target():
    model = TwoBlockPhaseNet(8, 0, CPU, torch.float64, readout_init="fan_in", input_dim=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    metrics = measure8(model, build_cube8(CPU, torch.float64), decay_coefficients(1e-3, 1.0),
                       2.0, 2.0)
    assert metrics["phase"] == "suppressed" and metrics["A8"] == 0.0
    assert metrics["route_reconstruction_error"] == 0.0
    assert metrics["objective"] == 2.0 + 1.0   # gamma4/2 * (1 + 1) + gamma8/2 * 1


def test_trainer_runs_every_cell_for_exactly_max_steps(tmp_path):
    main(["--ratios", "1", "--lambda0s", "1e-3", "--tag", "t0", "--output-dir", str(tmp_path),
          "--width", "16", "--device", "cpu", "--max-steps", "60", "--min-steps", "60",
          "--snapshot-every", "20", "--pretrain-max-steps", "30", "--pretrain-min-steps", "30",
          "--save-checkpoints"])
    results = pd.read_csv(tmp_path / "results_t0.csv")
    assert sorted(results.cell_id) == ["joint_s0_l0.001_r1", "parent_warm_s0_l0.001_r1"]
    assert (results.steps == 60).all() and (results.task == "d4d8").all()
    manifest = json.loads((tmp_path / "manifest_t0.json").read_text())
    assert manifest["task"] == "d4d8" and manifest["arguments"]["readout_init"] == "fan_in"
    state = torch.load(tmp_path / "checkpoints" / "joint_s0_l0.001_r1.pt",
                       weights_only=True)["state_dict"]
    assert state["E"].shape == (16, 8)
    pretrain = json.loads((tmp_path / "pretrain_t0.json").read_text())
    assert pretrain["0"]["steps"] == 30


def test_route_shares_of_a_d4_d8_network_are_fractions():
    model = TwoBlockPhaseNet(32, 1, CPU, torch.float64, readout_init="fan_in", input_dim=8)
    shares = neuron_route_shares_d4_d8(model)
    for key in ("phi_out", "phi_sq", "f_C_sq"):
        assert np.isnan(shares[key]) or 0.0 <= shares[key] <= 1.0


def test_block_splits_reconstruct_the_amplitudes_and_read_shares_are_a_partition():
    model = TwoBlockPhaseNet(32, 2, CPU, torch.float64, readout_init="fan_in", input_dim=8)
    metrics = measure8(model, build_cube8(CPU, torch.float64), decay_coefficients(1e-3, 1.0),
                       2.0, 2.0)
    row = network_diagnostics(model)
    assert np.isclose(row["A4_block1"] + row["A4_block2"], metrics["A4_mean"], atol=1e-12)
    assert np.isclose(row["A8_block1"] + row["A8_block2"], metrics["A8"], atol=1e-12)
    reads = [row[f"read_{name}"] for name in READ_GROUPS] + [row["read_rest"]]
    assert all(-1e-12 <= value <= 1 + 1e-12 for value in reads)
    assert np.isclose(sum(reads), 1.0)


def test_theory_phase_follows_the_free_energy_minima():
    free_d, free_c, _, _ = free_energy_surfaces(1e-2, 1.0)
    assert solve_cell(1e-2, 1.0)[0] == 0 and free_c.min() < free_d.min() < F_ZERO
    free_d, free_c, _, _ = free_energy_surfaces(1.0, 1.0)
    assert solve_cell(1.0, 1.0)[0] == 3 and min(free_c.min(), free_d.min()) >= F_ZERO - 2e-4


def test_base_readout_init_rescales_the_same_draws():
    def net(**kwargs):
        return TwoBlockPhaseNet(64, 3, CPU, torch.float64, input_dim=8, **kwargs)

    mup, fan_in = net(readout_init="mup"), net(readout_init="fan_in")
    base16, base64 = net(readout_init="base", base_width=16), net(readout_init="base", base_width=64)
    for name in ("E", "V1", "V2"):
        assert torch.equal(getattr(base16, name), getattr(mup, name))
    for name in ("Wp", "w4"):   # Var N0/N^2: sqrt(16) x muP, and fan-in at N0 = N
        assert torch.allclose(getattr(base16, name), 4.0 * getattr(mup, name))
        assert torch.allclose(getattr(base64, name), getattr(fan_in, name))
    with pytest.raises(ValueError):
        net(readout_init="base")
    with pytest.raises(ValueError):
        net(readout_init="mup", base_width=16)


@pytest.mark.parametrize("device", [CPU] + ([torch.device("cuda")] if torch.cuda.is_available()
                                            else []))
def test_a_separate_embedding_group_at_the_hidden_rate_changes_nothing(device):
    data = build_cube8(device, torch.float64)
    decays = decay_coefficients(1e-2, 1.0)
    states = []
    for separate in (False, True):
        model = TwoBlockPhaseNet(16, 0, device, torch.float64, readout_init="fan_in", input_dim=8)
        if separate:
            optimizer = build_sgd([model.V1, model.V2], [model.Wp, model.w4], 1e-3, 1e-3, 0.9,
                                  embedding=([model.E], 1e-3))
        else:
            optimizer = build_sgd([model.E, model.V1, model.V2], [model.Wp, model.w4], 1e-3, 1e-3,
                                  0.9)
        for _ in range(20):
            optimizer.zero_grad()
            loss_terms8(model, data, decays, 2.0, 2.0)["objective"].backward()
            optimizer.step()
        states.append({name: value.clone() for name, value in model.state_dict().items()})
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name])


@pytest.mark.parametrize("extra, readout_lr, embedding_lr", [
    ([], 1e-3, 1e-3),                                                  # fan_in: every layer lr
    (["--readout-init", "mup"], 1e-3 / 16, 1e-3),                      # muP-like readout
    (["--readout-init", "base", "--base-width", "4"], 1e-3 * 4 / 16, 1e-3 * 16 / 4),
    (["--readout-init", "base", "--base-width", "4", "--embedding-lr", "2e-3"],
     1e-3 * 4 / 16, 2e-3),
])
def test_parametrizations_derive_the_readout_and_embedding_rates(tmp_path, extra, readout_lr,
                                                                 embedding_lr):
    main(["--ratios", "1", "--lambda0s", "1e-2", "--protocols", "joint", "--tag", "t",
          "--output-dir", str(tmp_path), "--width", "16", "--device", "cpu", "--lr", "1e-3",
          "--max-steps", "5", "--min-steps", "5", "--snapshot-every", "5"] + extra)
    manifest = json.loads((tmp_path / "manifest_t.json").read_text())
    assert manifest["derived_readout_lr"] == pytest.approx(readout_lr, rel=1e-12)
    assert manifest["derived_embedding_lr"] == pytest.approx(embedding_lr, rel=1e-12)
    row = pd.read_csv(tmp_path / "results_t.csv").iloc[0]
    assert row.readout_lr == pytest.approx(readout_lr, rel=1e-12)
    assert row.embedding_lr == pytest.approx(embedding_lr, rel=1e-12)


def test_base_readout_needs_a_base_width(tmp_path):
    with pytest.raises(ValueError):
        main(["--ratios", "1", "--lambda0s", "1e-2", "--protocols", "joint", "--tag", "t",
              "--output-dir", str(tmp_path), "--width", "16", "--device", "cpu",
              "--readout-init", "base", "--max-steps", "5", "--min-steps", "5"])
