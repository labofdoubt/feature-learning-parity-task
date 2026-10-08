"""Langevin annealing in the two-block phase-diagram trainer."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest
import torch
from torch import nn

from analysis.accessibility.run_two_block_phase_diagram import (
    TemperatureSchedule,
    TwoBlockPhaseNet,
    add_langevin_noise,
    build_cube,
    build_sgd,
    decay_coefficients,
    main,
    noise_stream_seed,
    train_cell,
)

CPU = torch.device("cpu")

# train_cell of the pre-Langevin trainer (commit c85776b) on CPU with one thread:
# width 16, seed 0, momentum 0.9, exactly 200 steps -> objective, A4, sum of all parameters.
LEGACY_GOLDEN = {
    (1e-3, 1.0): (0.04036027521105054, 0.989482977985341, -6.680522228123857),
    (1e-2, 5.0): (0.40206531701703985, 0.9140625406498509, -7.43035677767588),
}


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_schedule_holds_then_anneals_exponentially_then_switches_off():
    schedule = TemperatureSchedule(1e-7, hold_steps=5000, anneal_steps=25000, final_ratio=1e-3)
    assert schedule(1) == 1e-7 and schedule(5000) == 1e-7
    assert math.isclose(schedule(5000 + 12500), 1e-7 * 1e-3 ** 0.5, rel_tol=1e-12)
    assert math.isclose(schedule(30000), 1e-10, rel_tol=1e-12)
    assert schedule(30001) == 0.0 and schedule(50000) == 0.0
    assert all(schedule(t) > schedule(t + 1) for t in range(5000, 30000, 997))
    deterministic = TemperatureSchedule(0.0)
    assert not deterministic.active and deterministic(1) == 0.0


def test_noise_variance_follows_each_optimizer_group_learning_rate():
    hidden = nn.Parameter(torch.zeros(400, 500, dtype=torch.float64))
    readout = nn.Parameter(torch.zeros(400, 500, dtype=torch.float64))
    hidden_lr, readout_lr, temperature = 1e-2, 1e-2 / 1024, 1e-7
    optimizer = build_sgd([hidden], [readout], hidden_lr, readout_lr, momentum=0.0)
    add_langevin_noise(optimizer, temperature, torch.Generator().manual_seed(1))
    for parameter, lr in ((hidden, hidden_lr), (readout, readout_lr)):
        expected = math.sqrt(2 * lr * temperature)
        assert abs(float(parameter.std()) / expected - 1) < 0.01
        assert abs(float(parameter.mean())) < 0.01 * expected
    assert math.isclose(float(hidden.std() / readout.std()), math.sqrt(1024), rel_tol=0.02)


def test_noise_streams_are_stable_reproducible_and_cell_specific():
    seed = noise_stream_seed(0, 0, "joint", 1e-3, 0.5)
    assert seed == 7188044834443795187   # SHA-256 based, so identical in every process
    variants = {
        seed,
        noise_stream_seed(1, 0, "joint", 1e-3, 0.5),
        noise_stream_seed(0, 1, "joint", 1e-3, 0.5),
        noise_stream_seed(0, 0, "parent_warm", 1e-3, 0.5),
        noise_stream_seed(0, 0, "joint", 3e-3, 0.5),
        noise_stream_seed(0, 0, "joint", 1e-3, 1.0),
    }
    assert len(variants) == 6
    draws = []
    for _ in range(2):
        hidden = nn.Parameter(torch.zeros(64, dtype=torch.float64))
        readout = nn.Parameter(torch.zeros(8, dtype=torch.float64))
        optimizer = build_sgd([hidden], [readout], 1e-2, 1e-5, momentum=0.0)
        add_langevin_noise(optimizer, 1e-7, torch.Generator().manual_seed(seed))
        draws.append(torch.cat([hidden.detach(), readout.detach()]))
    assert torch.equal(draws[0], draws[1])


def _tiny_cell(temperature=None, generator=None, momentum=0.0):
    """Width-8 cell whose huge plateau tolerance stops deterministic runs at step 10."""
    model = TwoBlockPhaseNet(8, 0, CPU, torch.float64)
    return train_cell(model, build_cube(CPU, torch.float64), decay_coefficients(1e-3, 1.0),
                      2.0, 2.0, 1e-2, 1e-2 / 8, momentum, 40, 10, 5, 1.0, 10,
                      temperature=temperature, noise_generator=generator)


def test_noisy_runs_take_exactly_max_steps_and_record_the_temperature():
    deterministic, _ = _tiny_cell()
    assert deterministic["steps"] == 10 and "temperature" not in deterministic
    schedule = TemperatureSchedule(1e-7, hold_steps=10, anneal_steps=20)
    noisy, trajectory = _tiny_cell(schedule, torch.Generator().manual_seed(3))
    assert noisy["steps"] == 40 and noisy["converged"] == 0.0 and noisy["temperature"] == 0.0
    temperatures = {row["step"]: row["temperature"] for row in trajectory}
    assert temperatures[10] == 1e-7 and temperatures[40] == 0.0
    assert math.isclose(temperatures[30], 1e-10, rel_tol=1e-12)


def test_fan_in_readout_init_only_rescales_the_readouts():
    mup = TwoBlockPhaseNet(256, 0, CPU, torch.float64)
    fan_in = TwoBlockPhaseNet(256, 0, CPU, torch.float64, readout_init="fan_in")
    for name in ("E", "V1", "V2"):
        assert torch.equal(getattr(mup, name), getattr(fan_in, name))
    # Same random draws; std 1/sqrt(N) instead of 1/N, i.e. sqrt(256) = 16 times larger.
    assert torch.allclose(fan_in.w4, 16 * mup.w4) and torch.allclose(fan_in.Wp, 16 * mup.Wp)


def test_noisy_training_requires_zero_momentum():
    with pytest.raises(ValueError):
        _tiny_cell(TemperatureSchedule(1e-7, 10, 20), torch.Generator().manual_seed(3),
                   momentum=0.9)


@pytest.mark.parametrize("cell", sorted(LEGACY_GOLDEN))
def test_zero_temperature_reproduces_the_pre_langevin_trainer(cell, single_thread):
    model = TwoBlockPhaseNet(16, 0, CPU, torch.float64)
    final, _ = train_cell(model, build_cube(CPU, torch.float64), decay_coefficients(*cell),
                          2.0, 2.0, 0.01, 0.01 / 16, 0.9, 200, 200, 50, 1e-4, 50,
                          temperature=TemperatureSchedule(0.0))
    objective, a4, parameter_sum = LEGACY_GOLDEN[cell]
    assert final["steps"] == 200 and "temperature" not in final
    assert math.isclose(final["objective"], objective, rel_tol=1e-10)
    assert math.isclose(final["A4"], a4, rel_tol=1e-10)
    total = float(sum(parameter.sum() for parameter in model.parameters()))
    assert math.isclose(total, parameter_sum, rel_tol=1e-10)


def _run_main(directory, *extra):
    main(["--ratios", "1", "--lambda0s", "1e-3", "--seeds", "0",
          "--protocols", "joint,parent_warm", "--tag", "t0", "--output-dir", str(directory),
          "--width", "8", "--device", "cpu", "--max-steps", "40", "--min-steps", "10",
          "--snapshot-every", "10", "--pretrain-max-steps", "20",
          "--pretrain-min-steps", "10", "--save-checkpoints", *extra])
    return (pd.read_csv(directory / "results_t0.csv"),
            pd.read_csv(directory / "trajectories_t0.csv"),
            json.loads((directory / "manifest_t0.json").read_text()))


def test_zero_temperature_main_keeps_the_legacy_ids_and_columns(tmp_path):
    results, trajectories, manifest = _run_main(tmp_path)
    assert sorted(results.cell_id) == ["joint_s0_l0.001_r1", "parent_warm_s0_l0.001_r1"]
    columns = results.columns.union(trajectories.columns)
    assert not [column for column in columns if column.startswith(("temperature", "noise"))]
    assert "temperature" not in manifest


def test_noisy_main_records_the_schedule_everywhere_without_id_collisions(tmp_path):
    noisy = ("--momentum", "0", "--pretrain-momentum", "0.9", "--temperature-initial", "1e-7",
             "--temperature-hold-steps", "10", "--temperature-anneal-steps", "20",
             "--noise-seed", "4")
    results, trajectories, manifest = _run_main(tmp_path, *noisy)
    assert sorted(results.cell_id) == ["joint_s0_l0.001_r1_T1e-07_n4",
                                       "parent_warm_s0_l0.001_r1_T1e-07_n4"]
    assert (results.steps == 40).all() and (results.temperature_initial == 1e-7).all()
    assert (results.noise_seed == 4).all() and results.noise_stream_seed.nunique() == 2
    joint = trajectories[trajectories.protocol == "joint"].set_index("step").temperature
    assert joint[10] == 1e-7 and joint[40] == 0.0
    assert manifest["temperature"] == {"temperature_initial": 1e-7, "temperature_hold_steps": 10,
                                       "temperature_anneal_steps": 20,
                                       "temperature_final_ratio": 1e-3, "noise_seed": 4}
    payload = torch.load(tmp_path / "checkpoints" / "joint_s0_l0.001_r1_T1e-07_n4.pt",
                         weights_only=True)
    assert payload["metadata"]["temperature_initial"] == 1e-7
    assert payload["metadata"]["noise_seed"] == 4
    with pytest.raises(ValueError):   # default momentum 0.9 is rejected for noisy runs
        _run_main(tmp_path / "rejected", "--temperature-initial", "1e-7")
