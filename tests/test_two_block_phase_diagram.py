"""Focused checks for the fully trainable two-block phase sweep."""
from __future__ import annotations

import math

import torch

from analysis.accessibility.run_two_block_phase_diagram import (
    TwoBlockPhaseNet,
    build_cube,
    build_sgd,
    decay_coefficients,
    loss_terms,
    measure,
)


def test_decay_axis_matches_report_definition():
    decays = decay_coefficients(lambda0=0.03, ratio=5.0)
    assert decays.embedding == 0.03
    assert decays.block1 == 0.15
    assert decays.block2 == 0.03
    assert decays.parent_readout == 0.03
    assert decays.target_readout == 0.03


def test_explicit_regularizer_taxes_the_five_groups_independently():
    device = torch.device("cpu")
    dtype = torch.float64
    model = TwoBlockPhaseNet(width=4, seed=0, device=device, dtype=dtype)
    data = build_cube(device, dtype)
    decays = decay_coefficients(lambda0=0.2, ratio=3.0)
    terms = loss_terms(model, data, decays, gamma2=2.0, gamma4=2.0)
    expected = 0.5 * (
        decays.embedding * model.E.square().sum()
        + decays.block1 * model.V1.square().sum()
        + decays.block2 * model.V2.square().sum()
        + decays.parent_readout * model.Wp.square().sum()
        + decays.target_readout * model.w4.square().sum()
    )
    assert torch.allclose(terms["regularizer"], expected)
    assert torch.allclose(
        terms["objective"], terms["data_loss"] + terms["regularizer"]
    )


def test_mup_sgd_uses_distinct_hidden_and_readout_learning_rates():
    model = TwoBlockPhaseNet(
        width=4, seed=0, device=torch.device("cpu"), dtype=torch.float64
    )
    optimizer = build_sgd(
        [model.E, model.V1, model.V2], [model.Wp, model.w4], 1e-2, 2.5e-3, 0.9
    )
    assert optimizer.param_groups[0]["lr"] == 1e-2
    assert optimizer.param_groups[1]["lr"] == 2.5e-3


def test_zero_model_is_classified_as_suppressed_and_reconstructs_target_route():
    device = torch.device("cpu")
    dtype = torch.float64
    model = TwoBlockPhaseNet(width=4, seed=0, device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    data = build_cube(device, dtype)
    metrics = measure(
        model,
        data,
        decay_coefficients(lambda0=0.1, ratio=1.0),
        gamma2=2.0,
        gamma4=2.0,
    )
    assert metrics["phase"] == "suppressed"
    assert metrics["A4"] == 0.0
    assert abs(float(metrics["route_reconstruction_error"])) < 1e-12
    assert math.isnan(float(metrics["gate_agrees_with_R"]))
