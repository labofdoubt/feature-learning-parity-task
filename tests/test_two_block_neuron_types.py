from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from analysis.accessibility.run_two_block_phase_diagram import TwoBlockPhaseNet, build_cube
from analysis.accessibility.two_block_neuron_types import (
    classify,
    log_edges,
    neuron_features,
    phase_boundary_segments,
    walsh_coefficients,
    walsh_matrix,
)


def test_walsh_matrix_is_orthonormal_on_the_cube():
    x = build_cube(torch.device("cpu"), torch.float64).x
    chi = walsh_matrix(x)
    assert torch.allclose(chi.T @ chi / x.shape[0], torch.eye(16, dtype=torch.float64))


def _hand_built_model() -> TwoBlockPhaseNet:
    """Width-16 network whose neuron types are known by construction.

    Residual coordinates 0-3 carry the raw bits (h0 = x, block-1 neurons 0-3 dead).
    Block 1: 4 = pair-A corner, 5 = pair-B corner, 6 = balanced quad gate,
    7 = unbalanced pair-A (other), 8/9 = +-(x0+x1) and 10/11 = +-(x2+x3) corners;
    relu(s) + relu(-s) = |x0+x1| = 1 + x0x1 makes the parents available to block 2.
    Block 2: 0 = 2+2->4 on (x0x1 - x2x3), 1 = pair-A corner on raw bits,
    2 = balanced quad gate on raw bits; everything else is dead.
    """
    gamma, gamma2 = 0.5, 1.0
    model = TwoBlockPhaseNet(16, 0, torch.device("cpu"), torch.float64)
    E = torch.zeros(16, 4, dtype=torch.float64)
    E[:4] = torch.eye(4, dtype=torch.float64)
    V1 = torch.zeros(16, 16, dtype=torch.float64)
    V1[4, :4] = gamma * torch.tensor([1.0, -1.0, 0.0, 0.0])
    V1[5, :4] = gamma * torch.tensor([0.0, 0.0, 1.0, 1.0])
    V1[6, :4] = gamma * torch.tensor([1.0, -1.0, -1.0, -1.0])
    V1[7, :4] = gamma * torch.tensor([1.0, 0.5, 0.0, 0.0])
    V1[8, :4] = gamma * torch.tensor([1.0, 1.0, 0.0, 0.0])
    V1[9, :4] = -gamma * torch.tensor([1.0, 1.0, 0.0, 0.0])
    V1[10, :4] = gamma * torch.tensor([0.0, 0.0, 1.0, 1.0])
    V1[11, :4] = -gamma * torch.tensor([0.0, 0.0, 1.0, 1.0])
    V2 = torch.zeros(16, 16, dtype=torch.float64)
    V2[0, 8:12] = gamma2 * torch.tensor([1.0, 1.0, -1.0, -1.0])
    V2[1, :4] = gamma2 * torch.tensor([1.0, -1.0, 0.0, 0.0])
    V2[2, :4] = gamma2 * torch.tensor([1.0, 1.0, 1.0, -1.0])
    with torch.no_grad():
        model.E.copy_(E)
        model.V1.copy_(V1)
        model.V2.copy_(V2)
    return model


def _labels(model, block, tau):
    z_coef, a_coef = walsh_coefficients(model)[block]
    return list(classify(neuron_features(z_coef, a_coef), block, tau))


def test_block1_types_follow_the_input_structure():
    labels = _labels(_hand_built_model(), 1, tau=1e-3)
    assert [labels[n] for n in (4, 5, 8, 9, 10, 11)] == ["1+1->2"] * 6
    assert labels[6] == "1+1+1+1->4"
    assert labels[7] == "other"
    assert [labels[n] for n in (0, 1, 2, 3, 12, 13, 14, 15)] == ["dead"] * 8


def test_block2_detects_composition_and_raw_bit_circuits():
    labels = _labels(_hand_built_model(), 2, tau=1e-3)
    assert labels[:3] == ["2+2->4", "1+1->2", "1+1+1+1->4"]
    assert labels[3:] == ["dead"] * 13


def test_threshold_above_every_coefficient_marks_all_neurons_dead():
    model = _hand_built_model()
    assert set(_labels(model, 1, tau=10.0)) == {"dead"}
    assert set(_labels(model, 2, tau=10.0)) == {"dead"}


def test_phase_boundary_follows_edges_between_hc_and_hd_cells_only():
    lambda0s, ratios = np.array([1.0, 10.0, 100.0]), np.array([1.0, 10.0])
    phases = pd.DataFrame({
        "lambda0": [1.0, 10.0, 100.0, 1.0, 10.0, 100.0],
        "ratio": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
        "phase": ["H_C", "H_C", "suppressed", "H_C", "H_D", "suppressed"],
    })
    x, y = log_edges(lambda0s), log_edges(ratios)
    segments = phase_boundary_segments(phases, lambda0s, ratios, x, y)
    expected = [
        [(x[1], y[1]), (x[2], y[1])],   # below the H_D cell
        [(x[1], y[1]), (x[1], y[2])],   # left of the H_D cell
    ]
    assert len(segments) == len(expected)
    for got, want in zip(sorted(segments), sorted(expected)):
        assert np.allclose(got, want)
