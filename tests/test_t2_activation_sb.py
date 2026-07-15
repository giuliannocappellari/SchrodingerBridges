from __future__ import annotations

import torch

from scripts.train_t2_activation_sb import (
    apply_ridge,
    brownian_bridge_training_rows,
    integrate_bridge_drift,
    prediction_metrics,
    ridge,
)


def test_t2_ridge_and_metrics_are_finite() -> None:
    inputs = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    targets = torch.tensor([[1.0], [2.0], [3.0], [0.0]])
    weights = ridge(inputs, targets)
    predictions = apply_ridge(inputs, weights)
    assert torch.isfinite(predictions).all()
    z0 = torch.zeros((4, 2))
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    positive = torch.tensor([True, True, False, False])
    metrics = prediction_metrics(z0, z1, z1, positive)
    assert metrics["endpoint_cosine"] == 1.0
    assert metrics["identity_drift_norm"] == 0.0


def test_t2_brownian_bridge_rows_and_dynamic_integration_are_finite() -> None:
    z0 = torch.zeros((6, 3))
    z1 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    condition = torch.eye(6)
    features, targets = brownian_bridge_training_rows(
        z0, z1, condition, steps=4, sigma=0.05
    )
    assert features.shape == (24, 13)
    assert targets.shape == (24, 3)
    weights = ridge(features, targets, alpha=0.1)
    delta, energy = integrate_bridge_drift(z0, condition, weights, steps=4)
    assert torch.isfinite(delta).all()
    assert energy >= 0.0
    assert delta[:3].norm(dim=1).mean() > delta[3:].norm(dim=1).mean()
