from __future__ import annotations

import torch

from scripts.train_t2_activation_sb import apply_ridge, prediction_metrics, ridge


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
