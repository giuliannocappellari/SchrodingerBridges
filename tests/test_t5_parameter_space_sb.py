from __future__ import annotations

import math

import torch

from scripts.train_t5_direct_adapters import (
    adapter_residual,
    optimize_adapter,
    project_residual,
)
from scripts.train_t5_parameter_sb import decode, encode, pca_fit


def test_rank_two_answer_position_adapter_shapes_and_storage() -> None:
    hidden = torch.randn(3, 16)
    left = torch.randn(16, 2)
    right = torch.randn(16, 2)
    residual = adapter_residual(hidden, left, right)
    assert residual.shape == hidden.shape
    assert 2 * 16 * 2 * 2 < 1_000_000


def test_sparse_endpoint_adapter_optimization_is_finite() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(4, 16)
    output_weight = torch.randn(32, 16)
    base_logits = project_residual(hidden, output_weight)
    specs = [
        {
            "access": "train",
            "positive": index < 2,
            "target_new_token_id": 3,
            "target_true_token_id": 4,
        }
        for index in range(4)
    ]
    left, right, losses = optimize_adapter(
        hidden,
        base_logits,
        output_weight,
        specs,
        rank=2,
        logit_scale=1.0,
        steps=20,
        top_k=8,
    )
    assert left.shape == right.shape == (16, 2)
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]


def test_parameter_adapter_pca_uses_train_rank_ceiling() -> None:
    torch.manual_seed(11)
    train = torch.randn(6, 20)
    mean, components, retained = pca_fit(train, nominal_dim=64)
    assert components.shape == (20, 5)
    reconstructed = decode(encode(train, mean, components), mean, components)
    assert reconstructed.shape == train.shape
    assert 0.0 < retained <= 1.000001
