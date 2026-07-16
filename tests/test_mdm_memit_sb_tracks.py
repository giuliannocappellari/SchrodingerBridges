from __future__ import annotations

import math

import torch

from scripts.mdm_memit_editor import sparse_support_kl
from scripts.run_mask_pattern_sb_track import _scheduled_bridge
from scripts.run_toy_text_csbm_fallback import (
    _kernel,
    _make_dataset,
    _sinkhorn,
)


def test_sparse_path_kl_has_gradient_and_is_zero_at_identity():
    base = torch.tensor([0.2, -0.1, 1.2, 0.4])
    identical = base.clone().requires_grad_(True)
    zero = sparse_support_kl(identical, base, top_k=4)
    assert abs(float(zero.detach())) < 1e-7
    edited = torch.tensor([0.8, -0.1, 0.7, 0.4], requires_grad=True)
    loss = sparse_support_kl(edited, base, top_k=4)
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert edited.grad is not None
    assert float(edited.grad.abs().sum()) > 0


def test_state_dependent_bridge_schedules_normalize():
    n = 3
    terminal = (1 << n) - 1
    costs = {
        (mask, index): float(index + 1)
        for mask in range(terminal)
        for index in range(n)
        if not mask & (1 << index)
    }
    reference = {
        (mask, index): 1.0 / (n - mask.bit_count())
        for mask in range(terminal)
        for index in range(n)
        if not mask & (1 << index)
    }
    for schedule in ("early_strong", "late_strong"):
        policy = _scheduled_bridge(costs, n, reference, beta=1.0, schedule=schedule)
        assert all(abs(sum(row.values()) - 1.0) < 1e-10 for row in policy.values())


def test_toy_csbm_dataset_and_sinkhorn_endpoint_constraints():
    dataset = _make_dataset()
    assert {key: len(value) for key, value in dataset.items()} == {
        "train": 5000,
        "validation": 1000,
        "test": 1000,
    }
    entities = [{row["entity"] for row in dataset[split]} for split in dataset]
    assert entities[0].isdisjoint(entities[1])
    assert entities[0].isdisjoint(entities[2])
    mu = [0.1, 0.2, 0.3, 0.4]
    nu = [0.4, 0.1, 0.2, 0.3]
    policy, _ = _sinkhorn(_kernel([1, 2, 3, 0], beta=5.0), mu, nu)
    induced = [sum(mu[i] * policy[i][j] for i in range(4)) for j in range(4)]
    assert max(abs(left - right) for left, right in zip(induced, nu)) < 1e-8
