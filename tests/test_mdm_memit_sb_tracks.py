from __future__ import annotations

import math
from pathlib import Path

import torch

from scripts.mdm_memit_editor import sparse_support_kl
from scripts.run_mask_pattern_sb_track import _analytical_tests, _scheduled_bridge
from scripts.run_sb_regularized_memit_track import (
    _nearest_lower_path_weight,
    _retains_dev_efficacy,
)
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


def test_m4_analytical_fixture_has_order_dependent_path_cost(tmp_path: Path):
    assert _analytical_tests(tmp_path)


def test_m3_bounded_rescue_uses_nearest_lower_predeclared_path_weight():
    assert _nearest_lower_path_weight(0.25) == 0.1
    assert _nearest_lower_path_weight(0.1) == 0.05
    assert _nearest_lower_path_weight(0.05) == 0.01
    assert _nearest_lower_path_weight(0.01) is None


def test_m3_dev_efficacy_gate_applies_all_frozen_limits():
    baseline = {"rewrite_exact": 0.8, "paraphrase_exact": 0.5}
    passing = {"rewrite_exact": 0.75, "paraphrase_exact": 0.45, "malformed_rate": 0.05}
    assert _retains_dev_efficacy(passing, baseline)
    assert not _retains_dev_efficacy({**passing, "rewrite_exact": 0.749}, baseline)
    assert not _retains_dev_efficacy({**passing, "paraphrase_exact": 0.449}, baseline)
    assert not _retains_dev_efficacy({**passing, "malformed_rate": 0.051}, baseline)


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
