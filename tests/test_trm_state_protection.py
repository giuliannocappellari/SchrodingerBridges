from __future__ import annotations

from scripts.run_trm_state_conditioned_protection import (
    distribution_kl,
    paired_tfpr_bootstrap,
    selected_shared_policy,
)

import torch


def test_selected_shared_policy_maps_every_d1_variant() -> None:
    assert selected_shared_policy({"selected_partial_method": "uniform_partial_state_delta"}) == (
        "uniform",
        "random",
        0.1,
    )
    assert selected_shared_policy({"selected_partial_method": "state_bucketed_delta"}) == (
        "cycle",
        "base_confidence",
        0.1,
    )


def test_distribution_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(4, 8)
    assert abs(distribution_kl(logits, logits)) < 1e-6


def test_same_subject_bootstrap_pairs_by_case() -> None:
    left = [
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": False},
        {"case_id": "b", "bucket": "same_subject", "target_new_hit": False},
    ]
    right = [
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": True},
        {"case_id": "b", "bucket": "same_subject", "target_new_hit": False},
    ]
    result = paired_tfpr_bootstrap(left, right, trials=200, seed=9)
    assert result["num_cases"] == 2
    assert result["delta"] == -0.5


def test_same_subject_bootstrap_averages_prompt_rows_within_case() -> None:
    left = [
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": True},
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": False},
        {"case_id": "b", "bucket": "same_subject", "target_new_hit": False},
    ]
    right = [
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": False},
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": False},
        {"case_id": "b", "bucket": "same_subject", "target_new_hit": False},
    ]
    result = paired_tfpr_bootstrap(left, right, trials=200, seed=11)
    assert result["num_cases"] == 2
    assert result["delta"] == 0.25
