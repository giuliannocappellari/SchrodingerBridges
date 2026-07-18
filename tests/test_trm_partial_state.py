from __future__ import annotations

from scripts.run_trm_partial_state_target_delta import paired_bootstrap_rewrite_delta
from scripts.trm_editor import state_bucket_from_counts


def test_state_bucket_from_mask_fraction() -> None:
    assert state_bucket_from_counts(4, 4) == "early"
    assert state_bucket_from_counts(2, 4) == "middle"
    assert state_bucket_from_counts(1, 4) == "late"
    assert state_bucket_from_counts(1, 1) == "early"


def test_paired_bootstrap_is_by_case_id() -> None:
    anchor = [
        {"case_id": "a", "bucket": "rewrite", "expected_hit": True},
        {"case_id": "b", "bucket": "rewrite", "expected_hit": True},
    ]
    comparison = [
        {"case_id": "a", "bucket": "rewrite", "expected_hit": False},
        {"case_id": "b", "bucket": "rewrite", "expected_hit": True},
    ]
    result = paired_bootstrap_rewrite_delta(anchor, comparison, trials=200, seed=3)
    assert result["num_cases"] == 2
    assert result["delta"] == 0.5
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]
