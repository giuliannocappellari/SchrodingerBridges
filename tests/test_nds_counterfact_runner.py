from pathlib import Path

import pytest
import torch

from scripts.run_nds_counterfact_editor import (
    aggregate_rows,
    harmonic_mean,
    js_divergence,
    per_prompt_distribution_rows,
    validate_manifest_access,
)


def test_locked_historical_and_fresh_confirmation_guards():
    with pytest.raises(PermissionError):
        validate_manifest_access(Path("analysis_500.jsonl"), allow_confirmation=False)
    with pytest.raises(PermissionError):
        validate_manifest_access(Path("cf_nds_confirmation_200.jsonl"), allow_confirmation=False)
    validate_manifest_access(Path("cf_nds_confirmation_200.jsonl"), allow_confirmation=True)


def test_common_aggregate_and_harmonic_score():
    rows = [
        {
            "case_id": "a",
            "bucket": "rewrite",
            "expected_hit": True,
            "target_new_hit": True,
            "target_true_hit": False,
            "target_token_f1": 1.0,
            "malformed": False,
        },
        {
            "case_id": "b",
            "bucket": "rewrite",
            "expected_hit": False,
            "target_new_hit": False,
            "target_true_hit": True,
            "target_token_f1": 0.0,
            "malformed": False,
        },
    ]
    assert aggregate_rows(rows)["rewrite"]["expected_exact"] == 0.5
    assert harmonic_mean((0.5, 0.5, 1.0)) == pytest.approx(0.6)


def test_js_divergence_is_zero_for_equal_logits():
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    assert js_divergence(logits, logits) == pytest.approx(0.0, abs=1e-7)
    rows = per_prompt_distribution_rows(
        [{"case_id": "a", "bucket": "same_subject", "prompt": "p"}],
        logits,
        logits,
    )
    assert rows[0]["protected_kl"] == pytest.approx(0.0, abs=1e-7)
    assert rows[0]["protected_js"] == pytest.approx(0.0, abs=1e-7)
