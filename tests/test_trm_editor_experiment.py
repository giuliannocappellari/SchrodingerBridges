from __future__ import annotations

import pytest

from scripts.run_trm_editor_experiment import grouped_diagnostics, validate_mode


def test_editor_experiment_mode_matrix() -> None:
    validate_mode("shared", "none")
    validate_mode("shared", "static")
    validate_mode("shared", "shared")
    validate_mode("bucketed", "none")
    validate_mode("bucketed", "state")
    with pytest.raises(ValueError):
        validate_mode("shared", "state")
    with pytest.raises(ValueError):
        validate_mode("bucketed", "static")


def test_grouped_diagnostics_preserves_exact_bins() -> None:
    rows = [
        {
            "case_id": "a",
            "target_length": 1,
            "relation_id": "P1",
            "bucket": "rewrite",
            "expected_hit": True,
            "target_new_hit": True,
            "malformed": False,
        },
        {
            "case_id": "b",
            "target_length": 2,
            "relation_id": "P1",
            "bucket": "rewrite",
            "expected_hit": False,
            "target_new_hit": False,
            "malformed": False,
        },
    ]
    length, relation = grouped_diagnostics(rows)
    assert {row["target_length"] for row in length} == {1, 2}
    assert relation[0]["num_edits"] == 2
