from __future__ import annotations

from scripts.run_trm_e1_smoke import REQUIRED_METHODS, normalized_report_row


def test_e1_registry_contains_every_frozen_family() -> None:
    assert len(REQUIRED_METHODS) == 13
    assert "timerome_partial_state_state_protected" in REQUIRED_METHODS
    assert "ordinary_mdm_memit" in REQUIRED_METHODS


def test_normalized_report_recomputes_self_normalized_score(tmp_path) -> None:
    report = {
        "num_edits": 20,
        "rewrite_exact": 0.8,
        "declarative_paraphrase_exact": 0.4,
        "same_subject_tfpr": 0.1,
        "near_tfpr": 0.0,
        "far_tfpr": 0.0,
        "malformed_rate": 0.0,
        "base_summary": {
            "near_locality": {"expected_exact": 0.5},
            "far_locality": {"expected_exact": 0.5},
        },
        "edited_summary": {
            "near_locality": {"expected_exact": 0.25},
            "far_locality": {"expected_exact": 0.25},
        },
    }
    path = tmp_path / "run"
    path.mkdir()
    row = normalized_report_row("method", path, report=report)
    assert row["clipped_self_normalized_locality"] == 0.5
    assert row["selection_score"] > 0
