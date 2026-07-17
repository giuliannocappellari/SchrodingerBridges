from __future__ import annotations

import pytest

from scripts.report_dnpe_selection import harmonic, score_report


def test_harmonic_zeroes_inactive_editor() -> None:
    assert harmonic((0.8, 0.5, 1.0)) > 0
    assert harmonic((0.0, 0.5, 1.0)) == 0.0


def test_constraints_precede_selection_score(tmp_path) -> None:
    report = {
        "method": "unsafe",
        "rewrite_exact": 0.9,
        "declarative_paraphrase_exact": 0.8,
        "same_subject_tfpr": 0.5,
        "near_tfpr": 0.0,
        "far_tfpr": 0.0,
        "malformed_rate": 0.0,
        "gpu_minutes_per_edit": 0.1,
        "base_summary": {
            "same_subject": {"target_new_tfpr_or_exact": 0.0},
            "near_locality": {"target_new_tfpr_or_exact": 0.0},
            "far_locality": {"target_new_tfpr_or_exact": 0.0},
        },
        "edited_summary": {
            "same_subject": {"base_agreement": 0.9},
            "near_locality": {"base_agreement": 1.0},
            "far_locality": {"base_agreement": 1.0},
        },
    }
    scored = score_report(report, tmp_path / "report_summary.json")
    assert scored["selection_score"] > 0
    assert scored["constraint_pass"] is False
    assert scored["feasible_selection_score"] == ""
