from __future__ import annotations

from scripts.report_dnpe_nullspace_baseline import _locality_score


def test_locality_score_uses_base_agreement_buckets() -> None:
    report = {
        "edited_summary": {
            "same_subject": {"base_agreement": 0.8},
            "near_locality": {"base_agreement": 0.9},
            "far_locality": {"base_agreement": 1.0},
        }
    }
    assert _locality_score(report) == 0.9
