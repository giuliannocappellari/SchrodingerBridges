from __future__ import annotations

from scripts.run_dnpe_pilot_selection import stress_score


def test_stress_score_is_zero_when_efficacy_is_zero() -> None:
    report = {
        "rewrite_exact": 0.0,
        "declarative_paraphrase_exact": 0.5,
        "edited_summary": {
            "same_subject": {"base_agreement": 1.0},
            "near_locality": {"base_agreement": 1.0},
            "far_locality": {"base_agreement": 1.0},
        },
    }
    assert stress_score(report) == 0.0
