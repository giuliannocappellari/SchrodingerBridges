from __future__ import annotations

from scripts.resolve_cl_conditionals import spectral_repair_evidence


def test_spectral_repair_must_be_efficacy_matched_and_improve_stability() -> None:
    baseline = {
        "current_rewrite_exact": 0.90,
        "average_forgetting": 0.20,
        "past_retention": 0.60,
    }
    improved = {
        "current_rewrite_exact": 0.88,
        "average_forgetting": 0.10,
        "past_retention": 0.72,
    }
    assert spectral_repair_evidence(baseline, improved)["conditional_pass"]


def test_observed_c0_lowrank_direction_is_not_a_repair() -> None:
    baseline = {
        "current_rewrite_exact": 0.93,
        "average_forgetting": 0.14,
        "past_retention": 0.80,
    }
    lowrank = {
        "current_rewrite_exact": 0.89,
        "average_forgetting": 0.17,
        "past_retention": 0.7444444444444445,
    }
    result = spectral_repair_evidence(baseline, lowrank)
    assert not result["conditional_pass"]
    assert result["forgetting_delta_lowrank_minus_baseline"] > 0
    assert result["retention_delta_lowrank_minus_baseline"] < 0
