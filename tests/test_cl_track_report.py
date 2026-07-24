from __future__ import annotations

import pytest

from scripts.report_cl_track import paired_bootstrap_delta, track_mechanism_signal


def test_paired_bootstrap_is_edit_aligned_and_deterministic() -> None:
    baseline = {"a": 0.0, "b": 1.0, "c": 0.0}
    candidate = {"a": 1.0, "b": 1.0, "c": 1.0}
    first = paired_bootstrap_delta(baseline, candidate, trials=200, seed=7)
    second = paired_bootstrap_delta(baseline, candidate, trials=200, seed=7)
    assert first == second
    assert first["num_paired_edits"] == 3
    assert first["mean_delta"] == pytest.approx(2 / 3)


def test_c2_mechanism_signal_requires_matched_efficacy_and_positive_pairing() -> None:
    baseline = {
        "current_rewrite_exact": 0.80,
        "average_forgetting": 0.20,
        "past_retention": 0.50,
    }
    candidate = {
        "current_rewrite_exact": 0.79,
        "average_forgetting": 0.10,
        "past_retention": 0.62,
    }
    passed, _ = track_mechanism_signal("C2", candidate, baseline, {"ci_low": 0.01})
    assert passed
    passed, _ = track_mechanism_signal("C2", candidate, baseline, {"ci_low": 0.0})
    assert not passed
