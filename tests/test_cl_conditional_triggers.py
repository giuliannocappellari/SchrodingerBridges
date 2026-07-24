from __future__ import annotations

from scripts.evaluate_cl_conditional_triggers import trigger_decisions


def test_duplicate_confirmed_mechanisms_do_not_trigger_integration(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.evaluate_cl_conditional_triggers.confirmed_equivalence_classes",
        lambda _rows: {"same_router"},
    )
    rows = [
        {"track_id": "C1", "confirmation_pass": "True"},
        {"track_id": "C3", "confirmation_pass": "True"},
        {"track_id": "C4", "confirmation_pass": "True"},
    ]
    decisions = {row["track_id"]: row for row in trigger_decisions(rows, [])}
    assert decisions["C10"]["triggered"]
    assert not decisions["C14"]["triggered"]


def test_two_distinct_confirmed_mechanisms_trigger_integration(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.evaluate_cl_conditional_triggers.confirmed_equivalence_classes",
        lambda _rows: {"growth", "replay"},
    )
    rows = [
        {"track_id": "C1", "confirmation_pass": "True"},
        {"track_id": "C2", "confirmation_pass": "True"},
    ]
    decisions = {row["track_id"]: row for row in trigger_decisions(rows, [])}
    assert decisions["C14"]["triggered"]


def test_spectral_trigger_requires_strong_acquisition_and_forgetting(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.evaluate_cl_conditional_triggers.confirmed_equivalence_classes",
        lambda _rows: set(),
    )
    pilot = [
        {
            "track_id": "C5",
            "current_rewrite_exact": "0.85",
            "average_forgetting": "0.12",
            "mechanism_signal_pass": "False",
        }
    ]
    decisions = {row["track_id"]: row for row in trigger_decisions([], pilot)}
    assert decisions["C12"]["triggered"]
    assert not decisions["C11"]["triggered"]
