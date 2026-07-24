from __future__ import annotations

from scripts.finalize_cl_campaign import REQUIRED_FILES, select_recommendation


def test_terminal_package_has_every_frozen_required_file() -> None:
    assert len(REQUIRED_FILES) == 16
    assert "terminal_package_validation.json" in REQUIRED_FILES
    assert "SELECTED_CONTINUAL_DIRECTION_FULL_CAMPAIGN_DRAFT.md" in REQUIRED_FILES


def test_selection_deduplicates_equivalent_confirmed_methods() -> None:
    rows = [
        {
            "track_id": "C1",
            "method": "growth",
            "implementation_equivalence_class": "same_router",
            "success_classes": "A",
            "past_retention": 0.80,
            "average_forgetting": 0.05,
            "same_subject_tfpr": 0.02,
            "current_rewrite_exact": 0.85,
        },
        {
            "track_id": "C3",
            "method": "sparse_alias",
            "implementation_equivalence_class": "same_router",
            "success_classes": "A",
            "past_retention": 0.79,
            "average_forgetting": 0.05,
            "same_subject_tfpr": 0.02,
            "current_rewrite_exact": 0.86,
        },
    ]
    recommendation, selected, claim = select_recommendation(rows, mechanism_signal_count=0)
    assert recommendation == "pursue_diffusiongrow_continual_editor"
    assert selected["track_id"] == "C1"
    assert claim == "full_continual_editor"


def test_no_confirmation_returns_mechanism_or_negative() -> None:
    assert select_recommendation([], mechanism_signal_count=1)[0] == "mechanism_only_result"
    assert (
        select_recommendation([], mechanism_signal_count=0)[0]
        == "no_promising_continual_direction"
    )
