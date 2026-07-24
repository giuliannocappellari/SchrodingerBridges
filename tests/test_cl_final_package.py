from __future__ import annotations

from scripts.finalize_cl_campaign import (
    REQUIRED_FILES,
    candidate_result_paths,
    paired_bootstrap_paths,
    select_recommendation,
)


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


def test_terminal_evidence_paths_include_bounded_rescues(tmp_path) -> None:
    pilot_root = tmp_path / "pilots"
    rescue_root = tmp_path / "rescues"
    for index in range(1, 10):
        report = pilot_root / "track_reports" / f"C{index}_pilot_v1"
        report.mkdir(parents=True)
        (report / "candidate_results.csv").write_text("track_id\n", encoding="utf-8")
        (report / "paired_bootstrap.csv").write_text("track_id\n", encoding="utf-8")
    rescue = rescue_root / "track_reports" / "C5_rescue_v1"
    rescue.mkdir(parents=True)
    rescue_candidate = rescue / "candidate_results.csv"
    rescue_bootstrap = rescue / "paired_bootstrap.csv"
    rescue_candidate.write_text("track_id\n", encoding="utf-8")
    rescue_bootstrap.write_text("track_id\n", encoding="utf-8")

    candidate_paths = candidate_result_paths(pilot_root, rescue_root)
    bootstrap_paths = paired_bootstrap_paths(pilot_root, rescue_root)

    assert len(candidate_paths) == 10
    assert len(bootstrap_paths) == 10
    assert rescue_candidate in candidate_paths
    assert rescue_bootstrap in bootstrap_paths
