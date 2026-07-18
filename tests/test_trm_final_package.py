from pathlib import Path

import pytest

from scripts import finalize_trm_campaign as finalizer


def test_required_terminal_package_matches_frozen_plan():
    required = set(finalizer.REQUIRED_PACKAGE_FILES)
    assert {
        "main_results_table.csv",
        "multi_token_table.csv",
        "same_subject_stress_table.csv",
        "locality_table.csv",
        "causal_localization_table.csv",
        "state_bucket_ablation.csv",
        "relation_table.csv",
        "compute_storage_table.csv",
        "paired_bootstrap.csv",
        "rewrite_locality_pareto.png",
        "state_bucket_plot.png",
        "multi_token_plot.png",
        "causal_heatmap.png",
        "terminal_package_validation.json",
    } <= required


def test_auto_outcome_requires_failed_e2_without_positive_class(monkeypatch, tmp_path):
    campaign = tmp_path / "runs" / finalizer.CAMPAIGN_ID
    e2 = campaign / "E2_pilot100_v1"
    e2.mkdir(parents=True)
    finalizer.write_json(
        e2 / "report_summary.json",
        {
            "acceptance_pass": False,
            "positive_classes": {
                "full_editor": False,
                "pareto_locality": False,
                "diffusion_specific_partial_state": False,
                "state_conditioning": False,
            },
        },
    )
    monkeypatch.setattr(finalizer, "CAMPAIGN_ROOT", campaign)
    outcome, claim, reason = finalizer.determine_outcome("auto")
    assert outcome == "formal_negative"
    assert claim == "formal_bounded_negative"
    assert "pilot100" in reason


def test_auto_outcome_does_not_hide_integrity_failure(monkeypatch, tmp_path):
    campaign = tmp_path / "runs" / finalizer.CAMPAIGN_ID
    e2 = campaign / "E2_pilot100_v1"
    e2.mkdir(parents=True)
    finalizer.write_json(
        e2 / "report_summary.json",
        {"acceptance_pass": False, "positive_classes": {"full_editor": True}},
    )
    monkeypatch.setattr(finalizer, "CAMPAIGN_ROOT", campaign)
    outcome, claim, _reason = finalizer.determine_outcome("auto")
    assert outcome == "infrastructure_blocked"
    assert claim == "infrastructure_blocked"


def test_terminal_finalizer_requires_idle_pod(tmp_path):
    with pytest.raises(RuntimeError, match="idleness"):
        finalizer.build_package(
            tmp_path / "package",
            requested_outcome="formal_negative",
            pod_idle_verified=False,
        )


def test_terminal_status_set_is_bounded():
    assert "pending" not in finalizer.TERMINAL_STAGE_STATUSES
    assert "running" not in finalizer.TERMINAL_STAGE_STATUSES
    assert "not_run_due_formal_pilot_stop" in finalizer.TERMINAL_STAGE_STATUSES
