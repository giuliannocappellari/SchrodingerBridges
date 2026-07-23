from __future__ import annotations

import pytest

from scripts import cl_common
from scripts.cl_common import assert_no_locked_path, sequential_metrics, success_classes


def test_sequential_metrics_forgetting_and_backward_transfer() -> None:
    scores = {
        0: {0: 0.9, 1: 0.1},
        1: {0: 0.8, 1: 0.85, 2: 0.1},
        2: {0: 0.7, 1: 0.75, 2: 0.8},
    }
    metrics = sequential_metrics(scores, pre_edit_scores={1: 0.05, 2: 0.05})
    assert metrics["average_retention"] == pytest.approx(0.75)
    assert metrics["average_forgetting"] == pytest.approx((0.2 + 0.1 + 0.0) / 3)
    assert metrics["backward_transfer"] == pytest.approx((-0.2 - 0.1) / 2)
    assert metrics["forward_transfer"] == pytest.approx(0.05)


def test_locked_paths_are_rejected() -> None:
    with pytest.raises(PermissionError):
        assert_no_locked_path("runs/old/analysis_500/results.json")
    with pytest.raises(PermissionError):
        assert_no_locked_path("runs/old/final_test_full/manifest.jsonl")


def test_success_class_a_uses_frozen_thresholds() -> None:
    row = {
        "current_rewrite_exact": 0.80,
        "current_paraphrase_exact": 0.45,
        "past_retention": 0.75,
        "average_forgetting": 0.10,
        "same_subject_tfpr": 0.03,
        "near_locality_pass": True,
        "far_locality_pass": True,
        "base_retention_loss_fraction": 0.05,
        "malformed_rate": 0.05,
        "storage_mb_per_edit": 2.0,
        "inference_overhead_fraction": 1.0,
    }
    assert "A" in success_classes(row, {})
    row["same_subject_tfpr"] = 0.031
    assert "A" not in success_classes(row, {})


def test_pareto_class_requires_positive_paired_lower_bound() -> None:
    baseline = {
        "current_rewrite_exact": 0.80,
        "average_forgetting": 0.20,
        "past_retention": 0.50,
        "same_subject_tfpr": 0.03,
    }
    candidate = {
        "current_rewrite_exact": 0.79,
        "average_forgetting": 0.10,
        "past_retention": 0.62,
        "same_subject_tfpr": 0.03,
    }
    assert "B" not in success_classes(candidate, baseline)
    candidate["paired_lower_bound_positive"] = True
    assert "B" in success_classes(candidate, baseline)


def test_successful_stage_rerun_clears_current_failure_marker(tmp_path, monkeypatch) -> None:
    state_root = tmp_path / "state"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(cl_common, "STATE_ROOT", state_root)
    monkeypatch.setattr(cl_common, "ROOT", tmp_path)
    monkeypatch.setattr(cl_common, "git_commit", lambda: "test-commit")

    cl_common.record_stage(
        "A0_source_audit",
        status="failed",
        acceptance_pass=False,
        output_dir=output_dir,
        started_at_utc="2026-01-01T00:00:00Z",
    )
    state = cl_common.record_stage(
        "A0_source_audit",
        status="passed",
        acceptance_pass=True,
        output_dir=output_dir,
        started_at_utc="2026-01-01T00:01:00Z",
    )

    assert "A0_source_audit" in state["completed_stages"]
    assert "A0_source_audit" not in state["failed_stages"]
