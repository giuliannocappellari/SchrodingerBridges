from __future__ import annotations

import math

import pytest
import torch

from scripts.run_trm_temporal_localization import stratified_prefix
from scripts.trm_localization import (
    TraceCandidate,
    aggregate_coordinates,
    build_site_policy_rows,
    candidate_grid,
    causal_recovery_metrics,
    shortlist_candidates,
    stability_summary,
    target_support_metrics,
    temporal_state_specs,
)


def test_temporal_state_family_preserves_all_labels_for_short_spans() -> None:
    states = temporal_state_specs(2, seed=7, confidence_order=[1, 0])
    assert [row["state_label"] for row in states] == [
        "fully_masked",
        "early",
        "middle",
        "late",
        "actual_confidence_trajectory",
    ]
    assert all(row["masked_count"] >= 1 for row in states)
    trajectory = next(row for row in states if row["state_label"] == "actual_confidence_trajectory")
    assert trajectory["revealed_positions"] == [1]


def test_target_support_and_causal_recovery_are_finite() -> None:
    logits = torch.tensor([[[0.0, 3.0, 1.0], [2.0, 0.0, 1.0]]])
    metrics = target_support_metrics(logits, [0, 1], [1, 0], [0, 1])[0]
    assert metrics["decoded_support"] == 1.0
    restored = dict(metrics)
    corrupted = {"target_probability": 0.1, "target_margin": -1.0, "decoded_support": 0.0}
    recovery = causal_recovery_metrics(metrics, corrupted, restored)
    assert recovery["distance_recovery_target_margin"] == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in recovery.values())


def _trace_row(candidate: TraceCandidate, case: str, value: float, *, seed: int = 1) -> dict:
    return {
        "case_id": case,
        **candidate.to_dict(),
        "target_role": "target_new",
        "distance_recovery_target_margin": value,
        "corruption_effect_target_margin": 2.0,
        "restoration_delta_target_probability": value / 10,
        "restoration_delta_target_margin": value,
        "restoration_delta_decoded_support": 0.0,
        "prompt_type": "rewrite",
        "state_label": "fully_masked",
        "noise_seed": seed,
    }


def test_shortlist_preserves_fixed_and_data_driven_coordinates() -> None:
    candidates = candidate_grid(range(8), ("mlp", "hidden"), ("last_subject", "first_answer_mask"))
    rows = [_trace_row(candidate, "a", candidate.layer / 10) for candidate in candidates]
    aggregate = aggregate_coordinates(rows)
    shortlist = shortlist_candidates(aggregate, all_candidates=candidates, limit=16, seed=3)
    ids = {candidate.candidate_id for candidate in shortlist}
    assert {f"L{layer:02d}:mlp:last_subject" for layer in (3, 4, 5, 6)}.issubset(ids)
    assert len(shortlist) == 16


def test_stability_and_policy_builder_freeze_required_controls() -> None:
    candidates = [
        TraceCandidate(layer, component, position)
        for layer, component, position in (
            (3, "mlp", "last_subject"),
            (4, "mlp", "last_subject"),
            (5, "mlp", "last_subject"),
            (6, "mlp", "last_subject"),
            (20, "hidden", "first_answer_mask"),
        )
    ]
    rows = []
    for case in ("a", "b"):
        for seed in (1, 2):
            for candidate in candidates:
                rows.append(_trace_row(candidate, case, 0.5 + candidate.layer / 100, seed=seed))
    stability = stability_summary(rows)
    policies = build_site_policy_rows(stability, rows, num_layers=32, seed=4)
    assert {row["policy_id"] for row in policies} == {
        "source_paper_compatible_fixed_site",
        "per_edit_highest_tie_site",
        "stable_temporal_site_set",
        "last_subject_early_mid_mlp_site",
        "random_site",
        "late_answer_mask_site",
    }


def test_stratified_prefix_covers_available_target_lengths() -> None:
    rows = [
        {"case_id": "a", "target_length": 1},
        {"case_id": "b", "target_length": 1},
        {"case_id": "c", "target_length": 2},
        {"case_id": "d", "target_length": 3},
    ]
    selected = stratified_prefix(rows, 3)
    assert {row["target_length"] for row in selected} == {1, 2, 3}
