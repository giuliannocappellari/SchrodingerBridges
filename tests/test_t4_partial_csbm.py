from __future__ import annotations

from scripts.train_t4_partial_csbm import mass_metrics, mix_with_identity


def test_partial_mass_metrics_reward_identity_preservation() -> None:
    rows = [
        {"transport_label": 1, "prompt_type": "rewrite"},
        {"transport_label": 1, "prompt_type": "paraphrase"},
        {"transport_label": 0, "prompt_type": "same_subject_different_relation"},
        {"transport_label": 0, "prompt_type": "near_locality"},
    ]
    result = mass_metrics(rows, [0.9, 0.85, 0.01, 0.01])
    assert result["positive_mean_rho"] >= 0.85
    assert result["same_subject_mean_rho"] <= 0.05
    assert result["negative_mean_rho"] <= 0.05


def test_partial_mixture_returns_identity_at_zero_mass() -> None:
    rows = [
        {
            "x0_token_ids": [1],
            "target_new_token_ids": [2],
            "candidate_support_by_position": [[1, 2, 99]],
        }
    ]
    mixed = mix_with_identity(rows, [[{1: 0.1, 2: 0.8, 99: 0.1}]], [0.0])
    assert mixed == [[{1: 1.0, 2: 0.0, 99: 0.0}]]
