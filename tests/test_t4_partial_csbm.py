from __future__ import annotations

from scripts.train_t4_partial_csbm import metrics


def test_partial_mass_metrics_reward_identity_preservation() -> None:
    rows = [
        {"transport_label": 1, "prompt_type": "rewrite"},
        {"transport_label": 1, "prompt_type": "paraphrase"},
        {"transport_label": 0, "prompt_type": "same_subject_different_relation"},
        {"transport_label": 0, "prompt_type": "near_locality"},
    ]
    result = metrics(rows, [0.9, 0.85, 0.01, 0.01])
    assert result["positive_mean_rho"] >= 0.85
    assert result["same_subject_mean_rho"] <= 0.05
    assert result["identity_sparse_kl"] <= 0.05
