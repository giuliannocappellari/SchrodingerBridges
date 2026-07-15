from __future__ import annotations

from scripts.train_t3_csbm import evaluate_scores


def test_t3_identity_aware_metrics() -> None:
    rows = [
        {"transport_label": 1, "prompt_type": "rewrite"},
        {"transport_label": 1, "prompt_type": "paraphrase"},
        {"transport_label": 0, "prompt_type": "same_subject_different_relation"},
        {"transport_label": 0, "prompt_type": "near_locality"},
    ]
    metrics = evaluate_scores(rows, [0.95, 0.90, 0.01, 0.02])
    assert metrics["endpoint_accuracy"] == 1.0
    assert metrics["identity_sparse_kl"] < 0.05
    assert metrics["same_subject_target_advantage"] < 0.0
