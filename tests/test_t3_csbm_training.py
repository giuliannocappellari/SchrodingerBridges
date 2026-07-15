from __future__ import annotations

import math

from scripts.train_t3_csbm import evaluate_predictions


def test_t3_identity_aware_metrics() -> None:
    rows = [
        {
            "identity": False,
            "prompt_type": "rewrite",
            "endpoint_token_ids": [2],
            "x0_token_ids": [1],
            "target_new_token_ids": [2],
        },
        {
            "identity": False,
            "prompt_type": "paraphrase",
            "endpoint_token_ids": [2],
            "x0_token_ids": [1],
            "target_new_token_ids": [2],
        },
        {
            "identity": True,
            "prompt_type": "same_subject_different_relation",
            "endpoint_token_ids": [1],
            "x0_token_ids": [1],
            "target_new_token_ids": [2],
        },
        {
            "identity": True,
            "prompt_type": "near_locality",
            "endpoint_token_ids": [1],
            "x0_token_ids": [1],
            "target_new_token_ids": [2],
        },
    ]
    predictions = [
        [{1: 0.05, 2: 0.95}],
        [{1: 0.10, 2: 0.90}],
        [{1: 0.99, 2: 0.01}],
        [{1: 0.98, 2: 0.02}],
    ]
    metrics = evaluate_predictions(rows, predictions)
    assert metrics["endpoint_accuracy"] == 1.0
    assert metrics["identity_sparse_kl"] < 0.05
    assert metrics["same_subject_target_advantage"] < 0.0
    assert math.isclose(metrics["span_exact"], 1.0)
