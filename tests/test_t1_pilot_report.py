from __future__ import annotations

from scripts.report_t1_pilot import aggregate, classify


def row(method: str, rewrite: float, paraphrase: float, stress: float) -> dict:
    return {
        "method": method,
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "near_tfpr": 0.0,
        "far_tfpr": 0.0,
        "same_subject_tfpr": stress,
        "malformed_rate": 0.0,
    }


def test_t1_green_decision_requires_efficacy_and_safety() -> None:
    rows = classify(
        [
            row("base", 0.05, 0.05, 0.0),
            row("myopic_score", 0.50, 0.40, 0.30),
            row("learned_gate_myopic", 0.30, 0.20, 0.01),
        ]
    )
    learned = next(item for item in rows if item["method"] == "learned_gate_myopic")
    assert learned["pilot_color"] == "green"
    assert learned["common_hard_constraints_pass"] is True


def test_t1_red_decision_when_same_subject_leaks() -> None:
    rows = classify(
        [
            row("base", 0.05, 0.05, 0.0),
            row("mc_bridge", 0.50, 0.40, 0.30),
            row("learned_gate_mc_bridge", 0.30, 0.20, 0.15),
        ]
    )
    learned = next(item for item in rows if item["method"] == "learned_gate_mc_bridge")
    assert learned["pilot_color"] == "red"
    assert learned["common_hard_constraints_pass"] is False


def test_t1_malformed_rate_is_worst_bucket_mean_not_row_maximum() -> None:
    rows = [
        {
            "method": "base",
            "edit_id": f"case_{index}",
            "split_role": "smoke20",
            "bucket": "rewrite",
            "exact_rate": 0.0,
            "target_false_positive_rate": 0.0,
            "malformed_rate": 1.0 if index == 0 else 0.0,
            "sparse_guidance_kl": 0.0,
            "gate_activation_rate": 0.0,
        }
        for index in range(20)
    ]
    result = aggregate(rows, runtime_seconds=60.0)[0]
    assert result["malformed_rate"] == 0.05
