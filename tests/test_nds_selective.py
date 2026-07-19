from scripts.run_nds_selective_conformal import (
    accepted_metrics,
    isotonic_fit,
    isotonic_predict,
    per_case_outcomes,
)


def test_isotonic_calibration_is_monotone():
    blocks = isotonic_fit([0.1, 0.2, 0.3, 0.4], [False, True, False, True])
    risks = [block["risk"] for block in blocks]
    assert risks == sorted(risks)
    predictions = isotonic_predict(blocks, [0.1, 0.4])
    assert predictions[0] <= predictions[1]


def test_selective_outcomes_report_abstained_rows_in_denominator():
    rows = [
        {"case_id": "a", "bucket": "rewrite", "expected_hit": "True", "target_new_hit": "True", "malformed": "False"},
        {"case_id": "a", "bucket": "declarative_paraphrase", "expected_hit": "True", "target_new_hit": "True", "malformed": "False"},
        {"case_id": "a", "bucket": "same_subject", "expected_hit": "", "target_new_hit": "False", "malformed": "False"},
        {"case_id": "b", "bucket": "rewrite", "expected_hit": "False", "target_new_hit": "False", "malformed": "False"},
    ]
    outcomes = per_case_outcomes(rows)
    metrics = accepted_metrics(outcomes, {"a"})
    assert metrics["accepted"] == 1
    assert metrics["accepted_rewrite_exact"] == 1.0
    assert outcomes["b"]["unsafe"] is True
