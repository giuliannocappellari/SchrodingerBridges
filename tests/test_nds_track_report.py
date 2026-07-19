from scripts.report_nds_counterfact_track import case_metric


def test_case_metric_bootstrap_unit_is_edit_id():
    rows = [
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": "True"},
        {"case_id": "a", "bucket": "same_subject", "target_new_hit": "False"},
        {"case_id": "b", "bucket": "same_subject", "target_new_hit": "False"},
    ]
    assert case_metric(rows, {"same_subject"}, "target_new_hit") == {"a": 0.5, "b": 0.0}
    assert case_metric(rows, {"same_subject"}, "target_new_hit", invert=True) == {
        "a": 0.5,
        "b": 1.0,
    }
