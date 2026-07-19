from scripts.run_next_direction_selection_campaign import baseline_floor


def test_baseline_floor_is_frozen():
    passing = {
        "rewrite_exact": 0.75,
        "declarative_paraphrase_exact": 0.40,
        "malformed_rate": 0.05,
        "base_summary": {"rewrite": {"target_new_rate": 0.10}},
    }
    assert baseline_floor(passing)
    failing = dict(passing, rewrite_exact=0.74)
    assert not baseline_floor(failing)
