from __future__ import annotations

from scripts.run_dnpe_partial_state_sweep import POLICIES


def test_partial_state_policy_grid_is_exactly_predeclared() -> None:
    assert set(POLICIES) == {
        "fully_masked_only",
        "all_mask_counts_random_positions",
        "uniform_mask_count_states",
        "confidence_trajectory_states",
    }
    assert POLICIES["all_mask_counts_random_positions"][:2] == ("cycle", "random")
    assert POLICIES["confidence_trajectory_states"][:2] == ("cycle", "base_confidence")
