from __future__ import annotations

from scripts.run_dnpe_target_value_sweep import config_id, objective_configs, staged_configs


def test_target_value_grid_is_staged_and_bounded() -> None:
    first = staged_configs()
    assert len(first) == 4
    assert {row["learning_rate"] for row in first} == {0.05, 0.10}
    assert {row["target_optimization_steps"] for row in first} == {25, 50}
    second = objective_configs(first[0])
    assert len({config_id(row) for row in first + second}) <= 7
    assert {row["state_consistency_weight"] for row in second} == {0.0, 0.1}
    assert {row["old_target_suppression_weight"] for row in second} == {0.0, 0.25}
