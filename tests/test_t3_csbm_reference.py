from __future__ import annotations

import math

from scripts.t3_csbm_reference import (
    next_step_distribution,
    reciprocal_bridge_distribution,
    seeded_sample,
)


def test_reciprocal_reference_is_normalized_and_endpoint_consistent() -> None:
    start = reciprocal_bridge_distribution(
        x0=1, xT=2, mask_id=99, support=[1, 2, 3], time=0.0, epsilon=0.01
    )
    end = reciprocal_bridge_distribution(
        x0=1, xT=2, mask_id=99, support=[1, 2, 3], time=1.0, epsilon=0.01
    )
    assert start[1] == 1.0
    assert end[2] > 0.98
    assert math.isclose(sum(end.values()), 1.0)
    assert all(math.isfinite(value) and value >= 0 for value in end.values())


def test_reference_next_step_and_sampling_are_deterministic() -> None:
    distribution = next_step_distribution(
        x_t=99,
        x0=1,
        xT=2,
        mask_id=99,
        support=[1, 2, 3],
        time=0.25,
        next_time=0.5,
        epsilon=0.05,
    )
    assert math.isclose(sum(distribution.values()), 1.0)
    assert seeded_sample(distribution, 7) == seeded_sample(distribution, 7)
