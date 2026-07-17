from __future__ import annotations

from scripts.run_dnpe_scaling import COUNTS


def test_scaling_counts_are_exactly_predeclared() -> None:
    assert COUNTS == (1, 10, 50, 100)
