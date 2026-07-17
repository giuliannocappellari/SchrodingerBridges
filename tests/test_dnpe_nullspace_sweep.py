from __future__ import annotations

from scripts.run_dnpe_nullspace_sweep import RIDGES, VARIANCES, _name


def test_nullspace_grid_is_bounded_and_staged() -> None:
    assert VARIANCES == (0.90, 0.95, 0.99)
    assert RIDGES == (1e-4, 1e-3, 1e-2)
    assert _name("smoke", 0.95, 1e-3) == "smoke_variance_0.95_ridge_1e-03"
