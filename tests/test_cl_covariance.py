from __future__ import annotations

from pathlib import Path

from scripts.build_cl_covariance import ROOT, resolve_output_dir


def test_relative_covariance_output_is_rooted_in_repository() -> None:
    relative = Path("runs/example_covariance")
    assert resolve_output_dir(relative) == (ROOT / relative).resolve()


def test_absolute_covariance_output_is_preserved() -> None:
    absolute = (ROOT / "runs/example_covariance").resolve()
    assert resolve_output_dir(absolute) == absolute
