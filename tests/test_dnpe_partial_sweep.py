from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_dnpe_partial_state_sweep import POLICIES, should_run
from scripts.run_dnpe_nullspace_sweep import should_run as nullspace_should_run


def test_partial_state_policy_grid_is_exactly_predeclared() -> None:
    assert set(POLICIES) == {
        "fully_masked_only",
        "all_mask_counts_random_positions",
        "uniform_mask_count_states",
        "confidence_trajectory_states",
    }
    assert POLICIES["all_mask_counts_random_positions"][:2] == ("cycle", "random")
    assert POLICIES["confidence_trajectory_states"][:2] == ("cycle", "base_confidence")


def test_resume_skips_only_complete_outputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert should_run(missing, resume=True)

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "report_summary.json").write_text("{}", encoding="utf-8")
    assert not should_run(complete, resume=True)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileExistsError):
        should_run(incomplete, resume=True)
    assert not nullspace_should_run(complete, resume=True)
    with pytest.raises(FileExistsError):
        nullspace_should_run(incomplete, resume=True)
