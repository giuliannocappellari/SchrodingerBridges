from __future__ import annotations

from scripts.run_cl_baseline_suite import BASELINE_METHODS, REQUIRED_PLAN_METHODS, run_specs


def test_c0_suite_covers_every_required_plan_baseline(tmp_path) -> None:
    assert REQUIRED_PLAN_METHODS <= set(BASELINE_METHODS)
    specs = run_specs(tmp_path)
    assert len(specs) == 2 * len(BASELINE_METHODS)
    assert {(row["method"], row["scale"]) for row in specs} == {
        (method, scale)
        for method in BASELINE_METHODS
        for scale in ("smoke20", "pilot100")
    }


def test_c0_suite_never_uses_locked_split_names(tmp_path) -> None:
    text = " ".join(str(row["manifest"]) for row in run_specs(tmp_path)).casefold()
    assert "analysis_500" not in text
    assert "final_test" not in text
