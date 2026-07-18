from __future__ import annotations

from scripts.run_trm_e2_pilot import deduplicate_candidates, method_is_deployable


def _row(method: str) -> dict:
    return {
        "method": method,
        "all_metrics_finite": True,
        "runtime_schema_present": True,
        "malformed_rate": 0.0,
        "gpu_minutes_per_edit": 0.2,
        "utility_base_agreement": 0.9,
        "analysis_500_used": False,
        "final_test_used": False,
    }


def test_method_deployability_enforces_hard_limits() -> None:
    row = _row("a")
    assert method_is_deployable(row)
    row["malformed_rate"] = 0.051
    assert not method_is_deployable(row)


def test_candidate_deduplication_is_ordered_and_bounded() -> None:
    rows = [_row("a"), _row("a"), _row("b"), _row("c"), _row("d")]
    assert [row["method"] for row in deduplicate_candidates(rows)] == ["a", "b", "c"]
