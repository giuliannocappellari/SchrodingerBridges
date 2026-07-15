from __future__ import annotations

from scripts.finalize_sb_alt_budget_completion import budget_completion_required


def budget(remaining: float) -> dict:
    return {
        "remaining_budget_usd": remaining,
        "reserve_usd": 5.0,
        "pilot_estimates": {"T1": 0.75, "T2": 1.5, "T3": 1.5, "T4": 1.25, "T5": 3.0},
    }


def test_budget_completion_requires_all_untested_pilots_plus_reserve() -> None:
    result = budget_completion_required(budget(10.47), ["T2", "T3", "T4", "T5"])
    assert result["required_available_usd"] == 12.25
    assert result["shortfall_usd"] == 1.78
    assert result["budget_completion_required"] is True


def test_budget_completion_not_allowed_when_reserve_is_covered() -> None:
    result = budget_completion_required(budget(12.25), ["T2", "T3", "T4", "T5"])
    assert result["budget_completion_required"] is False
