from __future__ import annotations

from scripts.run_dnpe_dev_selection import deduplicate_candidates


def test_dev_nominations_are_deduplicated_and_capped() -> None:
    rows = [
        {"path": "a", "nomination": "A"},
        {"path": "a", "nomination": "B"},
        {"path": "b", "nomination": "C"},
        {"path": "c", "nomination": "D"},
        {"path": "d", "nomination": "E"},
    ]
    result = deduplicate_candidates(rows)
    assert [row["path"] for row in result] == ["a", "b", "c"]
