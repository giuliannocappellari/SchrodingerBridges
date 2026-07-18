from __future__ import annotations

from scripts.run_trm_relation_conditioned_rescue import (
    relation_groups,
    relation_protection_records,
)
from scripts.trm_protection import REQUIRED_PROTECTION_FAMILIES


def test_relation_groups_are_exact_and_deterministic() -> None:
    rows = [
        {"case_id": "a", "relation_id": "P2"},
        {"case_id": "b", "relation_id": "P1"},
        {"case_id": "c", "relation_id": "P2"},
    ]
    grouped = relation_groups(rows)
    assert list(grouped) == ["P1", "P2"]
    assert [row["case_id"] for row in grouped["P2"]] == ["a", "c"]


def test_relation_protection_uses_training_relation_then_backoff() -> None:
    records = []
    for family in REQUIRED_PROTECTION_FAMILIES:
        for index, relation in enumerate(("P1", "P1", "P2", "P3")):
            records.append(
                {
                    "anchor_id": f"{family}-{index}",
                    "family": family,
                    "relation_id": relation,
                }
            )
    selected, report = relation_protection_records(
        records, "P1", maximum_per_family=2, minimum_per_family=3
    )
    assert report["all_families_present"]
    assert report["exact_relation_counts"] == {
        family: 2 for family in REQUIRED_PROTECTION_FAMILIES
    }
    assert report["fallback_counts"] == {
        family: 1 for family in REQUIRED_PROTECTION_FAMILIES
    }
    assert len(selected) == 3 * len(REQUIRED_PROTECTION_FAMILIES)
