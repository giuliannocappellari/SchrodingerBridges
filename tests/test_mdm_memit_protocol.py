from __future__ import annotations

from scripts.build_mdm_memit_protocol import (
    _paraphrase_template,
    overlap_audit,
    round_robin_stratified,
)


def make_rows(count: int):
    return [
        {
            "case_id": f"case_{index}",
            "relation_id": f"P{index % 3}",
            "target_length_bin": str(1 + index % 2),
        }
        for index in range(count)
    ]


def test_stratified_selection_is_deterministic_and_disjoint():
    rows = make_rows(30)
    used_a: set[str] = set()
    first_a = round_robin_stratified(
        rows, 10, seed=9, used=used_a, group_fields=("target_length_bin", "relation_id")
    )
    second_a = round_robin_stratified(
        rows, 10, seed=10, used=used_a, group_fields=("target_length_bin", "relation_id")
    )
    used_b: set[str] = set()
    first_b = round_robin_stratified(
        rows, 10, seed=9, used=used_b, group_fields=("target_length_bin", "relation_id")
    )
    assert [row["case_id"] for row in first_a] == [row["case_id"] for row in first_b]
    assert {row["case_id"] for row in first_a}.isdisjoint(
        {row["case_id"] for row in second_a}
    )


def test_overlap_audit_detects_cross_split_reuse():
    rows = make_rows(3)
    audit = overlap_audit({"left": rows[:2], "right": rows[1:]})
    assert audit == [{"left": "left", "right": "right", "overlap_count": 1}]


def test_kamel_paraphrase_is_evaluation_only_rewrite_not_same_template():
    source = "What country is [S] located in?"
    paraphrase = _paraphrase_template(source)
    assert "[S]" in paraphrase
    assert paraphrase != source
    assert paraphrase.startswith("Regarding [S],")

