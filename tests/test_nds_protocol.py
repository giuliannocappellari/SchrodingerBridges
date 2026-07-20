from scripts.build_next_direction_protocol import (
    allocate_auxiliary_prompts,
    candidate_has_disjoint_prompt_coverage,
    deduplicate_rewrite_prompts,
    normalize_counterfact_candidates,
    overlap_audit,
    prompt_fingerprint,
    select_counterfact,
)


def _row(index: int, relation: str = "P1", length: int = 1):
    return {
        "case_id": f"trm_cf_{index}",
        "source_index": index,
        "source_fingerprint": f"source-{index}",
        "fact_fingerprint": f"fact-{index}",
        "fact_target_fingerprint": f"target-{index}",
        "relation_id": relation,
        "subject": f"Subject {index}",
        "rewrite_prompt": f"Subject {index} relation",
        "rewrite_template": "{} relation",
        "target_new": "new",
        "target_true": "old",
        "target_new_token_ids": [10] * length,
        "target_true_token_ids": [11] * length,
        "target_length": length,
        "paraphrase_prompts": [f"Regarding Subject {index}, relation"],
        "near_locality_prompts": [f"near {index}"],
        "attribute_prompts": [f"attribute {index}"],
        "generation_prompts": [f"generation {index}"],
        "same_subject_prompt_candidates": [
            {"relation_id": "P2", "prompt": f"Subject {index} other relation"}
        ],
    }


def test_counterfact_normalization_filters_to_single_token():
    rows = normalize_counterfact_candidates([_row(1, length=1), _row(2, length=2)])
    assert [row["case_id"] for row in rows] == ["nds_cf_1"]
    assert rows[0]["protocol_version"] == "diffusion_editor_next_direction_selection_v1"


def test_rewrite_prompt_deduplication_is_deterministic():
    left = _row(1)
    duplicate = _row(2)
    duplicate["rewrite_prompt"] = left["rewrite_prompt"]
    distinct = _row(3)

    first, summary = deduplicate_rewrite_prompts(
        [left, duplicate, distinct], namespace="test"
    )
    second, second_summary = deduplicate_rewrite_prompts(
        [duplicate, distinct, left], namespace="test"
    )

    assert {row["case_id"] for row in first} == {row["case_id"] for row in second}
    assert summary == second_summary
    assert summary["kept_count"] == 2
    assert summary["duplicate_rows_dropped"] == 1
    assert summary["duplicate_prompt_groups"] == 1


def test_replacement_coverage_requires_every_real_prompt_family():
    row = _row(5)
    assert candidate_has_disjoint_prompt_coverage(row, set())

    near_fingerprint = prompt_fingerprint(row["near_locality_prompts"][0])
    assert not candidate_has_disjoint_prompt_coverage(row, {near_fingerprint})


def test_auxiliary_allocation_and_overlap_are_disjoint():
    left = _row(1)
    right = _row(2, relation="P2")
    donor = _row(3, relation="P3")
    donor_two = _row(4, relation="P4")
    splits = {"left": [left], "right": [right]}
    # Exercise allocation using the campaign's required role names.
    role_splits = {
        "cf_nds_statistics_train_500": [left],
        "cf_nds_calibration_200": [right],
        "cf_nds_smoke_20": [],
        "cf_nds_pilot_100": [],
        "cf_nds_confirmation_200": [],
    }
    allocate_auxiliary_prompts(
        role_splits,
        [left, right, donor, donor_two],
        {left["case_id"], right["case_id"]},
    )
    source, prompts = overlap_audit({"train": [left], "calibration": [right]})
    assert all(row["overlap_count"] == 0 for row in source)
    assert all(row["prompt_overlap_count"] == 0 for row in prompts)


def test_auxiliary_allocation_assigns_duplicate_paraphrase_once():
    train = _row(10)
    confirmation = _row(20, relation="P2")
    duplicate = "A shared real paraphrase"
    train["paraphrase_prompts"] = [duplicate]
    confirmation["paraphrase_prompts"] = [duplicate]
    donors = [_row(30, relation="P3"), _row(40, relation="P4")]
    role_splits = {
        "cf_nds_statistics_train_500": [train],
        "cf_nds_calibration_200": [],
        "cf_nds_smoke_20": [],
        "cf_nds_pilot_100": [],
        "cf_nds_confirmation_200": [confirmation],
    }

    allocation = allocate_auxiliary_prompts(
        role_splits,
        [train, confirmation, *donors],
        {train["case_id"], confirmation["case_id"]},
    )

    assert confirmation["paraphrase_prompts"] == [duplicate]
    assert train["paraphrase_prompts"] == []
    assert allocation["duplicate_or_rewrite_colliding_paraphrases_dropped"] == 1
    source, prompts = overlap_audit(
        {"train": [train], "confirmation": [confirmation]}
    )
    assert all(row["overlap_count"] == 0 for row in source)
    assert all(row["prompt_overlap_count"] == 0 for row in prompts)
