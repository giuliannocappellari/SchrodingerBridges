from scripts.build_next_direction_protocol import (
    allocate_auxiliary_prompts,
    normalize_counterfact_candidates,
    overlap_audit,
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
