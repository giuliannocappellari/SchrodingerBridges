from __future__ import annotations

from scripts.build_dnpe_preservation_basis import build_prompt_specs, stratified_limit


def _row(index: int) -> dict:
    return {
        "case_id": f"c{index}",
        "split_role": "dnpe_anchor_train_500",
        "subject": f"Subject {index}",
        "rewrite_prompt": f"Subject {index} relation",
        "same_subject_prompts": [f"Subject {index} other relation"],
        "near_locality_prompts": [f"Near {index} one", f"Near {index} two"],
        "attribute_prompts": [f"Subject {index} attribute"],
        "generation_prompts": [f"Subject {index} generation"],
    }


def test_preservation_prompt_specs_cover_required_categories() -> None:
    specs = build_prompt_specs([_row(index) for index in range(10)])
    selected = stratified_limit(specs, 70)
    categories = {row["category"] for row in selected}
    assert categories == {
        "same_subject_different_relation",
        "different_subject_same_relation",
        "near_locality",
        "far_locality",
        "attribute",
        "generation",
        "unrelated",
    }
    assert all(row["source_manifest_role"] == "dnpe_anchor_train_500" for row in selected)
