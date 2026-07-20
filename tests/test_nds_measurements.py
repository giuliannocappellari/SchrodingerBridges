import pytest

from scripts.build_nds_shared_measurements import (
    build_protected_requests,
    build_subject_anchor_requests,
    relation_template_bank,
    validate_training_manifest,
)


def _row(case_id, relation, template):
    subject = f"Subject {case_id}"
    return {
        "case_id": case_id,
        "subject": subject,
        "relation_id": relation,
        "rewrite_template": template,
        "rewrite_prompt": template.format(subject),
        "target_new": "new",
        "target_true": "old",
        "target_length": 1,
        "same_subject_prompts": [f"{subject} other"],
        "near_locality_prompts": [f"near {case_id}"],
        "far_locality_cases": [{"prompt": f"far {case_id}", "target": "far"}],
        "attribute_prompts": [f"attribute {case_id}"],
        "generation_prompts": [f"generation {case_id}"],
    }


def test_subject_anchors_use_training_relation_templates_not_eval_prompts():
    rows = [_row("a", "P1", "{} works as"), _row("b", "P2", "{} lives in")]
    anchors = build_subject_anchor_requests(rows, relation_template_bank(rows))
    assert all(row["evaluation_prompt_used"] is False for row in anchors)
    assert all(row["prompt_provenance"] == "training_relation_template_runtime_subject" for row in anchors)
    assert anchors[0]["relation_id"] != rows[0]["relation_id"]


def test_protected_families_are_explicit_and_training_only():
    row = _row("a", "P1", "{} works as")
    for family in ("same_subject", "near", "far", "unrelated"):
        request = build_protected_requests([row], family)[0]
        assert request["prompt_provenance"] == f"allowed_{family}_training_prompt"
        expected = (
            "last_subject"
            if row["subject"] in request["rewrite_prompt"]
            else "last_prompt_token"
        )
        assert request["lookup_mode"] == expected


def test_measurement_manifest_guard_rejects_locked_roles():
    validate_training_manifest(__import__("pathlib").Path("cf_nds_statistics_train_500.jsonl"))
    with pytest.raises(PermissionError):
        validate_training_manifest(__import__("pathlib").Path("cf_nds_confirmation_200.jsonl"))
