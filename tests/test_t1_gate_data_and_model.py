from __future__ import annotations

from scripts.build_t1_gate_data import gate_rows_for_split
from scripts.t1_gate_model import FEATURE_DIM, FORBIDDEN_RUNTIME_FIELDS, featurize
from scripts.train_t1_gate import hard_gate_pass, pr_auc, rank_auc, select_threshold


def sample_edit(index: int, relation: str) -> dict:
    return {
        "case_id": f"case-{index}",
        "split_role": "train",
        "subject": f"Subject {index}",
        "relation_id": relation,
        "rewrite_template": "{} works in",
        "rewrite_prompt": f"Subject {index} works in",
        "target_new": "new",
        "target_true": "old",
        "paraphrase_prompts": [f"The field of Subject {index} is"],
        "near_locality_prompts": [f"Neighbor {index} works in"],
        "generation_prompts": [f"Subject {index} is known for"],
        "attribute_prompts": [f"Attribute {index} works in"],
    }


def test_gate_data_has_required_positive_and_negative_categories() -> None:
    rows = gate_rows_for_split([sample_edit(0, "P1"), sample_edit(1, "P2")], "train")
    types = {row["prompt_type"] for row in rows}
    assert {"rewrite", "declarative_paraphrase"}.issubset(types)
    assert {
        "same_subject_different_relation",
        "near_locality",
        "far_locality",
        "generation",
        "attribute",
        "unrelated",
    }.issubset(types)
    same_subject = next(row for row in rows if row["prompt_type"] == "same_subject_different_relation")
    assert same_subject["synthetic_from_metadata"] is True
    assert "Subject" in same_subject["prompt"]


def test_gate_features_are_fixed_and_runtime_deployable() -> None:
    features = featurize("Ada works in mathematics", "Ada", "{} works in", "P101")
    assert features.shape == (FEATURE_DIM,)
    assert features.isfinite().all()
    runtime_inputs = {"prompt", "subject", "relation_template", "relation_id"}
    assert not runtime_inputs & FORBIDDEN_RUNTIME_FIELDS


def test_t1_metrics_and_threshold_selection() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert rank_auc(labels, scores) == 1.0
    assert pr_auc(labels, scores) == 1.0
    passing = {
        "threshold": 0.5,
        "roc_auc": 0.95,
        "pr_auc": 0.92,
        "relation_shuffle_auc_drop": 0.07,
        "rewrite_activation": 0.95,
        "declarative_paraphrase_activation": 0.90,
        "same_subject_different_relation_activation": 0.03,
        "near_locality_activation": 0.01,
        "far_locality_activation": 0.0,
    }
    passing["hard_acceptance_pass"] = hard_gate_pass(passing)
    assert passing["hard_acceptance_pass"]
    assert select_threshold([passing])["threshold"] == 0.5
