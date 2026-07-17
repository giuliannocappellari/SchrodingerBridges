from __future__ import annotations

from scripts.materialize_dnpe_locked_manifest import adapt_row, text_value


def test_locked_row_adapter_preserves_targets_and_prompts() -> None:
    row = {
        "case_id": "c1",
        "source_dataset_split": "train",
        "source_index": 1,
        "relation_id": "P1",
        "subject": "Ada",
        "rewrite_template": "{} works as",
        "prompt": "Ada works as",
        "target_new": {"text": " engineer", "context_token_ids": [4, 5]},
        "target_true": {"text": " writer", "context_token_ids": [6]},
        "target_new_token_len": 2,
        "declarative_paraphrase_prompts": ["Ada's job is"],
        "near_locality_cases": [{"prompt": "Bob works as", "target": " writer"}],
        "far_locality_cases": [{"prompt": "Paris is in", "target": " France"}],
        "attribute_prompts": ["Ada was born in"],
        "generation_prompts": ["Ada is known for"],
        "protocol_version": "old",
    }
    adapted = adapt_row(row, "dnpe_analysis_500")
    assert adapted["target_new"] == " engineer"
    assert adapted["target_new_token_ids"] == [4, 5]
    assert adapted["same_subject_prompts"] == ["Ada was born in"]
    assert adapted["near_locality_cases"][0]["target"] == " writer"
    assert adapted["far_locality_cases"][0]["target"] == " France"
    assert text_value({"text": "x"}) == "x"
