from __future__ import annotations

from scripts.build_dnpe_state_banks import (
    frozen_layers,
    normalized_prompt_hash,
    prompt_hashes,
    select_state_bank_inputs,
)


def test_state_bank_prompt_hashes_normalize_text() -> None:
    left = [{"rewrite_prompt": "Ada   works as"}]
    right = [{"rewrite_prompt": "  ada works AS  "}]
    assert prompt_hashes(left) == prompt_hashes(right)


def test_frozen_layers_are_union_of_precommitted_policies() -> None:
    lock = {
        "policies": [
            {"policy_id": "a", "layers": [6, 4]},
            {"policy_id": "b", "layers": [4, 8]},
        ]
    }
    assert frozen_layers(lock) == [4, 6, 8]


def test_state_bank_selection_excludes_only_actual_prompt_collisions() -> None:
    train = [
        {
            "case_id": "train-1",
            "split_role": "dnpe_anchor_train_500",
            "subject": "Ada",
            "rewrite_prompt": "Ada works as",
            "same_subject_prompts": ["Ada was born in"],
            "near_locality_prompts": ["Grace works as", "Grace was born in"],
            "attribute_prompts": ["Ada is known for"],
            "generation_prompts": ["Write about Ada"],
        },
        {
            "case_id": "train-2",
            "split_role": "dnpe_anchor_train_500",
            "subject": "Linus",
            "rewrite_prompt": "Linus works as",
            "same_subject_prompts": ["Linus was born in"],
            "near_locality_prompts": ["Ken works as", "Ken was born in"],
            "attribute_prompts": ["Linus is known for"],
            "generation_prompts": ["Write about Linus"],
        },
    ]
    evaluation = [{"rewrite_prompt": "Ada works as"}]
    positive, preservation, audit = select_state_bank_inputs(
        train,
        evaluation,
        maximum_positive_keys=1,
        maximum_preservation_keys=20,
    )
    assert [row["case_id"] for row in positive] == ["train-2"]
    assert audit["source_manifest_prompt_overlap_count"] == 1
    assert audit["state_bank_prompt_overlap_count"] == 0
    evaluation_hashes = prompt_hashes(evaluation)
    selected_hashes = {
        normalized_prompt_hash(row["prompt"]) for row in preservation
    }
    assert not selected_hashes & evaluation_hashes
