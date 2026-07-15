from __future__ import annotations

import torch

from scripts.build_t2_activation_endpoints import endpoint_specs, fake_tensors, validate


def edit(index: int, relation: str) -> dict:
    return {
        "case_id": f"case-{index}",
        "subject": f"Subject {index}",
        "relation_id": relation,
        "rewrite_template": "{} works in",
        "rewrite_prompt": f"Subject {index} works in",
        "target_new": "math",
        "target_true": "physics",
        "paraphrase_prompts": [f"The field of Subject {index} is"],
        "near_locality_prompts": [f"Neighbor {index} works in"],
        "generation_prompts": [f"Subject {index} is known for"],
        "attribute_prompts": [f"Attribute {index} is"],
    }


def test_t2_fake_endpoint_cache_has_identity_and_trajectory_shape() -> None:
    rows = [edit(index, f"P{index}") for index in range(24)]
    specs = endpoint_specs(rows, "train")
    tensors = fake_tensors(specs)
    summary = validate(specs, tensors)
    assert summary["all_vectors_finite"]
    assert summary["positive_nonidentical_rate"] == 1.0
    assert summary["identity_max_delta_norm"] == 0.0
    assert set(tensors) == {
        "h0_middle",
        "h1_middle",
        "h0_final",
        "h1_final",
        "base_target_logit",
        "endpoint_target_logit",
    }
    assert all(torch.isfinite(value).all() for value in tensors.values())
    assert {
        "same_subject_different_relation",
        "near_locality",
        "far_locality",
        "generation",
        "attribute",
        "unrelated",
    }.issubset(summary["prompt_type_histogram"])
