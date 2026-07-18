from __future__ import annotations

import torch

from scripts.build_trm_covariance import (
    finalize_diagonal_covariance,
    training_prompt_texts,
)


def test_finalize_diagonal_covariance_is_finite_positive() -> None:
    value = finalize_diagonal_covariance(torch.tensor([2.0, 0.0, 6.0]), 2)
    assert torch.isfinite(value).all()
    assert bool((value > 0).all())
    assert value[0] == 1.0
    assert value[2] == 3.0


def test_covariance_prompts_are_train_fields_only() -> None:
    rows = [
        {
            "rewrite_prompt": "A is",
            "same_subject_prompts": ["A was", "held out"],
            "generation_prompts": ["A writes", "held out"],
            "attribute_prompts": ["A has", "held out"],
        }
    ]
    assert training_prompt_texts(rows) == ["A is", "A was", "A writes", "A has"]
