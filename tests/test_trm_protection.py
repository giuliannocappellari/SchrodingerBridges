from __future__ import annotations

import torch

from scripts.trm_editor import build_input_protection_basis, fit_factorized_residual_memory
from scripts.trm_protection import (
    REQUIRED_PROTECTION_FAMILIES,
    build_protection_prompt_records,
    state_revealed_count,
)


def _anchor(index: int, relation: str) -> dict:
    return {
        "case_id": f"a{index}",
        "split_role": "cf_trm_anchor_train_500",
        "relation_id": relation,
        "subject": f"Subject {index}",
        "rewrite_prompt": f"Subject {index} works in",
        "same_subject_prompts": [f"Subject {index} was born in"],
        "generation_prompts": [f"Subject {index} is known for"],
        "attribute_prompts": [f"Attribute subject {index} works in"],
    }


def test_protection_prompt_builder_covers_every_required_family() -> None:
    rows = [_anchor(index, f"r{index % 2}") for index in range(8)]
    records, summary = build_protection_prompt_records(rows, max_per_family=4)
    assert summary["all_required_families_present"]
    assert set(record["family"] for record in records) == set(REQUIRED_PROTECTION_FAMILIES)
    assert all(record["synthetic"] is False for record in records)


def test_state_revealed_counts_cover_early_middle_late() -> None:
    assert state_revealed_count("early", 4) == 0
    assert state_revealed_count("middle", 4) == 2
    assert state_revealed_count("late", 4) == 3


def test_soft_preservation_reduces_protected_prediction() -> None:
    torch.manual_seed(5)
    edit_keys = torch.randn(8, 6)
    residuals = torch.randn(8, 4)
    protected = torch.randn(20, 6)
    plain = fit_factorized_residual_memory(edit_keys, residuals, ridge=0.1)
    soft = fit_factorized_residual_memory(
        edit_keys,
        residuals,
        ridge=0.1,
        protect_keys=protected,
        preservation_strength=10.0,
    )
    assert soft.predict(protected).norm() < plain.predict(protected).norm()
    assert soft.protect_row_count == len(protected)


def test_soft_preservation_matches_augmented_kernel_solution() -> None:
    torch.manual_seed(17)
    edit_keys = torch.randn(5, 4)
    residuals = torch.randn(5, 3)
    protected = torch.randn(7, 4)
    ridge = 0.2
    strength = 2.5
    memory = fit_factorized_residual_memory(
        edit_keys,
        residuals,
        ridge=ridge,
        protect_keys=protected,
        preservation_strength=strength,
    )
    augmented_keys = torch.cat((edit_keys, protected * strength**0.5), dim=0)
    augmented_residuals = torch.cat((residuals, torch.zeros(7, 3)), dim=0)
    expected_dual = torch.linalg.solve(
        augmented_keys @ augmented_keys.T
        + torch.eye(len(augmented_keys)) * ridge,
        augmented_keys,
    )
    probes = torch.randn(6, 4)
    expected = (probes @ expected_dual.T) @ augmented_residuals
    assert torch.allclose(memory.predict(probes), expected, atol=1e-5, rtol=1e-5)


def test_static_input_basis_removes_protected_directions() -> None:
    torch.manual_seed(9)
    protected = torch.randn(30, 8)
    basis, report = build_input_protection_basis(
        protected, explained_variance=0.8, maximum_rank=4
    )
    memory = fit_factorized_residual_memory(
        torch.randn(6, 8),
        torch.randn(6, 3),
        ridge=0.1,
        input_projection_basis=basis,
    )
    assert basis.shape == (8, report["protected_rank"])
    assert memory.input_projection_basis is not None
    assert torch.isfinite(memory.predict(torch.randn(2, 8))).all()
