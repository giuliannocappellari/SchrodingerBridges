from __future__ import annotations

import math

import pytest
import torch

from scripts.dnpe_editor import (
    explicit_partial_state,
    normalized_aie,
    protected_key_drift,
    state_bank,
)
from scripts.mdm_memit_editor import build_protected_basis, project_update_to_nullspace


def test_state_bank_has_every_mask_count() -> None:
    bank = state_bank([10, 11, 12, 13], policy="all_mask_counts_random_positions", seed=7)
    assert [row["revealed_count"] for row in bank] == [0, 1, 2, 3]
    assert [row["masked_count"] for row in bank] == [4, 3, 2, 1]


def test_explicit_partial_state_preserves_revealed_tokens() -> None:
    state, supervised = explicit_partial_state([10, 11, 12], [1], 99)
    assert state == [99, 11, 99]
    assert supervised == [0, 2]


def test_normalized_aie_is_finite_and_has_expected_scale() -> None:
    assert normalized_aie(0.8, 0.2, 0.5) == pytest.approx(0.5)
    with pytest.raises(FloatingPointError):
        normalized_aie(float("nan"), 0.2, 0.5)


def test_nullspace_projection_removes_protected_energy() -> None:
    protected = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    basis, report = build_protected_basis(protected, 0.9)
    update = torch.tensor([[2.0, 3.0, 4.0], [-1.0, 5.0, 2.0]])
    projected, geometry = project_update_to_nullspace(update, basis)
    assert report["protected_rank"] == 1
    assert geometry["protected_energy_after"] < 1e-5
    assert protected_key_drift(projected, protected) < 1e-5
    assert torch.isfinite(projected).all()


def test_projector_rejects_invalid_variance() -> None:
    with pytest.raises(ValueError):
        build_protected_basis(torch.randn(3, 4), 0.0)
