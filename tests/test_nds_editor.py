import torch

from scripts.nds_editor import (
    diagonal_fisher_update,
    fit_relation_key_statistics,
    fixed_penalty_update,
    low_rank_fisher_update,
    primal_dual_update,
    residualize_runtime_keys,
)
from scripts.nds_methods import fisher_low_rank, protected_response


def test_runtime_relation_residualization_uses_frozen_statistics():
    train = torch.tensor([[2.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
    stats = fit_relation_key_statistics(train, ["P1", "P1", "P2", "P2"])
    keys = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    anchors = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    residual, report = residualize_runtime_keys(
        keys, ["P1", "P2"], stats, subject_anchor_keys=anchors, mode="full"
    )
    assert residual.shape == keys.shape
    assert report["evaluation_prompt_used"] is False
    assert report["runtime_features"] == ["subject", "relation_id", "rewrite_prompt"]


def test_fisher_transforms_are_finite_and_preserve_gain_direction():
    torch.manual_seed(3)
    update = torch.randn(4, 6)
    keys = torch.randn(5, 6)
    residuals = torch.randn(5, 4)
    diagonal = torch.rand(6) + 0.2
    transformed, report = diagonal_fisher_update(update, diagonal, keys, residuals)
    assert torch.isfinite(transformed).all()
    assert report["linearized_gain_before"] * report["linearized_gain_after"] >= 0
    low_rank = fisher_low_rank(torch.randn(20, 6), rank=3, damping=1e-2)
    transformed, report = low_rank_fisher_update(
        update,
        low_rank["basis"],
        low_rank["eigenvalues"],
        low_rank["damping"],
        keys,
        residuals,
    )
    assert torch.isfinite(transformed).all()
    assert report["rank"] == 3


def test_primal_dual_reduces_protected_response_beyond_initial_value():
    torch.manual_seed(7)
    update = torch.randn(3, 5)
    families = {"same_subject": torch.randn(6, 5), "near": torch.randn(5, 5)}
    limits = {
        name: protected_response(update, keys) * 0.8 for name, keys in families.items()
    }
    fixed, fixed_report = fixed_penalty_update(update, families, strength=0.05)
    candidate, report = primal_dual_update(
        update,
        families,
        limits,
        multiplier_step=0.05,
        penalty_growth=1.5,
        iterations=30,
    )
    assert torch.isfinite(fixed).all() and fixed_report["finite"]
    assert torch.isfinite(candidate).all() and report["finite"]
    assert max(report["final_responses"].values()) < max(
        protected_response(update, keys) for keys in families.values()
    )
