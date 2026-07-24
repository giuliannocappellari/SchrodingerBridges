from __future__ import annotations

from pathlib import Path

import torch

from scripts.cl_lora import LoRABranch
from scripts.run_cl_sequential_editor import (
    METHOD_EQUIVALENCE_CLASS,
    OEDIT_INITIAL_BASIS_RANK,
    OEDIT_RESCUE_BASIS_RANKS,
    RELATION_GATE_INITIAL_THRESHOLD,
    RELATION_GATE_RESCUE_THRESHOLDS,
    _retention_mask_states,
    rank_truncate,
    select_covariance_representation,
)


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.o_proj(value)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()


def test_lora_branch_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(1)
    model = _Model()
    value = torch.randn(2, 3, 4)
    expected = model.model.layers[0].self_attn(value).detach()
    branch = LoRABranch(model, [0], rank=2)
    actual = model.model.layers[0].self_attn(value)
    assert torch.equal(expected, actual)
    with torch.no_grad():
        branch.parameters_by_layer[0][1].fill_(0.1)
    changed = model.model.layers[0].self_attn(value)
    assert not torch.equal(expected, changed)
    branch.close()


def test_rank_truncate_obeys_rank_and_is_finite() -> None:
    torch.manual_seed(2)
    update = torch.randn(8, 6)
    truncated, report = rank_truncate(update, 2)
    assert torch.linalg.matrix_rank(truncated, atol=1e-5) <= 2
    assert torch.isfinite(truncated).all()
    assert report["rank"] == 2
    assert report["decomposition"] == "exact_svd"
    assert 0.0 < report["explained_update_energy"] <= 1.0


def test_editor_source_freezes_python_torch_and_cuda_seeds() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_cl_sequential_editor.py"
    ).read_text(encoding="utf-8")
    assert "random.seed(args.seed)" in source
    assert "torch.manual_seed(args.seed)" in source
    assert "torch.cuda.manual_seed_all(args.seed)" in source
    assert '"seed": args.seed' in source


def test_conceptual_clone_methods_have_explicit_equivalence_class() -> None:
    expected = "rank8_block_delta_subject_relation_router"
    assert METHOD_EQUIVALENCE_CLASS["growth_block_gate"] == expected
    assert METHOD_EQUIVALENCE_CLASS["sparse_routed_memory"] == expected
    assert METHOD_EQUIVALENCE_CLASS["gated_adapter_expansion"] == expected


def test_relation_gate_rescue_grid_is_frozen_before_pilots() -> None:
    assert RELATION_GATE_INITIAL_THRESHOLD == 0.20
    assert RELATION_GATE_RESCUE_THRESHOLDS == (0.35, 0.50)
    assert OEDIT_INITIAL_BASIS_RANK == 64
    assert OEDIT_RESCUE_BASIS_RANKS == (16, 32)
    assert (
        METHOD_EQUIVALENCE_CLASS["gated_adapter_shared_basis"]
        == "rank8_merged_delta_subject_relation_router"
    )


def test_diagonal_covariance_representation_is_explicit() -> None:
    covariance = torch.tensor([[2.0, 0.5], [0.5, 3.0]])
    diagonal = select_covariance_representation(covariance, "diagonal")
    assert torch.equal(diagonal, torch.tensor([2.0, 3.0]))
    assert torch.equal(select_covariance_representation(covariance, "full"), covariance)


def test_retention_mask_states_are_deterministic_and_cover_ratios() -> None:
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(1, len(str(text).split()) + 1))}

    rows = [{"case_id": "case-1", "rewrite_prompt": "one two three", "target_true": "four"}]
    first = _retention_mask_states(Tokenizer(), rows, 99)
    second = _retention_mask_states(Tokenizer(), rows, 99)
    assert first == second
    assert {row["mask_ratio"] for row in first} == {0.25, 0.5, 1.0}
    assert all(row["masked_positions"] for row in first)
    assert all(99 in row["input_ids"] for row in first)
