from __future__ import annotations

import math
import random

import torch
from torch import nn

from scripts.mdm_memit_editor import (
    WeightRollback,
    exact_mask_pattern_bridge,
    find_last_subject_token,
    get_module,
    partial_mask_state,
    render_masked_input,
    solve_memit_update,
    sparse_support_kl,
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        tokens = str(text).split()
        ids = [index + 10 for index in range(len(tokens))]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for token in tokens:
                start = str(text).find(token, cursor)
                offsets.append((start, start + len(token)))
                cursor = start + len(token)
            result["offset_mapping"] = offsets
        return result


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_out = nn.Linear(5, 3, bias=False)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])


class TinyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TinyTransformer()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyInner()


def test_subject_locator_returns_last_subject_token():
    tokenizer = TinyTokenizer()
    assert find_last_subject_token(tokenizer, "Before Ada Lovelace wrote", "Ada Lovelace") == 2


def test_mask_count_equals_target_length_and_context_is_preserved():
    tokenizer = TinyTokenizer()
    rendered = render_masked_input(tokenizer, "Ada works as", [40, 41], 99)
    assert rendered["input_ids"][-2:] == [99, 99]
    assert len(rendered["answer_positions"]) == 2
    assert rendered["supervised_positions"] == rendered["answer_positions"]


def test_partial_mask_cycle_and_n1_reduction():
    rng = random.Random(7)
    target = [10, 11, 12, 13]
    revealed_counts = []
    supervised_union = set()
    for step in range(4):
        state, supervised, revealed = partial_mask_state(
            target,
            step=step,
            mask_id=99,
            schedule="cycle",
            reveal_policy="random",
            rng=rng,
        )
        revealed_counts.append(len(revealed))
        supervised_union.update(supervised)
        assert all(state[index] == target[index] for index in revealed)
        assert all(state[index] == 99 for index in supervised)
    assert revealed_counts == [0, 1, 2, 3]
    assert supervised_union == {0, 1, 2, 3}
    assert partial_mask_state(
        [10], step=8, mask_id=99, schedule="cycle", reveal_policy="random", rng=rng
    ) == ([99], [0], [])


def test_random_partial_mask_is_seed_reproducible():
    left = partial_mask_state(
        [1, 2, 3, 4],
        step=3,
        mask_id=9,
        schedule="cycle",
        reveal_policy="random",
        rng=random.Random(55),
    )
    right = partial_mask_state(
        [1, 2, 3, 4],
        step=3,
        mask_id=9,
        schedule="cycle",
        reveal_policy="random",
        rng=random.Random(55),
    )
    assert left == right


def test_closed_form_update_has_expected_shape_and_reduces_residual():
    torch.manual_seed(0)
    keys = torch.randn(4, 5)
    residuals = torch.randn(4, 3)
    covariance = torch.eye(5)
    update = solve_memit_update(keys, residuals, covariance, covariance_weight=0.1)
    assert update.shape == (3, 5)
    before = residuals.norm()
    after = (residuals - (update @ keys.T).T).norm()
    assert after < before


def test_weight_rollback_restores_exactly():
    model = TinyModel()
    original = get_module(model, "model.transformer.blocks.0.ff_out").weight.detach().clone()
    rollback = WeightRollback(model, [0])
    rollback.apply(0, torch.ones_like(original))
    assert not torch.equal(get_module(model, "model.transformer.blocks.0.ff_out").weight, original)
    rollback.rollback()
    assert rollback.checksum_matches()
    assert torch.equal(get_module(model, "model.transformer.blocks.0.ff_out").weight, original)


def test_sparse_kl_zero_for_equal_logits():
    logits = torch.tensor([0.5, 1.0, -2.0, 3.0])
    assert float(sparse_support_kl(logits, logits, top_k=2).abs()) < 1e-7


def test_exact_mask_pattern_dp_normalizes_and_beta_zero_matches_reference():
    n = 3
    terminal = (1 << n) - 1
    costs = {
        (mask, index): float(index + 1) if mask == 0 else 0.0
        for mask in range(terminal)
        for index in range(n)
        if not mask & (1 << index)
    }
    beta_zero = exact_mask_pattern_bridge(costs, n, beta=0.0)
    assert all(math.isclose(sum(row.values()), 1.0) for row in beta_zero.values())
    assert beta_zero[0] == {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    controlled = exact_mask_pattern_bridge(costs, n, beta=2.0)
    assert controlled[0][0] > controlled[0][2]
