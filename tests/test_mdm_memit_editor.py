from __future__ import annotations

import math
import random
from types import SimpleNamespace

import torch
from torch import nn

from scripts.mdm_memit_editor import (
    WeightRollback,
    base_target_confidence,
    denoise_answer_span,
    denoise_answer_spans_batch,
    exact_mask_pattern_bridge,
    find_last_subject_token,
    get_module,
    partial_mask_state,
    render_masked_input,
    request_lookup_index,
    solve_memit_update,
    sparse_support_kl,
)


def test_memit_transform_callback_contract_is_identity_safe(monkeypatch):
    import scripts.mdm_memit_editor as editor

    model = TinyModel()
    tokenizer = TinyTokenizer()
    request = {
        "case_id": "tiny-1",
        "subject": "Ada",
        "rewrite_prompt": "Ada works as",
        "target_new": "mathematician",
        "target_new_token_ids": [7],
    }
    target = torch.zeros(3)
    monkeypatch.setattr(editor, "optimize_target_value", lambda *_args, **_kwargs: (target, {}))
    monkeypatch.setattr(
        editor,
        "extract_keys_and_outputs",
        lambda *_args, **_kwargs: (torch.ones(1, 5), torch.zeros(1, 3)),
    )
    seen = {"key": 0, "update": 0}

    def key_transform(layer, keys, requests):
        seen["key"] += 1
        assert layer == 0 and requests[0]["case_id"] == "tiny-1"
        return keys, {"mode": "identity"}

    def update_transform(layer, update, context):
        seen["update"] += 1
        assert layer == 0 and context["keys"].shape == (1, 5)
        return update, {"mode": "identity"}

    config = editor.MemitConfig(layers=(0,), covariance_weight=1.0)
    rollback, report = editor.apply_memit_batch(
        model,
        tokenizer,
        [request],
        config,
        lambda _layer: torch.ones(5),
        key_transform=key_transform,
        update_transform=update_transform,
    )
    assert seen == {"key": 1, "update": 1}
    assert report["layer_updates"][0]["key_transform"]["mode"] == "identity"
    assert report["layer_updates"][0]["update_transform"]["mode"] == "identity"
    rollback.rollback()
    assert rollback.checksum_matches()


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

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


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


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(mask_token_id=99)

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        logits[..., 7] = 10.0 + self.anchor
        return SimpleNamespace(logits=logits)


def test_subject_locator_returns_last_subject_token():
    tokenizer = TinyTokenizer()
    assert find_last_subject_token(tokenizer, "Before Ada Lovelace wrote", "Ada Lovelace") == 2


def test_request_lookup_index_supports_prompt_local_fallback():
    tokenizer = TinyTokenizer()
    prompt = "Detroit City Hall, by"
    assert request_lookup_index(
        tokenizer, prompt, "Joe Louis Arena", lookup_mode="last_prompt_token"
    ) == len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) - 1


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


def test_base_target_confidence_returns_one_finite_score_per_target_token():
    scores = base_target_confidence(
        TinyDenoiser(), TinyTokenizer(), "Ada works as", [7, 8], 99
    )
    assert len(scores) == 2
    assert all(math.isfinite(score) for score in scores)
    assert scores[0] > scores[1]


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


def test_diagonal_covariance_woodbury_matches_dense_diagonal_solve():
    torch.manual_seed(3)
    keys = torch.randn(4, 7)
    residuals = torch.randn(4, 3)
    diagonal = torch.rand(7) + 0.5
    dense = solve_memit_update(
        keys, residuals, torch.diag(diagonal), covariance_weight=2.5
    )
    woodbury = solve_memit_update(
        keys, residuals, diagonal, covariance_weight=2.5
    )
    assert torch.allclose(woodbury, dense, atol=1e-5, rtol=1e-5)


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


def test_batched_denoising_matches_scalar_for_variable_prompt_and_span_lengths():
    model = TinyDenoiser()
    tokenizer = TinyTokenizer()
    prompts = ["Ada works as", "Grace Hopper worked as a"]
    lengths = [1, 2]
    scalar = [
        denoise_answer_span(model, tokenizer, prompt, length)
        for prompt, length in zip(prompts, lengths)
    ]
    batched = denoise_answer_spans_batch(
        model, tokenizer, prompts, lengths, batch_size=2
    )
    assert [row["output_token_ids"] for row in batched] == [
        row["output_token_ids"] for row in scalar
    ]
    assert [row["model_eval_count"] for row in batched] == [1, 2]
