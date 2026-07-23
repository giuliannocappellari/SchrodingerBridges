from __future__ import annotations

import torch

from scripts.cl_delta_bank import DeltaBranchBank, _contains_subsequence


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 31 + 1 for char in str(text)]}


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value):
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

    def forward(self, input_ids):
        values = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        return self.model.layers[0].self_attn(values)


def test_subsequence_match_is_exact() -> None:
    assert _contains_subsequence([1, 2, 3, 4], [2, 3])
    assert not _contains_subsequence([1, 2, 3, 4], [2, 4])


def test_subject_route_uses_prompt_tokens_only() -> None:
    tokenizer = _Tokenizer()
    model = _Model()
    active_ids = torch.tensor([tokenizer("Ada")["input_ids"]])
    inactive_ids = torch.tensor([tokenizer("Bob")["input_ids"]])
    active_base = model(input_ids=active_ids).detach().clone()
    inactive_base = model(input_ids=inactive_ids).detach().clone()
    bank = DeltaBranchBank(model, tokenizer, [0], route_mode="subject")
    request = {"subject": "Ada", "rewrite_template": "{} works as", "rewrite_prompt": "Ada works as"}
    bank.add_branch({0: torch.eye(4)}, [request], block_index=0)

    active_output = model(input_ids=active_ids)
    active_gate = bank.current_gates.clone()
    inactive_output = model(input_ids=inactive_ids)

    assert not torch.equal(active_output, active_base)
    assert torch.equal(inactive_output, inactive_base)
    assert active_gate.item() == 1.0
    assert bank.current_gates.item() == 0.0
    assert bank.activation_summary()["forbidden_runtime_features_used"] is False
    bank.close()
