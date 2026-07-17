from __future__ import annotations

from scripts.build_dnpe_state_banks import frozen_layers, prompt_hashes


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
