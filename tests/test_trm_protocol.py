from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_trm_protocol import (
    _overlap_audit,
    _sanitize_prompt_fields,
    prompt_fingerprint,
    target_length_bin,
)
from scripts.trm_common import is_forbidden_historical_locked_path


def test_target_length_bins_are_exact_until_four_plus() -> None:
    assert [target_length_bin(value) for value in (1, 2, 3, 4, 6)] == ["1", "2", "3", ">=4", ">=4"]


def test_prompt_fingerprint_normalizes_case_and_whitespace() -> None:
    assert prompt_fingerprint("  Ada   works AS ") == prompt_fingerprint("ada works as")


def test_overlap_audit_rejects_reused_source_rows() -> None:
    with pytest.raises(RuntimeError, match="Fresh protocol overlap"):
        _overlap_audit(
            {
                "a": [{"source_fingerprint": "same"}],
                "b": [{"source_fingerprint": "same"}],
            }
        )


def test_historical_locked_names_are_skipped_without_opening() -> None:
    assert is_forbidden_historical_locked_path(Path("runs/old/protocol/analysis_500.jsonl"))
    assert is_forbidden_historical_locked_path(Path("runs/old/final_test_full.jsonl"))
    assert not is_forbidden_historical_locked_path(Path("runs/old/protocol/dev_200.jsonl"))


def test_auxiliary_prompts_are_allocated_to_only_one_role() -> None:
    roles = (
        "cf_trm_locked_500",
        "cf_trm_dev_200",
        "cf_trm_pilot_100",
        "cf_trm_smoke_20",
        "cf_trm_scaling_100",
        "cf_trm_localize_50",
        "cf_trm_anchor_train_500",
    )
    splits = {}
    for index, role in enumerate(roles):
        splits[role] = [
            {
                "case_id": role,
                "rewrite_prompt": f"subject {index} relation",
                "paraphrase_prompts": [f"paraphrase {index}"],
                "same_subject_prompt_candidates": [
                    {"relation_id": f"P{candidate}", "prompt": f"subject {index} other {candidate}"}
                    for candidate in range(8)
                ],
                "near_locality_prompts": ["shared auxiliary", f"near {index}"],
                "attribute_prompts": ["shared auxiliary", f"attribute {index}"],
                "generation_prompts": ["shared generation", f"generation {index}"],
            }
        ]
    _sanitize_prompt_fields(splits)
    fingerprints = []
    for rows in splits.values():
        row = rows[0]
        for field in ("rewrite_prompt",):
            fingerprints.append(prompt_fingerprint(row[field]))
        for field in ("paraphrase_prompts", "same_subject_prompts", "near_locality_prompts", "attribute_prompts", "generation_prompts"):
            fingerprints.extend(prompt_fingerprint(value) for value in row[field])
    assert len(fingerprints) == len(set(fingerprints))
