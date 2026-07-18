from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_trm_protocol import _overlap_audit, prompt_fingerprint, target_length_bin
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
