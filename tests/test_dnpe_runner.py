from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.run_dnpe_editor import _forbid_locked_manifest, aggregate


def test_runner_rejects_locked_manifests_without_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_METHOD_LOCKED", raising=False)
    monkeypatch.delenv("FINAL_METHOD_LOCKED", raising=False)
    monkeypatch.delenv("DNPE_KAMEL_LOCKED", raising=False)
    with pytest.raises(PermissionError):
        _forbid_locked_manifest(Path("analysis_500.jsonl"))
    with pytest.raises(PermissionError):
        _forbid_locked_manifest(Path("final_test_500.jsonl"))
    with pytest.raises(PermissionError):
        _forbid_locked_manifest(Path("dnpe_kamel_locked_200_n2.jsonl"))


def test_aggregate_reports_tfpr_and_base_agreement() -> None:
    rows = [
        {
            "bucket": "same_subject",
            "case_id": "a",
            "expected_hit": None,
            "target_new_hit": True,
            "target_true_hit": False,
            "target_token_f1": 1.0,
            "malformed": False,
            "base_agreement": False,
            "model_eval_count": 2,
        },
        {
            "bucket": "same_subject",
            "case_id": "b",
            "expected_hit": None,
            "target_new_hit": False,
            "target_true_hit": True,
            "target_token_f1": 0.0,
            "malformed": False,
            "base_agreement": True,
            "model_eval_count": 2,
        },
    ]
    summary = aggregate(rows)["same_subject"]
    assert summary["target_new_tfpr_or_exact"] == 0.5
    assert summary["base_agreement"] == 0.5
