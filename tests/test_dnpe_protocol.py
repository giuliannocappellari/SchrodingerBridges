from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import dnpe_common
from scripts.build_dnpe_protocol import _overlap_audit


def test_dnpe_identity_matches_active_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    active = json.loads((root / "ACTIVE_RESEARCH_CAMPAIGN.json").read_text())
    registry = json.loads((root / "EXPERIMENT_PROTOCOL_REGISTRY.json").read_text())
    assert active["active_protocol"] == dnpe_common.CAMPAIGN_ID
    assert registry["protocol_version"] == dnpe_common.CAMPAIGN_ID


def test_overlap_audit_rejects_duplicate_source_fingerprint() -> None:
    with pytest.raises(RuntimeError, match="Protocol overlap"):
        _overlap_audit(
            {
                "left": [{"source_fingerprint": "same"}],
                "right": [{"source_fingerprint": "same"}],
            }
        )


def test_overlap_audit_accepts_disjoint_rows() -> None:
    audit = _overlap_audit(
        {
            "left": [{"source_fingerprint": "a"}],
            "right": [{"source_fingerprint": "b"}],
        }
    )
    assert audit == [{"left": "left", "right": "right", "overlap_count": 0}]


def test_locked_names_never_count_as_tuning_roles() -> None:
    assert set(dnpe_common.LOCKED_SPLIT_NAMES) == {
        "analysis_500",
        "final_test_500",
        "final_test_full",
    }
    assert all("dev" not in name for name in dnpe_common.LOCKED_SPLIT_NAMES)
