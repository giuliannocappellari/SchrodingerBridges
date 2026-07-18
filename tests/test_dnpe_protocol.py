from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import dnpe_common
from scripts.build_dnpe_protocol import CF_COUNTS, _overlap_audit


def test_dnpe_identity_remains_registered_as_closed_history() -> None:
    root = Path(__file__).resolve().parents[1]
    active = json.loads((root / "ACTIVE_RESEARCH_CAMPAIGN.json").read_text())
    registry = json.loads((root / "EXPERIMENT_PROTOCOL_REGISTRY.json").read_text())
    assert active["active_protocol"] == "partial_state_temporal_residual_editor_v1"
    assert active["historical_protocols"][dnpe_common.CAMPAIGN_ID] == "closed_bounded_negative"
    assert registry["protocol_version"] == active["active_protocol"]


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


def test_protocol_reserves_disjoint_far_locality_pool() -> None:
    assert CF_COUNTS["dnpe_locality_eval_300"] == 300
    assert CF_COUNTS["dnpe_anchor_train_500"] == 500
