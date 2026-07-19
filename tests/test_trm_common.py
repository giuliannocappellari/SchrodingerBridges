from __future__ import annotations

import json
from pathlib import Path

from scripts import nds_common, trm_common


def test_trm_identity_remains_registered_as_closed_history() -> None:
    root = Path(__file__).resolve().parents[1]
    active = json.loads((root / "ACTIVE_RESEARCH_CAMPAIGN.json").read_text())
    registry = json.loads((root / "EXPERIMENT_PROTOCOL_REGISTRY.json").read_text())
    assert active["active_protocol"] == nds_common.CAMPAIGN_ID
    assert active["historical_protocols_immutable"] is True
    assert registry["protocol_version"] == trm_common.CAMPAIGN_ID
    assert trm_common.CAMPAIGN_ID in nds_common.HISTORICAL_CAMPAIGNS
    assert "diffusion_native_causal_partial_state_editor_v1" in trm_common.HISTORICAL_CAMPAIGNS


def test_trm_locked_splits_are_never_stage_roles() -> None:
    assert set(trm_common.LOCKED_SPLIT_NAMES) == {
        "analysis_500",
        "final_test_500",
        "final_test_full",
    }
    assert not any(name in stage for name in trm_common.LOCKED_SPLIT_NAMES for stage in trm_common.STAGES)


def test_trm_autonomy_requires_protocol_specific_flag(monkeypatch) -> None:
    monkeypatch.delenv("PS_TRM_AUTONOMOUS_MODE", raising=False)
    monkeypatch.setenv("SB_ALT_AUTONOMOUS_MODE", "1")
    assert not trm_common.autonomous_enabled()
    monkeypatch.setenv("PS_TRM_AUTONOMOUS_MODE", "1")
    assert trm_common.autonomous_enabled()
