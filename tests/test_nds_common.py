import json
from pathlib import Path

import pytest

from scripts import nds_common as common


def test_initial_state_has_all_mandatory_tracks(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setenv("NEXT_DIRECTION_AUTONOMOUS_MODE", "1")
    state = common.initialize_state()
    assert state["autonomous_mode"] is True
    assert all(state["track_status"][track] == "pending" for track in common.MANDATORY_TRACKS)
    assert state["track_status"]["N6"] == "conditional_pending"
    assert state["analysis_500_used"] is False
    assert state["final_test_used"] is False


def test_historical_exclusion_uses_identity_fields_only(monkeypatch, tmp_path):
    root = tmp_path / "runs" / "historical"
    protocol = root / "protocol" / "analysis_500"
    protocol.mkdir(parents=True)
    path = protocol / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "source_split": "train",
                "source_index": 7,
                "source_fingerprint": "source-fp",
                "prompt": "forbidden prompt content",
                "target_new": "forbidden label",
                "metric": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "HISTORICAL_CAMPAIGNS", ("historical",))
    exclusions = common.collect_historical_exclusions()
    assert exclusions["case_ids"] == ["case-1"]
    assert exclusions["source_keys"] == ["train:7"]
    assert exclusions["source_fingerprints"] == ["source-fp"]
    assert exclusions["prompt_fingerprints"] == []
    assert exclusions["historical_locked_content_fields_used"] is False
    assert exclusions["audit"][0]["locked_manifest"] is True


def test_runtime_guard_rejects_historical_locked_path(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "HISTORICAL_CAMPAIGNS", ("historical",))
    path = tmp_path / "runs" / "historical" / "analysis_500" / "rows.jsonl"
    with pytest.raises(PermissionError):
        common.forbid_historical_locked_content(path)


def test_track_registry_preserves_rich_pilot_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "STATE_ROOT", tmp_path / "state")
    output = tmp_path / "pilot"
    output.mkdir()
    common.write_json(output / "report_summary.json", {"acceptance_pass": True})
    common.initialize_state()
    common.update_track(
        "N1",
        status="pilot_passed",
        candidate_id="relation_full",
        mechanism_pass=True,
        pilot_pass=True,
        success_class="C",
        output_dir=output,
    )
    state = common.read_json(common.STATE_ROOT / "campaign_state.json")
    assert state["track_details"]["N1"]["candidate_id"] == "relation_full"
    registry = common.read_json(common.STATE_ROOT / "track_registry.json")
    row = next(item for item in registry["tracks"] if item["track_id"] == "N1")
    assert row["nominated_candidate"] == "relation_full"
    assert row["pilot_pass"] is True
