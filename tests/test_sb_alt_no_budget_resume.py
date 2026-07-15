from __future__ import annotations

from scripts.resume_sb_alt_no_budget import migrate_registry_rows, migrate_state_payload


def test_resume_preserves_t1_and_reopens_only_budget_skipped_tracks() -> None:
    state = {
        "campaign_protocol": "counterfact_sb_alternatives_campaign_v1",
        "campaign_status": "budget_completion",
        "analysis_500_used": False,
        "final_test_used": False,
        "completed_tracks": ["T1", "T2", "T3", "T4", "T5"],
        "failed_tracks": ["T1"],
        "passed_tracks": [],
        "terminal_reason": "old budget stop",
        "terminal_at_utc": "then",
    }
    migrated = migrate_state_payload(state, "new-pod", "/workspace/SB")
    assert migrated["campaign_status"] == "running"
    assert migrated["completed_tracks"] == ["T1"]
    assert migrated["current_track"] == "T2"
    assert migrated["current_stage"] == "T2.1_activation_endpoint_collection"
    assert migrated["previous_budget_stop_superseded"] is True
    assert "terminal_reason" not in migrated


def test_registry_reopens_budget_not_run_without_resetting_t1() -> None:
    rows = [
        {"track_id": "T1", "status": "formal_negative", "evidence_path": "t1", "rescue_used": "False"},
        *[
            {"track_id": track_id, "status": "budget_not_run", "evidence_path": "old", "rescue_used": "False"}
            for track_id in ("T2", "T3", "T4", "T5")
        ],
    ]
    migrated = migrate_registry_rows(rows)
    assert migrated[0]["status"] == "formal_negative"
    assert [row["status"] for row in migrated[1:]] == ["pending"] * 4
    assert all(not row["evidence_path"] for row in migrated[1:])
