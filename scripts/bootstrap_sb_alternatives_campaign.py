#!/usr/bin/env python3
"""Initialize and validate the autonomous SB alternatives campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    STATE_ROOT,
    TRACKS,
    collect_locked_exclusions,
    git_commit,
    initialize_campaign_state,
    now_utc,
    repo_path,
    require_autonomous_environment,
    write_json,
)


AUTHORITATIVE_FILES = [
    "AGENTS.md",
    "ACTIVE_RESEARCH_CAMPAIGN.json",
    "ALTERNATIVE_PROTOCOL_REGISTRY.json",
    "SB_ALTERNATIVES_AUTONOMOUS_RESEARCH_PLAN.md",
    *[str(track["plan"]) for track in TRACKS],
]


def main() -> None:
    config = require_autonomous_environment()
    missing = [path for path in AUTHORITATIVE_FILES if not repo_path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative campaign files: {missing}")

    active = json.loads(repo_path("ACTIVE_RESEARCH_CAMPAIGN.json").read_text(encoding="utf-8"))
    registry = json.loads(repo_path("ALTERNATIVE_PROTOCOL_REGISTRY.json").read_text(encoding="utf-8"))
    if active.get("campaign_protocol") != CAMPAIGN_PROTOCOL:
        raise RuntimeError("ACTIVE_RESEARCH_CAMPAIGN.json does not select the alternatives campaign")
    if registry.get("campaign_protocol") != CAMPAIGN_PROTOCOL:
        raise RuntimeError("ALTERNATIVE_PROTOCOL_REGISTRY.json protocol mismatch")
    if active.get("analysis_500_locked") is not True or active.get("final_test_500_locked") is not True:
        raise RuntimeError("Locked analysis/final flags are not enabled")

    exclusions = collect_locked_exclusions()
    initialize_campaign_state(config)
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "stage": "Phase A campaign bootstrap",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "autonomous_mode": True,
        "campaign_configuration_pass": True,
        "analysis_500_used": False,
        "final_test_used": False,
        "authoritative_files": AUTHORITATIVE_FILES,
        "cost_tracking": {
            "policy": "informational_only_non_blocking",
            "hourly_rate_usd": config["hourly_rate_usd"],
            "budget_guard_enabled": False,
        },
        "locked_manifest_exclusion_audit": exclusions["manifests"],
        "locked_prompt_label_output_metric_fields_used": False,
        "artifacts": {
            "campaign_state": str(STATE_ROOT / "campaign_state.json"),
            "cost_state": str(STATE_ROOT / "cost_state.json"),
            "track_registry": str(STATE_ROOT / "track_registry.csv"),
            "stage_history": str(STATE_ROOT / "stage_history.csv"),
            "autonomous_log": str(STATE_ROOT / "autonomous_log.md"),
        },
    }
    write_json(STATE_ROOT / "bootstrap_report.json", report)
    print("campaign_configuration_pass=True")
    print("budget_guard_enabled=False")


if __name__ == "__main__":
    main()
