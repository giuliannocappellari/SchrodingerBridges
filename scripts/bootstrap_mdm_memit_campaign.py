#!/usr/bin/env python3
"""Initialize the autonomous masked-diffusion MEMIT campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ID,
    STATE_ROOT,
    collect_historical_exclusions,
    git_commit,
    initialize_campaign,
    now_utc,
    record_stage,
    repo_path,
    require_autonomous_environment,
    write_csv,
    write_json,
)


AUTHORITIES = (
    "AGENTS.md",
    "ACTIVE_RESEARCH_CAMPAIGN.json",
    "EXPERIMENT_PROTOCOL_REGISTRY.json",
    "MDM_MEMIT_SB_AUTONOMOUS_RESEARCH_PLAN.md",
    "MEMIT_REPRODUCTION_PLAN.md",
    "PARTIAL_MASK_MEMIT_PLAN.md",
    "SB_REGULARIZED_MEMIT_PLAN.md",
    "MASK_PATTERN_SB_PLAN.md",
    "ADAPTIVE_EDIT_MEMORY_FALLBACK_PLAN.md",
    "TOY_TEXT_CSBM_FALLBACK_PLAN.md",
)


def main() -> None:
    started = now_utc()
    config = require_autonomous_environment()
    missing = [path for path in AUTHORITIES if not repo_path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing campaign authorities: {missing}")

    active = json.loads(repo_path("ACTIVE_RESEARCH_CAMPAIGN.json").read_text(encoding="utf-8"))
    registry = json.loads(repo_path("EXPERIMENT_PROTOCOL_REGISTRY.json").read_text(encoding="utf-8"))
    if active.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("ACTIVE_RESEARCH_CAMPAIGN.json selects another campaign")
    if registry.get("registry_version") != CAMPAIGN_ID:
        raise RuntimeError("EXPERIMENT_PROTOCOL_REGISTRY.json protocol mismatch")
    if active.get("old_analysis_500_locked") is not True:
        raise RuntimeError("Historical analysis lock is not enabled")
    if active.get("old_final_test_500_locked") is not True:
        raise RuntimeError("Historical final lock is not enabled")

    exclusions = collect_historical_exclusions()
    state = initialize_campaign(config)
    audit_rows = exclusions["audit"]
    write_csv(STATE_ROOT / "historical_exclusion_file_audit.csv", audit_rows)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "A1_campaign_bootstrap",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "autonomous_mode": True,
        "campaign_configuration_pass": True,
        "pod_running": True,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "monetary_budget_guard_enabled": False,
        "cost_tracking_informational_only": True,
        "historical_protocols_immutable": True,
        "authoritative_files": list(AUTHORITIES),
        "historical_exclusion_case_id_count": len(exclusions["case_ids"]),
        "historical_exclusion_source_key_count": len(exclusions["source_keys"]),
        "historical_exclusion_files_inspected": len(audit_rows),
        "locked_prompt_label_output_metric_fields_used": False,
        "pod": {
            "pod_id": config["pod_id"],
            "ssh_host": config["ssh_host"],
            "ssh_port": config["ssh_port"],
            "remote_repo_dir": config["remote_repo_dir"],
        },
        "state": state,
    }
    write_json(STATE_ROOT / "bootstrap_report.json", report)
    record_stage(
        stage="A1_campaign_bootstrap",
        status="passed",
        output_dir=STATE_ROOT,
        acceptance_pass=True,
        started_at_utc=started,
        notes="Autonomous state initialized; historical data projected to exclusion fields only.",
    )
    print("campaign_configuration_pass=True")
    print(f"historical_exclusion_case_id_count={len(exclusions['case_ids'])}")


if __name__ == "__main__":
    main()
