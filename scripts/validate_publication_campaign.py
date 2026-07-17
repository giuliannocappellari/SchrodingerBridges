#!/usr/bin/env python3
"""Validate that every mandatory publication track and terminal artifact exists."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_publication_package import REQUIRED_OUTPUTS
from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    STATE_ROOT,
    now_utc,
    read_json,
    sha256_file,
    write_json,
)


REPORTS = {
    "P0_source_audit": CAMPAIGN_ROOT / "source_audit" / "report_summary.json",
    "P1_partial_state": CAMPAIGN_ROOT / "partial_state_memit_audit_v1" / "report_summary.json",
    "P2_theory": CAMPAIGN_ROOT / "theory_and_naming_v1" / "report_summary.json",
    "P3_planners": CAMPAIGN_ROOT / "planner_baselines_dev_v1" / "report_summary.json",
    "P4_llada_locked": CAMPAIGN_ROOT / "llada_locked_confirmation_v1" / "report_summary.json",
    "P5_dream_locked": CAMPAIGN_ROOT / "dream_confirmation_v1" / "report_summary.json",
    "P6_editor_generality": CAMPAIGN_ROOT / "editor_generality_v1" / "report_summary.json",
    "P7_approximation": CAMPAIGN_ROOT / "approximate_solver_v1" / "report_summary.json",
    "P8_package": CAMPAIGN_ROOT / "final_publication_package_v1" / "report_summary.json",
}


def main() -> None:
    missing_reports = [name for name, path in REPORTS.items() if not path.exists()]
    if missing_reports:
        raise RuntimeError(f"Missing mandatory terminal reports: {missing_reports}")
    reports = {name: read_json(path) for name, path in REPORTS.items()}
    leakage_failures = []
    for name, report in reports.items():
        for key in (
            "historical_analysis_500_used",
            "historical_final_test_used",
            "locked_outcomes_used_for_tuning",
        ):
            if bool(report.get(key, False)):
                leakage_failures.append(f"{name}:{key}")
    package = CAMPAIGN_ROOT / "final_publication_package_v1"
    missing_outputs = [name for name in REQUIRED_OUTPUTS if not (package / name).exists()]
    state_path = STATE_ROOT / "campaign_state.json"
    state = read_json(state_path)
    pending_tracks = [
        track
        for track, status in state.get("track_status", {}).items()
        if status in {"pending", "running", ""}
    ]
    checks = {
        "campaign_id_match": state.get("campaign_id") == CAMPAIGN_ID,
        "campaign_status_completed": state.get("campaign_status") == "completed",
        "all_mandatory_reports_exist": not missing_reports,
        "all_required_package_outputs_exist": not missing_outputs,
        "no_historical_locked_tuning": not leakage_failures,
        "no_pending_tracks": not pending_tracks,
        "package_validation_pass": bool(reports["P8_package"]["package_validation_pass"]),
    }
    acceptance = all(checks.values())
    validation = {
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": now_utc(),
        "checks": checks,
        "missing_reports": missing_reports,
        "missing_outputs": missing_outputs,
        "leakage_failures": leakage_failures,
        "pending_tracks": pending_tracks,
        "report_hashes": {
            name: sha256_file(REPORTS[name]) for name in sorted(REPORTS)
        },
        "acceptance_pass": acceptance,
    }
    write_json(package / "terminal_validation.json", validation)
    if not acceptance:
        raise RuntimeError(validation)
    state["terminal_validation_pass"] = True
    state["pod_status"] = "ready_to_stop"
    state["updated_at_utc"] = now_utc()
    write_json(state_path, state)
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
