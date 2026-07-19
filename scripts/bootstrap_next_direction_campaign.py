#!/usr/bin/env python3
"""Initialize and validate the next-direction selection campaign."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    MANDATORY_TRACKS,
    PLAN_FILES,
    STATE_ROOT,
    TRACKS,
    git_commit,
    initialize_state,
    now_utc,
    record_stage,
    sha256_file,
    write_json,
)


def build_report(*, require_autonomous: bool = True) -> dict:
    state = initialize_state()
    plans = []
    for name in PLAN_FILES:
        path = ROOT / name
        plans.append(
            {
                "path": name,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    registry_path = ROOT / "CANDIDATE_DIRECTION_REGISTRY.json"
    active_path = ROOT / "ACTIVE_RESEARCH_CAMPAIGN.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    active = json.loads(active_path.read_text(encoding="utf-8"))
    registered = {str(row["track_id"]) for row in registry.get("tracks", [])}
    checks = {
        "autonomous_mode": bool(state["autonomous_mode"]),
        "all_plan_files_exist": all(row["exists"] for row in plans),
        "active_protocol_matches": active.get("active_protocol") == CAMPAIGN_ID,
        "registry_protocol_matches": registry.get("protocol_version") == CAMPAIGN_ID,
        "all_mandatory_tracks_registered": set(MANDATORY_TRACKS) <= registered,
        "conditional_track_registered": "N6" in registered,
        "historical_protocols_read_only": bool(state["historical_protocols_read_only"]),
        "analysis_500_used": False,
        "final_test_used": False,
    }
    ignored_checks = {"analysis_500_used", "final_test_used"}
    if not require_autonomous:
        ignored_checks.add("autonomous_mode")
    pass_values = [value for key, value in checks.items() if key not in ignored_checks]
    return {
        "campaign_id": CAMPAIGN_ID,
        "stage": "S0_bootstrap",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "plans": plans,
        "tracks": TRACKS,
        "checks": checks,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": all(pass_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_non_autonomous_test", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    started = now_utc()
    output = CAMPAIGN_ROOT / "S0_bootstrap_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = build_report(require_autonomous=not bool(args.allow_non_autonomous_test))
    write_json(output / "report_summary.json", report)
    write_json(output / "validation_report.json", report["checks"])
    write_json(
        output / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "autonomous_mode": report["checks"]["autonomous_mode"],
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    record_stage(
        "S0_bootstrap",
        status="passed" if report["acceptance_pass"] else "failed",
        acceptance_pass=bool(report["acceptance_pass"]),
        output_dir=output,
        started_at_utc=started,
        notes="Authoritative plans, registry, autonomous state, and immutable-history policy validated.",
        next_stage="S0_fresh_manifests" if report["acceptance_pass"] else None,
        exit_code=0 if report["acceptance_pass"] else 2,
    )
    if not report["acceptance_pass"]:
        raise SystemExit(2)
    print(f"S0 bootstrap passed: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
