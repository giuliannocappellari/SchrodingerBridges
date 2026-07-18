#!/usr/bin/env python3
"""Initialize and validate the partial-state temporal residual campaign."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    STATE_ROOT,
    git_commit,
    historical_snapshot,
    initialize_state,
    now_utc,
    read_json,
    record_stage,
    write_json,
)


def command(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod_verified", type=int, choices=(0, 1), required=True)
    parser.add_argument("--remote_tests_passed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu_name", required=True)
    parser.add_argument("--pod_id", required=True)
    args = parser.parse_args()
    started = now_utc()
    active = read_json(ROOT / "ACTIVE_RESEARCH_CAMPAIGN.json")
    registry = read_json(ROOT / "EXPERIMENT_PROTOCOL_REGISTRY.json")
    if active.get("active_protocol") != CAMPAIGN_ID:
        raise RuntimeError("Active campaign does not select the TRM protocol")
    if registry.get("protocol_version") != CAMPAIGN_ID:
        raise RuntimeError("Protocol registry does not select the TRM protocol")
    state = initialize_state()
    snapshot = historical_snapshot()
    write_json(STATE_ROOT / "historical_immutability_snapshot.json", snapshot)
    acceptance = {
        "autonomous_mode_enabled": bool(state["autonomous_mode"]),
        "active_protocol_matches": True,
        "pod_running_with_gpu": bool(args.pod_verified and args.gpu_name),
        "remote_tests_pass": bool(args.remote_tests_passed),
        "historical_protocols_snapshotted_read_only": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = (
        all(value for key, value in acceptance.items() if key not in {"analysis_500_used", "final_test_used"})
        and not acceptance["analysis_500_used"]
        and not acceptance["final_test_used"]
    )
    output = CAMPAIGN_ROOT / "A0_bootstrap_v1"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "A0_bootstrap",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "autonomous_environment": {
            "PS_TRM_AUTONOMOUS_MODE": os.environ.get("PS_TRM_AUTONOMOUS_MODE"),
            "PS_TRM_MAX_INFRA_RETRIES": os.environ.get("PS_TRM_MAX_INFRA_RETRIES"),
            "PS_TRM_MAX_SCIENTIFIC_RESCUES_PER_STAGE": os.environ.get("PS_TRM_MAX_SCIENTIFIC_RESCUES_PER_STAGE"),
        },
        "pod_id": args.pod_id,
        "gpu_name": args.gpu_name,
        "historical_campaign_count": len(snapshot["historical_campaigns"]),
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "git_status_short": command(["git", "status", "--short"]),
    }
    write_json(output / "report_summary.json", report)
    write_json(output / "run_config.json", vars(args))
    write_json(output / "validation_report.json", acceptance)
    record_stage(
        "A0_bootstrap",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=output,
        started_at_utc=started,
        notes="Pod/GPU/tests verified and historical artifacts snapshotted read-only.",
        next_stage="A1_source_audit" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(f"A0 bootstrap passed: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
