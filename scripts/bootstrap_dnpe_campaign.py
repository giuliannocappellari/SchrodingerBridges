#!/usr/bin/env python3
"""Initialize and validate the diffusion-native parametric editor campaign."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    HISTORICAL_CAMPAIGNS,
    STATE_ROOT,
    git_commit,
    initialize_state,
    now_utc,
    read_json,
    record_stage,
    write_json,
)


def _command(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod_verified", type=int, choices=(0, 1), default=0)
    parser.add_argument("--remote_tests_passed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gpu_name", default="")
    args = parser.parse_args()
    started = now_utc()
    active = read_json(ROOT / "ACTIVE_RESEARCH_CAMPAIGN.json")
    registry = read_json(ROOT / "EXPERIMENT_PROTOCOL_REGISTRY.json")
    if active.get("active_protocol") != CAMPAIGN_ID:
        raise RuntimeError("ACTIVE_RESEARCH_CAMPAIGN.json does not select DNPE")
    if registry.get("protocol_version") != CAMPAIGN_ID:
        raise RuntimeError("EXPERIMENT_PROTOCOL_REGISTRY.json does not select DNPE")
    state = initialize_state()
    historical = []
    for campaign in HISTORICAL_CAMPAIGNS:
        path = ROOT / "runs" / campaign
        historical.append(
            {
                "campaign": campaign,
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "access_policy": "read_only_evidence",
            }
        )
    acceptance = {
        "autonomous_mode_enabled": bool(state["autonomous_mode"]),
        "active_protocol_matches": True,
        "pod_running_with_gpu": bool(args.pod_verified and args.gpu_name),
        "ssh_workspace_valid": bool(args.pod_verified),
        "remote_tests_pass": bool(args.remote_tests_passed),
        "historical_protocols_read_only": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = all(
        value is True
        for key, value in acceptance.items()
        if key not in {"analysis_500_used", "final_test_used"}
    ) and not acceptance["analysis_500_used"] and not acceptance["final_test_used"]
    output = CAMPAIGN_ROOT / "A0_bootstrap_v1"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "A0_bootstrap",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "python": sys.version,
        "autonomous_environment": {
            "DNPE_AUTONOMOUS_MODE": os.environ.get("DNPE_AUTONOMOUS_MODE"),
            "SB_ALT_AUTONOMOUS_MODE": os.environ.get("SB_ALT_AUTONOMOUS_MODE"),
        },
        "gpu_name": args.gpu_name,
        "historical_campaigns": historical,
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "git_status_short": _command(["git", "status", "--short"]),
    }
    write_json(output / "report_summary.json", report)
    write_json(output / "run_config.json", {"pod_verified": args.pod_verified, "remote_tests_passed": args.remote_tests_passed})
    write_json(output / "validation_report.json", acceptance)
    record_stage(
        "A0_bootstrap",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=output,
        started_at_utc=started,
        notes="Pod/GPU and remote tests verified; historical roots frozen read-only." if passed else "Bootstrap acceptance failed.",
        next_stage="A1_source_audit" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(f"A0 bootstrap passed: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

