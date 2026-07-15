#!/usr/bin/env python3
"""Migrate a historical budget stop into the active no-budget campaign state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import CAMPAIGN_PROTOCOL, now_utc


CAMPAIGN_REL = Path("runs/counterfact_sb_alternatives_campaign_v1")
STATE_REL = CAMPAIGN_REL / "autonomous_campaign_v1"
TRACK_ROOTS = {
    "T2": Path("runs/counterfact_activation_space_sb_v1"),
    "T3": Path("runs/counterfact_conditional_answer_span_csbm_v1"),
    "T4": Path("runs/counterfact_unbalanced_partial_csbm_v1"),
    "T5": Path("runs/counterfact_parameter_space_sb_v1"),
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def migrate_state_payload(state: Mapping[str, Any], pod_id: str, remote_repo_dir: str) -> dict[str, Any]:
    result = deepcopy(dict(state))
    if result.get("campaign_protocol") != CAMPAIGN_PROTOCOL:
        raise RuntimeError("Persisted state has the wrong campaign protocol")
    if result.get("analysis_500_used") or result.get("final_test_used"):
        raise RuntimeError("Cannot resume: locked analysis/final state is already consumed")
    prior = {
        "campaign_status": result.get("campaign_status"),
        "terminal_reason": result.get("terminal_reason"),
        "terminal_at_utc": result.get("terminal_at_utc"),
        "last_event": result.get("last_event"),
        "superseded_at_utc": now_utc(),
        "superseded_by": "no_budget_guard_campaign_migration",
    }
    result.setdefault("historical_terminal_events", []).append(prior)
    result.update(
        {
            "campaign_status": "running",
            "current_track": "T2",
            "current_stage": "T2.1_activation_endpoint_collection",
            "completed_tracks": ["T1"],
            "failed_tracks": ["T1"],
            "passed_tracks": [],
            "runpod_pod_id": pod_id,
            "remote_repo_dir": remote_repo_dir,
            "budget_guard_enabled": False,
            "cost_tracking_policy": "informational_only_non_blocking",
            "previous_budget_stop_superseded": True,
            "resume_count": int(result.get("resume_count", 0)) + 1,
            "last_event": "budget_stop_superseded_campaign_resumed",
            "last_stage_status": "pass",
            "updated_at_utc": now_utc(),
        }
    )
    result.pop("terminal_reason", None)
    result.pop("terminal_at_utc", None)
    return result


def migrate_registry_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        track_id = row["track_id"]
        if track_id == "T1":
            if row.get("status") != "formal_negative":
                raise RuntimeError("T1 validated formal-negative status was not preserved")
        elif track_id in TRACK_ROOTS:
            if row.get("status") != "budget_not_run":
                raise RuntimeError(f"{track_id} is not a superseded monetary stop")
            row["status"] = "pending"
            row["evidence_path"] = ""
            row["rescue_used"] = "False"
        output.append(row)
    if [row["track_id"] for row in output] != ["T1", "T2", "T3", "T4", "T5"]:
        raise RuntimeError("Track registry is incomplete or reordered")
    return output


def move_preserving(source: Path, destination: Path, manifest: list[dict[str, Any]]) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    before = tree_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    after = tree_hash(destination)
    if before != after:
        raise RuntimeError(f"Artifact hash changed while moving {source}")
    manifest.append(
        {
            "source_path": str(source.relative_to(ROOT)),
            "historical_path": str(destination.relative_to(ROOT)),
            "sha256_tree": after,
            "preserved": True,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod_id", required=True)
    parser.add_argument("--remote_repo_dir", default="/workspace/SB")
    args = parser.parse_args()

    campaign = ROOT / CAMPAIGN_REL
    state_root = ROOT / STATE_REL
    output = campaign / "resume_migration_v1"
    report_path = output / "report_summary.json"
    if report_path.exists():
        report = read_json(report_path)
        if report.get("migration_pass") is not True:
            raise RuntimeError("Existing migration report did not pass")
        print("migration_already_applied=True")
        return

    state_path = state_root / "campaign_state.json"
    registry_path = state_root / "track_registry.csv"
    state_before = read_json(state_path)
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry_before = list(csv.DictReader(handle))
    if state_before.get("campaign_status") != "budget_completion":
        raise RuntimeError("Expected the persisted historical budget-completion state")

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "state_before.json", state_before)
    manifest: list[dict[str, Any]] = []
    historical = campaign / "historical_budget_stop_v1"
    for name in (
        "final_research_package_v1",
        "campaign_terminal_status.json",
        "pilot_registry_lock.json",
    ):
        move_preserving(campaign / name, historical / name, manifest)
    budget_path = state_root / "budget_state.json"
    historical_budget = read_json(budget_path)
    move_preserving(budget_path, historical / "budget_state.json", manifest)
    for track_id, track_root_rel in TRACK_ROOTS.items():
        track_root = ROOT / track_root_rel
        move_preserving(
            track_root / "pilot_stop_package_v1",
            track_root / "historical_budget_stop_package_v1",
            manifest,
        )

    state_after = migrate_state_payload(state_before, args.pod_id, args.remote_repo_dir)
    registry_after = migrate_registry_rows(registry_before)
    write_json(state_path, state_after)
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registry_after[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(registry_after)

    cost = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "cost_tracking_policy": "informational_only_non_blocking",
        "budget_guard_enabled": False,
        "hourly_rate_usd": historical_budget.get("hourly_rate_usd"),
        "estimated_spend_usd": historical_budget.get("estimated_spend_usd"),
        "pod_running_seconds": sum(
            float(row.get("running_seconds") or 0.0)
            for row in historical_budget.get("stage_costs", [])
        ),
        "stage_costs": historical_budget.get("stage_costs", []),
        "last_cost_epoch": time.time(),
        "updated_at_utc": now_utc(),
        "historical_budget_state": str((historical / "budget_state.json").relative_to(ROOT)),
    }
    write_json(state_root / "cost_state.json", cost)

    timestamp = now_utc()
    with (state_root / "stage_history.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                timestamp,
                "campaign",
                "resume_no_budget_guard",
                "historical_budget_stop_superseded",
                "pass",
                "Preserved T1 formal negative; reopened T2-T5 at pending without repeating validated work.",
            ]
        )
    with (state_root / "autonomous_log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n- {timestamp}: [campaign/resume_no_budget_guard] historical budget stop superseded; "
            "T1 preserved and T2-T5 reopened. Cost tracking is informational only.\n"
        )

    write_json(output / "state_after.json", state_after)
    with (output / "artifact_migration_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "stage": "no-budget-guard resume migration",
        "created_at_utc": now_utc(),
        "migration_pass": True,
        "previous_budget_stop_superseded": True,
        "budget_guard_enabled": False,
        "cost_tracking_policy": "informational_only_non_blocking",
        "t1_status_preserved": "formal_negative",
        "reopened_tracks": ["T2", "T3", "T4", "T5"],
        "first_incomplete_stage": "T2.1_activation_endpoint_collection",
        "analysis_500_used": False,
        "final_test_used": False,
        "historical_artifacts_preserved": len(manifest),
        "pod_id": args.pod_id,
        "remote_repo_dir": args.remote_repo_dir,
    }
    write_json(report_path, report)
    write_json(campaign / "campaign_resume_status.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
