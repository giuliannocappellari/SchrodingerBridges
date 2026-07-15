#!/usr/bin/env python3
"""Validate and finalize a bounded-negative Direction 3 campaign.

This is a local, summary-only operation. It never imports model code, starts a
pod, or reads analysis/final split artifacts. The validator checks that the
preserved stop package supports the protocol-mandated decision, records which
raw source artifacts remain available, and normalizes campaign metadata to a
single terminal state.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D3_PROTOCOL_VERSION, ROOT, now_utc, sha256_file, write_csv, write_json


DEFAULT_D3_ROOT = Path("runs/counterfact_direction3_controller_v1")
DEFAULT_CAMPAIGN_DIR = DEFAULT_D3_ROOT / "autonomous_campaign_v1"
DEFAULT_STOP_DIR = DEFAULT_D3_ROOT / "direction3_autonomous_stop_checkpoint_v1"
REQUIRED_STOP_FILES = (
    "report_summary.json",
    "direction3_autonomous_stop_checkpoint.md",
    "direction3_evidence_table.csv",
    "negative_result_report.md",
    "next_direction_recommendation.md",
)
REQUIRED_FAILURES = {
    "value_top3_pass",
    "representation_beats_target_indicator_pass",
    "state_shuffle_hurts_pass",
}


def _full(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _metric_row(rows: Sequence[Mapping[str, str]], stage_fragment: str, metric: str) -> Mapping[str, str]:
    matches = [row for row in rows if stage_fragment in row.get("stage", "") and row.get("metric") == metric]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one evidence row for stage={stage_fragment!r}, metric={metric!r}; found {len(matches)}"
        )
    return matches[0]


def validate_stop_package(root: Path, stop_dir: Path) -> Dict[str, Any]:
    full_stop = _full(root, stop_dir)
    missing_required = [name for name in REQUIRED_STOP_FILES if not (full_stop / name).is_file()]
    if missing_required:
        raise AssertionError(f"Missing required stop-package files: {missing_required}")

    summary = _read_json(full_stop / "report_summary.json")
    evidence = _read_csv(full_stop / "direction3_evidence_table.csv")
    stop_text = (full_stop / "direction3_autonomous_stop_checkpoint.md").read_text(encoding="utf-8")
    negative_text = (full_stop / "negative_result_report.md").read_text(encoding="utf-8")

    checks = {
        "protocol_version_pass": summary.get("protocol_version") == D3_PROTOCOL_VERSION,
        "formal_negative_completion_pass": summary.get("campaign_status") == "formal_negative_completion",
        "negative_completion_pass": _as_bool(summary.get("negative_completion")),
        "positive_completion_false_pass": not _as_bool(summary.get("positive_completion")),
        "bounded_rescue_consumed_pass": _as_bool(summary.get("bounded_scientific_rescue_used"))
        and int((summary.get("rescue_attempts_used") or {}).get("stage_1b4_value", 0)) == 1,
        "stage_2a_blocked_pass": _as_bool(summary.get("do_not_run_stage_2a"))
        and not _as_bool(summary.get("stage_2a_run")),
        "no_actual_decode_pass": not _as_bool(summary.get("actual_decode_run")),
        "analysis_lock_preserved_pass": not _as_bool(summary.get("analysis_500_used"))
        and _as_bool(summary.get("do_not_run_analysis_500")),
        "final_lock_preserved_pass": not _as_bool(summary.get("final_test_used"))
        and _as_bool(summary.get("do_not_run_final_test")),
        "required_hard_failures_recorded_pass": REQUIRED_FAILURES.issubset(
            set(summary.get("hard_criteria_failed_after_rescue") or [])
        ),
        "checkpoint_markdown_pass": "formal_negative_completion" in stop_text
        and "Stage 2A actual D3 decoding was not run" in stop_text,
        "negative_report_markdown_pass": "No analysis or final split was used" in negative_text
        and "Stage 2A actual decoding would be scientifically unjustified" in negative_text,
    }

    feature_audit = _metric_row(evidence, "1B.4A feature audit", "audit_pass")
    rescue_status = _metric_row(evidence, "1B.4/1B.5 rescue1", "scientific_acceptance_pass")
    rescue_top3 = _metric_row(evidence, "1B.4/1B.5 rescue1", "d3_value_repr_teacher_top3_overlap")
    rescue_full = _metric_row(evidence, "1B.4/1B.5 rescue1", "d3_value_repr_macro_spearman")
    rescue_target = _metric_row(evidence, "1B.4/1B.5 rescue1", "target_indicator_only_macro_spearman")
    rescue_shuffle = _metric_row(evidence, "1B.4/1B.5 rescue1", "state_shuffle_hurts")
    leakage = _metric_row(evidence, "1B.4/1B.5 rescue1", "feature_leakage_audit_pass")

    checks.update(
        {
            "feature_audit_evidence_pass": _as_bool(feature_audit.get("value")) and _as_bool(feature_audit.get("pass")),
            "rescue_scientific_failure_evidence_pass": not _as_bool(rescue_status.get("value"))
            and not _as_bool(rescue_status.get("pass")),
            "rescue_top3_failure_evidence_pass": float(rescue_top3["value"]) < 0.65
            and not _as_bool(rescue_top3.get("pass")),
            "target_indicator_shortcut_evidence_pass": float(rescue_target["value"])
            > float(rescue_full["value"]) + 0.05
            and not _as_bool(rescue_target.get("pass")),
            "state_shuffle_failure_evidence_pass": not _as_bool(rescue_shuffle.get("value"))
            and not _as_bool(rescue_shuffle.get("pass")),
            "feature_leakage_evidence_pass": _as_bool(leakage.get("value")) and _as_bool(leakage.get("pass")),
        }
    )

    referenced = sorted(set(str(value).rstrip("/") for value in (summary.get("artifacts") or {}).values()))
    source_artifacts = [
        {
            "path": path,
            "exists": _full(root, Path(path)).exists(),
        }
        for path in referenced
    ]
    missing_source_artifacts = [row["path"] for row in source_artifacts if not row["exists"]]
    package_validation_pass = all(checks.values())
    if not package_validation_pass:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Direction 3 stop-package validation failed: {failed}")

    return {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 terminal stop-package validation",
        "validated_at_utc": now_utc(),
        "validation_scope": "durable_stop_summary_consistency",
        "package_validation_pass": True,
        "terminal_state_valid": True,
        "analysis_500_used": False,
        "final_test_used": False,
        "actual_decode_performed": False,
        "bounded_scientific_rescue_used": True,
        "stage_2a_allowed": False,
        "checks": checks,
        "raw_source_metric_rederivation_available": not missing_source_artifacts,
        "referenced_source_artifacts": source_artifacts,
        "missing_source_artifacts": missing_source_artifacts,
        "warnings": (
            [
                "The old pod's raw v3 train/replay directories are not present in this workspace. "
                "The terminal decision is validated from the preserved stop summaries and cannot be rederived from raw rows here."
            ]
            if missing_source_artifacts
            else []
        ),
    }


def _integrity_rows(stop_dir: Path) -> list[Dict[str, Any]]:
    rows = []
    for name in REQUIRED_STOP_FILES:
        path = stop_dir / name
        rows.append(
            {
                "artifact": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "required": True,
                "status": "present",
            }
        )
    return rows


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _append_stage_history(path: Path, timestamp: str, note: str) -> None:
    rows = _read_csv(path) if path.exists() else []
    event = "terminal_package_validated_and_pod_stopped"
    if not any(row.get("event") == event for row in rows):
        rows.append(
            {
                "timestamp_utc": timestamp,
                "stage": "direction3_autonomous_stop_checkpoint_v1",
                "event": event,
                "status": "pass",
                "notes": note,
            }
        )
    write_csv(path, rows, fieldnames=["timestamp_utc", "stage", "event", "status", "notes"])


def finalize_terminal_campaign(
    *,
    root: Path,
    active_campaign_path: Path,
    campaign_dir: Path,
    stop_dir: Path,
    budget_usd: float,
    hourly_rate_usd: float,
    reserve_usd: float,
    current_pod_running_seconds: float,
    runpod_pod_id: str,
    runpod_ssh_host: str,
    runpod_ssh_port: str,
) -> Dict[str, Any]:
    if min(budget_usd, hourly_rate_usd) <= 0 or reserve_usd < 0:
        raise AssertionError("Budget and hourly rate must be positive; reserve must be nonnegative")

    full_stop = _full(root, stop_dir)
    full_campaign = _full(root, campaign_dir)
    full_active = _full(root, active_campaign_path)
    validation = validate_stop_package(root, stop_dir)
    write_json(full_stop / "terminal_package_validation.json", validation)
    write_csv(full_stop / "artifact_integrity_manifest.csv", _integrity_rows(full_stop))

    state_path = full_campaign / "campaign_state.json"
    budget_path = full_campaign / "budget_state.json"
    history_path = full_campaign / "stage_history.csv"
    log_path = full_campaign / "autonomous_log.md"
    for required in (state_path, budget_path, history_path, log_path, full_active):
        if not required.exists():
            raise AssertionError(f"Required campaign state artifact is missing: {required}")

    timestamp = now_utc()
    state = _read_json(state_path)
    state.update(
        {
            "protocol_version": D3_PROTOCOL_VERSION,
            "autonomous_mode": True,
            "campaign_status": "formal_negative_completion",
            "status": "complete_negative",
            "current_stage": "complete_negative",
            "current_stage_status": "completed",
            "next_stage": None,
            "scientific_claim_status": "negative_result",
            "analysis_500_used": False,
            "final_test_used": False,
            "last_event": "terminal_package_validated_and_pod_stopped",
            "last_stage_status": "terminal",
            "last_updated_utc": timestamp,
            "updated_at_utc": timestamp,
            "runpod_pod_id": runpod_pod_id,
            "runpod_ssh_host": runpod_ssh_host,
            "runpod_ssh_port": runpod_ssh_port,
            "runpod_status": "EXITED",
            "stage_2a_allowed": False,
            "terminal_package_validation_pass": True,
            "terminal_package_validation_path": str(stop_dir / "terminal_package_validation.json"),
            "raw_source_metric_rederivation_available": validation["raw_source_metric_rederivation_available"],
            "failed_stages": ["stage_1b4_representation_aware_value_controller"],
            "rescues_used": {"stage_1b4_value": 1},
        }
    )
    completed = list(state.get("completed_stages") or [])
    for stage in (
        "stage_1b4a_feature_cache_readiness_audit",
        "stage_1b4_value_attempt1",
        "stage_1b4_value_rescue1",
        "direction3_autonomous_stop_checkpoint_v1",
        "terminal_package_validation",
    ):
        if stage not in completed:
            completed.append(stage)
    state["completed_stages"] = completed
    write_json(state_path, state)

    stop_summary = _read_json(full_stop / "report_summary.json")
    campaign_start = _parse_time(str(state["campaign_start_utc"]))
    scientific_stop = _parse_time(str(stop_summary["created_at_utc"]))
    prior_running_seconds = max(0.0, (scientific_stop - campaign_start).total_seconds())
    total_running_seconds = prior_running_seconds + max(0.0, current_pod_running_seconds)
    estimated_spend = total_running_seconds / 3600.0 * hourly_rate_usd
    remaining = max(0.0, budget_usd - estimated_spend)
    budget = _read_json(budget_path)
    budget.update(
        {
            "protocol_version": D3_PROTOCOL_VERSION,
            "autonomous_mode": True,
            "budget_usd": budget_usd,
            "hourly_rate_usd": hourly_rate_usd,
            "reserve_usd": reserve_usd,
            "estimated_spend_usd": round(estimated_spend, 6),
            "remaining_budget_usd": round(remaining, 6),
            "budget_guard_pass": remaining >= reserve_usd,
            "last_updated_utc": timestamp,
            "last_budget_check_utc": timestamp,
            "runpod_allowed_next": False,
            "runpod_pod_id": runpod_pod_id,
            "pod_status": "EXITED",
            "terminal_state": "formal_negative_completion",
            "spend_basis": "bounded_campaign_wall_clock_plus_terminal_resume_pod_wall_clock",
            "stage_costs": [
                {
                    "stage": "stage_1b4a_through_formal_negative_completion",
                    "running_seconds": round(prior_running_seconds, 3),
                    "estimated_cost_usd": round(prior_running_seconds / 3600.0 * hourly_rate_usd, 6),
                },
                {
                    "stage": "terminal_resume_integrity_audit",
                    "running_seconds": round(max(0.0, current_pod_running_seconds), 3),
                    "estimated_cost_usd": round(max(0.0, current_pod_running_seconds) / 3600.0 * hourly_rate_usd, 6),
                },
            ],
        }
    )
    write_json(budget_path, budget)

    _append_stage_history(
        history_path,
        timestamp,
        "Durable negative stop package validated; raw v3 source directories unavailable; idle replacement pod stopped.",
    )
    log_text = log_path.read_text(encoding="utf-8")
    marker = "Terminal package revalidation and pod shutdown"
    if marker not in log_text:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## {timestamp} - {marker}\n"
                "- Durable stop summaries passed internal consistency validation.\n"
                "- Raw v3 train/replay directories from the old pod are unavailable for metric rederivation.\n"
                "- No rescue or Stage 2A work was rerun.\n"
                f"- Replacement pod `{runpod_pod_id}` was confirmed idle and stopped.\n"
            )

    active = _read_json(full_active)
    active.update(
        {
            "updated_at_utc": timestamp,
            "campaign_status": "completed_negative",
            "current_stage": "complete_negative",
            "next_stage_on_pass": None,
            "terminal_state": "formal_negative_completion",
            "terminal_package": str(stop_dir),
            "terminal_package_validation": str(stop_dir / "terminal_package_validation.json"),
        }
    )
    direction3_state = dict(active.get("direction3_state") or {})
    direction3_state.update(
        {
            "representation_aware_v3": "failed_after_bounded_rescue",
            "stage_2a": "not_run_blocked_by_offline_hard_criteria",
            "terminal_package_validation": "passed_with_raw_source_rederivation_unavailable",
        }
    )
    active["direction3_state"] = direction3_state
    write_json(full_active, active)
    return {"validation": validation, "campaign_state": state, "budget_state": budget, "active_campaign": active}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--active_campaign", type=Path, default=Path("ACTIVE_RESEARCH_CAMPAIGN.json"))
    parser.add_argument("--campaign_dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--stop_dir", type=Path, default=DEFAULT_STOP_DIR)
    parser.add_argument("--budget_usd", type=float, default=float(os.environ.get("D3_AUTONOMOUS_BUDGET_USD", "0")))
    parser.add_argument("--hourly_rate_usd", type=float, default=float(os.environ.get("RUNPOD_HOURLY_RATE_USD", "0")))
    parser.add_argument("--reserve_usd", type=float, default=float(os.environ.get("D3_AUTONOMOUS_BUDGET_RESERVE_USD", "0")))
    parser.add_argument("--current_pod_running_seconds", type=float, default=0.0)
    parser.add_argument("--runpod_pod_id", default=os.environ.get("RUNPOD_POD_ID", ""))
    parser.add_argument("--runpod_ssh_host", default=os.environ.get("RUNPOD_SSH_HOST", ""))
    parser.add_argument("--runpod_ssh_port", default=os.environ.get("RUNPOD_SSH_PORT", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.runpod_pod_id:
        raise AssertionError("RUNPOD_POD_ID is required for terminal campaign finalization")
    result = finalize_terminal_campaign(
        root=args.root.resolve(),
        active_campaign_path=args.active_campaign,
        campaign_dir=args.campaign_dir,
        stop_dir=args.stop_dir,
        budget_usd=args.budget_usd,
        hourly_rate_usd=args.hourly_rate_usd,
        reserve_usd=args.reserve_usd,
        current_pod_running_seconds=args.current_pod_running_seconds,
        runpod_pod_id=args.runpod_pod_id,
        runpod_ssh_host=args.runpod_ssh_host,
        runpod_ssh_port=args.runpod_ssh_port,
    )
    print(f"[INFO] terminal_package_validation_pass={result['validation']['package_validation_pass']}")
    print(f"[INFO] campaign_status={result['campaign_state']['campaign_status']}")
    print(f"[INFO] runpod_status={result['campaign_state']['runpod_status']}")


if __name__ == "__main__":
    main()
