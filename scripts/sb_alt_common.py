#!/usr/bin/env python3
"""Shared state, budget, and provenance utilities for the SB alternatives campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PROTOCOL = "counterfact_sb_alternatives_campaign_v1"
CAMPAIGN_ROOT = Path("runs/counterfact_sb_alternatives_campaign_v1")
STATE_ROOT = CAMPAIGN_ROOT / "autonomous_campaign_v1"
COMMON_ROOT = CAMPAIGN_ROOT / "common_protocol_v1"
D1_PROTOCOL_ROOT = Path("runs/counterfact_direction1_v1/protocol")

TRACKS = [
    {
        "id": "T1",
        "key": "learned_gate_raw_bridge",
        "protocol": "counterfact_learned_gate_raw_bridge_v1",
        "plan": "LEARNED_GATE_RAW_BRIDGE_PLAN.md",
        "pilot_estimate_usd": 0.75,
    },
    {
        "id": "T2",
        "key": "activation_space_sb",
        "protocol": "counterfact_activation_space_sb_v1",
        "plan": "ACTIVATION_SPACE_SB_PLAN.md",
        "pilot_estimate_usd": 1.50,
    },
    {
        "id": "T3",
        "key": "conditional_answer_span_csbm",
        "protocol": "counterfact_conditional_answer_span_csbm_v1",
        "plan": "CONDITIONAL_ANSWER_SPAN_CSBM_PLAN.md",
        "pilot_estimate_usd": 1.50,
    },
    {
        "id": "T4",
        "key": "unbalanced_partial_csbm",
        "protocol": "counterfact_unbalanced_partial_csbm_v1",
        "plan": "UNBALANCED_PARTIAL_CSBM_PLAN.md",
        "pilot_estimate_usd": 1.25,
    },
    {
        "id": "T5",
        "key": "parameter_space_sb",
        "protocol": "counterfact_parameter_space_sb_v1",
        "plan": "PARAMETER_SPACE_SB_PLAN.md",
        "pilot_estimate_usd": 3.00,
    },
]

LOCKED_MANIFEST_NAMES = (
    "dev_tune_200",
    "ablation_500",
    "analysis_500",
    "final_test_500",
    "final_test_full",
)
LOCKED_ALLOWED_FIELDS = {
    "case_id",
    "id",
    "source_dataset_split",
    "source_split",
    "source_index",
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def read_json(path: str | Path) -> Any:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with repo_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with full.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with full.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_csv(path: str | Path, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    exists = full.exists() and full.stat().st_size > 0
    with full.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_log(message: str) -> None:
    full = repo_path(STATE_ROOT / "autonomous_log.md")
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("a", encoding="utf-8") as handle:
        handle.write(f"\n- {now_utc()}: {message}\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: Any) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def env_float(name: str, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def require_autonomous_environment() -> dict[str, Any]:
    if os.environ.get("SB_ALT_AUTONOMOUS_MODE") != "1":
        raise RuntimeError("SB_ALT_AUTONOMOUS_MODE must equal 1")
    raw_rate = os.environ.get("RUNPOD_HOURLY_RATE_USD")
    rate = float(raw_rate) if raw_rate not in {None, ""} else None
    if rate is not None and rate <= 0.0:
        raise RuntimeError("RUNPOD_HOURLY_RATE_USD must be positive when supplied")
    required_strings = [
        "RUNPOD_POD_ID",
        "RUNPOD_SSH_KEY",
        "RUNPOD_SSH_USER",
        "RUNPOD_SSH_HOST",
        "RUNPOD_SSH_PORT",
        "REMOTE_REPO_DIR",
    ]
    missing = [name for name in required_strings if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing RunPod environment variables: {missing}")
    return {
        "hourly_rate_usd": rate,
        "cost_tracking_policy": "informational_only_non_blocking",
        "budget_guard_enabled": False,
        "runpod_pod_id": os.environ["RUNPOD_POD_ID"],
        "runpod_ssh_host": os.environ["RUNPOD_SSH_HOST"],
        "runpod_ssh_port": int(os.environ["RUNPOD_SSH_PORT"]),
        "remote_repo_dir": os.environ["REMOTE_REPO_DIR"],
    }


def collect_locked_exclusions(
    protocol_root: str | Path = D1_PROTOCOL_ROOT,
) -> dict[str, Any]:
    """Collect only IDs/source coordinates and hashes from locked manifests."""

    protocol_root = repo_path(protocol_root)
    case_ids: set[str] = set()
    source_keys: set[tuple[str, int]] = set()
    manifests: dict[str, Any] = {}
    for name in LOCKED_MANIFEST_NAMES:
        path = protocol_root / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Deliberately project immediately to the allowed exclusion fields.
                projected = {key: row.get(key) for key in LOCKED_ALLOWED_FIELDS}
                case_id = str(projected.get("case_id") or projected.get("id"))
                source_split = str(
                    projected.get("source_dataset_split")
                    or projected.get("source_split")
                    or ""
                )
                source_index = int(projected.get("source_index") or 0)
                case_ids.add(case_id)
                source_keys.add((source_split, source_index))
                count += 1
        manifests[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "count": count,
            "fields_used": sorted(LOCKED_ALLOWED_FIELDS),
            "prompt_label_output_metric_fields_used": False,
        }
    return {
        "case_ids": sorted(case_ids),
        "source_keys": sorted([f"{split}:{index}" for split, index in source_keys]),
        "manifests": manifests,
    }


def initialize_campaign_state(config: Mapping[str, Any]) -> None:
    state_path = repo_path(STATE_ROOT / "campaign_state.json")
    if state_path.exists():
        existing = read_json(state_path)
        if existing.get("campaign_protocol") != CAMPAIGN_PROTOCOL:
            raise RuntimeError("Existing campaign state has the wrong protocol")
        return

    started = now_utc()
    state = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "autonomous_mode": True,
        "campaign_status": "running",
        "current_track": None,
        "current_stage": "phase_a_bootstrap",
        "analysis_500_used": False,
        "final_test_used": False,
        "completed_tracks": [],
        "failed_tracks": [],
        "passed_tracks": [],
        "rescues_used": {},
        "last_git_commit": git_commit(),
        "campaign_start_utc": started,
        "campaign_start_epoch": time.time(),
        "runpod_pod_id": config["runpod_pod_id"],
        "remote_repo_dir": config["remote_repo_dir"],
        "updated_at_utc": started,
    }
    cost = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "hourly_rate_usd": config["hourly_rate_usd"],
        "cost_tracking_policy": "informational_only_non_blocking",
        "budget_guard_enabled": False,
        "estimated_spend_usd": 0.0,
        "pod_running_seconds": 0.0,
        "stage_costs": [],
        "last_cost_epoch": time.time(),
        "updated_at_utc": started,
    }
    write_json(STATE_ROOT / "campaign_state.json", state)
    write_json(STATE_ROOT / "cost_state.json", cost)
    write_csv(
        STATE_ROOT / "track_registry.csv",
        [
            {
                "track_id": track["id"],
                "track_key": track["key"],
                "protocol": track["protocol"],
                "plan": track["plan"],
                "status": "pending",
                "pilot_estimate_usd": track["pilot_estimate_usd"],
                "rescue_used": False,
                "evidence_path": "",
            }
            for track in TRACKS
        ],
    )
    write_csv(
        STATE_ROOT / "stage_history.csv",
        [
            {
                "timestamp_utc": started,
                "track": "campaign",
                "stage": "phase_a_bootstrap",
                "event": "campaign_initialized",
                "status": "pass",
                "notes": "Autonomous alternatives campaign state initialized.",
            }
        ],
    )
    repo_path(STATE_ROOT / "autonomous_log.md").write_text(
        "# SB Alternatives Autonomous Campaign Log\n\n"
        f"- {started}: campaign initialized at commit `{git_commit()}`.\n",
        encoding="utf-8",
    )


def summarize(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with repo_path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def refresh_cost(stage: str, notes: str = "") -> dict[str, Any]:
    cost_path = STATE_ROOT / "cost_state.json"
    if repo_path(cost_path).exists():
        cost = read_json(cost_path)
    else:
        raw_rate = os.environ.get("RUNPOD_HOURLY_RATE_USD")
        cost = {
            "campaign_protocol": CAMPAIGN_PROTOCOL,
            "hourly_rate_usd": float(raw_rate) if raw_rate not in {None, ""} else None,
            "cost_tracking_policy": "informational_only_non_blocking",
            "budget_guard_enabled": False,
            "estimated_spend_usd": 0.0,
            "pod_running_seconds": 0.0,
            "stage_costs": [],
            "last_cost_epoch": time.time(),
        }
    now_epoch = time.time()
    last_epoch = float(cost.get("last_cost_epoch") or now_epoch)
    elapsed = max(0.0, now_epoch - last_epoch)
    rate = cost.get("hourly_rate_usd")
    stage_cost = elapsed / 3600.0 * float(rate) if rate not in {None, ""} else None
    cost["pod_running_seconds"] = round(
        float(cost.get("pod_running_seconds", 0.0)) + elapsed, 3
    )
    if stage_cost is not None:
        cost["estimated_spend_usd"] = round(
            float(cost.get("estimated_spend_usd", 0.0)) + stage_cost, 6
        )
    cost["last_cost_epoch"] = now_epoch
    cost["updated_at_utc"] = now_utc()
    if elapsed > 0.0:
        cost.setdefault("stage_costs", []).append(
            {
                "stage": stage,
                "running_seconds": round(elapsed, 3),
                "estimated_cost_usd": round(stage_cost, 6) if stage_cost is not None else None,
                "notes": notes,
            }
        )
    write_json(cost_path, cost)
    return cost


def refresh_budget(stage: str, notes: str = "") -> dict[str, Any]:
    """Compatibility alias for historical reporters; never blocks execution."""

    cost = refresh_cost(stage, notes)
    return {
        **cost,
        "remaining_budget_usd": None,
        "budget_guard_pass": True,
    }


def budget_guard(track_id: str) -> dict[str, Any]:
    cost = refresh_cost(f"preflight_{track_id}", "Non-blocking cost refresh.")
    return {
        "track_id": track_id,
        "pass": True,
        "budget_guard_enabled": False,
        "cost_tracking_policy": "informational_only_non_blocking",
        "estimated_spend_usd": cost.get("estimated_spend_usd"),
    }


def record_stage_event(
    *,
    track: str,
    stage: str,
    event: str,
    status: str,
    notes: str,
) -> None:
    timestamp = now_utc()
    append_csv(
        STATE_ROOT / "stage_history.csv",
        {
            "timestamp_utc": timestamp,
            "track": track,
            "stage": stage,
            "event": event,
            "status": status,
            "notes": notes,
        },
        ["timestamp_utc", "track", "stage", "event", "status", "notes"],
    )
    append_log(f"[{track}/{stage}] {event}: {status}. {notes}")
    state = read_json(STATE_ROOT / "campaign_state.json")
    state["current_track"] = None if track == "campaign" else track
    state["current_stage"] = stage
    state["last_event"] = event
    state["last_stage_status"] = status
    state["last_git_commit"] = git_commit()
    state["updated_at_utc"] = timestamp
    write_json(STATE_ROOT / "campaign_state.json", state)


def set_track_status(
    track_id: str,
    status: str,
    *,
    evidence_path: str = "",
    rescue_used: bool | None = None,
) -> None:
    rows = read_csv(STATE_ROOT / "track_registry.csv")
    matched = False
    for row in rows:
        if row["track_id"] == track_id:
            row["status"] = status
            row["evidence_path"] = evidence_path
            if rescue_used is not None:
                row["rescue_used"] = str(bool(rescue_used))
            matched = True
    if not matched:
        raise KeyError(track_id)
    write_csv(STATE_ROOT / "track_registry.csv", rows)

    state = read_json(STATE_ROOT / "campaign_state.json")
    state["current_track"] = track_id
    if status in {"pilot_passed", "pilot_failed", "formal_negative", "budget_not_run"}:
        completed = list(state.get("completed_tracks", []))
        if track_id not in completed:
            completed.append(track_id)
        state["completed_tracks"] = completed
    if status == "pilot_passed":
        passed = list(state.get("passed_tracks", []))
        if track_id not in passed:
            passed.append(track_id)
        state["passed_tracks"] = passed
    if status in {"pilot_failed", "formal_negative"}:
        failed = list(state.get("failed_tracks", []))
        if track_id not in failed:
            failed.append(track_id)
        state["failed_tracks"] = failed
    state["updated_at_utc"] = now_utc()
    write_json(STATE_ROOT / "campaign_state.json", state)

    cost_path = repo_path(STATE_ROOT / "cost_state.json")
    if cost_path.exists():
        cost = read_json(cost_path)
        cost["last_track_status"] = {"track_id": track_id, "status": status}
        cost["updated_at_utc"] = now_utc()
        write_json(STATE_ROOT / "cost_state.json", cost)
