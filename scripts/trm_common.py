#!/usr/bin/env python3
"""Shared state, provenance, and artifact helpers for the TRM campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "partial_state_temporal_residual_editor_v1"
CAMPAIGN_ROOT = ROOT / "runs" / CAMPAIGN_ID
STATE_ROOT = CAMPAIGN_ROOT / "autonomous_campaign_v1"
PROTOCOL_ROOT = CAMPAIGN_ROOT / "protocol_v1"

PRIMARY_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
PRIMARY_MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
SOURCE_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
SOURCE_MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
SECONDARY_MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
SECONDARY_MODEL_REVISION = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"

HISTORICAL_CAMPAIGNS = (
    "counterfact_direction1_v1",
    "counterfact_direction2_bridge_adapter_v1",
    "counterfact_direction3_controller_v1",
    "counterfact_sb_alternatives_campaign_v1",
    "masked_diffusion_memit_sb_positive_result_v1",
    "mask_pattern_sb_publication_confirmation_v1",
    "diffusion_native_causal_partial_state_editor_v1",
)
LOCKED_SPLIT_NAMES = ("analysis_500", "final_test_500", "final_test_full")
STAGES = (
    "A0_bootstrap",
    "A1_source_audit",
    "B0_fresh_protocol",
    "C0_timerome_source_reproduction",
    "C1_temporal_localization",
    "C2_fullmask_temporal_residual",
    "D1_partial_state_target_delta",
    "D2_state_conditioned_protection",
    "E1_smoke20",
    "E2_pilot100",
    "E3_kamel_multi_token",
    "F1_dev200_selection",
    "F2_dev_lock",
    "F3_locked_confirmation",
    "G1_edit_scaling",
    "G2_second_backbone",
    "H1_final_package",
)
STAGE_HISTORY_FIELDS = (
    "stage",
    "status",
    "started_at_utc",
    "ended_at_utc",
    "git_commit",
    "output_dir",
    "log_path",
    "exit_code",
    "acceptance_pass",
    "notes",
)
ARTIFACT_FIELDS = (
    "stage",
    "path",
    "sha256",
    "size_bytes",
    "created_at_utc",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: str | Path) -> Any:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or sorted({key for row in rows for key in row}))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()


def autonomous_enabled() -> bool:
    return os.environ.get("PS_TRM_AUTONOMOUS_MODE") == "1"


def initial_campaign_state() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "protocol_version": CAMPAIGN_ID,
        "autonomous_mode": autonomous_enabled(),
        "campaign_status": "running",
        "current_stage": "A0_bootstrap",
        "next_stage": "A1_source_audit",
        "analysis_500_used": False,
        "final_test_used": False,
        "historical_protocols_read_only": True,
        "completed_stages": [],
        "failed_stages": [],
        "rescues_used": {},
        "stage_status": {stage: "pending" for stage in STAGES},
        "last_git_commit": git_commit(),
        "pod_status": "running",
        "created_at_utc": now_utc(),
        "updated_at_utc": now_utc(),
    }


def initialize_state(*, allow_existing: bool = True) -> dict[str, Any]:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path = STATE_ROOT / "campaign_state.json"
    if path.exists():
        if not allow_existing:
            raise FileExistsError(path)
        state = read_json(path)
        if state.get("protocol_version") != CAMPAIGN_ID:
            raise RuntimeError("Persisted campaign protocol does not match active TRM protocol")
        return state
    state = initial_campaign_state()
    write_json(path, state)
    write_csv(STATE_ROOT / "stage_history.csv", [], STAGE_HISTORY_FIELDS)
    (STATE_ROOT / "autonomous_log.md").write_text(
        f"# TRM Autonomous Log\n\n- {now_utc()}: campaign initialized.\n",
        encoding="utf-8",
    )
    write_json(
        STATE_ROOT / "cost_state.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "monetary_budget_guard_enabled": False,
            "cost_logging_informational_only": True,
            "pod_hourly_rate_usd": _optional_float("RUNPOD_HOURLY_RATE_USD"),
            "pod_started_at_utc": now_utc(),
            "estimated_cost_usd": 0.0,
            "stage_costs": [],
            "updated_at_utc": now_utc(),
        },
    )
    write_json(STATE_ROOT / "artifact_registry.json", {"campaign_id": CAMPAIGN_ID, "artifacts": []})
    return state


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in {None, ""} else None


def append_log(message: str) -> None:
    initialize_state()
    with (STATE_ROOT / "autonomous_log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- {now_utc()}: {message}\n")


def register_artifacts(stage: str, paths: Iterable[str | Path]) -> None:
    initialize_state()
    registry_path = STATE_ROOT / "artifact_registry.json"
    registry = read_json(registry_path)
    by_path = {row["path"]: row for row in registry.get("artifacts", [])}
    for value in paths:
        path = repo_path(value)
        if not path.is_file():
            continue
        relative = str(path.relative_to(ROOT))
        by_path[relative] = {
            "stage": stage,
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "created_at_utc": now_utc(),
        }
    registry["artifacts"] = sorted(by_path.values(), key=lambda row: row["path"])
    registry["updated_at_utc"] = now_utc()
    write_json(registry_path, registry)


def record_stage(
    stage: str,
    *,
    status: str,
    acceptance_pass: bool,
    output_dir: str | Path,
    started_at_utc: str,
    notes: str = "",
    log_path: str = "",
    exit_code: int = 0,
    next_stage: str | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unknown TRM stage: {stage}")
    state = initialize_state()
    state["stage_status"][stage] = status
    state["current_stage"] = stage
    state["next_stage"] = next_stage
    state["updated_at_utc"] = now_utc()
    state["last_git_commit"] = git_commit()
    target = "completed_stages" if acceptance_pass else "failed_stages"
    if stage not in state[target]:
        state[target].append(stage)
    write_json(STATE_ROOT / "campaign_state.json", state)
    history = STATE_ROOT / "stage_history.csv"
    exists = history.exists() and history.stat().st_size > 0
    with history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_HISTORY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "stage": stage,
                "status": status,
                "started_at_utc": started_at_utc,
                "ended_at_utc": now_utc(),
                "git_commit": git_commit(),
                "output_dir": str(repo_path(output_dir).relative_to(ROOT)),
                "log_path": log_path,
                "exit_code": exit_code,
                "acceptance_pass": acceptance_pass,
                "notes": notes,
            }
        )
    append_log(f"{stage} -> {status}; acceptance={acceptance_pass}; {notes}")
    output = repo_path(output_dir)
    register_artifacts(stage, sorted(path for path in output.rglob("*") if path.is_file()))
    return state


def historical_snapshot() -> dict[str, Any]:
    rows = []
    for name in HISTORICAL_CAMPAIGNS:
        root = ROOT / "runs" / name
        files = sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []
        digest = hashlib.sha256()
        for path in files:
            relative = str(path.relative_to(root))
            digest.update(relative.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(sha256_file(path).encode("ascii"))
        rows.append(
            {
                "campaign": name,
                "path": str(root.relative_to(ROOT)),
                "exists": root.exists(),
                "file_count": len(files),
                "tree_sha256": digest.hexdigest(),
                "access_policy": "read_only_evidence",
            }
        )
    return {"created_at_utc": now_utc(), "historical_campaigns": rows}
