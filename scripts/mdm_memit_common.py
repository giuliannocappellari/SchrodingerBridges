#!/usr/bin/env python3
"""Shared state and provenance helpers for the MDM-MEMIT campaign."""

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
CAMPAIGN_ID = "masked_diffusion_memit_sb_positive_result_v1"
CAMPAIGN_ROOT = Path("runs") / CAMPAIGN_ID
STATE_ROOT = CAMPAIGN_ROOT / "autonomous_campaign_v1"
PROTOCOL_ROOT = CAMPAIGN_ROOT / "protocol"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"

TRACKS = {
    "M1": "M1_mdm_memit_reproduction_v1",
    "M2": "M2_partial_mask_memit_v1",
    "M3": "M3_schrodinger_regularized_memit_v1",
    "M4": "M4_mask_pattern_sb_v1",
    "F1": "F1_adaptive_edit_memory_v1",
    "F2": "F2_toy_text_csbm_v1",
}

HISTORICAL_ROOTS = (
    Path("runs/counterfact_direction1_v1"),
    Path("runs/counterfact_direction2_bridge_adapter_v1"),
    Path("runs/counterfact_direction3_controller_v1"),
    Path("runs/counterfact_sb_alternatives_campaign_v1"),
)

LOCKED_NAMES = {
    "analysis_500",
    "final_test_500",
    "final_test_full",
}

STATE_HISTORY_FIELDS = (
    "stage",
    "track",
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


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def read_json(path: str | Path) -> Any:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
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
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
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
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    with full.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_csv(path: str | Path, row: Mapping[str, Any]) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    exists = full.exists() and full.stat().st_size > 0
    with full.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(STATE_HISTORY_FIELDS), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in STATE_HISTORY_FIELDS})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()


def append_log(message: str) -> None:
    path = repo_path(STATE_ROOT / "autonomous_log.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n- {now_utc()}: {message}\n")


def require_autonomous_environment() -> dict[str, Any]:
    enabled = (
        os.environ.get("MDM_MEMIT_SB_AUTONOMOUS_MODE") == "1"
        or os.environ.get("SB_ALT_AUTONOMOUS_MODE") == "1"
    )
    if not enabled:
        raise RuntimeError("MDM_MEMIT_SB_AUTONOMOUS_MODE must equal 1")
    names = (
        "RUNPOD_POD_ID",
        "RUNPOD_SSH_KEY",
        "RUNPOD_SSH_USER",
        "RUNPOD_SSH_HOST",
        "RUNPOD_SSH_PORT",
        "REMOTE_REPO_DIR",
    )
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing RunPod environment variables: {missing}")
    return {
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "ssh_host": os.environ["RUNPOD_SSH_HOST"],
        "ssh_port": int(os.environ["RUNPOD_SSH_PORT"]),
        "remote_repo_dir": os.environ["REMOTE_REPO_DIR"],
        "hourly_rate_usd": float(os.environ.get("RUNPOD_HOURLY_RATE_USD", "0") or 0),
        "max_infra_retries": int(
            os.environ.get("MDM_MEMIT_SB_MAX_INFRA_RETRIES")
            or os.environ.get("SB_ALT_AUTONOMOUS_MAX_INFRA_RETRIES", "3")
        ),
    }


def _manifest_kind(path: Path) -> str | None:
    stem = path.stem.lower()
    if any(name in stem for name in LOCKED_NAMES):
        return "locked"
    manifest_markers = ("manifest", "split", "protocol", "test50", "smoke", "dev", "val", "train")
    return "historical" if any(marker in stem for marker in manifest_markers) else None


def collect_historical_exclusions() -> dict[str, Any]:
    """Read only ID/source-coordinate fields from historical manifest-like files."""

    case_ids: set[str] = set()
    source_keys: set[str] = set()
    audit: list[dict[str, Any]] = []
    allowed = {
        "case_id",
        "id",
        "source_split",
        "source_dataset_split",
        "source_index",
        "counterfact_raw_case_id",
    }
    for root_rel in HISTORICAL_ROOTS:
        root = repo_path(root_rel)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            kind = _manifest_kind(path)
            if kind is None:
                continue
            row_count = 0
            parsed = False
            try:
                if path.suffix == ".jsonl":
                    iterator: Iterable[Any] = (
                        json.loads(line)
                        for line in path.open("r", encoding="utf-8")
                        if line.strip()
                    )
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        iterator = payload
                    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                        iterator = payload["rows"]
                    else:
                        iterator = []
                for raw in iterator:
                    if not isinstance(raw, dict):
                        continue
                    projected = {key: raw.get(key) for key in allowed}
                    case_id = projected.get("case_id") or projected.get("id")
                    split = projected.get("source_dataset_split") or projected.get("source_split")
                    index = projected.get("source_index")
                    if case_id not in {None, ""}:
                        case_ids.add(str(case_id))
                    if split not in {None, ""} and index is not None:
                        source_keys.add(f"{split}:{int(index)}")
                    row_count += 1
                parsed = True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                parsed = False
            audit.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "kind": kind,
                    "sha256": sha256_file(path),
                    "rows_projected": row_count,
                    "parsed": parsed,
                    "fields_used": sorted(allowed),
                    "prompt_label_output_metric_fields_used": False,
                }
            )
    return {
        "case_ids": sorted(case_ids),
        "source_keys": sorted(source_keys),
        "audit": audit,
        "prompt_label_output_metric_fields_used": False,
    }


def initialize_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    state_path = repo_path(STATE_ROOT / "campaign_state.json")
    if state_path.exists():
        state = read_json(state_path)
        if state.get("campaign_id") != CAMPAIGN_ID:
            raise RuntimeError("Persisted state belongs to a different campaign")
        return state

    started = now_utc()
    state = {
        "campaign_id": CAMPAIGN_ID,
        "autonomous_mode": True,
        "campaign_status": "running",
        "current_stage": "A1_campaign_bootstrap",
        "next_stage": "A2_source_audit",
        "completed_stages": [],
        "failed_stages": [],
        "rescues_used": {},
        "track_status": {
            "M1": "pending",
            "M2": "pending",
            "M3": "pending",
            "M4": "pending",
            "F1": "conditional",
            "F2": "conditional",
        },
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "last_git_commit": git_commit(),
        "pod_status": "running",
        "pod_id": config["pod_id"],
        "started_at_utc": started,
        "started_epoch": time.time(),
        "updated_at_utc": started,
    }
    write_json(STATE_ROOT / "campaign_state.json", state)
    write_json(
        STATE_ROOT / "cost_state.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "informational_only": True,
            "budget_guard_enabled": False,
            "hourly_rate_usd": config["hourly_rate_usd"],
            "started_at_utc": started,
            "started_epoch": time.time(),
            "estimated_spend_usd": 0.0,
            "stage_costs": [],
        },
    )
    write_json(
        STATE_ROOT / "artifact_availability.json",
        {"campaign_id": CAMPAIGN_ID, "artifacts": [], "updated_at_utc": started},
    )
    append_log("Campaign initialized; historical campaigns remain immutable.")
    return state


def update_campaign_state(**updates: Any) -> dict[str, Any]:
    state = read_json(STATE_ROOT / "campaign_state.json")
    state.update(updates)
    state["updated_at_utc"] = now_utc()
    state["last_git_commit"] = git_commit()
    write_json(STATE_ROOT / "campaign_state.json", state)
    return state


def record_stage(
    *,
    stage: str,
    track: str = "campaign",
    status: str,
    output_dir: str | Path,
    acceptance_pass: bool,
    started_at_utc: str,
    log_path: str = "",
    exit_code: int = 0,
    notes: str = "",
) -> None:
    append_csv(
        STATE_ROOT / "stage_history.csv",
        {
            "stage": stage,
            "track": track,
            "status": status,
            "started_at_utc": started_at_utc,
            "ended_at_utc": now_utc(),
            "git_commit": git_commit(),
            "output_dir": str(output_dir),
            "log_path": log_path,
            "exit_code": exit_code,
            "acceptance_pass": acceptance_pass,
            "notes": notes,
        },
    )
    append_log(f"{stage}: {status}; acceptance_pass={acceptance_pass}. {notes}")


def histogram(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))
