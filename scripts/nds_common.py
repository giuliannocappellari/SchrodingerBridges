#!/usr/bin/env python3
"""Shared state, provenance, and safety helpers for next-direction selection."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "diffusion_editor_next_direction_selection_v1"
CAMPAIGN_ROOT = ROOT / "runs" / CAMPAIGN_ID
STATE_ROOT = CAMPAIGN_ROOT / "autonomous_campaign_v1"
PROTOCOL_ROOT = CAMPAIGN_ROOT / "protocol_v1"

PRIMARY_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
PRIMARY_MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"

PLAN_FILES = (
    "NEXT_DIRECTION_SELECTION_AUTONOMOUS_PLAN.md",
    "RELATION_RESIDUALIZATION_PLAN.md",
    "FISHER_CONSTRAINED_EDITING_PLAN.md",
    "PRIMAL_DUAL_LOCALITY_PLAN.md",
    "SELECTIVE_CONFORMAL_EDITING_PLAN.md",
    "JOINT_ANSWER_SPAN_COUPLING_PLAN.md",
    "INTEGRATED_CANDIDATE_PLAN.md",
    "FINAL_SELECTION_AND_REPORTING_PLAN.md",
)

HISTORICAL_CAMPAIGNS = (
    "counterfact_direction1_v1",
    "counterfact_direction2_bridge_adapter_v1",
    "counterfact_direction3_controller_v1",
    "counterfact_sb_alternatives_campaign_v1",
    "masked_diffusion_memit_sb_positive_result_v1",
    "mask_pattern_sb_publication_confirmation_v1",
    "diffusion_native_causal_partial_state_editor_v1",
    "partial_state_temporal_residual_editor_v1",
)

TRACKS = {
    "N1": "relation_residualized_editing",
    "N2": "fisher_constrained_editing",
    "N3": "primal_dual_locality",
    "N4": "selective_conformal_editing",
    "N5": "joint_answer_span_coupling",
    "N6": "integrated_candidate",
}
MANDATORY_TRACKS = tuple(f"N{index}" for index in range(1, 6))

STAGES = (
    "S0_bootstrap",
    "S0_fresh_manifests",
    "S1_common_baselines",
    "S1_shared_measurements",
    "N1_pilot",
    "N2_pilot",
    "N3_pilot",
    "N4_pilot",
    "N5_pilot",
    "N6_integrated_pilot",
    "S4_fresh_confirmation",
    "S5_final_selection",
    "S6_final_package",
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

TRACK_TERMINAL_STATUSES = {
    "pilot_passed",
    "pilot_failed",
    "confirmation_passed",
    "confirmation_failed",
    "protocol_infeasible",
    "infrastructure_blocked",
    "not_triggered",
}

LOCKED_TOKENS = (
    "analysis_500",
    "final_test_500",
    "final_test_full",
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


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with repo_path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    return os.environ.get("NEXT_DIRECTION_AUTONOMOUS_MODE") == "1"


def initial_campaign_state() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "protocol_version": CAMPAIGN_ID,
        "autonomous_mode": autonomous_enabled(),
        "campaign_status": "running",
        "current_stage": "S0_bootstrap",
        "next_stage": "S0_fresh_manifests",
        "analysis_500_used": False,
        "final_test_used": False,
        "historical_protocols_read_only": True,
        "completed_stages": [],
        "failed_stages": [],
        "rescues_used": {track: 0 for track in TRACKS},
        "stage_status": {stage: "pending" for stage in STAGES},
        "track_status": {
            **{track: "pending" for track in MANDATORY_TRACKS},
            "N6": "conditional_pending",
        },
        "last_git_commit": git_commit(),
        "pod_status": "running",
        "created_at_utc": now_utc(),
        "updated_at_utc": now_utc(),
    }


def initialize_state(*, allow_existing: bool = True) -> dict[str, Any]:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = STATE_ROOT / "campaign_state.json"
    if state_path.exists():
        if not allow_existing:
            raise FileExistsError(state_path)
        state = read_json(state_path)
        if state.get("protocol_version") != CAMPAIGN_ID:
            raise RuntimeError("Persisted campaign protocol does not match active protocol")
        return state
    state = initial_campaign_state()
    write_json(state_path, state)
    write_csv(STATE_ROOT / "stage_history.csv", [], STAGE_HISTORY_FIELDS)
    write_json(
        STATE_ROOT / "track_registry.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "tracks": [
                {
                    "track_id": track,
                    "name": name,
                    "mandatory": track in MANDATORY_TRACKS,
                    "status": state["track_status"][track],
                    "rescue_used": False,
                    "nominated_candidate": None,
                }
                for track, name in TRACKS.items()
            ],
        },
    )
    write_csv(
        STATE_ROOT / "infrastructure_events.csv",
        [],
        ("created_at_utc", "event", "attempt", "status", "details"),
    )
    (STATE_ROOT / "autonomous_log.md").write_text(
        f"# Next-Direction Autonomous Log\n\n- {now_utc()}: campaign initialized.\n",
        encoding="utf-8",
    )
    return state


def append_log(message: str) -> None:
    initialize_state()
    with (STATE_ROOT / "autonomous_log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- {now_utc()}: {message}\n")


def record_infrastructure_event(
    event: str, *, attempt: int, status: str, details: str
) -> None:
    initialize_state()
    path = STATE_ROOT / "infrastructure_events.csv"
    exists = path.exists() and path.stat().st_size > 0
    fields = ("created_at_utc", "event", "attempt", "status", "details")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "created_at_utc": now_utc(),
                "event": event,
                "attempt": int(attempt),
                "status": status,
                "details": details,
            }
        )


def register_artifacts(stage: str, paths: Iterable[str | Path]) -> None:
    initialize_state()
    registry_path = STATE_ROOT / "artifact_registry.json"
    registry = read_json(registry_path) if registry_path.exists() else {
        "campaign_id": CAMPAIGN_ID,
        "artifacts": [],
    }
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
        raise ValueError(f"Unknown next-direction stage: {stage}")
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
        writer = csv.DictWriter(handle, fieldnames=STAGE_HISTORY_FIELDS)
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
                "exit_code": int(exit_code),
                "acceptance_pass": bool(acceptance_pass),
                "notes": notes,
            }
        )
    append_log(f"{stage} -> {status}; acceptance={acceptance_pass}; {notes}")
    output = repo_path(output_dir)
    register_artifacts(stage, sorted(path for path in output.rglob("*") if path.is_file()))
    return state


def update_track(
    track_id: str,
    *,
    status: str,
    rescue_used: bool | None = None,
    nominated_candidate: str | None = None,
    report_path: str | None = None,
    **metadata: Any,
) -> None:
    if track_id not in TRACKS:
        raise ValueError(f"Unknown track: {track_id}")
    state = initialize_state()
    state["track_status"][track_id] = status
    candidate = nominated_candidate or metadata.get("candidate_id")
    state.setdefault("track_details", {}).setdefault(track_id, {}).update(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"output_dir"} and value is not None
        }
    )
    if candidate is not None:
        state["track_details"][track_id]["nominated_candidate"] = candidate
    output_dir = metadata.get("output_dir")
    if report_path is None and output_dir is not None:
        report = repo_path(output_dir) / "report_summary.json"
        if report.is_file():
            report_path = str(report.relative_to(ROOT))
    if rescue_used is not None:
        state["rescues_used"][track_id] = int(bool(rescue_used))
    state["updated_at_utc"] = now_utc()
    write_json(STATE_ROOT / "campaign_state.json", state)
    registry = read_json(STATE_ROOT / "track_registry.json")
    for row in registry["tracks"]:
        if row["track_id"] == track_id:
            row["status"] = status
            if rescue_used is not None:
                row["rescue_used"] = bool(rescue_used)
            if nominated_candidate is not None:
                row["nominated_candidate"] = nominated_candidate
            elif candidate is not None:
                row["nominated_candidate"] = candidate
            if report_path is not None:
                row["report_path"] = report_path
            row.update(
                {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"output_dir", "candidate_id"} and value is not None
                }
            )
            break
    write_json(STATE_ROOT / "track_registry.json", registry)
    append_log(f"{track_id} -> {status}; candidate={candidate}")


def is_historical_locked_path(path: Path) -> bool:
    lower = str(path).casefold()
    return any(token in lower for token in LOCKED_TOKENS)


def forbid_historical_locked_content(path: str | Path) -> None:
    candidate = repo_path(path).resolve()
    for name in HISTORICAL_CAMPAIGNS:
        root = (ROOT / "runs" / name).resolve()
        if root == candidate or root in candidate.parents:
            if is_historical_locked_path(candidate):
                raise PermissionError(f"Historical locked content is forbidden: {candidate}")


def collect_historical_exclusions() -> dict[str, Any]:
    """Collect identity/fingerprint fields without using historical prompt content."""

    case_ids: set[str] = set()
    source_keys: set[str] = set()
    source_fingerprints: set[str] = set()
    fact_fingerprints: set[str] = set()
    fact_target_fingerprints: set[str] = set()
    prompt_fingerprints: set[str] = set()
    audit: list[dict[str, Any]] = []
    candidates: set[Path] = set()
    for name in HISTORICAL_CAMPAIGNS:
        root = ROOT / "runs" / name
        if not root.exists():
            audit.append(
                {
                    "campaign": name,
                    "path": str(root.relative_to(ROOT)),
                    "status": "historical_root_unavailable",
                    "fields_used": "none",
                }
            )
            continue
        for pattern in ("**/protocol/**/*.jsonl", "**/protocol_v1/*.jsonl", "**/*manifest*.jsonl"):
            candidates.update(root.glob(pattern))
    allowed_fields = {
        "case_id",
        "id",
        "source_split",
        "source_dataset_split",
        "source_index",
        "source_fingerprint",
        "fact_fingerprint",
        "fact_target_fingerprint",
        "prompt_fingerprint",
    }
    for path in sorted(candidates):
        rows = 0
        added: Counter[str] = Counter()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    parsed = json.loads(line)
                    row = {key: parsed.get(key) for key in allowed_fields if key in parsed}
                    rows += 1
                    case_id = row.get("case_id") or row.get("id")
                    if case_id is not None:
                        before = len(case_ids)
                        case_ids.add(str(case_id))
                        added["case_ids"] += len(case_ids) - before
                    split = row.get("source_split") or row.get("source_dataset_split")
                    index = row.get("source_index")
                    if split is not None and index is not None:
                        before = len(source_keys)
                        source_keys.add(f"{split}:{int(index)}")
                        added["source_keys"] += len(source_keys) - before
                    for field, target in (
                        ("source_fingerprint", source_fingerprints),
                        ("fact_fingerprint", fact_fingerprints),
                        ("fact_target_fingerprint", fact_target_fingerprints),
                        ("prompt_fingerprint", prompt_fingerprints),
                    ):
                        value = row.get(field)
                        if value:
                            before = len(target)
                            target.add(str(value))
                            added[field] += len(target) - before
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            audit.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "status": "unreadable_identity_manifest",
                    "rows": rows,
                    "error": str(exc),
                    "fields_used": ",".join(sorted(allowed_fields)),
                }
            )
            continue
        audit.append(
            {
                "path": str(path.relative_to(ROOT)),
                "status": "read_identity_fingerprint_fields_only",
                "locked_manifest": is_historical_locked_path(path),
                "rows": rows,
                "case_ids_added": added["case_ids"],
                "source_keys_added": added["source_keys"],
                "source_fingerprints_added": added["source_fingerprint"],
                "fact_fingerprints_added": added["fact_fingerprint"],
                "fact_target_fingerprints_added": added["fact_target_fingerprint"],
                "prompt_fingerprints_added": added["prompt_fingerprint"],
                "fields_used": ",".join(sorted(allowed_fields)),
                "prompt_label_output_metric_fields_used": False,
            }
        )
    return {
        "case_ids": sorted(case_ids),
        "source_keys": sorted(source_keys),
        "source_fingerprints": sorted(source_fingerprints),
        "fact_fingerprints": sorted(fact_fingerprints),
        "fact_target_fingerprints": sorted(fact_target_fingerprints),
        "prompt_fingerprints": sorted(prompt_fingerprints),
        "audit": audit,
        "historical_locked_content_fields_used": False,
    }


def all_mandatory_tracks_terminal() -> bool:
    state = initialize_state()
    return all(state["track_status"].get(track) in TRACK_TERMINAL_STATUSES for track in MANDATORY_TRACKS)
