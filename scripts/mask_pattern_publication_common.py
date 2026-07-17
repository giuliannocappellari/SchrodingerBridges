#!/usr/bin/env python3
"""Shared, leakage-safe utilities for the publication confirmation campaign."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "mask_pattern_sb_publication_confirmation_v1"
CAMPAIGN_ROOT = REPO_ROOT / "runs" / CAMPAIGN_ID
STATE_ROOT = CAMPAIGN_ROOT / "autonomous_campaign_v1"
PROTOCOL_ROOT = CAMPAIGN_ROOT / "protocol_v1"
HISTORICAL_ROOT = REPO_ROOT / "runs" / "masked_diffusion_memit_sb_positive_result_v1"
PRIMARY_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
PRIMARY_MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
SECONDARY_MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
TRACKS = tuple(f"P{index}" for index in range(9))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    payload = "::".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def histogram(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def contextual_target_ids(tokenizer: Any, prompt: str, target: str) -> list[int]:
    """Tokenize a target in prompt context without inspecting evaluation outcomes."""

    prefix = str(prompt).rstrip()
    target_text = str(target).strip()
    combined = list(
        map(int, tokenizer(prefix + " " + target_text, add_special_tokens=False)["input_ids"])
    )
    prefix_ids = list(map(int, tokenizer(prefix, add_special_tokens=False)["input_ids"]))
    if len(combined) > len(prefix_ids) and combined[: len(prefix_ids)] == prefix_ids:
        return combined[len(prefix_ids) :]
    return list(map(int, tokenizer(" " + target_text, add_special_tokens=False)["input_ids"]))


def autonomous_enabled() -> bool:
    return os.environ.get("MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_MODE") == "1"


def initial_campaign_state() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "autonomous_mode": autonomous_enabled(),
        "campaign_status": "running",
        "current_stage": "P0_bootstrap_and_source_audit",
        "next_stage": "P1_partial_state_memit_discrepancy",
        "completed_stages": [],
        "failed_stages": [],
        "rescues_used": {},
        "track_status": {track: "pending" for track in TRACKS},
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "locked_confirmation_opened": False,
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
        return read_json(state_path)
    state = initial_campaign_state()
    write_json(state_path, state)
    write_csv(
        STATE_ROOT / "stage_history.csv",
        [],
        fieldnames=(
            "stage",
            "track",
            "status",
            "started_at_utc",
            "ended_at_utc",
            "git_commit",
            "output_dir",
            "acceptance_pass",
            "notes",
        ),
    )
    (STATE_ROOT / "autonomous_log.md").write_text(
        f"# Autonomous Log\n\n- {now_utc()}: campaign initialized.\n", encoding="utf-8"
    )
    write_json(
        STATE_ROOT / "cost_state.json",
        {
            "informational_only": True,
            "campaign_started_at_utc": now_utc(),
            "hourly_rate_usd": None,
            "estimated_spend_usd": 0.0,
            "stage_costs": [],
        },
    )
    write_json(STATE_ROOT / "artifact_availability.json", {"artifacts": []})
    return state


def _append_csv(path: Path, row: Mapping[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def record_stage(
    *,
    stage: str,
    track: str,
    status: str,
    output_dir: Path,
    acceptance_pass: bool,
    started_at_utc: str,
    notes: str = "",
    next_stage: str | None = None,
) -> dict[str, Any]:
    state = initialize_state()
    ended = now_utc()
    row = {
        "stage": stage,
        "track": track,
        "status": status,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended,
        "git_commit": git_commit(),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "acceptance_pass": acceptance_pass,
        "notes": notes,
    }
    _append_csv(STATE_ROOT / "stage_history.csv", row)
    completed = list(state.get("completed_stages", []))
    failed = list(state.get("failed_stages", []))
    if acceptance_pass and stage not in completed:
        completed.append(stage)
    if acceptance_pass and stage in failed:
        failed.remove(stage)
    if not acceptance_pass and stage not in failed:
        failed.append(stage)
    if not acceptance_pass and stage in completed:
        completed.remove(stage)
    state["completed_stages"] = completed
    state["failed_stages"] = failed
    state.setdefault("track_status", {})[track] = status
    state["current_stage"] = stage
    state["next_stage"] = next_stage or ""
    state["last_git_commit"] = git_commit()
    state["updated_at_utc"] = ended
    write_json(STATE_ROOT / "campaign_state.json", state)
    with (STATE_ROOT / "autonomous_log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"- {ended}: `{stage}` ({track}) -> `{status}`; "
            f"acceptance={acceptance_pass}. {notes}\n"
        )
    return state


def immutable_historical_paths() -> tuple[Path, ...]:
    return (
        REPO_ROOT / "runs" / "counterfact_direction1_v1",
        REPO_ROOT / "runs" / "counterfact_direction2_bridge_adapter_v1",
        REPO_ROOT / "runs" / "counterfact_direction3_controller_v1",
        REPO_ROOT / "runs" / "counterfact_sb_alternatives_campaign_v1",
        HISTORICAL_ROOT,
    )


def collect_historical_kamel_exclusions() -> dict[str, Any]:
    """Read historical KAMEL manifests only for source/fingerprint exclusion."""

    protocol = HISTORICAL_ROOT / "protocol"
    patterns = (
        "kamel_smoke_20_per_length.jsonl",
        "kamel_dev_50_per_length.jsonl",
        "kamel_repro_200_per_length.jsonl",
    )
    source_fingerprints: set[str] = set()
    case_ids: set[str] = set()
    source_keys: set[str] = set()
    fact_fingerprints: set[str] = set()
    fact_target_fingerprints: set[str] = set()
    audit: list[dict[str, Any]] = []
    for name in patterns:
        path = protocol / name
        if not path.exists():
            raise FileNotFoundError(f"Required historical exclusion manifest missing: {path}")
        rows = read_jsonl(path)
        for row in rows:
            case_ids.add(str(row["case_id"]))
            source_fingerprints.add(str(row["source_fingerprint"]))
            source_keys.add(
                f"{row.get('source_split', '')}:{row.get('relation_id', '')}:{row.get('source_index', '')}"
            )
            fact_fingerprints.add(
                stable_hash(
                    row.get("relation_id"),
                    str(row.get("subject", "")).casefold(),
                    str(row.get("target_true", "")).casefold(),
                )
            )
            fact_target_fingerprints.add(
                stable_hash(
                    row.get("relation_id"),
                    str(row.get("subject", "")).casefold(),
                    str(row.get("target_new", "")).casefold(),
                )
            )
        audit.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "row_count": len(rows),
                "fields_used": "case_id,source_split,source_index,source_fingerprint,relation_id,subject,target_true,target_new",
                "prompt_or_metric_fields_used": False,
            }
        )
    return {
        "case_ids": sorted(case_ids),
        "source_fingerprints": sorted(source_fingerprints),
        "source_keys": sorted(source_keys),
        "fact_fingerprints": sorted(fact_fingerprints),
        "fact_target_fingerprints": sorted(fact_target_fingerprints),
        "audit": audit,
    }
