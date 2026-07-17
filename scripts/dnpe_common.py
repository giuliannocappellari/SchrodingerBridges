#!/usr/bin/env python3
"""Shared state, provenance, and validation helpers for the DNPE campaign."""

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
CAMPAIGN_ID = "diffusion_native_causal_partial_state_editor_v1"
CAMPAIGN_ROOT = ROOT / "runs" / CAMPAIGN_ID
STATE_ROOT = CAMPAIGN_ROOT / "autonomous_campaign_v1"
PROTOCOL_ROOT = CAMPAIGN_ROOT / "protocol_v1"
PRIMARY_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
PRIMARY_MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
SECONDARY_MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
SECONDARY_MODEL_REVISION = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
SECONDARY_FALLBACK_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
SECONDARY_FALLBACK_MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"

HISTORICAL_CAMPAIGNS = (
    "counterfact_direction1_v1",
    "counterfact_direction2_bridge_adapter_v1",
    "counterfact_direction3_controller_v1",
    "counterfact_sb_alternatives_campaign_v1",
    "masked_diffusion_memit_sb_positive_result_v1",
    "mask_pattern_sb_publication_confirmation_v1",
)
LOCKED_SPLIT_NAMES = ("analysis_500", "final_test_500", "final_test_full")
STAGES = (
    "A0_bootstrap",
    "A1_source_audit",
    "A2_fresh_protocol",
    "B1_mdm_memit_reproduction",
    "B2_partial_state_reproduction",
    "B3_alphaedit_style",
    "B4_timerome_style",
    "C1_standard_causal_tracing",
    "C2_temporal_causal_tracing",
    "C3_site_policy_lock",
    "D1_state_banks",
    "D2_target_value_optimization",
    "D3_causal_update",
    "D4_nullspace_main",
    "D5_state_conditioned_rescue",
    "E1_smoke20",
    "E2_pilot100",
    "F1_dev200",
    "F2_scaling",
    "F3_second_backbone",
    "G1_dev_lock",
    "G2_analysis500",
    "G3_final500",
    "H_final_package",
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


def histogram(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def autonomous_enabled() -> bool:
    return (
        os.environ.get("DNPE_AUTONOMOUS_MODE") == "1"
        or os.environ.get("SB_ALT_AUTONOMOUS_MODE") == "1"
    )


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
        return read_json(path)
    state = initial_campaign_state()
    write_json(path, state)
    write_csv(STATE_ROOT / "stage_history.csv", [], STAGE_HISTORY_FIELDS)
    (STATE_ROOT / "autonomous_log.md").write_text(
        f"# DNPE Autonomous Log\n\n- {now_utc()}: campaign initialized.\n",
        encoding="utf-8",
    )
    write_json(
        STATE_ROOT / "cost_state.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "monetary_budget_guard_enabled": False,
            "cost_logging_informational_only": True,
            "pod_hourly_rate_usd": None,
            "estimated_cost_usd": 0.0,
            "stage_costs": [],
            "updated_at_utc": now_utc(),
        },
    )
    return state


def _append_stage_history(row: Mapping[str, Any]) -> None:
    path = STATE_ROOT / "stage_history.csv"
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_HISTORY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in STAGE_HISTORY_FIELDS})


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
        raise ValueError(f"Unknown DNPE stage: {stage}")
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
    _append_stage_history(
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
    with (STATE_ROOT / "autonomous_log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"- {now_utc()}: {stage} -> {status}; acceptance={acceptance_pass}; {notes}\n"
        )
    return state


def immutable_historical_roots() -> tuple[Path, ...]:
    return tuple(ROOT / "runs" / name for name in HISTORICAL_CAMPAIGNS)


def _manifest_kind(path: Path) -> str:
    lower = path.name.lower()
    if "analysis_500" in lower:
        return "analysis_500"
    if "final_test_500" in lower:
        return "final_test_500"
    if "final_test_full" in lower:
        return "final_test_full"
    return "historical_manifest"


def collect_historical_exclusions() -> dict[str, Any]:
    """Collect only identity/fingerprint fields from historical manifest rows.

    Locked split rows are opened solely to collect exclusion identities. Prompt,
    target, label, output, and metric fields are never copied into this campaign.
    """

    case_ids: set[str] = set()
    source_keys: set[str] = set()
    source_fingerprints: set[str] = set()
    fact_fingerprints: set[str] = set()
    fact_target_fingerprints: set[str] = set()
    audit: list[dict[str, Any]] = []
    candidate_files: set[Path] = set()
    for root in immutable_historical_roots():
        if not root.exists():
            continue
        for pattern in ("**/protocol/**/*.jsonl", "**/protocol_v1/*.jsonl", "**/*manifest*.jsonl"):
            candidate_files.update(root.glob(pattern))
    for path in sorted(candidate_files):
        rows = 0
        added = Counter()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows += 1
                    case_id = row.get("case_id") or row.get("id")
                    if case_id is not None:
                        before = len(case_ids)
                        case_ids.add(str(case_id))
                        added["case_ids"] += len(case_ids) - before
                    split = row.get("source_split") or row.get("source_dataset_split")
                    index = row.get("source_index")
                    if split is not None and index is not None:
                        key = f"{split}:{int(index)}"
                        before = len(source_keys)
                        source_keys.add(key)
                        added["source_keys"] += len(source_keys) - before
                    for field, target in (
                        ("source_fingerprint", source_fingerprints),
                        ("fact_fingerprint", fact_fingerprints),
                        ("fact_target_fingerprint", fact_target_fingerprints),
                    ):
                        value = row.get(field)
                        if value:
                            before = len(target)
                            target.add(str(value))
                            added[field] += len(target) - before
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            audit.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "manifest_kind": _manifest_kind(path),
                    "status": "unreadable",
                    "error": str(exc),
                    "rows": rows,
                }
            )
            continue
        audit.append(
            {
                "path": str(path.relative_to(ROOT)),
                "manifest_kind": _manifest_kind(path),
                "status": "read_identity_fields_only",
                "rows": rows,
                "case_ids_added": added["case_ids"],
                "source_keys_added": added["source_keys"],
                "source_fingerprints_added": added["source_fingerprint"],
                "fact_fingerprints_added": added["fact_fingerprint"],
                "fact_target_fingerprints_added": added["fact_target_fingerprint"],
                "prompt_label_output_metric_fields_used": False,
            }
        )
    # Historical exclusion manifests sometimes compact the original split rows.
    # Import identity arrays when present, while never importing prompt/label data.
    for root in immutable_historical_roots():
        if not root.exists():
            continue
        for path in sorted(root.glob("**/historical_exclusion_manifest.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                audit.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "manifest_kind": "historical_exclusion_manifest",
                        "status": "unreadable",
                        "error": str(exc),
                    }
                )
                continue
            imported = {}
            for key, target in (
                ("case_ids", case_ids),
                ("source_keys", source_keys),
                ("source_fingerprints", source_fingerprints),
                ("fact_fingerprints", fact_fingerprints),
                ("fact_target_fingerprints", fact_target_fingerprints),
            ):
                values = payload.get(key, [])
                if not isinstance(values, list):
                    values = []
                before = len(target)
                target.update(map(str, values))
                imported[key] = len(target) - before
            audit.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "manifest_kind": "historical_exclusion_manifest",
                    "status": "read_identity_arrays_only",
                    **{f"{key}_added": value for key, value in imported.items()},
                    "prompt_label_output_metric_fields_used": False,
                }
            )
    return {
        "case_ids": sorted(case_ids),
        "source_keys": sorted(source_keys),
        "source_fingerprints": sorted(source_fingerprints),
        "fact_fingerprints": sorted(fact_fingerprints),
        "fact_target_fingerprints": sorted(fact_target_fingerprints),
        "audit": audit,
    }


def artifact_manifest(root: str | Path) -> list[dict[str, Any]]:
    base = repo_path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(value for value in base.rglob("*") if value.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows
