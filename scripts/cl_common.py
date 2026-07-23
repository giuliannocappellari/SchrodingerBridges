#!/usr/bin/env python3
"""Shared provenance, state, and metric helpers for the continual campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "continual_diffusion_editing_sb_selection_v1"
CAMPAIGN_ROOT = ROOT / "runs" / CAMPAIGN_ID
STATE_ROOT = CAMPAIGN_ROOT / "autonomous_campaign_v1"
PROTOCOL_ROOT = CAMPAIGN_ROOT / "protocol_v1"
SOURCE_AUDIT_ROOT = CAMPAIGN_ROOT / "A0_source_audit_v1"

PRIMARY_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
PRIMARY_MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
SECONDARY_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
SEED = 260723101

HISTORICAL_CAMPAIGNS = (
    "counterfact_direction1_v1",
    "counterfact_direction2_bridge_adapter_v1",
    "counterfact_direction3_controller_v1",
    "counterfact_sb_alternatives_campaign_v1",
    "masked_diffusion_memit_sb_positive_result_v1",
    "mask_pattern_sb_publication_confirmation_v1",
    "diffusion_native_causal_partial_state_editor_v1",
    "partial_state_temporal_residual_editor_v1",
    "diffusion_editor_next_direction_selection_v1",
)
LOCKED_TOKENS = ("analysis_500", "final_test_500", "final_test_full")

TRACKS = {
    "C0": "common_sequential_baselines",
    "C1": "function_preserving_growth",
    "C2": "partial_state_replay",
    "C3": "sparse_routed_memory",
    "C4": "gated_adapter_expansion",
    "C5": "orthogonal_fisher_growth",
    "C6": "functional_replay",
    "C7": "bridge_trajectory_replay",
    "C8": "function_space_sb_consolidation",
    "C9": "dual_memory_consolidation",
    "C10": "parameter_space_sb",
    "C11": "online_laplace",
    "C12": "spectral_repair",
    "C13": "selective_routing",
    "C14": "integrated_candidate",
}
MANDATORY_TRACKS = tuple(f"C{index}" for index in range(10))
CONDITIONAL_TRACKS = tuple(f"C{index}" for index in range(10, 15))

STAGES = (
    "A0_source_audit",
    "A1_campaign_bootstrap",
    "B0_fresh_streams",
    "B1_sequential_harness",
    "C0_common_baselines",
    *(f"C{index}_pilot" for index in range(1, 10)),
    "E_pilot_eligibility",
    "F_fresh_confirmation",
    "G_conditional_tracks",
    "H_final_selection",
    "I_final_package",
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


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    with repo_path(path).open(encoding="utf-8") as handle:
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


def autonomous_enabled() -> bool:
    return os.environ.get("CL_DLLM_AUTONOMOUS_MODE") == "1"


def _initial_state() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "protocol_version": CAMPAIGN_ID,
        "campaign_status": "running",
        "autonomous_mode": autonomous_enabled(),
        "current_stage": "A0_source_audit",
        "next_stage": "A1_campaign_bootstrap",
        "analysis_500_used": False,
        "final_test_used": False,
        "historical_protocols_read_only": True,
        "completed_stages": [],
        "failed_stages": [],
        "stage_status": {stage: "pending" for stage in STAGES},
        "track_status": {
            **{track: "pending" for track in MANDATORY_TRACKS},
            **{track: "not_triggered" for track in CONDITIONAL_TRACKS},
        },
        "rescues_used": {track: 0 for track in TRACKS},
        "pod_status": "not_verified",
        "created_at_utc": now_utc(),
        "updated_at_utc": now_utc(),
        "last_git_commit": git_commit(),
    }


def initialize_state(*, allow_existing: bool = True) -> dict[str, Any]:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = STATE_ROOT / "campaign_state.json"
    if state_path.exists():
        if not allow_existing:
            raise FileExistsError(state_path)
        state = read_json(state_path)
        if state.get("protocol_version") != CAMPAIGN_ID:
            raise RuntimeError("Persisted campaign protocol mismatch")
        return state
    state = _initial_state()
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
    write_json(STATE_ROOT / "artifact_registry.json", {"campaign_id": CAMPAIGN_ID, "artifacts": []})
    (STATE_ROOT / "autonomous_log.md").write_text(
        f"# Continual Diffusion Editing Autonomous Log\n\n- {now_utc()}: campaign initialized.\n",
        encoding="utf-8",
    )
    return state


def append_log(message: str) -> None:
    initialize_state()
    with (STATE_ROOT / "autonomous_log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- {now_utc()}: {message}\n")


def register_artifacts(stage: str, paths: Iterable[str | Path]) -> None:
    initialize_state()
    path = STATE_ROOT / "artifact_registry.json"
    registry = read_json(path)
    indexed = {row["path"]: row for row in registry.get("artifacts", [])}
    for raw in paths:
        artifact = repo_path(raw)
        if artifact.is_file():
            relative = str(artifact.relative_to(ROOT))
            indexed[relative] = {
                "stage": stage,
                "path": relative,
                "sha256": sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
                "registered_at_utc": now_utc(),
            }
    registry["artifacts"] = sorted(indexed.values(), key=lambda row: row["path"])
    registry["updated_at_utc"] = now_utc()
    write_json(path, registry)


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
        raise ValueError(f"Unknown continual stage: {stage}")
    state = initialize_state()
    state["stage_status"][stage] = status
    state["current_stage"] = stage
    state["next_stage"] = next_stage
    state["updated_at_utc"] = now_utc()
    state["last_git_commit"] = git_commit()
    target = "completed_stages" if acceptance_pass else "failed_stages"
    opposite = "failed_stages" if acceptance_pass else "completed_stages"
    state[opposite] = [item for item in state[opposite] if item != stage]
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
                "exit_code": exit_code,
                "acceptance_pass": acceptance_pass,
                "notes": notes,
            }
        )
    append_log(f"{stage} -> {status}; acceptance={acceptance_pass}; {notes}")
    output = repo_path(output_dir)
    if output.exists():
        register_artifacts(stage, (item for item in output.rglob("*") if item.is_file()))
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
    if rescue_used is not None:
        state["rescues_used"][track_id] = int(rescue_used)
    details = state.setdefault("track_details", {}).setdefault(track_id, {})
    details.update({key: value for key, value in metadata.items() if value is not None})
    if nominated_candidate is not None:
        details["nominated_candidate"] = nominated_candidate
    if report_path is not None:
        details["report_path"] = report_path
    state["updated_at_utc"] = now_utc()
    write_json(STATE_ROOT / "campaign_state.json", state)
    registry = read_json(STATE_ROOT / "track_registry.json")
    for row in registry["tracks"]:
        if row["track_id"] == track_id:
            row["status"] = status
            if rescue_used is not None:
                row["rescue_used"] = rescue_used
            if nominated_candidate is not None:
                row["nominated_candidate"] = nominated_candidate
            if report_path is not None:
                row["report_path"] = report_path
            row.update({key: value for key, value in metadata.items() if value is not None})
            break
    write_json(STATE_ROOT / "track_registry.json", registry)
    append_log(f"{track_id} -> {status}; candidate={nominated_candidate}")


def historical_identity_manifest_candidates(root: Path) -> set[Path]:
    candidates: set[Path] = set()
    for pattern in (
        "**/protocol/**/*.jsonl",
        "**/protocol_v1/*.jsonl",
        "**/*_protocol_v1/*.jsonl",
        "**/*manifest*.jsonl",
        "controller_train*.jsonl",
        "controller_val*.jsonl",
        "dev_smoke*.jsonl",
    ):
        candidates.update(path for path in root.glob(pattern) if path.is_file())
    return candidates


def collect_historical_exclusions() -> dict[str, Any]:
    """Read identity/fingerprint fields only from immutable historical manifests."""

    fields = {
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
    sets: dict[str, set[str]] = defaultdict(set)
    audit: list[dict[str, Any]] = []
    for campaign in HISTORICAL_CAMPAIGNS:
        root = ROOT / "runs" / campaign
        if not root.exists():
            audit.append({"campaign": campaign, "status": "unavailable", "fields_used": "none"})
            continue
        for path in sorted(historical_identity_manifest_candidates(root)):
            rows = 0
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        source = json.loads(line)
                        row = {key: source.get(key) for key in fields if key in source}
                        rows += 1
                        case_id = row.get("case_id") or row.get("id")
                        if case_id is not None:
                            sets["case_ids"].add(str(case_id))
                        split = row.get("source_split") or row.get("source_dataset_split")
                        index = row.get("source_index")
                        if split is not None and index is not None:
                            sets["source_keys"].add(f"{split}:{int(index)}")
                        for key in (
                            "source_fingerprint",
                            "fact_fingerprint",
                            "fact_target_fingerprint",
                            "prompt_fingerprint",
                        ):
                            if row.get(key):
                                sets[key + "s"].add(str(row[key]))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                audit.append({
                    "campaign": campaign,
                    "path": str(path.relative_to(ROOT)),
                    "status": "unreadable_identity_manifest",
                    "error": str(exc),
                    "fields_used": ",".join(sorted(fields)),
                })
                continue
            audit.append({
                "campaign": campaign,
                "path": str(path.relative_to(ROOT)),
                "status": "identity_fields_only",
                "rows": rows,
                "fields_used": ",".join(sorted(fields)),
                "prompt_label_output_metric_fields_used": False,
            })
    return {
        **{
            key: sorted(sets.get(key, set()))
            for key in (
                "case_ids",
                "source_keys",
                "source_fingerprints",
                "fact_fingerprints",
                "fact_target_fingerprints",
                "prompt_fingerprints",
            )
        },
        "audit": audit,
        "historical_locked_content_fields_used": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }


def assert_no_locked_path(path: str | Path) -> None:
    lower = str(repo_path(path)).casefold()
    if any(token in lower for token in LOCKED_TOKENS):
        raise PermissionError(f"Locked historical split content is forbidden: {path}")


def sequential_metrics(
    scores: Mapping[int, Mapping[int, float]],
    *,
    pre_edit_scores: Mapping[int, float] | None = None,
) -> dict[str, float]:
    """Compute retention, forgetting, BWT, and FWT from a block accuracy matrix.

    ``scores[t][j]`` is the score on block ``j`` after training through block ``t``.
    """

    if not scores:
        raise ValueError("scores cannot be empty")
    terminal = max(scores)
    seen = sorted(index for index in scores[terminal] if index <= terminal)
    if not seen:
        raise ValueError("terminal row has no seen blocks")
    final_values = [float(scores[terminal][index]) for index in seen]
    forgetting = []
    bwt = []
    for index in seen:
        observed = [float(scores[t][index]) for t in sorted(scores) if t >= index and index in scores[t]]
        if not observed:
            continue
        forgetting.append(max(observed) - observed[-1])
        if index < terminal and index in scores.get(index, {}):
            bwt.append(float(scores[terminal][index]) - float(scores[index][index]))
    fwt = []
    if pre_edit_scores:
        for index in sorted(pre_edit_scores):
            prior_t = index - 1
            if prior_t in scores and index in scores[prior_t]:
                fwt.append(float(scores[prior_t][index]) - float(pre_edit_scores[index]))
    return {
        "average_retention": sum(final_values) / len(final_values),
        "average_forgetting": sum(forgetting) / max(len(forgetting), 1),
        "backward_transfer": sum(bwt) / max(len(bwt), 1),
        "forward_transfer": sum(fwt) / max(len(fwt), 1),
    }


def success_classes(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Apply the frozen A-D pilot rules without post-hoc threshold changes."""

    classes: list[str] = []
    rewrite = float(candidate.get("current_rewrite_exact", 0.0))
    para = float(candidate.get("current_paraphrase_exact", 0.0))
    retention = float(candidate.get("past_retention", 0.0))
    forgetting = float(candidate.get("average_forgetting", 1.0))
    same_tfpr = float(candidate.get("same_subject_tfpr", 1.0))
    near_pass = bool(candidate.get("near_locality_pass", False))
    far_pass = bool(candidate.get("far_locality_pass", False))
    base_loss = float(candidate.get("base_retention_loss_fraction", 1.0))
    malformed = float(candidate.get("malformed_rate", 1.0))
    if (
        rewrite >= 0.80
        and para >= 0.45
        and retention >= 0.75
        and forgetting <= 0.10
        and same_tfpr <= 0.03
        and near_pass
        and far_pass
        and base_loss <= 0.05
        and malformed <= 0.05
    ):
        classes.append("A")
    matched = abs(rewrite - float(baseline.get("current_rewrite_exact", 0.0))) <= 0.03
    base_forgetting = float(baseline.get("average_forgetting", 0.0))
    base_retention = float(baseline.get("past_retention", 0.0))
    base_kl = float(baseline.get("protected_kl", 0.0))
    protected_kl = float(candidate.get("protected_kl", 0.0))
    forgetting_reduction = (
        (base_forgetting - forgetting) / base_forgetting if base_forgetting > 0 else 0.0
    )
    kl_reduction = (base_kl - protected_kl) / base_kl if base_kl > 0 else 0.0
    if matched and (
        forgetting_reduction >= 0.30
        or retention - base_retention >= 0.10
        or kl_reduction >= 0.20
    ) and same_tfpr <= float(baseline.get("same_subject_tfpr", 1.0)) and bool(
        candidate.get("paired_lower_bound_positive", False)
    ):
        classes.append("B")
    sb_baseline = candidate.get("matched_non_sb") or {}
    sb_retention_gain = retention - float(sb_baseline.get("past_retention", retention))
    sb_forgetting = float(sb_baseline.get("average_forgetting", forgetting))
    sb_forgetting_reduction = (
        (sb_forgetting - forgetting) / sb_forgetting if sb_forgetting > 0 else 0.0
    )
    storage_reduction = float(candidate.get("matched_storage_reduction", 0.0))
    if bool(candidate.get("is_sb", False)) and (
        sb_retention_gain >= 0.05
        or sb_forgetting_reduction >= 0.25
        or storage_reduction >= 0.25
    ) and bool(candidate.get("paired_lower_bound_positive", False)):
        classes.append("C")
    storage = float(candidate.get("storage_mb_per_edit", float("inf")))
    overhead = float(candidate.get("inference_overhead_fraction", float("inf")))
    if storage <= 1.0 and overhead <= 0.25 and retention >= base_retention - 0.03:
        classes.append("D")
    return classes
