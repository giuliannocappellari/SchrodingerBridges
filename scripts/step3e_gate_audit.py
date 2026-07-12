#!/usr/bin/env python3
"""Local Step 3E gate audit utilities.

This script intentionally does not load LLaDA and does not run decoding. It
recomputes hybrid-gate features from existing dev artifacts using the runtime
gate implementation, then writes Step 3E.2 and Step 3E.3 reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llada_counterfact_protocol import PROTOCOL_VERSION
from llada_runtime_editor_eval import (
    RolloutConfig,
    gate_should_activate,
    hybrid_relation_gate_scores,
    load_raw_jsonl,
    normalize_gate_text,
    raw_relation_id,
    raw_subject,
    relation_content_text,
    relation_text_for_record,
)


RUN_ROOT = Path("runs/counterfact_direction1_v1")
DEV_PATH = RUN_ROOT / "protocol/dev_tune_200.jsonl"
STRESS_PATH = RUN_ROOT / "same_subject_stress_inputs/dev_tune_200_same_subject_stress.jsonl"
REPLAY_DIR = RUN_ROOT / "dev_tune_200_hybrid_gate_replay_v1"
ACTUAL_REPORT_DIR = RUN_ROOT / "dev_tune_200_hybrid_gate_decode_v1"
PARITY_OUT_DIR = RUN_ROOT / "dev_tune_200_hybrid_gate_parity_audit_v1"
GRID_OUT_DIR = RUN_ROOT / "dev_tune_200_actual_gate_activation_grid_v1"

GATE_ID = "hybrid_or_rel0.45_bank0.10"
DEFAULT_REL_THRESHOLD = 0.45
DEFAULT_BANK_THRESHOLD = 0.10
RELATION_BANK_SOURCE = "frozen_dev_tune_200_rewrite_templates"

BUCKET_ORDER = [
    "rewrite",
    "declarative_paraphrases",
    "qa_format_generalization",
    "near_locality",
    "far_locality",
    "same_subject_template",
    "generation",
]

ACTUAL_RUN_DIRS = [
    RUN_ROOT / "dev_tune_200_hybrid_decode_prompt_memory",
    RUN_ROOT / "dev_tune_200_hybrid_decode_myopic_gs175",
    RUN_ROOT / "dev_tune_200_hybrid_decode_myopic_gs200",
    RUN_ROOT / "dev_tune_200_hybrid_decode_mc_bridge_gs175",
    RUN_ROOT / "dev_tune_200_hybrid_decode_mc_bridge_gs200",
    RUN_ROOT / "dev_tune_200_hybrid_decode_no_rollout_gs200",
    RUN_ROOT / "dev_tune_200_hybrid_decode_stress_prompt_memory",
    RUN_ROOT / "dev_tune_200_hybrid_decode_stress_myopic_gs175",
    RUN_ROOT / "dev_tune_200_hybrid_decode_stress_myopic_gs200",
    RUN_ROOT / "dev_tune_200_hybrid_decode_stress_mc_bridge_gs175",
    RUN_ROOT / "dev_tune_200_hybrid_decode_stress_mc_bridge_gs200",
    RUN_ROOT / "dev_tune_200_hybrid_decode_stress_no_rollout_gs200",
]

GRID_CONFIGS = [
    ("hybrid_or_rel0.45_bank0.15", "hybrid_relation_or", 0.45, 0.15),
    ("hybrid_or_rel0.45_bank0.20", "hybrid_relation_or", 0.45, 0.20),
    ("hybrid_or_rel0.45_bank0.25", "hybrid_relation_or", 0.45, 0.25),
    ("hybrid_or_rel0.45_bank0.30", "hybrid_relation_or", 0.45, 0.30),
    ("hybrid_or_rel0.55_bank0.15", "hybrid_relation_or", 0.55, 0.15),
    ("hybrid_or_rel0.55_bank0.20", "hybrid_relation_or", 0.55, 0.20),
    ("hybrid_or_rel0.55_bank0.25", "hybrid_relation_or", 0.55, 0.25),
    ("hybrid_or_rel0.55_bank0.30", "hybrid_relation_or", 0.55, 0.30),
    ("hybrid_and_rel0.30_bank0.10", "hybrid_relation_and", 0.30, 0.10),
    ("hybrid_and_rel0.30_bank0.15", "hybrid_relation_and", 0.30, 0.15),
    ("hybrid_and_rel0.40_bank0.10", "hybrid_relation_and", 0.40, 0.10),
    ("hybrid_and_rel0.40_bank0.15", "hybrid_relation_and", 0.40, 0.15),
]


def repo_path(path: Path) -> Path:
    return ROOT / path


def read_csv(path: Path) -> List[Dict[str, str]]:
    with repo_path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with full.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def assert_dev_only_path(path: Path) -> None:
    path_text = str(path)
    disallowed = ["analysis_500", "final_test_500", "final_test_full"]
    if any(token in path_text for token in disallowed):
        raise AssertionError(f"Disallowed locked split path for local gate audit: {path}")


def assert_dev_only_report(path: Path) -> None:
    assert_dev_only_path(path)
    data = json.loads(repo_path(path).read_text(encoding="utf-8"))
    assert data.get("analysis_500_used") is False, f"{path}: analysis_500_used is not false"
    assert data.get("final_test_used") is False, f"{path}: final_test_used is not false"


def prepare_output_dir(path: Path, allow_overwrite: bool) -> None:
    full = repo_path(path)
    if full.exists() and not allow_overwrite:
        raise FileExistsError(f"Output directory already exists: {full}. Use --allow-overwrite 1 to replace files.")
    full.mkdir(parents=True, exist_ok=True)


def load_edit_maps() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    assert_dev_only_path(DEV_PATH)
    assert_dev_only_path(STRESS_PATH)
    dev_rows = load_raw_jsonl(str(repo_path(DEV_PATH)))
    stress_rows = load_raw_jsonl(str(repo_path(STRESS_PATH)))
    for row in dev_rows + stress_rows:
        assert row.get("protocol_version") == PROTOCOL_VERSION
        assert row.get("split_role") == "dev_tune_200"
    dev = {str(row["id"]): row for row in dev_rows}
    stress = {str(row["id"]): row for row in stress_rows}
    return dev, stress


def resolve_raw_edit(edit_id: str, dev: Mapping[str, Dict[str, Any]], stress: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    if edit_id in stress:
        return stress[edit_id]
    if edit_id in dev:
        return dev[edit_id]
    if edit_id.endswith("_same_subject_stress") and edit_id in stress:
        return stress[edit_id]
    raise KeyError(f"Cannot resolve raw edit for edit_id={edit_id}")


def canonical_bucket(row: Mapping[str, Any]) -> str:
    prompt_id = str(row.get("prompt_id") or row.get("prompt_uid") or row.get("case_id") or "")
    edit_id = str(row.get("edit_id") or "")
    if "_same_subject_template_" in prompt_id:
        return "same_subject_template"
    if "_generation_" in prompt_id and edit_id.endswith("_same_subject_stress"):
        return "generation"
    return str(row.get("bucket") or "")


def prompt_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    edit_id = str(row.get("edit_id") or "")
    bucket = canonical_bucket(row)
    prompt_id = str(row.get("prompt_id") or row.get("prompt_uid") or row.get("case_id") or "")
    return edit_id, bucket, prompt_id


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def float_or_zero(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def load_replay_prompt_rows() -> Dict[Tuple[str, str, str], Dict[str, str]]:
    path = REPLAY_DIR / "gate_replay_dataset.csv"
    assert_dev_only_report(REPLAY_DIR / "report_summary.json")
    rows = read_csv(path)
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        key = prompt_key(row)
        if key in out:
            existing = out[key]
            if existing.get("prompt") != row.get("prompt"):
                raise AssertionError(f"Prompt collision for {key}")
            continue
        out[key] = row
    expected_buckets = set(BUCKET_ORDER)
    observed = {key[1] for key in out}
    missing = expected_buckets - observed
    if missing:
        raise AssertionError(f"Missing replay buckets: {sorted(missing)}")
    return out


def make_cfg(gate_mode: str, rel_threshold: float, bank_threshold: float) -> RolloutConfig:
    return RolloutConfig(
        steps=4,
        bridge_topk=4,
        mc_rollouts=0,
        guidance_scale=1.0,
        reward_mode="soft_overlap",
        reward_beta=6.0,
        target_logit_bias=0.0,
        gate_mode=gate_mode,
        temperature=1.0,
        relation_sim_rewrite_threshold=rel_threshold,
        relation_sim_bank_threshold=bank_threshold,
        relation_bank_path=str(repo_path(DEV_PATH)),
        relation_bank_source=RELATION_BANK_SOURCE,
    )


def recompute_features(row: Mapping[str, Any], raw_edit: Dict[str, Any], cfg: RolloutConfig) -> Dict[str, Any]:
    prompt = str(row.get("prompt") or row.get("prompt_text") or row.get("rendered_prompt") or "")
    subject_match, rewrite_sim, bank_sim = hybrid_relation_gate_scores(raw_edit, prompt, cfg)
    prompt_relation = relation_content_text(
        prompt,
        subject=raw_subject(raw_edit),
        target_new=str(raw_edit.get("target") or ""),
        target_true=str(raw_edit.get("old_target") or ""),
    )
    return {
        "subject": raw_subject(raw_edit),
        "relation_id": raw_relation_id(raw_edit),
        "subject_match": bool(subject_match),
        "relation_sim_rewrite": float(rewrite_sim),
        "relation_sim_bank": float(bank_sim),
        "prompt_relation_text": prompt_relation,
        "rewrite_relation_text": relation_text_for_record(raw_edit),
    }


def gate_active_from_features(
    *,
    gate_mode: str,
    subject_match: bool,
    rewrite_sim: float,
    bank_sim: float,
    rel_threshold: float,
    bank_threshold: float,
) -> bool:
    if not subject_match:
        return False
    if gate_mode == "hybrid_relation_or":
        return rewrite_sim >= rel_threshold or bank_sim >= bank_threshold
    if gate_mode == "hybrid_relation_and":
        return rewrite_sim >= rel_threshold and bank_sim >= bank_threshold
    raise ValueError(f"Unsupported grid gate mode: {gate_mode}")


def load_actual_activation_by_key() -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    assert_dev_only_report(ACTUAL_REPORT_DIR / "report_summary.json")
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    labels: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for run_dir in ACTUAL_RUN_DIRS:
        rows_path = run_dir / "per_case_results.jsonl"
        if not repo_path(rows_path).exists():
            raise FileNotFoundError(f"Missing actual decode rows: {repo_path(rows_path)}")
        cfg_path = run_dir / "run_config.json"
        if repo_path(cfg_path).exists():
            cfg = json.loads(repo_path(cfg_path).read_text(encoding="utf-8"))
            assert cfg.get("split_role") == "dev_tune_200"
            assert cfg.get("analysis_500_used") is False
            assert cfg.get("final_test_used") is False
        for row in load_raw_jsonl(str(repo_path(rows_path))):
            assert row.get("protocol_version") == PROTOCOL_VERSION
            assert row.get("split_role") == "dev_tune_200"
            key = prompt_key(row)
            grouped[key].append(float_or_zero(row.get("gate_activation_rate")))
            labels[key].append(str(row.get("method_variant") or row.get("method") or run_dir.name))
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, values in grouped.items():
        out[key] = {
            "gate_activation_mean": sum(values) / len(values),
            "gate_activation_min": min(values),
            "gate_activation_max": max(values),
            "num_method_rows": len(values),
            "method_labels": sorted(set(labels[key])),
        }
    return out


def parity_audit(allow_overwrite: bool) -> Dict[str, Any]:
    prepare_output_dir(PARITY_OUT_DIR, allow_overwrite)
    dev, stress = load_edit_maps()
    replay_rows = load_replay_prompt_rows()
    actual_by_key = load_actual_activation_by_key()
    cfg = make_cfg("hybrid_relation_or", DEFAULT_REL_THRESHOLD, DEFAULT_BANK_THRESHOLD)

    feature_rows: List[Dict[str, Any]] = []
    missing_actual: List[Tuple[str, str, str]] = []
    for key, replay_row in sorted(replay_rows.items()):
        edit_id, bucket, prompt_id = key
        raw_edit = resolve_raw_edit(edit_id, dev, stress)
        prompt = replay_row.get("prompt", "")
        replay_subject_match = boolish(replay_row.get("subject_match"))
        replay_rewrite_sim = float_or_zero(replay_row.get("relation_sim_rewrite"))
        replay_bank_sim = float_or_zero(replay_row.get("relation_sim_bank"))
        gate_active_replay = gate_active_from_features(
            gate_mode="hybrid_relation_or",
            subject_match=replay_subject_match,
            rewrite_sim=replay_rewrite_sim,
            bank_sim=replay_bank_sim,
            rel_threshold=DEFAULT_REL_THRESHOLD,
            bank_threshold=DEFAULT_BANK_THRESHOLD,
        )
        recomputed = recompute_features(replay_row, raw_edit, cfg)
        gate_active_recomputed = gate_should_activate(raw_edit, prompt, "hybrid_relation_or", cfg)
        actual = actual_by_key.get(key)
        if actual is None:
            missing_actual.append(key)
            actual_mean = math.nan
            gate_active_actual = False
            actual_min = math.nan
            actual_max = math.nan
            method_row_count = 0
        else:
            actual_mean = float(actual["gate_activation_mean"])
            gate_active_actual = actual_mean >= 0.5
            actual_min = float(actual["gate_activation_min"])
            actual_max = float(actual["gate_activation_max"])
            method_row_count = int(actual["num_method_rows"])
        feature_rows.append(
            {
                "edit_id": edit_id,
                "bucket": bucket,
                "prompt_id": prompt_id,
                "prompt_sha1": sha1_text(prompt),
                "normalized_prompt_sha1": sha1_text(normalize_gate_text(prompt)),
                "prompt": prompt,
                "subject": recomputed["subject"],
                "relation_id": recomputed["relation_id"],
                "subject_match": recomputed["subject_match"],
                "relation_sim_rewrite": recomputed["relation_sim_rewrite"],
                "relation_sim_bank": recomputed["relation_sim_bank"],
                "threshold_rel": DEFAULT_REL_THRESHOLD,
                "threshold_bank": DEFAULT_BANK_THRESHOLD,
                "gate_active_replay": gate_active_replay,
                "gate_active_recomputed": gate_active_recomputed,
                "gate_active_actual": gate_active_actual,
                "actual_gate_activation_mean": actual_mean,
                "actual_gate_activation_min": actual_min,
                "actual_gate_activation_max": actual_max,
                "actual_method_row_count": method_row_count,
                "replay_subject_match": replay_subject_match,
                "replay_relation_sim_rewrite": replay_rewrite_sim,
                "replay_relation_sim_bank": replay_bank_sim,
                "prompt_relation_text": recomputed["prompt_relation_text"],
                "rewrite_relation_text": recomputed["rewrite_relation_text"],
                "replay_actual_mismatch": gate_active_replay != gate_active_actual,
                "recomputed_actual_mismatch": gate_active_recomputed != gate_active_actual,
            }
        )

    if missing_actual:
        raise AssertionError(f"Missing actual activation for {len(missing_actual)} prompt rows; first={missing_actual[:5]}")

    summary_rows = activation_parity_summary(feature_rows)
    threshold_rows = threshold_diagnostics(feature_rows)
    mismatch_rows = [row for row in feature_rows if row["replay_actual_mismatch"]][:200]

    max_replay_drift = max(abs(float(row["replay_actual_drift"])) for row in summary_rows)
    max_recomputed_drift = max(abs(float(row["recomputed_actual_drift"])) for row in summary_rows)
    acceptance_pass = max_replay_drift <= 0.01
    bug_identified = (not acceptance_pass) and max_recomputed_drift <= 0.01
    status = "pass" if acceptance_pass else "bug_identified" if bug_identified else "fail"

    write_csv(PARITY_OUT_DIR / "gate_feature_parity.csv", feature_rows)
    write_csv(PARITY_OUT_DIR / "gate_activation_parity_summary.csv", summary_rows)
    write_csv(PARITY_OUT_DIR / "gate_mismatch_samples.csv", mismatch_rows)
    write_csv(PARITY_OUT_DIR / "gate_threshold_diagnostics.csv", threshold_rows)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "split_role": "dev_tune_200",
        "analysis_500_used": False,
        "final_test_used": False,
        "stage": "Step 3E.2 Hybrid Gate Parity Audit",
        "git_commit": git_commit(),
        "gate_id": GATE_ID,
        "gate_mode": "hybrid_relation_or",
        "relation_sim_rewrite_threshold": DEFAULT_REL_THRESHOLD,
        "relation_sim_bank_threshold": DEFAULT_BANK_THRESHOLD,
        "relation_bank_path": str(DEV_PATH),
        "relation_bank_source": RELATION_BANK_SOURCE,
        "num_prompt_rows": len(feature_rows),
        "max_replay_actual_activation_drift": max_replay_drift,
        "max_recomputed_actual_activation_drift": max_recomputed_drift,
        "acceptance_criterion": "activation drift <= 0.01 per bucket or concrete bug identified",
        "acceptance_pass": acceptance_pass,
        "bug_identified": bug_identified,
        "status": status,
        "bug_summary": (
            "Step 3E.0 replay features disagree with actual runtime features, but recomputed runtime features "
            "match actual activation; replay feature extraction is the likely bug."
            if bug_identified
            else ""
        ),
        "artifacts": {
            "gate_feature_parity": str(PARITY_OUT_DIR / "gate_feature_parity.csv"),
            "gate_activation_parity_summary": str(PARITY_OUT_DIR / "gate_activation_parity_summary.csv"),
            "gate_mismatch_samples": str(PARITY_OUT_DIR / "gate_mismatch_samples.csv"),
            "gate_threshold_diagnostics": str(PARITY_OUT_DIR / "gate_threshold_diagnostics.csv"),
        },
    }
    write_json(PARITY_OUT_DIR / "report_summary.json", report)
    return report


def activation_parity_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        group = [row for row in rows if row["bucket"] == bucket]
        if not group:
            continue
        replay = sum(float(row["gate_active_replay"]) for row in group) / len(group)
        recomputed = sum(float(row["gate_active_recomputed"]) for row in group) / len(group)
        actual = sum(float(row["gate_active_actual"]) for row in group) / len(group)
        replay_mismatch = sum(bool(row["replay_actual_mismatch"]) for row in group)
        recomputed_mismatch = sum(bool(row["recomputed_actual_mismatch"]) for row in group)
        out.append(
            {
                "bucket": bucket,
                "num_prompt_rows": len(group),
                "num_edits": len({row["edit_id"] for row in group}),
                "replay_activation_rate": replay,
                "recomputed_activation_rate": recomputed,
                "actual_activation_rate": actual,
                "replay_actual_drift": replay - actual,
                "recomputed_actual_drift": recomputed - actual,
                "replay_actual_mismatch_count": replay_mismatch,
                "recomputed_actual_mismatch_count": recomputed_mismatch,
                "activation_drift_pass": abs(replay - actual) <= 0.01,
                "recomputed_matches_actual": abs(recomputed - actual) <= 0.01,
            }
        )
    return out


def threshold_diagnostics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        group = [row for row in rows if row["bucket"] == bucket]
        if not group:
            continue
        rewrite_scores = [float(row["relation_sim_rewrite"]) for row in group]
        bank_scores = [float(row["relation_sim_bank"]) for row in group]
        out.append(
            {
                "bucket": bucket,
                "num_prompt_rows": len(group),
                "subject_match_rate": sum(float(row["subject_match"]) for row in group) / len(group),
                "rewrite_branch_rate": sum(score >= DEFAULT_REL_THRESHOLD for score in rewrite_scores) / len(group),
                "bank_branch_rate": sum(score >= DEFAULT_BANK_THRESHOLD for score in bank_scores) / len(group),
                "rewrite_sim_mean": statistics.mean(rewrite_scores),
                "rewrite_sim_p50": statistics.median(rewrite_scores),
                "rewrite_sim_max": max(rewrite_scores),
                "bank_sim_mean": statistics.mean(bank_scores),
                "bank_sim_p50": statistics.median(bank_scores),
                "bank_sim_max": max(bank_scores),
            }
        )
    return out


def activation_grid(allow_overwrite: bool) -> Dict[str, Any]:
    prepare_output_dir(GRID_OUT_DIR, allow_overwrite)
    dev, stress = load_edit_maps()
    replay_rows = load_replay_prompt_rows()

    feature_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, row in replay_rows.items():
        raw_edit = resolve_raw_edit(key[0], dev, stress)
        cfg = make_cfg("hybrid_relation_or", DEFAULT_REL_THRESHOLD, DEFAULT_BANK_THRESHOLD)
        feature_cache[key] = {
            **row,
            **recompute_features(row, raw_edit, cfg),
        }

    grid_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []

    for gate_id, gate_mode, rel_threshold, bank_threshold in GRID_CONFIGS:
        prompt_results: List[Dict[str, Any]] = []
        for key, features in sorted(feature_cache.items()):
            active = gate_active_from_features(
                gate_mode=gate_mode,
                subject_match=bool(features["subject_match"]),
                rewrite_sim=float(features["relation_sim_rewrite"]),
                bank_sim=float(features["relation_sim_bank"]),
                rel_threshold=rel_threshold,
                bank_threshold=bank_threshold,
            )
            prompt_results.append(
                {
                    "gate_id": gate_id,
                    "gate_mode": gate_mode,
                    "threshold_rel": rel_threshold,
                    "threshold_bank": bank_threshold,
                    "edit_id": key[0],
                    "bucket": key[1],
                    "prompt_id": key[2],
                    "prompt": features.get("prompt", ""),
                    "subject": features["subject"],
                    "relation_id": features["relation_id"],
                    "subject_match": features["subject_match"],
                    "relation_sim_rewrite": features["relation_sim_rewrite"],
                    "relation_sim_bank": features["relation_sim_bank"],
                    "gate_active": active,
                }
            )
        bucket_summary = summarize_grid_buckets(prompt_results)
        grid_rows.extend(bucket_summary)
        candidate = candidate_row_from_bucket_summary(gate_id, gate_mode, rel_threshold, bank_threshold, bucket_summary)
        candidate_rows.append(candidate)
        sample_rows.extend(grid_sample_rows(prompt_results, gate_id))
        threshold_rows.extend(grid_threshold_diagnostics(prompt_results, gate_id))

    best_rows = [row for row in candidate_rows if row["gate_candidate_pass"]]
    write_csv(GRID_OUT_DIR / "gate_activation_grid.csv", grid_rows)
    write_csv(GRID_OUT_DIR / "best_actual_gate_candidates.csv", best_rows, fieldnames=list(candidate_rows[0].keys()))
    write_csv(GRID_OUT_DIR / "gate_threshold_diagnostics.csv", threshold_rows)
    write_csv(GRID_OUT_DIR / "gate_activation_samples.csv", sample_rows)
    write_activation_png(GRID_OUT_DIR / "gate_activation_plot.png", candidate_rows)

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "split_role": "dev_tune_200",
        "analysis_500_used": False,
        "final_test_used": False,
        "stage": "Step 3E.3 Actual-Gate Activation Grid",
        "git_commit": git_commit(),
        "relation_bank_path": str(DEV_PATH),
        "relation_bank_source": RELATION_BANK_SOURCE,
        "num_gate_configs": len(GRID_CONFIGS),
        "num_prompt_rows": len(feature_cache),
        "num_passing_gates": len(best_rows),
        "acceptance_criteria": {
            "rewrite_activation_min": 0.95,
            "declarative_paraphrases_activation_min": 0.85,
            "same_subject_template_activation_max": 0.05,
            "generation_activation_max": 0.10,
            "near_locality_activation_max": 0.02,
            "far_locality_activation_max": 0.0,
        },
        "status": "pass" if best_rows else "no_gate_passed",
        "best_gate_ids": [row["gate_id"] for row in best_rows],
        "artifacts": {
            "gate_activation_grid": str(GRID_OUT_DIR / "gate_activation_grid.csv"),
            "best_actual_gate_candidates": str(GRID_OUT_DIR / "best_actual_gate_candidates.csv"),
            "gate_threshold_diagnostics": str(GRID_OUT_DIR / "gate_threshold_diagnostics.csv"),
            "gate_activation_samples": str(GRID_OUT_DIR / "gate_activation_samples.csv"),
            "gate_activation_plot": str(GRID_OUT_DIR / "gate_activation_plot.png"),
        },
    }
    write_json(GRID_OUT_DIR / "report_summary.json", report)
    return report


def summarize_grid_buckets(prompt_results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    first = prompt_results[0]
    for bucket in BUCKET_ORDER:
        group = [row for row in prompt_results if row["bucket"] == bucket]
        if not group:
            continue
        out.append(
            {
                "gate_id": first["gate_id"],
                "gate_mode": first["gate_mode"],
                "threshold_rel": first["threshold_rel"],
                "threshold_bank": first["threshold_bank"],
                "bucket": bucket,
                "gate_activation_rate": sum(float(row["gate_active"]) for row in group) / len(group),
                "num_edits": len({row["edit_id"] for row in group}),
                "num_prompt_rows": len(group),
            }
        )
    return out


def candidate_row_from_bucket_summary(
    gate_id: str,
    gate_mode: str,
    rel_threshold: float,
    bank_threshold: float,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_bucket = {row["bucket"]: float(row["gate_activation_rate"]) for row in rows}
    checks = {
        "rewrite_pass": by_bucket.get("rewrite", 0.0) >= 0.95,
        "declarative_pass": by_bucket.get("declarative_paraphrases", 0.0) >= 0.85,
        "same_subject_template_pass": by_bucket.get("same_subject_template", 1.0) <= 0.05,
        "generation_pass": by_bucket.get("generation", 1.0) <= 0.10,
        "near_locality_pass": by_bucket.get("near_locality", 1.0) <= 0.02,
        "far_locality_pass": by_bucket.get("far_locality", 1.0) == 0.0,
    }
    violations = [key for key, passed in checks.items() if not passed]
    return {
        "gate_id": gate_id,
        "gate_mode": gate_mode,
        "threshold_rel": rel_threshold,
        "threshold_bank": bank_threshold,
        "rewrite_activation": by_bucket.get("rewrite", 0.0),
        "declarative_paraphrases_activation": by_bucket.get("declarative_paraphrases", 0.0),
        "qa_format_generalization_activation": by_bucket.get("qa_format_generalization", 0.0),
        "near_locality_activation": by_bucket.get("near_locality", 0.0),
        "far_locality_activation": by_bucket.get("far_locality", 0.0),
        "same_subject_template_activation": by_bucket.get("same_subject_template", 0.0),
        "generation_activation": by_bucket.get("generation", 0.0),
        **checks,
        "gate_candidate_pass": not violations,
        "constraint_violations": ";".join(violations),
    }


def grid_sample_rows(prompt_results: Sequence[Mapping[str, Any]], gate_id: str) -> List[Dict[str, Any]]:
    interesting = [
        row
        for row in prompt_results
        if row["bucket"] in {"same_subject_template", "generation", "declarative_paraphrases"}
    ]
    active_stress = [row for row in interesting if row["gate_active"] and row["bucket"] in {"same_subject_template", "generation"}]
    inactive_para = [row for row in interesting if not row["gate_active"] and row["bucket"] == "declarative_paraphrases"]
    selected = active_stress[:10] + inactive_para[:10]
    return [
        {
            "gate_id": gate_id,
            "sample_type": "active_stress" if row["bucket"] in {"same_subject_template", "generation"} else "inactive_paraphrase",
            **row,
        }
        for row in selected
    ]


def grid_threshold_diagnostics(prompt_results: Sequence[Mapping[str, Any]], gate_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        group = [row for row in prompt_results if row["bucket"] == bucket]
        if not group:
            continue
        rewrite_scores = [float(row["relation_sim_rewrite"]) for row in group]
        bank_scores = [float(row["relation_sim_bank"]) for row in group]
        out.append(
            {
                "gate_id": gate_id,
                "bucket": bucket,
                "num_prompt_rows": len(group),
                "subject_match_rate": sum(float(row["subject_match"]) for row in group) / len(group),
                "rewrite_sim_mean": statistics.mean(rewrite_scores),
                "rewrite_sim_p50": statistics.median(rewrite_scores),
                "rewrite_sim_max": max(rewrite_scores),
                "bank_sim_mean": statistics.mean(bank_scores),
                "bank_sim_p50": statistics.median(bank_scores),
                "bank_sim_max": max(bank_scores),
            }
        )
    return out


def write_activation_png(path: Path, candidate_rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 1200, 520
    pixels = [(255, 255, 255)] * (width * height)

    def set_px(x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            pixels[y * width + x] = color

    def rect(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                set_px(x, y, color)

    # Axes.
    rect(55, 40, 58, 470, (0, 0, 0))
    rect(55, 467, 1160, 470, (0, 0, 0))
    for frac in [0.05, 0.10, 0.85, 0.95]:
        y = int(467 - frac * 400)
        rect(55, y, 1160, y + 1, (220, 220, 220))

    colors = {
        "rewrite_activation": (36, 120, 200),
        "declarative_paraphrases_activation": (35, 150, 90),
        "same_subject_template_activation": (210, 70, 60),
        "generation_activation": (230, 150, 40),
    }
    bar_w = max(8, int(1000 / max(1, len(candidate_rows) * 5)))
    x = 70
    for row in candidate_rows:
        for metric, color in colors.items():
            value = float(row.get(metric) or 0.0)
            y0 = int(467 - value * 400)
            rect(x, y0, x + bar_w, 467, color)
            x += bar_w + 2
        x += 10

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels[y * width + x])
    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")

    def chunk(kind: bytes, data: bytes) -> None:
        png.extend(struct.pack(">I", len(data)))
        png.extend(kind)
        png.extend(data)
        crc = zlib.crc32(kind)
        crc = zlib.crc32(data, crc)
        png.extend(struct.pack(">I", crc & 0xFFFFFFFF))

    chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    chunk(b"IEND", b"")
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(bytes(png))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["parity-audit", "activation-grid"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--allow-overwrite", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "parity-audit":
        report = parity_audit(allow_overwrite=bool(args.allow_overwrite))
    elif args.command == "activation-grid":
        report = activation_grid(allow_overwrite=bool(args.allow_overwrite))
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
