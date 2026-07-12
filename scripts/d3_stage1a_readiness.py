#!/usr/bin/env python3
"""Audit Direction 3 Stage 1A local readiness.

This script is local-only. It does not start RunPod, does not load LLaDA, and
does not run decoding. It verifies that the Direction 3 scaffold has enough
local structure for a later, explicitly approved GPU teacher-cache smoke.
"""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import (
    D3_PROTOCOL_VERSION,
    D3_ROOT,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    repo_path,
    summarize_counter,
    write_csv,
    write_json,
)


READINESS_DIR = D3_ROOT / "stage1a_teacher_cache_readiness_v1"
REQUIRED_ARTIFACTS = [
    ("controller_train_100", "controller_train_100.jsonl"),
    ("controller_val_50", "controller_val_50.jsonl"),
    ("dev_smoke_50", "dev_smoke_50.jsonl"),
    ("controller_train_10", "controller_train_10.jsonl"),
    ("controller_val_5", "controller_val_5.jsonl"),
    ("split_summary", "split_summary.json"),
    ("gate_train", "gate_train.jsonl"),
    ("gate_val", "gate_val.jsonl"),
    ("gate_dev_smoke", "gate_dev_smoke.jsonl"),
    ("gate_data_summary", "gate_data_summary.json"),
    ("fake_teacher_cache_summary", "fake_teacher_cache_v1/report_summary.json"),
    ("fake_controller_train_summary", "fake_controller_train_v1/report_summary.json"),
    ("fake_offline_replay_summary", "fake_offline_replay_v1/report_summary.json"),
]
REQUIRED_NEGATIVE_TYPES = [
    "same_subject_different_relation",
    "near_locality",
    "far_locality",
    "generation",
    "attribute",
    "unrelated",
]


def artifact_audit(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, rel_path in REQUIRED_ARTIFACTS:
        path = input_dir / rel_path
        exists = repo_path(path).exists()
        status = "ok" if exists else "missing"
        notes = ""
        if exists and rel_path.endswith(".json"):
            try:
                read_json(path)
            except Exception as exc:
                status = "invalid_json"
                notes = str(exc)
        rows.append(
            {
                "artifact_name": name,
                "path": str(path),
                "exists": exists,
                "required": True,
                "status": status,
                "notes": notes,
            }
        )
    return rows


def require_audit_pass(audit_rows: Sequence[Mapping[str, Any]]) -> None:
    bad = [row for row in audit_rows if row["required"] and row["status"] != "ok"]
    if bad:
        sample = [(row["artifact_name"], row["status"], row["path"]) for row in bad[:10]]
        raise AssertionError(f"Missing or invalid required Stage 1A artifacts: {sample}")


def target_bins_for(path: Path) -> Dict[str, int]:
    rows = read_jsonl(path)
    return summarize_counter(str(row.get("target_length_bin")) for row in rows)


def validate_split_manifests(input_dir: Path) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for split_name in ["controller_train_100", "controller_val_50", "dev_smoke_50"]:
        path = input_dir / f"{split_name}.jsonl"
        rows = read_jsonl(path)
        hist = target_bins_for(path)
        has_bins = "1" in hist and "2" in hist
        checks[split_name] = {
            "count": len(rows),
            "target_length_bins": hist,
            "required_bins_1_and_2_present": has_bins,
            "relation_histogram": summarize_counter(str(row.get("relation_id")) for row in rows),
        }
        if not has_bins:
            raise AssertionError(f"{split_name} must include target-length bins 1 and 2 when possible")

    subset_checks: Dict[str, Any] = {}
    for subset_name, parent_name, expected_count in [
        ("controller_train_10", "controller_train_100", 10),
        ("controller_val_5", "controller_val_50", 5),
    ]:
        subset_rows = read_jsonl(input_dir / f"{subset_name}.jsonl")
        parent_rows = read_jsonl(input_dir / f"{parent_name}.jsonl")
        subset_ids = [row["case_id"] for row in subset_rows]
        parent_ids = [row["case_id"] for row in parent_rows]
        parent_id_set = set(parent_ids)
        hist = summarize_counter(str(row.get("target_length_bin")) for row in subset_rows)
        is_subset = all(case_id in parent_id_set for case_id in subset_ids)
        parent_order = {case_id: i for i, case_id in enumerate(parent_ids)}
        order_positions = [parent_order[case_id] for case_id in subset_ids]
        parent_order_preserved = order_positions == sorted(order_positions)
        subset_checks[subset_name] = {
            "parent_split": parent_name,
            "count": len(subset_rows),
            "expected_count": expected_count,
            "subset_of_parent": is_subset,
            "parent_order_preserved": parent_order_preserved,
            "target_length_bins": hist,
            "required_bins_1_and_2_present": "1" in hist and "2" in hist,
        }
        if len(subset_rows) != expected_count or not is_subset or not parent_order_preserved:
            raise AssertionError(f"Invalid smoke subset: {subset_name}")
        if "1" not in hist or "2" not in hist:
            raise AssertionError(f"{subset_name} must include target-length bins 1 and 2 when possible")

    return {"splits": checks, "smoke_subsets": subset_checks}


def validate_gate_summary(input_dir: Path) -> Dict[str, Any]:
    summary = read_json(input_dir / "gate_data_summary.json")
    negative_counts = summary.get("negative_type_counts", {})
    unavailable = summary.get("category_unavailable_reason_counts", {})
    split_results: Dict[str, Any] = {}
    for split, counts in negative_counts.items():
        missing = [name for name in REQUIRED_NEGATIVE_TYPES if name not in counts]
        split_results[split] = {
            "negative_type_counts": counts,
            "missing_negative_types": missing,
            "category_unavailable_reason_counts": unavailable.get(split, {}),
        }
        if missing:
            raise AssertionError(f"Gate data missing required negative types for {split}: {missing}")
    if not summary.get("synthetic_fallback_marked_explicitly"):
        raise AssertionError("Gate data synthetic fallbacks must be explicitly marked")
    return {
        "uses_real_prompt_fields_when_available": bool(summary.get("uses_real_prompt_fields_when_available")),
        "synthetic_fallback_marked_explicitly": bool(summary.get("synthetic_fallback_marked_explicitly")),
        "split_results": split_results,
    }


def validate_fake_summary_flags(input_dir: Path) -> Dict[str, Any]:
    report_paths = {
        "fake_teacher_cache": input_dir / "fake_teacher_cache_v1/report_summary.json",
        "fake_controller_train": input_dir / "fake_controller_train_v1/report_summary.json",
        "fake_offline_replay": input_dir / "fake_offline_replay_v1/report_summary.json",
    }
    flags: Dict[str, Any] = {}
    for name, path in report_paths.items():
        payload = read_json(path)
        required = {
            "fake_model": True,
            "llada_loaded": False,
            "analysis_500_used": False,
            "final_test_used": False,
        }
        mismatches = {
            key: {"expected": expected, "got": payload.get(key)}
            for key, expected in required.items()
            if payload.get(key) != expected
        }
        if mismatches:
            raise AssertionError(f"Invalid fake-mode summary flags for {name}: {mismatches}")
        flags[name] = {
            "path": str(path),
            "required_flags_ok": True,
            "acceptance_pass": payload.get("acceptance_pass"),
        }
    return flags


def validate_fake_teacher_cache(input_dir: Path) -> Dict[str, Any]:
    summary = read_json(input_dir / "fake_teacher_cache_v1/report_summary.json")
    target_hist = {str(k): int(v) for k, v in summary.get("target_len_histogram", {}).items()}
    prompt_hist = summary.get("prompt_type_histogram", {})
    active_hist = {str(k): int(v) for k, v in summary.get("active_mask_count_histogram", {}).items()}
    step_hist = summary.get("step_histogram", {})
    checks = {
        "distinct_steps": len(step_hist),
        "active_mask_count_gt_1_present": any(int(k) > 1 for k in active_hist),
        "target_bins_1_and_2_present": "1" in target_hist and "2" in target_hist,
        "same_subject_negative_present": "same_subject_different_relation" in prompt_hist,
        "locality_negative_present": "near_locality" in prompt_hist or "far_locality" in prompt_hist,
    }
    if checks["distinct_steps"] < 3:
        raise AssertionError("Fake teacher cache must include at least 3 distinct steps")
    for key, value in checks.items():
        if key != "distinct_steps" and not value:
            raise AssertionError(f"Fake teacher cache failed readiness check: {key}")
    return {"summary_checks": checks, "target_len_histogram": target_hist, "prompt_type_histogram": prompt_hist}


def write_runpod_template(output_dir: Path) -> Path:
    path = output_dir / "runpod_teacher_cache_smoke_command.sh"
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(
        """#!/usr/bin/env bash
# TEMPLATE ONLY. DO NOT EXECUTE AUTOMATICALLY.
# Requires explicit user approval before starting RunPod or running GPU work.
# Scope: Direction 3 teacher-cache smoke only.

set -euo pipefail

cd /workspace/SB
python --version
python -m pytest tests -q
nvidia-smi

python scripts/build_d3_teacher_cache.py \\
  --input_dir runs/counterfact_direction3_controller_v1 \\
  --output_dir runs/counterfact_direction3_controller_v1/teacher_cache_smoke_v1 \\
  --fake_model 0 \\
  --split_train controller_train_10.jsonl \\
  --split_val controller_val_5.jsonl \\
  --model_id GSAI-ML/LLaDA-8B-Base \\
  --dtype float16 \\
  --use_4bit 1 \\
  --device_map auto \\
  --top_k 8 \\
  --steps 4 \\
  --mc_rollouts 2 \\
  --methods base,myopic_score,no_rollout_bridge,mc_bridge
""",
        encoding="utf-8",
    )
    full.chmod(0o644)
    mode = stat.S_IMODE(full.stat().st_mode)
    if mode != 0o644:
        raise AssertionError(f"RunPod template must be chmod 644, got {oct(mode)}")
    return path


def build_readiness_report(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    audit_rows = artifact_audit(input_dir)
    require_audit_pass(audit_rows)
    split_checks = validate_split_manifests(input_dir)
    gate_checks = validate_gate_summary(input_dir)
    fake_flags = validate_fake_summary_flags(input_dir)
    fake_cache_checks = validate_fake_teacher_cache(input_dir)
    offline_replay = read_json(input_dir / "fake_offline_replay_v1/report_summary.json")
    scientific_acceptance_pass = bool(offline_replay.get("acceptance_pass", False))
    template_path = write_runpod_template(output_dir)

    report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1A teacher-cache readiness",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "split_role": "controller_train_100/controller_val_50/dev_smoke_50",
        "local_only": True,
        "runpod_used": False,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "local_scaffold_ready": True,
        "pipeline_readiness_pass": True,
        "scientific_acceptance_pass": scientific_acceptance_pass,
        "runpod_allowed_next": False,
        "next_gpu_step": "teacher_cache_smoke_only",
        "requires_user_approval_for_runpod": True,
        "teacher_cache_smoke_scope": {
            "train_manifest": "controller_train_10.jsonl",
            "val_manifest": "controller_val_5.jsonl",
            "top_k": 8,
            "steps": 4,
            "mc_rollouts": 2,
            "methods": ["base", "myopic_score", "no_rollout_bridge", "mc_bridge"],
        },
        "split_checks": split_checks,
        "gate_checks": gate_checks,
        "fake_summary_flags": fake_flags,
        "fake_teacher_cache_checks": fake_cache_checks,
        "artifacts": {
            "readiness_checklist": str(output_dir / "readiness_checklist.md"),
            "fake_artifact_audit": str(output_dir / "fake_artifact_audit.csv"),
            "runpod_teacher_cache_smoke_command": str(template_path),
        },
    }
    return report


def write_checklist(output_dir: Path, report: Mapping[str, Any]) -> None:
    text = f"""# Direction 3 Stage 1A Readiness

Status: {'PASSED' if report['pipeline_readiness_pass'] else 'FAILED'}

- Local scaffold ready: {report['local_scaffold_ready']}
- Pipeline readiness pass: {report['pipeline_readiness_pass']}
- Scientific acceptance pass: {report['scientific_acceptance_pass']}
- RunPod allowed next: {report['runpod_allowed_next']}
- Requires user approval for RunPod: {report['requires_user_approval_for_runpod']}
- Next GPU step: {report['next_gpu_step']}

This is pipeline readiness only. Direction 3 is not scientifically validated by
fake replay. Scientific validation starts with a real teacher-cache smoke after
explicit user approval.

Teacher-cache smoke scope:

- train manifest: `controller_train_10.jsonl`
- val manifest: `controller_val_5.jsonl`
- `top_k = 8`
- `steps = 4`
- `mc_rollouts = 2`
- methods: `base,myopic_score,no_rollout_bridge,mc_bridge`

Guards:

- No RunPod command was executed.
- No LLaDA load occurred.
- Current analysis/final artifacts were not used for tuning or reporting.
- The generated RunPod command is a non-executable template.
"""
    repo_path(output_dir / "readiness_checklist.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=D3_ROOT)
    parser.add_argument("--output_dir", type=Path, default=READINESS_DIR)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    full_output = repo_path(output_dir)
    if full_output.exists() and not bool(args.allow_overwrite):
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    full_output.mkdir(parents=True, exist_ok=True)

    audit_rows = artifact_audit(args.input_dir)
    write_csv(output_dir / "fake_artifact_audit.csv", audit_rows, fieldnames=[
        "artifact_name",
        "path",
        "exists",
        "required",
        "status",
        "notes",
    ])
    report = build_readiness_report(args.input_dir, output_dir)
    write_checklist(output_dir, report)
    write_json(output_dir / "report_summary.json", report)
    print(f"[INFO] Wrote Direction 3 Stage 1A readiness package to {output_dir}")


if __name__ == "__main__":
    main()
