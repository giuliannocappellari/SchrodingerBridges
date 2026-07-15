#!/usr/bin/env python3
"""Audit a Direction 3 deployable feature cache before representation training.

This is a local CPU/report step. It never imports or loads LLaDA. The audit
checks tensor shapes/finite values, feature-index alignment, prompt provenance,
distribution summaries, and runtime-feature leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, git_commit, now_utc, read_json, read_jsonl, repo_path, write_csv, write_json


DEFAULT_FEATURE_CACHE = D3_ROOT / "deployable_feature_cache_train100_val50_v1"
DEFAULT_TEACHER_CACHE = D3_ROOT / "teacher_cache_train100_val50_v1"
DEFAULT_OUTPUT = D3_ROOT / "deployable_feature_cache_train100_val50_v1_local_audit"
DEFAULT_EXPECTED_CANDIDATE_GROUPS = 2994
DEFAULT_EXPECTED_CANDIDATE_WIDTH = 8

REQUIRED_FILES = [
    "report_summary.json",
    "feature_schema.json",
    "feature_index.jsonl",
    "state_features.safetensors",
    "candidate_features.safetensors",
    "edit_features.safetensors",
    "gate_features.safetensors",
    "feature_quality_report.csv",
    "feature_alignment_report.csv",
    "runtime_feature_leakage_audit.json",
]
POSITIVE_PROMPT_TYPES = {"rewrite", "declarative_paraphrase"}
PROMPT_COVERAGE_TARGETS = {
    "rewrite": 0.95,
    "declarative_paraphrase": 0.95,
    "near_locality": 0.95,
    "far_locality": 0.95,
}
FORBIDDEN_RUNTIME_FIELDS = {
    "raw_bridge_scores_top_k",
    "mc_rollout_rewards_top_k",
    "myopic_scores_top_k",
    "no_rollout_scores_top_k",
    "target_myopic_margin",
    "target_no_rollout_margin",
    "chosen_teacher_token",
    "chosen_token",
    "chosen_token_id",
    "final_edit_success",
    "final_locality_success",
    "malformed",
    "prompt_type",
    "negative_type",
    "split_role",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with repo_path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def ids_for(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("edit_id") or row.get("case_id")) for row in rows}


def finite_tensor_summary(cache_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for filename in [
        "state_features.safetensors",
        "candidate_features.safetensors",
        "edit_features.safetensors",
        "gate_features.safetensors",
    ]:
        tensors = load_file(str(repo_path(cache_dir / filename)))
        for tensor_name, tensor in sorted(tensors.items()):
            is_float = bool(tensor.is_floating_point())
            finite = True
            mean = ""
            std = ""
            min_value = ""
            max_value = ""
            if is_float:
                finite = bool(tensor.isfinite().all().item())
                flat = tensor.float().reshape(-1)
                mean = float(flat.mean().item())
                std = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
                min_value = float(flat.min().item())
                max_value = float(flat.max().item())
            rows.append(
                {
                    "tensor_file": filename,
                    "tensor_name": tensor_name,
                    "shape": "x".join(str(v) for v in tensor.shape),
                    "dtype": str(tensor.dtype),
                    "is_float": is_float,
                    "finite": finite,
                    "mean": mean,
                    "std": std,
                    "min": min_value,
                    "max": max_value,
                }
            )
    return rows


def prompt_provenance_rows(teacher_rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any], index_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in teacher_rows:
        by_type[str(row.get("prompt_type") or "unknown")].append(row)
    schema_text = json.dumps(schema, sort_keys=True)
    index_has_prompt_source = any("prompt_source" in row or "source_manifest" in row for row in index_rows)
    out: List[Dict[str, Any]] = []
    for prompt_type, rows in sorted(by_type.items()):
        real_count = sum(1 for row in rows if str(row.get("prompt_text") or "").strip())
        synthetic_count = sum(1 for row in rows if "synthetic" in str(row.get("source_manifest") or row.get("prompt_source") or "").lower())
        total = len(rows)
        coverage = real_count / total if total else 0.0
        required = PROMPT_COVERAGE_TARGETS.get(prompt_type, None)
        pass_check = coverage >= required if required is not None else total > 0
        out.append(
            {
                "prompt_type": prompt_type,
                "num_rows": total,
                "real_prompt_text_rows": real_count,
                "synthetic_from_metadata_rows": synthetic_count,
                "real_prompt_coverage": coverage,
                "required_coverage": "" if required is None else required,
                "prompt_source_field_in_schema_or_index": ("prompt_source" in schema_text) or index_has_prompt_source,
                "source_manifest_field_in_schema_or_index": ("source_manifest" in schema_text) or index_has_prompt_source,
                "source_reported": "teacher_cache_prompt_text",
                "category_unavailable_reason": "",
                "pass": pass_check,
            }
        )
    for required_type in ["same_subject_different_relation", "generation"]:
        if required_type not in by_type:
            out.append(
                {
                    "prompt_type": required_type,
                    "num_rows": 0,
                    "real_prompt_text_rows": 0,
                    "synthetic_from_metadata_rows": 0,
                    "real_prompt_coverage": 0.0,
                    "required_coverage": "present",
                    "prompt_source_field_in_schema_or_index": ("prompt_source" in schema_text) or index_has_prompt_source,
                    "source_manifest_field_in_schema_or_index": ("source_manifest" in schema_text) or index_has_prompt_source,
                    "source_reported": "",
                    "category_unavailable_reason": "missing_from_teacher_cache",
                    "pass": False,
                }
            )
    return out


def count_rows(rows: Sequence[Mapping[str, Any]], key: str) -> List[Dict[str, Any]]:
    counter = Counter(str(row.get(key) or "unknown") for row in rows)
    return [{"value": value, "num_rows": count} for value, count in sorted(counter.items())]


def leakage_pass(schema: Mapping[str, Any], leakage: Mapping[str, Any]) -> tuple[bool, List[str]]:
    names: List[str] = []
    for key in ["state_features", "candidate_features", "edit_features", "gate_features"]:
        names.extend(str(v) for v in schema.get(key, []))
    leaked = [name for name in names if any(field in name for field in FORBIDDEN_RUNTIME_FIELDS)]
    if int(leakage.get("num_leaked_runtime_features", 0)) > 0:
        leaked.extend(str(v) for v in leakage.get("leaked_runtime_feature_names", []))
    return len(leaked) == 0 and bool(leakage.get("runtime_feature_leakage_audit_pass", True)), sorted(set(leaked))


def build_audit(
    feature_cache_dir: Path,
    teacher_cache_dir: Path,
    output_dir: Path,
    *,
    expected_candidate_groups: int = DEFAULT_EXPECTED_CANDIDATE_GROUPS,
    expected_candidate_width: int = DEFAULT_EXPECTED_CANDIDATE_WIDTH,
) -> Mapping[str, Any]:
    missing = [name for name in REQUIRED_FILES if not repo_path(feature_cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature-cache files: {missing}")

    feature_summary = read_json(feature_cache_dir / "report_summary.json")
    schema = read_json(feature_cache_dir / "feature_schema.json")
    leakage = read_json(feature_cache_dir / "runtime_feature_leakage_audit.json")
    index_rows = read_jsonl(feature_cache_dir / "feature_index.jsonl")
    train_rows = read_jsonl(teacher_cache_dir / "teacher_states_train.jsonl")
    val_rows = read_jsonl(teacher_cache_dir / "teacher_states_val.jsonl")
    teacher_rows = train_rows + val_rows

    train_ids = ids_for(train_rows)
    val_ids = ids_for(val_rows)
    tensor_rows = finite_tensor_summary(feature_cache_dir)
    prompt_rows = prompt_provenance_rows(teacher_rows, schema, index_rows)
    leakage_ok, leaked_names = leakage_pass(schema, leakage)
    alignment_csv = read_csv_rows(feature_cache_dir / "feature_alignment_report.csv")
    quality_csv = read_csv_rows(feature_cache_dir / "feature_quality_report.csv")

    tensor_shape_ok = all(as_bool(row["finite"]) for row in tensor_rows)
    candidate_width = int(feature_summary.get("candidate_width", 0))
    candidate_groups = int(feature_summary.get("num_candidate_groups", 0))
    integrity_pass = (
        int(feature_summary.get("num_train_edits", 0)) >= 100
        and int(feature_summary.get("num_val_edits", 0)) >= 50
        and len(train_ids & val_ids) == 0
        and candidate_groups == expected_candidate_groups
        and candidate_width == expected_candidate_width
        and tensor_shape_ok
        and all(as_bool(row.get("pass", False)) for row in quality_csv)
    )
    alignment_pass = bool(feature_summary.get("feature_alignment_pass", False)) and all(as_bool(row.get("pass", False)) for row in alignment_csv)
    prompt_pass = all(as_bool(row.get("pass", False)) for row in prompt_rows)
    protocol_pass = (
        feature_summary.get("analysis_500_used") is False
        and feature_summary.get("final_test_used") is False
        and feature_summary.get("actual_decode_performed") is False
    )

    output = repo_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tensor_schema_audit.csv", tensor_rows)
    write_csv(output_dir / "group_alignment_audit.csv", alignment_csv)
    write_csv(output_dir / "prompt_provenance_audit.csv", prompt_rows)
    write_csv(output_dir / "feature_distribution_summary.csv", tensor_rows)
    write_csv(output_dir / "relation_distribution_summary.csv", count_rows(teacher_rows, "relation_id"))
    write_csv(output_dir / "target_length_summary.csv", count_rows(teacher_rows, "target_length_bin"))

    report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1B.3A local deployable feature-cache readiness audit",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "feature_cache_dir": str(feature_cache_dir),
        "teacher_cache_dir": str(teacher_cache_dir),
        "analysis_500_used": False,
        "final_test_used": False,
        "llada_loaded": False,
        "actual_decode_performed": False,
        "num_train_edits": len(train_ids),
        "num_val_edits": len(val_ids),
        "train_val_overlap": len(train_ids & val_ids),
        "num_candidate_groups": candidate_groups,
        "expected_candidate_groups": expected_candidate_groups,
        "candidate_width": candidate_width,
        "expected_candidate_width": expected_candidate_width,
        "feature_integrity_pass": integrity_pass,
        "feature_alignment_pass": alignment_pass,
        "prompt_provenance_pass": prompt_pass,
        "runtime_feature_leakage_pass": leakage_ok,
        "num_leaked_runtime_features": len(leaked_names),
        "leaked_runtime_feature_names": leaked_names,
        "protocol_safety_pass": protocol_pass,
        "audit_pass": integrity_pass and alignment_pass and prompt_pass and leakage_ok and protocol_pass,
        "artifacts": {
            "tensor_schema_audit": str(output_dir / "tensor_schema_audit.csv"),
            "group_alignment_audit": str(output_dir / "group_alignment_audit.csv"),
            "prompt_provenance_audit": str(output_dir / "prompt_provenance_audit.csv"),
            "feature_distribution_summary": str(output_dir / "feature_distribution_summary.csv"),
            "relation_distribution_summary": str(output_dir / "relation_distribution_summary.csv"),
            "target_length_summary": str(output_dir / "target_length_summary.csv"),
        },
    }
    write_json(output_dir / "report_summary.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_cache_dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--teacher_cache_dir", type=Path, default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected_candidate_groups", type=int, default=DEFAULT_EXPECTED_CANDIDATE_GROUPS)
    parser.add_argument("--expected_candidate_width", type=int, default=DEFAULT_EXPECTED_CANDIDATE_WIDTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_audit(
        args.feature_cache_dir,
        args.teacher_cache_dir,
        args.output_dir,
        expected_candidate_groups=args.expected_candidate_groups,
        expected_candidate_width=args.expected_candidate_width,
    )
    print(f"[INFO] Wrote D3 feature-cache audit to {args.output_dir}")
    print(f"[INFO] audit_pass={report['audit_pass']}")


if __name__ == "__main__":
    main()
