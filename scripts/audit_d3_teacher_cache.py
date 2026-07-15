#!/usr/bin/env python3
"""Audit a real Direction 3 teacher-cache smoke.

This script is CPU/report-only. It reads existing teacher-cache JSONL files,
validates trajectory/schema quality, and writes an auditable Stage 1A.1 report.
It does not import or load LLaDA.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    write_jsonl,
)


DEFAULT_INPUT = D3_ROOT / "teacher_cache_smoke_v1"
DEFAULT_OUTPUT = D3_ROOT / "teacher_cache_smoke_v1_audit"
TOP_K_DEFAULT = 8

CANDIDATE_FIELDS = [
    ("top_k_candidate_token_ids", "candidate"),
    ("top_k_candidate_ids", "candidate"),
]
SCORE_FIELDS = [
    "base_logits_top_k",
    "base_probabilities_top_k",
    "raw_bridge_scores_top_k",
    "myopic_scores_top_k",
    "no_rollout_scores_top_k",
    "mc_rollout_rewards_top_k",
    "base_logits",
    "base_probs",
    "raw_bridge_scores",
    "myopic_scores",
    "no_rollout_scores",
    "mc_rollout_rewards",
]
PRIMARY_SCORE_FIELDS = [
    "myopic_scores_top_k",
    "no_rollout_scores_top_k",
    "mc_rollout_rewards_top_k",
]
POSITIVE_PROMPT_TYPES = {"rewrite", "declarative_paraphrase"}
SAME_SUBJECT_PROMPT_TYPES = {"same_subject_different_relation", "same_subject_template"}
LOCALITY_PROMPT_TYPES = {"near_locality", "far_locality"}
REQUIRED_PROMPT_TYPES = {
    "rewrite",
    "declarative_paraphrase",
    "same_subject_different_relation",
    "near_locality",
    "far_locality",
    "generation",
}


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def as_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        raise AssertionError("Expected a list")
    return [float(value) for value in values]


def load_cache_rows(cache_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_path = cache_dir / "teacher_states_train.jsonl"
    val_path = cache_dir / "teacher_states_val.jsonl"
    summary_path = cache_dir / "report_summary.json"
    if not repo_path(train_path).exists():
        raise FileNotFoundError(f"Missing teacher train cache: {train_path}")
    if not repo_path(val_path).exists():
        raise FileNotFoundError(f"Missing teacher val cache: {val_path}")
    if not repo_path(summary_path).exists():
        raise FileNotFoundError(f"Missing teacher cache summary: {summary_path}")
    return read_jsonl(train_path), read_jsonl(val_path), read_json(summary_path)


def target_positions(row: Mapping[str, Any]) -> List[int]:
    candidates = row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or []
    targets = {int(token_id) for token_id in (row.get("target_token_ids") or [])}
    return [idx for idx, token_id in enumerate(candidates) if int(token_id) in targets]


def target_score_for(row: Mapping[str, Any], field: str) -> Optional[float]:
    positions = target_positions(row)
    if not positions:
        return None
    scores = row.get(field)
    if not isinstance(scores, list):
        return None
    valid = [float(scores[idx]) for idx in positions if idx < len(scores) and finite_float(scores[idx])]
    return max(valid) if valid else None


def split_quality_rows(split_name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    edit_ids = {str(row.get("edit_id") or row.get("case_id")) for row in rows}
    prompt_types = summarize_counter(str(row.get("prompt_type")) for row in rows)
    target_bins = summarize_counter(str(row.get("target_length_bin")) for row in rows)
    active_counts = summarize_counter(str(row.get("active_mask_count")) for row in rows)
    step_hist = summarize_counter(str(row.get("step_index")) for row in rows)
    return {
        "split": split_name,
        "num_rows": len(rows),
        "num_edits": len(edit_ids),
        "distinct_steps": len(step_hist),
        "active_mask_count_gt_1_present": any(int(k) > 1 for k in active_counts if str(k).lstrip("-").isdigit()),
        "target_bins_1_and_2_present": "1" in target_bins and "2" in target_bins,
        "same_subject_negative_present": any(k in prompt_types for k in SAME_SUBJECT_PROMPT_TYPES),
        "locality_negative_present": any(k in prompt_types for k in LOCALITY_PROMPT_TYPES),
        "positive_prompt_present": any(k in prompt_types for k in POSITIVE_PROMPT_TYPES),
        "negative_prompt_present": any(k not in POSITIVE_PROMPT_TYPES for k in prompt_types),
        "prompt_types": ";".join(f"{k}:{v}" for k, v in prompt_types.items()),
        "target_length_bins": ";".join(f"{k}:{v}" for k, v in target_bins.items()),
        "active_mask_counts": ";".join(f"{k}:{v}" for k, v in active_counts.items()),
        "step_histogram": ";".join(f"{k}:{v}" for k, v in step_hist.items()),
    }


def field_audit(rows: Sequence[Mapping[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    audit_rows: List[Dict[str, Any]] = []
    fields = CANDIDATE_FIELDS + [(field, "score") for field in SCORE_FIELDS]
    for field, field_type in fields:
        present = 0
        correct_length = 0
        finite_arrays = 0
        invalid_examples: List[str] = []
        for row in rows:
            value = row.get(field)
            if isinstance(value, list):
                present += 1
                if len(value) == top_k:
                    correct_length += 1
                if field_type == "candidate":
                    finite_ok = all(str(v).lstrip("-").isdigit() for v in value)
                else:
                    finite_ok = all(finite_float(v) for v in value)
                if finite_ok:
                    finite_arrays += 1
                if (len(value) != top_k or not finite_ok) and len(invalid_examples) < 5:
                    invalid_examples.append(str(row.get("prompt_id")))
            elif len(invalid_examples) < 5:
                invalid_examples.append(str(row.get("prompt_id")))
        audit_rows.append(
            {
                "field": field,
                "field_type": field_type,
                "rows": len(rows),
                "present_rows": present,
                "correct_length_rows": correct_length,
                "finite_or_integer_rows": finite_arrays,
                "expected_top_k": top_k,
                "pass": present == len(rows) and correct_length == len(rows) and finite_arrays == len(rows),
                "invalid_prompt_examples": ";".join(invalid_examples),
            }
        )
    return audit_rows


def variance_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for field in PRIMARY_SCORE_FIELDS:
        all_values: List[float] = []
        row_variances: List[float] = []
        for row in rows:
            scores = row.get(field)
            if not isinstance(scores, list) or not scores:
                continue
            vals = [float(x) for x in scores if finite_float(x)]
            if len(vals) != len(scores):
                continue
            all_values.extend(vals)
            row_variances.append(pvariance(vals) if len(vals) > 1 else 0.0)
        global_variance = pvariance(all_values) if len(all_values) > 1 else 0.0
        nonzero_row_variance_rate = (
            sum(1 for value in row_variances if value > 0.0) / len(row_variances)
            if row_variances
            else 0.0
        )
        out.append(
            {
                "score_field": field,
                "num_values": len(all_values),
                "global_variance": global_variance,
                "mean_row_variance": mean(row_variances) if row_variances else 0.0,
                "nonzero_row_variance_rate": nonzero_row_variance_rate,
                "min": min(all_values) if all_values else "",
                "max": max(all_values) if all_values else "",
                "mean": mean(all_values) if all_values else "",
                "variance_pass": global_variance > 0.0 and nonzero_row_variance_rate > 0.0,
            }
        )
    return out


def separation_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for field in PRIMARY_SCORE_FIELDS:
        grouped: Dict[int, List[float]] = defaultdict(list)
        margins: Dict[int, List[float]] = defaultdict(list)
        for row in rows:
            label = int(row.get("label", 0))
            score = target_score_for(row, field)
            scores = row.get(field)
            if score is None or not isinstance(scores, list):
                continue
            grouped[label].append(score)
            positions = set(target_positions(row))
            non_target = [
                float(value)
                for idx, value in enumerate(scores)
                if idx not in positions and finite_float(value)
            ]
            if non_target:
                margins[label].append(score - max(non_target))
        pos = grouped.get(1, [])
        neg = grouped.get(0, [])
        pos_margin = margins.get(1, [])
        neg_margin = margins.get(0, [])
        out.append(
            {
                "score_field": field,
                "positive_rows_with_target": len(pos),
                "negative_rows_with_target": len(neg),
                "positive_target_score_mean": mean(pos) if pos else "",
                "negative_target_score_mean": mean(neg) if neg else "",
                "positive_minus_negative_target_score": (mean(pos) - mean(neg)) if pos and neg else "",
                "positive_target_margin_mean": mean(pos_margin) if pos_margin else "",
                "negative_target_margin_mean": mean(neg_margin) if neg_margin else "",
                "positive_minus_negative_margin": (mean(pos_margin) - mean(neg_margin)) if pos_margin and neg_margin else "",
                "positives_and_negatives_present": bool(pos and neg),
            }
        )
    return out


def select_sample_rows(rows: Sequence[Mapping[str, Any]], limit_per_group: int = 4) -> List[Dict[str, Any]]:
    groups = {
        "positive": lambda row: int(row.get("label", 0)) == 1,
        "same_subject_negative": lambda row: str(row.get("prompt_type")) in SAME_SUBJECT_PROMPT_TYPES,
        "locality_negative": lambda row: str(row.get("prompt_type")) in LOCALITY_PROMPT_TYPES,
        "generation_negative": lambda row: str(row.get("prompt_type")) == "generation",
    }
    samples: List[Dict[str, Any]] = []
    for group_name, predicate in groups.items():
        count = 0
        for row in rows:
            if not predicate(row):
                continue
            slim = {
                "sample_group": group_name,
                "split_role": row.get("split_role"),
                "edit_id": row.get("edit_id") or row.get("case_id"),
                "prompt_id": row.get("prompt_id"),
                "prompt_type": row.get("prompt_type"),
                "label": row.get("label"),
                "target_new": row.get("target_new"),
                "target_true": row.get("target_true"),
                "target_token_ids": row.get("target_token_ids"),
                "top_k_candidate_token_ids": row.get("top_k_candidate_token_ids"),
                "raw_bridge_scores_top_k": row.get("raw_bridge_scores_top_k"),
                "myopic_scores_top_k": row.get("myopic_scores_top_k"),
                "no_rollout_scores_top_k": row.get("no_rollout_scores_top_k"),
                "mc_rollout_rewards_top_k": row.get("mc_rollout_rewards_top_k"),
                "chosen_token_id": row.get("chosen_token_id"),
                "sparse_guidance_kl": row.get("sparse_guidance_kl"),
                "malformed": row.get("malformed"),
            }
            samples.append(slim)
            count += 1
            if count >= limit_per_group:
                break
    return samples


def require_acceptance(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    teacher_summary: Mapping[str, Any],
    quality: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    variances: Sequence[Mapping[str, Any]],
    separation: Sequence[Mapping[str, Any]],
    min_train_edits: int,
    min_val_edits: int,
) -> Dict[str, Any]:
    train_edits = {str(row.get("edit_id") or row.get("case_id")) for row in train_rows}
    val_edits = {str(row.get("edit_id") or row.get("case_id")) for row in val_rows}
    prompt_types = {str(row.get("prompt_type")) for row in all_rows}
    target_bins = {str(row.get("target_length_bin")) for row in all_rows}
    step_values = {int(row.get("step_index")) for row in all_rows}
    active_counts = {int(row.get("active_mask_count")) for row in all_rows}
    checks = {
        "analysis_500_unused": not bool(teacher_summary.get("analysis_500_used", False)),
        "final_test_unused": not bool(teacher_summary.get("final_test_used", False)),
        "fake_model_false": teacher_summary.get("fake_model") is False,
        "llada_loaded_true_for_teacher_generation": teacher_summary.get("llada_loaded") is True,
        "num_train_edits_ge_min": len(train_edits) >= int(min_train_edits),
        "num_val_edits_ge_min": len(val_edits) >= int(min_val_edits),
        "distinct_steps_ge_3": len(step_values) >= 3,
        "active_mask_count_gt_1_present": any(count > 1 for count in active_counts),
        "target_length_bins_1_and_2_present": "1" in target_bins and "2" in target_bins,
        "required_prompt_types_present": REQUIRED_PROMPT_TYPES.issubset(prompt_types),
        "top_k_and_score_schema_pass": all(bool(row.get("pass")) for row in scores),
        "teacher_score_variance_pass": all(bool(row.get("variance_pass")) for row in variances),
        "positive_negative_rows_present": all(bool(row.get("positives_and_negatives_present")) for row in separation),
        "same_subject_negative_present": any(pt in prompt_types for pt in SAME_SUBJECT_PROMPT_TYPES),
        "locality_negative_present": any(pt in prompt_types for pt in LOCALITY_PROMPT_TYPES),
    }
    checks["audit_pass"] = all(checks.values())
    return checks


def build_audit(
    cache_dir: Path,
    output_dir: Path,
    top_k: int,
    min_train_edits: int = 10,
    min_val_edits: int = 5,
) -> Dict[str, Any]:
    train_rows, val_rows, teacher_summary = load_cache_rows(cache_dir)
    all_rows = train_rows + val_rows
    if not all_rows:
        raise AssertionError("Teacher cache has no rows")

    quality = [
        split_quality_rows("train", train_rows),
        split_quality_rows("val", val_rows),
        split_quality_rows("all", all_rows),
    ]
    score_audit = field_audit(all_rows, top_k=top_k)
    variances = variance_rows(all_rows)
    separation = separation_rows(all_rows)
    acceptance = require_acceptance(
        train_rows=train_rows,
        val_rows=val_rows,
        all_rows=all_rows,
        teacher_summary=teacher_summary,
        quality=quality,
        scores=score_audit,
        variances=variances,
        separation=separation,
        min_train_edits=min_train_edits,
        min_val_edits=min_val_edits,
    )

    report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1A.1 real teacher-cache audit",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "teacher_cache_dir": str(cache_dir),
        "fake_model": False,
        "llada_loaded": False,
        "teacher_generation_llada_loaded": bool(teacher_summary.get("llada_loaded", False)),
        "analysis_500_used": False,
        "final_test_used": False,
        "top_k": top_k,
        "min_train_edits": int(min_train_edits),
        "min_val_edits": int(min_val_edits),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "num_rows": len(all_rows),
        "num_train_edits": len({str(row.get("edit_id") or row.get("case_id")) for row in train_rows}),
        "num_val_edits": len({str(row.get("edit_id") or row.get("case_id")) for row in val_rows}),
        "prompt_type_histogram": summarize_counter(str(row.get("prompt_type")) for row in all_rows),
        "target_len_histogram": summarize_counter(str(row.get("target_length_bin")) for row in all_rows),
        "step_histogram": summarize_counter(str(row.get("step_index")) for row in all_rows),
        "active_mask_count_histogram": summarize_counter(str(row.get("active_mask_count")) for row in all_rows),
        "acceptance_checks": acceptance,
        "audit_pass": bool(acceptance["audit_pass"]),
        "artifacts": {
            "cache_quality_table": str(output_dir / "cache_quality_table.csv"),
            "score_field_audit": str(output_dir / "score_field_audit.csv"),
            "teacher_score_variance": str(output_dir / "teacher_score_variance.csv"),
            "positive_negative_score_separation": str(output_dir / "positive_negative_score_separation.csv"),
            "sample_rows": str(output_dir / "sample_rows.jsonl"),
        },
    }
    return {
        "report": report,
        "quality": quality,
        "score_audit": score_audit,
        "variances": variances,
        "separation": separation,
        "samples": select_sample_rows(all_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_cache_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--min_train_edits", type=int, default=10)
    parser.add_argument("--min_val_edits", type=int, default=5)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result = build_audit(
        args.teacher_cache_dir,
        args.output_dir,
        top_k=int(args.top_k),
        min_train_edits=int(args.min_train_edits),
        min_val_edits=int(args.min_val_edits),
    )
    write_csv(args.output_dir / "cache_quality_table.csv", result["quality"])
    write_csv(args.output_dir / "score_field_audit.csv", result["score_audit"])
    write_csv(args.output_dir / "teacher_score_variance.csv", result["variances"])
    write_csv(args.output_dir / "positive_negative_score_separation.csv", result["separation"])
    write_jsonl(args.output_dir / "sample_rows.jsonl", result["samples"])
    write_json(args.output_dir / "report_summary.json", result["report"])
    if not result["report"]["audit_pass"]:
        raise AssertionError("Teacher-cache audit failed; see report_summary.json")
    print(f"[INFO] Wrote Direction 3 teacher-cache audit to {args.output_dir}")


if __name__ == "__main__":
    main()
