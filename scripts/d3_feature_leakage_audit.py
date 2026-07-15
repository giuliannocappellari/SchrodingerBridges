#!/usr/bin/env python3
"""Audit Direction 3 offline controller inputs for teacher/outcome leakage.

This is a CPU-only report step. It inspects the cached teacher rows and the
offline controller artifact, then writes an explicit pass/fail report. The
script does not import or load LLaDA and must not be used to justify GPU decode
unless ``audit_pass`` is true.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, git_commit, now_utc, read_json, read_jsonl, repo_path, write_csv, write_json


DEFAULT_CACHE_DIR = D3_ROOT / "teacher_cache_train100_val50_v1"
DEFAULT_CONTROLLER_DIR = D3_ROOT / "offline_train_value_gate_train100_val50_v1"
DEFAULT_REPLAY_DIR = D3_ROOT / "offline_replay_train100_val50_v1"
DEFAULT_OUTPUT_DIR = D3_ROOT / "stage1b_feature_leakage_audit_v1"

TEACHER_SCORE_FIELDS = {
    "raw_bridge_scores_top_k",
    "raw_bridge_scores",
    "mc_rollout_rewards_top_k",
    "mc_rollout_rewards",
    "myopic_scores_top_k",
    "myopic_scores",
    "no_rollout_scores_top_k",
    "no_rollout_scores",
}
FINAL_OUTCOME_FIELDS = {
    "chosen_token",
    "chosen_token_id",
    "final_decoded_output",
    "final_edit_success",
    "final_locality_success",
    "malformed",
    "sparse_guidance_kl",
}
LOCKED_TOKENS = ("analysis_500", "final_test_500", "final_test_full")

# The current offline trainer uses short feature names. Keep this explicit so
# the audit is inspectable instead of relying on fuzzy substring matching.
FEATURE_SOURCES: Dict[str, List[str]] = {
    "bias": [],
    "base_logit": ["base_logits_top_k"],
    "base_prob": ["base_probabilities_top_k"],
    "base_logprob": ["base_probabilities_top_k"],
    "candidate_rank": ["candidate_index"],
    "candidate_token_id": ["top_k_candidate_token_ids"],
    "candidate_token_id_embedding": ["top_k_candidate_token_ids"],
    "candidate_is_target_new_token": ["target_token_ids", "top_k_candidate_token_ids"],
    "candidate_is_target_true_token": ["target_true", "top_k_candidate_token_ids"],
    "target_token_position": ["selected_mask_position", "target_token_ids"],
    "target_length": ["target_token_ids", "target_length_bin"],
    "myopic_score": ["myopic_scores_top_k"],
    "no_rollout_score": ["no_rollout_scores_top_k"],
    "step": ["step_index"],
    "timestep": ["timestep"],
    "active_masks": ["active_mask_count"],
    "mask_ratio": ["mask_ratio"],
    "selected_mask_position": ["selected_mask_position"],
    "answer_position": ["selected_mask_position", "target_token_ids"],
    "candidate_position": ["candidate_index"],
    "subject_match": ["subject", "prompt_text"],
    "relation_token_jaccard_to_rewrite": ["prompt_text", "rewrite_prompt_text", "subject", "target_new", "target_true"],
    "relation_char3_jaccard_to_rewrite": ["prompt_text", "rewrite_prompt_text", "subject", "target_new", "target_true"],
    "prompt_token_len": ["prompt_text"],
    "subject_token_len": ["subject"],
    "question_indicator": ["prompt_text"],
    "possessive_indicator": ["prompt_text"],
    "subject_position_frac": ["subject", "prompt_text"],
    "relation_id_bucket": ["relation_id"],
    "target_base_margin": ["base_logits_top_k", "target_token_ids", "top_k_candidate_token_ids"],
    "target_myopic_margin": ["myopic_scores_top_k", "target_token_ids", "top_k_candidate_token_ids"],
    "target_no_rollout_margin": ["no_rollout_scores_top_k", "target_token_ids", "top_k_candidate_token_ids"],
    "target_base_prob": ["base_probabilities_top_k", "target_token_ids", "top_k_candidate_token_ids"],
    "target_in_topk": ["target_token_ids", "top_k_candidate_token_ids"],
}

LABEL_FIELDS = [
    {
        "field": "raw_bridge_scores_top_k",
        "usage": "value distillation teacher",
        "eligible_as_input": False,
    },
    {
        "field": "mc_rollout_rewards_top_k",
        "usage": "ranking teacher",
        "eligible_as_input": False,
    },
    {
        "field": "myopic_scores_top_k",
        "usage": "baseline/teacher diagnostic score",
        "eligible_as_input": False,
    },
    {
        "field": "no_rollout_scores_top_k",
        "usage": "baseline/teacher diagnostic score",
        "eligible_as_input": False,
    },
    {
        "field": "final_edit_success",
        "usage": "post-decode outcome label",
        "eligible_as_input": False,
    },
    {
        "field": "final_locality_success",
        "usage": "post-decode outcome label",
        "eligible_as_input": False,
    },
    {
        "field": "chosen_token_id",
        "usage": "teacher chosen action",
        "eligible_as_input": False,
    },
]


def feature_rows(controller_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    controllers = controller_payload.get("controllers")
    if not isinstance(controllers, dict):
        controllers = {
            str(controller_payload.get("default_controller") or "controller"): {
                "value_feature_names": controller_payload.get("value_feature_names")
                or controller_payload.get("feature_names")
                or [],
                "gate_feature_names": controller_payload.get("gate_feature_names") or [],
            }
        }
    for controller_name, spec in sorted(controllers.items()):
        for component, key in [("value", "value_feature_names"), ("gate", "gate_feature_names")]:
            features = spec.get(key)
            if not isinstance(features, list):
                continue
            for feature in features:
                sources = FEATURE_SOURCES.get(str(feature), [f"UNKNOWN_FEATURE_SOURCE:{feature}"])
                teacher_sources = sorted(set(sources) & TEACHER_SCORE_FIELDS)
                outcome_sources = sorted(set(sources) & FINAL_OUTCOME_FIELDS)
                unknown = [source for source in sources if str(source).startswith("UNKNOWN_FEATURE_SOURCE:")]
                leak = bool(teacher_sources or outcome_sources or unknown)
                rows.append(
                    {
                        "controller": controller_name,
                        "component": component,
                        "feature_name": str(feature),
                        "source_fields": ";".join(sources),
                        "teacher_score_sources": ";".join(teacher_sources),
                        "final_outcome_sources": ";".join(outcome_sources),
                        "unknown_source": ";".join(unknown),
                        "eligible_for_actual_decode": not leak,
                        "leakage_reason": (
                            "teacher_score_input"
                            if teacher_sources
                            else "final_outcome_input"
                            if outcome_sources
                            else "unknown_feature_source"
                            if unknown
                            else ""
                        ),
                    }
                )
    return rows


def ids_for(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("edit_id") or row.get("case_id")) for row in rows}


def locked_token_paths(paths: Iterable[Path]) -> List[str]:
    bad: List[str] = []
    for path in paths:
        text = str(path)
        if any(token in text for token in LOCKED_TOKENS):
            bad.append(text)
    return bad


def build_audit(cache_dir: Path, controller_dir: Path, replay_dir: Path, output_dir: Path) -> Dict[str, Any]:
    cache_summary = read_json(cache_dir / "report_summary.json")
    controller_summary = read_json(controller_dir / "report_summary.json")
    replay_summary = read_json(replay_dir / "report_summary.json")
    controller_payload = read_json(controller_dir / "controller_weights.json")
    train_rows = read_jsonl(cache_dir / "teacher_states_train.jsonl")
    val_rows = read_jsonl(cache_dir / "teacher_states_val.jsonl")
    train_ids = ids_for(train_rows)
    val_ids = ids_for(val_rows)
    feature_audit_rows = feature_rows(controller_payload)
    leaked_features = [row for row in feature_audit_rows if not bool(row["eligible_for_actual_decode"])]
    overlap = sorted(train_ids & val_ids)
    locked_paths = locked_token_paths([cache_dir, controller_dir, replay_dir, output_dir])
    analysis_flags = {
        "cache_analysis_500_used": bool(cache_summary.get("analysis_500_used", False)),
        "cache_final_test_used": bool(cache_summary.get("final_test_used", False)),
        "controller_analysis_500_used": bool(controller_summary.get("analysis_500_used", False)),
        "controller_final_test_used": bool(controller_summary.get("final_test_used", False)),
        "replay_analysis_500_used": bool(replay_summary.get("analysis_500_used", False)),
        "replay_final_test_used": bool(replay_summary.get("final_test_used", False)),
    }
    checks = {
        "no_teacher_score_fields_used_as_input": not any(row["teacher_score_sources"] for row in feature_audit_rows),
        "no_final_outcome_fields_used_as_input": not any(row["final_outcome_sources"] for row in feature_audit_rows),
        "no_unknown_feature_sources": not any(row["unknown_source"] for row in feature_audit_rows),
        "train_val_edit_ids_disjoint": not overlap,
        "analysis_500_used_false": not any(key.endswith("analysis_500_used") and value for key, value in analysis_flags.items()),
        "final_test_used_false": not any(key.endswith("final_test_used") and value for key, value in analysis_flags.items()),
        "no_locked_split_paths": not locked_paths,
    }
    audit_pass = all(checks.values())
    payload = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1B.1 feature leakage audit",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": False,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "teacher_cache_dir": str(cache_dir),
        "controller_dir": str(controller_dir),
        "offline_replay_dir": str(replay_dir),
        "num_controller_input_features": len(feature_audit_rows),
        "num_leaked_features": len(leaked_features),
        "leaked_features": leaked_features,
        "train_num_edit_ids": len(train_ids),
        "val_num_edit_ids": len(val_ids),
        "train_val_overlap_count": len(overlap),
        "train_val_overlap_examples": overlap[:20],
        "analysis_final_flags": analysis_flags,
        "locked_split_path_violations": locked_paths,
        "acceptance_checks": checks,
        "audit_pass": audit_pass,
        "actual_decode_allowed_next": audit_pass,
        "artifacts": {
            "feature_leakage_audit": str(output_dir / "feature_leakage_audit.json"),
            "controller_input_feature_list": str(output_dir / "controller_input_feature_list.csv"),
            "label_field_list": str(output_dir / "label_field_list.csv"),
        },
    }
    return {
        "payload": payload,
        "feature_rows": feature_audit_rows,
        "label_rows": LABEL_FIELDS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--controller_dir", type=Path, default=DEFAULT_CONTROLLER_DIR)
    parser.add_argument("--offline_replay_dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail_on_leak", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    repo_path(output_dir).mkdir(parents=True, exist_ok=True)
    result = build_audit(args.teacher_cache_dir, args.controller_dir, args.offline_replay_dir, output_dir)
    write_json(output_dir / "feature_leakage_audit.json", result["payload"])
    write_json(output_dir / "report_summary.json", result["payload"])
    write_csv(output_dir / "controller_input_feature_list.csv", result["feature_rows"])
    write_csv(output_dir / "label_field_list.csv", result["label_rows"])
    if bool(args.fail_on_leak) and not result["payload"]["audit_pass"]:
        raise AssertionError("D3 feature leakage audit failed; see report_summary.json")
    status = "PASSED" if result["payload"]["audit_pass"] else "FAILED"
    print(f"[INFO] D3 feature leakage audit {status}: {output_dir}")


if __name__ == "__main__":
    main()
