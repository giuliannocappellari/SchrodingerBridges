#!/usr/bin/env python3
"""Build local Direction 3 controller pilot split plans.

This script does not materialize full CounterFact prompts. It creates
metadata-only pilot manifests from the local Direction 1 validity report and
uses locked split manifests only to collect excluded IDs/fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import (
    D1_ROOT,
    D3_PROTOCOL_VERSION,
    D3_ROOT,
    collect_locked_manifest_exclusions,
    git_commit,
    load_valid_train_pool,
    now_utc,
    repo_path,
    sha256_file,
    summarize_counter,
    write_json,
    write_jsonl,
)


SPLIT_SIZES = {
    "controller_train_100": 100,
    "controller_val_50": 50,
    "dev_smoke_50": 50,
}

SMOKE_SUBSETS = {
    "controller_train_10": ("controller_train_100", 10),
    "controller_val_5": ("controller_val_50", 5),
}

REQUIRED_TARGET_BINS = ("1", "2")


def _stable_sort_key(seed: int, split_role: str, bin_name: str, row: Dict[str, Any]) -> str:
    return hashlib.sha1(f"{seed}:{split_role}:{bin_name}:{row['case_id']}".encode("utf-8")).hexdigest()


def _group_rows(rows: List[Dict[str, Any]], seed: int, split_role: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("target_length_bin", "1"))].append(dict(row))
    for bin_name, items in groups.items():
        items.sort(key=lambda row: _stable_sort_key(seed, split_role, bin_name, row))
    return groups


def select_split_rows(
    candidates: List[Dict[str, Any]],
    split_role: str,
    count: int,
    seed: int,
    used_case_ids: set[str],
    remaining_splits_after: int = 0,
) -> List[Dict[str, Any]]:
    """Select one disjoint split while forcing target bins 1 and 2 when available."""

    available = [row for row in candidates if row["case_id"] not in used_case_ids]
    groups = _group_rows(available, seed, split_role)
    selected: List[Dict[str, Any]] = []

    for bin_name in REQUIRED_TARGET_BINS:
        if groups.get(bin_name):
            row = groups[bin_name].pop(0)
            selected.append(row)
            used_case_ids.add(row["case_id"])

    ordered_bins = ["1", "2", "3", ">=4"]
    while len(selected) < count:
        made_progress = False
        for bin_name in ordered_bins:
            while groups.get(bin_name) and groups[bin_name][0]["case_id"] in used_case_ids:
                groups[bin_name].pop(0)
            if bin_name in REQUIRED_TARGET_BINS and len(groups.get(bin_name, [])) <= remaining_splits_after:
                continue
            if groups.get(bin_name):
                row = groups[bin_name].pop(0)
                selected.append(row)
                used_case_ids.add(row["case_id"])
                made_progress = True
                if len(selected) >= count:
                    break
        if not made_progress:
            break

    if len(selected) < count:
        raise ValueError(f"Could only select {len(selected)} rows for {split_role}; needed {count}")
    return selected


def choose_smoke_subset(parent_rows: List[Dict[str, Any]], subset_role: str, count: int) -> List[Dict[str, Any]]:
    """Choose a deterministic subset of a parent manifest and keep parent ordering."""

    selected_ids: set[str] = set()
    selected: List[Dict[str, Any]] = []

    for bin_name in REQUIRED_TARGET_BINS:
        for row in parent_rows:
            if str(row.get("target_length_bin")) == bin_name and row["case_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["case_id"])
                break

    for row in parent_rows:
        if len(selected) >= count:
            break
        if row["case_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["case_id"])

    if len(selected) < count:
        raise ValueError(f"Could only select {len(selected)} rows for {subset_role}; needed {count}")

    parent_order = {row["case_id"]: idx for idx, row in enumerate(parent_rows)}
    return sorted(selected, key=lambda row: parent_order[row["case_id"]])


def row_for_manifest(row: Dict[str, Any], split_role: str, rank: int, seed: int) -> Dict[str, Any]:
    return {
        "protocol_version": D3_PROTOCOL_VERSION,
        "split_role": split_role,
        "schema_version": 1,
        "id": row["case_id"],
        "case_id": row["case_id"],
        "source_dataset_split": row["source_split"],
        "source_index": row["source_index"],
        "relation_id": row["relation_id"],
        "subject": row["subject"],
        "target_length_bin": row["target_length_bin"],
        "target_new_context_token_len": row["target_new_context_token_len"],
        "target_true_context_token_len": row["target_true_context_token_len"],
        "prompt_token_len": row["prompt_token_len"],
        "paraphrase_prompt_count": row["paraphrase_prompt_count"],
        "neighborhood_prompt_count": row["neighborhood_prompt_count"],
        "subject_len_chars": row["subject_len_chars"],
        "subject_len_tokens": row["subject_len_tokens"],
        "subject_ambiguity_proxy": row["subject_ambiguity_proxy"],
        "selection_rank": rank,
        "selection_seed": seed,
        "materialization_status": "metadata_only_from_direction1_validity_report",
        "materialization_required_for_gpu_teacher_cache": True,
        "hf_materialization_key": {
            "dataset": "azhx/counterfact",
            "split": row["source_split"],
            "source_index": row["source_index"],
        },
    }


def write_protocol_docs(out_dir: Path) -> None:
    protocol = """# counterfact_direction3_controller_v1 Controller Pilot Protocol

## Definition

Direction 3 trains a small edit-conditioned runtime controller and edit-intent
gate that approximate bridge behavior more cheaply than MC rollouts.

This is not full CSBM and does not update base LLaDA weights.

```text
protocol_version = counterfact_direction3_controller_v1
base_model = GSAI-ML/LLaDA-8B-Base
theta0 = frozen
trained_parameters = controller_and_gate_only
base_model_weight_update = none
edit_access = given_at_edit_time
training_access = controller_train_only
analysis_500_used_for_tuning = false
final_test_500_used_for_tuning = false
```

## First Pilot

Start with metadata-only local manifests:

- `controller_train_100`
- `controller_val_50`
- `dev_smoke_50`

GPU teacher-cache generation is allowed only after local fake-mode tests pass.
"""
    teacher_schema = {
        "schema_version": 1,
        "protocol_version": D3_PROTOCOL_VERSION,
        "required_fields": [
            "case_id",
            "edit_id",
            "prompt_id",
            "prompt_type",
            "subject",
            "relation_id",
            "target_new",
            "target_true",
            "target_token_ids",
            "target_length_bin",
            "step_index",
            "timestep",
            "mask_ratio",
            "active_mask_count",
            "current_state",
            "selected_mask_positions",
            "top_k_candidate_token_ids",
            "base_logits_top_k",
            "base_probabilities_top_k",
            "raw_bridge_scores_top_k",
            "myopic_scores_top_k",
            "no_rollout_scores_top_k",
            "mc_rollout_rewards_top_k",
            "chosen_token_id",
            "final_decoded_output",
            "final_edit_success",
            "final_locality_success",
            "sparse_guidance_kl",
            "malformed",
            "fake_state",
            "selected_mask_position",
            "top_k_candidate_ids",
            "base_logits",
            "base_probs",
            "raw_bridge_scores",
            "myopic_scores",
            "no_rollout_scores",
            "mc_rollout_rewards",
            "chosen_token",
        ],
    }
    gate_schema = {
        "schema_version": 1,
        "protocol_version": D3_PROTOCOL_VERSION,
        "required_fields": [
            "gate_row_id",
            "split_role",
            "case_id",
            "prompt_id",
            "prompt_type",
            "label",
            "label_name",
            "negative_type",
            "subject",
            "relation_id",
            "target_new",
            "target_true",
            "prompt_text",
            "source_manifest",
            "category_unavailable_reason",
            "synthetic_from_metadata",
        ],
        "positive_prompt_types": ["rewrite", "declarative_paraphrase"],
        "negative_prompt_types": [
            "same_subject_different_relation",
            "near_locality",
            "far_locality",
            "generation",
            "attribute",
            "unrelated",
        ],
    }
    (repo_path(out_dir) / "direction3_controller_protocol.md").write_text(protocol, encoding="utf-8")
    write_json(out_dir / "teacher_cache_schema.json", teacher_schema)
    write_json(out_dir / "gate_dataset_schema.json", gate_schema)


def build_splits(out_dir: Path, seed: int) -> Dict[str, Any]:
    exclusions = collect_locked_manifest_exclusions(D1_ROOT / "protocol")
    excluded_ids = set(exclusions["excluded_case_ids"])
    excluded_source_keys = set(exclusions["excluded_source_keys"])
    pool = load_valid_train_pool(D1_ROOT / "protocol/validity_report.json")
    candidates = [
        row
        for row in pool
        if row["case_id"] not in excluded_ids
        and f"{row['source_split']}:{row['source_index']}" not in excluded_source_keys
    ]
    used_case_ids: set[str] = set()
    splits: Dict[str, List[Dict[str, Any]]] = {}
    split_items = list(SPLIT_SIZES.items())
    for idx, (split_role, count) in enumerate(split_items):
        remaining = len(split_items) - idx - 1
        splits[split_role] = select_split_rows(candidates, split_role, count, seed, used_case_ids, remaining)

    artifacts: Dict[str, Any] = {}
    for split_role, rows in splits.items():
        manifest_rows = [row_for_manifest(row, split_role, rank, seed) for rank, row in enumerate(rows)]
        path = out_dir / f"{split_role}.jsonl"
        write_jsonl(path, manifest_rows)
        target_hist = summarize_counter(row["target_length_bin"] for row in manifest_rows)
        relation_hist = summarize_counter(row["relation_id"] for row in manifest_rows)
        missing_required_bins = [bin_name for bin_name in REQUIRED_TARGET_BINS if bin_name not in target_hist]
        artifacts[split_role] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "count": len(manifest_rows),
            "target_length_bins": target_hist,
            "required_target_bins_present": not missing_required_bins,
            "missing_required_target_bins": missing_required_bins,
            "relation_count": len({row["relation_id"] for row in manifest_rows}),
            "relation_histogram": relation_hist,
        }

    subset_artifacts: Dict[str, Any] = {}
    for subset_role, (parent_role, count) in SMOKE_SUBSETS.items():
        parent_rows = [
            row_for_manifest(dict(row), subset_role, rank, seed)
            for rank, row in enumerate(choose_smoke_subset(splits[parent_role], subset_role, count))
        ]
        path = out_dir / f"{subset_role}.jsonl"
        write_jsonl(path, parent_rows)
        target_hist = summarize_counter(row["target_length_bin"] for row in parent_rows)
        missing_required_bins = [bin_name for bin_name in REQUIRED_TARGET_BINS if bin_name not in target_hist]
        subset_artifacts[subset_role] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "count": len(parent_rows),
            "parent_split": parent_role,
            "subset_of_parent": True,
            "target_length_bins": target_hist,
            "required_target_bins_present": not missing_required_bins,
            "missing_required_target_bins": missing_required_bins,
            "relation_count": len({row["relation_id"] for row in parent_rows}),
            "relation_histogram": summarize_counter(row["relation_id"] for row in parent_rows),
        }
        artifacts[subset_role] = subset_artifacts[subset_role]

    all_selected_ids = [row["case_id"] for rows in splits.values() for row in rows]
    overlap_with_locked = sorted(set(all_selected_ids) & excluded_ids)
    if overlap_with_locked:
        raise AssertionError(f"D3 split overlap with locked/current tuning splits: {overlap_with_locked[:10]}")

    split_overlap_checks: Dict[str, Any] = {}
    split_names = list(splits)
    for i, left in enumerate(split_names):
        left_ids = {row["case_id"] for row in splits[left]}
        for right in split_names[i + 1 :]:
            overlap = sorted(left_ids & {row["case_id"] for row in splits[right]})
            split_overlap_checks[f"{left}::{right}"] = {
                "count": len(overlap),
                "sample": overlap[:10],
            }
            if overlap:
                raise AssertionError(f"D3 split overlap between {left} and {right}: {overlap[:10]}")

    split_plan = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "seed": seed,
        "source_validity_report": str(D1_ROOT / "protocol/validity_report.json"),
        "source_pool": "CounterFact train validity rows, metadata-only",
        "locked_split_manifest_use": "case_id/source_index/fingerprint exclusion only",
        "locked_prompts_labels_outputs_or_metrics_used": False,
        "excluded_manifest_fingerprints": exclusions["manifests"],
        "candidate_pool_size_after_exclusion": len(candidates),
        "selection_strategy": "disjoint per-split deterministic interleaving with required target bins 1 and 2",
        "smoke_subset_strategy": "deterministic parent-order subset, forcing target bins 1 and 2 when available",
        "stratification_fields": [
            "target_length_bin",
            "relation_id",
            "subject_len_tokens",
            "subject_ambiguity_proxy",
        ],
        "base_target_new_success_available": False,
        "split_overlap_checks": split_overlap_checks,
        "target_length_histograms": {
            split_role: info["target_length_bins"] for split_role, info in artifacts.items()
        },
        "relation_histograms": {
            split_role: info["relation_histogram"] for split_role, info in artifacts.items()
        },
        "artifacts": artifacts,
    }
    write_json(out_dir / "controller_split_plan.json", split_plan)
    write_json(out_dir / "split_summary.json", {
        "protocol_version": D3_PROTOCOL_VERSION,
        "created_at_utc": now_utc(),
        "seed": seed,
        "locked_split_manifest_use": "case_id/source_index/fingerprint exclusion only",
        "locked_prompts_labels_outputs_or_metrics_used": False,
        "excluded_manifest_fingerprints": exclusions["manifests"],
        "candidate_pool_size_after_exclusion": len(candidates),
        "split_overlap_checks": split_overlap_checks,
        "artifacts": artifacts,
    })
    return split_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=D3_ROOT)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    full_out = repo_path(out_dir)
    if full_out.exists() and not bool(args.allow_overwrite):
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    full_out.mkdir(parents=True, exist_ok=True)
    write_protocol_docs(out_dir)
    split_plan = build_splits(out_dir, args.seed)
    report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 controller split scaffold",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "base_model": "GSAI-ML/LLaDA-8B-Base",
        "theta0": "frozen",
        "trained_parameters": "controller_and_gate_only",
        "base_model_weight_update": "none",
        "edit_access": "given_at_edit_time",
        "training_access": "controller_train_only",
        "analysis_500_used_for_tuning": False,
        "final_test_500_used_for_tuning": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "runpod_used": False,
        "llada_loaded": False,
        "artifacts": {
            "direction3_controller_protocol": str(out_dir / "direction3_controller_protocol.md"),
            "controller_split_plan": str(out_dir / "controller_split_plan.json"),
            "teacher_cache_schema": str(out_dir / "teacher_cache_schema.json"),
            "gate_dataset_schema": str(out_dir / "gate_dataset_schema.json"),
            **{name: info["path"] for name, info in split_plan["artifacts"].items()},
        },
    }
    write_json(out_dir / "report_summary.json", report)
    print(f"[INFO] Wrote Direction 3 split scaffold to {out_dir}")


if __name__ == "__main__":
    main()
