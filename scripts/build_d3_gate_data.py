#!/usr/bin/env python3
"""Build local Direction 3 gate datasets from controller pilot manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, git_commit, now_utc, read_jsonl, repo_path, summarize_counter, write_json, write_jsonl


SPLIT_MANIFESTS = {
    "controller_train_100": "gate_train.jsonl",
    "controller_val_50": "gate_val.jsonl",
    "dev_smoke_50": "gate_dev_smoke.jsonl",
}

POSITIVE_TYPES = ["rewrite", "declarative_paraphrase"]
NEGATIVE_TYPES = [
    "same_subject_different_relation",
    "near_locality",
    "far_locality",
    "generation",
    "attribute",
    "unrelated",
]


def target_text(row: Dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value:
        return str(value)
    requested = row.get("requested_rewrite")
    if isinstance(requested, dict):
        target = requested.get("target_new" if key == "target_new" else "target_true")
        if isinstance(target, dict) and target.get("str"):
            return str(target["str"])
    return f"[{key}_unmaterialized]"


def real_prompt_from_item(row: Dict[str, Any], prompt_type: str) -> Optional[Tuple[str, str]]:
    """Use real prompt fields when a future materialized manifest provides them."""

    direct_keys = {
        "rewrite": ["prompt", "rewrite_prompt"],
        "declarative_paraphrase": ["declarative_paraphrase", "paraphrase_prompt"],
        "same_subject_different_relation": ["same_subject_different_relation_prompt"],
        "near_locality": ["near_locality_prompt", "neighborhood_prompt"],
        "far_locality": ["far_locality_prompt"],
        "generation": ["generation_prompt"],
        "attribute": ["attribute_prompt"],
        "unrelated": ["unrelated_prompt"],
    }
    for key in direct_keys.get(prompt_type, []):
        if row.get(key):
            return str(row[key]), f"manifest_field:{key}"

    lists = {
        "declarative_paraphrase": ["paraphrase_prompts", "declarative_paraphrases"],
        "near_locality": ["neighborhood_prompts", "near_locality_prompts"],
        "far_locality": ["far_locality_prompts"],
        "generation": ["generation_prompts"],
        "attribute": ["attribute_prompts"],
    }
    for key in lists.get(prompt_type, []):
        values = row.get(key)
        if isinstance(values, list) and values:
            return str(values[0]), f"manifest_list:{key}[0]"

    requested = row.get("requested_rewrite")
    if prompt_type == "rewrite" and isinstance(requested, dict) and requested.get("prompt"):
        return str(requested["prompt"]), "requested_rewrite.prompt"
    return None


def prompt_for(row: Dict[str, Any], prompt_type: str) -> str:
    subject = str(row.get("subject", "SUBJECT"))
    relation = str(row.get("relation_id", "REL"))
    if prompt_type == "rewrite":
        return f"{subject} [relation {relation}]"
    if prompt_type == "declarative_paraphrase":
        return f"A paraphrase asks about {subject} and relation {relation}"
    if prompt_type == "same_subject_different_relation":
        return f"{subject} is mentioned, but the prompt asks for a different relation"
    if prompt_type == "near_locality":
        return f"A nearby entity asks about relation {relation}"
    if prompt_type == "far_locality":
        return f"A far-away entity asks about an unrelated relation"
    if prompt_type == "generation":
        return f"{subject} is known for"
    if prompt_type == "attribute":
        return f"{subject} has an unrelated attribute"
    if prompt_type == "unrelated":
        return "An unrelated entity is described by an unrelated relation"
    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def build_gate_rows(manifest_rows: Iterable[Dict[str, Any]], split_role: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in manifest_rows:
        for prompt_type in POSITIVE_TYPES + NEGATIVE_TYPES:
            label = 1 if prompt_type in POSITIVE_TYPES else 0
            prompt_id = f"{item['case_id']}::{prompt_type}"
            real_prompt = real_prompt_from_item(item, prompt_type)
            if real_prompt is None:
                prompt_text = prompt_for(item, prompt_type)
                synthetic_from_metadata = True
                source_manifest = f"{split_role}.jsonl"
                category_unavailable_reason = "source_prompt_unavailable_metadata_only"
            else:
                prompt_text, source_manifest = real_prompt
                synthetic_from_metadata = False
                category_unavailable_reason = ""
            rows.append(
                {
                    "protocol_version": D3_PROTOCOL_VERSION,
                    "schema_version": 1,
                    "gate_row_id": f"{split_role}::{prompt_id}",
                    "split_role": split_role,
                    "case_id": item["case_id"],
                    "edit_id": item["case_id"],
                    "source_dataset_split": item["source_dataset_split"],
                    "source_index": item["source_index"],
                    "prompt_id": prompt_id,
                    "prompt_type": prompt_type,
                    "negative_type": prompt_type if not label else "",
                    "label": label,
                    "label_name": "edit_intent" if label else "non_edit_intent",
                    "subject": item["subject"],
                    "relation_id": item["relation_id"],
                    "target_length_bin": item["target_length_bin"],
                    "target_new": target_text(item, "target_new"),
                    "target_true": target_text(item, "target_true"),
                    "prompt_text": prompt_text,
                    "source_manifest": source_manifest,
                    "category_unavailable_reason": category_unavailable_reason,
                    "synthetic_from_metadata": synthetic_from_metadata,
                    "no_train_eval_prompt_overlap_unit": "edit_id_split_disjoint",
                    "analysis_500_used": False,
                    "final_test_used": False,
                }
            )
    return rows


def validate_gate_rows(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    edit_sets = {split: {row["edit_id"] for row in rows} for split, rows in rows_by_split.items()}
    overlaps: Dict[str, List[str]] = {}
    names = sorted(edit_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(edit_sets[left] & edit_sets[right])
            overlaps[f"{left}::{right}"] = overlap
            if overlap:
                raise AssertionError(f"Gate dataset edit overlap between {left} and {right}: {overlap[:10]}")

    negative_counts = {
        split: summarize_counter(row["negative_type"] for row in rows if row["negative_type"])
        for split, rows in rows_by_split.items()
    }
    missing_negative_types = {
        split: [name for name in NEGATIVE_TYPES if name not in counts]
        for split, counts in negative_counts.items()
    }
    unavailable_reasons = {
        split: summarize_counter(
            f"{row['prompt_type']}::{row['category_unavailable_reason']}"
            for row in rows
            if row.get("category_unavailable_reason")
        )
        for split, rows in rows_by_split.items()
    }
    for split, missing in missing_negative_types.items():
        if missing:
            raise AssertionError(f"Missing required gate negative types for {split}: {missing}")

    summary = {
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "edit_counts": {split: len(edit_sets[split]) for split in rows_by_split},
        "label_counts": {
            split: summarize_counter(str(row["label"]) for row in rows)
            for split, rows in rows_by_split.items()
        },
        "prompt_type_counts": {
            split: summarize_counter(row["prompt_type"] for row in rows)
            for split, rows in rows_by_split.items()
        },
        "negative_type_counts": negative_counts,
        "missing_negative_types": missing_negative_types,
        "category_unavailable_reason_counts": unavailable_reasons,
        "uses_real_prompt_fields_when_available": True,
        "synthetic_fallback_marked_explicitly": all(
            (not row.get("synthetic_from_metadata")) or bool(row.get("category_unavailable_reason"))
            for rows in rows_by_split.values()
            for row in rows
        ),
        "same_subject_negatives_present": all(
            any(row["prompt_type"] == "same_subject_different_relation" for row in rows)
            for rows in rows_by_split.values()
        ),
        "locality_negatives_present": all(
            any(row["prompt_type"] in {"near_locality", "far_locality"} for row in rows)
            for rows in rows_by_split.values()
        ),
        "edit_overlaps": overlaps,
        "no_train_eval_prompt_overlap": True,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=D3_ROOT)
    parser.add_argument("--output_dir", type=Path, default=D3_ROOT)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    repo_path(output_dir).mkdir(parents=True, exist_ok=True)
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split_role, output_name in SPLIT_MANIFESTS.items():
        manifest_path = input_dir / f"{split_role}.jsonl"
        if not repo_path(manifest_path).exists():
            raise FileNotFoundError(f"Missing controller manifest: {manifest_path}")
        manifest_rows = read_jsonl(manifest_path)
        gate_rows = build_gate_rows(manifest_rows, split_role)
        rows_by_split[split_role] = gate_rows
        write_jsonl(output_dir / output_name, gate_rows)

    validation = validate_gate_rows(rows_by_split)
    summary = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 gate dataset scaffold",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": False,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "data_materialization_status": "synthetic_prompt_text_from_metadata_for_local_scaffold",
        "positive_prompt_types": POSITIVE_TYPES,
        "negative_prompt_types": NEGATIVE_TYPES,
        **validation,
        "artifacts": {
            "gate_train": str(output_dir / "gate_train.jsonl"),
            "gate_val": str(output_dir / "gate_val.jsonl"),
            "gate_dev_smoke": str(output_dir / "gate_dev_smoke.jsonl"),
        },
    }
    write_json(output_dir / "gate_data_summary.json", summary)
    print(f"[INFO] Wrote Direction 3 gate data scaffold to {output_dir}")


if __name__ == "__main__":
    main()
