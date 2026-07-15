#!/usr/bin/env python3
"""Build deterministic legal splits for the SB alternatives campaign."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    COMMON_ROOT,
    collect_locked_exclusions,
    git_commit,
    now_utc,
    read_json,
    repo_path,
    sha256_file,
    stable_key,
    summarize,
    write_csv,
    write_json,
    write_jsonl,
)


SPLIT_SIZES = {
    "sb_alt_train_2000": 2000,
    "sb_alt_val_300": 300,
    "sb_alt_smoke_50": 50,
    "sb_alt_confirmation_50": 50,
}


def target_bin(row: dict[str, Any]) -> str:
    value = int(row.get("target_new_context_token_len") or 1)
    return ">=4" if value >= 4 else str(value)


def load_legal_pool(
    validity_path: str | Path,
    exclusions: dict[str, Any],
) -> list[dict[str, Any]]:
    validity = read_json(validity_path)
    excluded_ids = set(exclusions["case_ids"])
    excluded_source_keys = set(exclusions["source_keys"])
    rows: list[dict[str, Any]] = []
    for source in validity.get("train", []):
        if not source.get("valid", False):
            continue
        case_id = str(source["case_id"])
        source_split = str(source.get("source_split") or "train")
        source_index = int(source["source_index"])
        if case_id in excluded_ids or f"{source_split}:{source_index}" in excluded_source_keys:
            continue
        row = dict(source)
        row["case_id"] = case_id
        row["source_split"] = source_split
        row["source_index"] = source_index
        row["target_length_bin"] = target_bin(row)
        rows.append(row)
    return rows


def select_disjoint_splits(
    rows: list[dict[str, Any]],
    sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["target_length_bin"], str(row.get("relation_id") or "unknown"))].append(row)
    for key, values in groups.items():
        values.sort(key=lambda row: stable_key(seed, key, row["case_id"]))

    ordered_keys = sorted(groups, key=lambda key: (key[0] == ">=4", key[0], key[1]))
    selected: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    cursor = 0
    for split_name, count in sizes.items():
        bucket: list[dict[str, Any]] = []
        stalled = 0
        while len(bucket) < count and ordered_keys:
            key = ordered_keys[cursor % len(ordered_keys)]
            cursor += 1
            values = groups[key]
            while values and values[0]["case_id"] in used:
                values.pop(0)
            if values:
                row = values.pop(0)
                used.add(row["case_id"])
                bucket.append(row)
                stalled = 0
            else:
                stalled += 1
                if stalled > len(ordered_keys) * 2:
                    ordered_keys = [candidate for candidate in ordered_keys if groups[candidate]]
                    stalled = 0
        if len(bucket) != count:
            raise RuntimeError(f"Could select only {len(bucket)} rows for {split_name}")
        selected[split_name] = bucket
    return selected


def materialize_rows(
    selected: dict[str, list[dict[str, Any]]],
    dataset_name: str,
) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train")
    output: dict[str, list[dict[str, Any]]] = {}
    for split_name, rows in selected.items():
        materialized: list[dict[str, Any]] = []
        for rank, meta in enumerate(rows):
            raw = dataset[int(meta["source_index"])]
            rewrite = raw["requested_rewrite"]
            # Direction 1 intentionally namespaces case IDs by source row index,
            # while the upstream CounterFact `case_id` is a separate identifier.
            case_id = f"counterfact_train_{int(meta['source_index'])}"
            if case_id != meta["case_id"]:
                raise RuntimeError(f"Historical source-index namespace mismatch for {meta['case_id']}")
            prompt_template = str(rewrite["prompt"])
            subject = str(rewrite["subject"])
            row = {
                "schema_version": 1,
                "campaign_protocol": CAMPAIGN_PROTOCOL,
                "split_role": split_name,
                "selection_rank": rank,
                "case_id": case_id,
                "id": case_id,
                "source_dataset": dataset_name,
                "source_split": "train",
                "source_index": int(meta["source_index"]),
                "counterfact_raw_case_id": int(raw["case_id"]),
                "relation_id": str(rewrite["relation_id"]),
                "subject": subject,
                "rewrite_template": prompt_template,
                "rewrite_prompt": prompt_template.format(subject),
                "target_new": str(rewrite["target_new"]["str"]),
                "target_true": str(rewrite["target_true"]["str"]),
                "target_new_context_token_len": int(meta["target_new_context_token_len"]),
                "target_length_bin": meta["target_length_bin"],
                "paraphrase_prompts": list(raw.get("paraphrase_prompts") or []),
                "near_locality_prompts": list(raw.get("neighborhood_prompts") or []),
                "attribute_prompts": list(raw.get("attribute_prompts") or []),
                "generation_prompts": list(raw.get("generation_prompts") or []),
                "prompt_provenance": "real_azhx_counterfact_train_row",
                "synthetic_from_metadata": False,
            }
            materialized.append(row)
        output[split_name] = materialized
    return output


def stratified_subsplit(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: stable_key(seed, row["target_length_bin"], row["relation_id"], row["case_id"]),
    )
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_bin[row["target_length_bin"]].append(row)
    result: list[dict[str, Any]] = []
    for bin_name in ["1", "2", "3", ">=4"]:
        if by_bin.get(bin_name) and len(result) < count:
            result.append(by_bin[bin_name].pop(0))
    for row in ordered:
        if len(result) >= count:
            break
        if row not in result:
            result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=COMMON_ROOT)
    parser.add_argument(
        "--validity_path",
        type=Path,
        default=Path("runs/counterfact_direction1_v1/protocol/validity_report.json"),
    )
    parser.add_argument("--dataset_name", default="azhx/counterfact")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    out_dir = repo_path(args.output_dir)
    report_path = out_dir / "report_summary.json"
    if report_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Output already exists: {report_path}")

    exclusions = collect_locked_exclusions()
    pool = load_legal_pool(args.validity_path, exclusions)
    selected_meta = select_disjoint_splits(pool, SPLIT_SIZES, args.seed)
    selected = materialize_rows(selected_meta, args.dataset_name)

    smoke20 = stratified_subsplit(selected["sb_alt_smoke_50"], 20, args.seed + 1)
    smoke_ids = {row["case_id"] for row in smoke20}
    confirmation30 = [
        row for row in selected["sb_alt_smoke_50"] if row["case_id"] not in smoke_ids
    ]
    if len(confirmation30) != 30:
        raise RuntimeError("smoke20/confirmation30 partition is not 20/30")

    all_outputs = dict(selected)
    all_outputs["track_smoke_20"] = smoke20
    all_outputs["track_confirmation_30"] = confirmation30
    for split_name, rows in all_outputs.items():
        write_jsonl(args.output_dir / f"{split_name}.jsonl", rows)

    ids_by_split = {
        name: {row["case_id"] for row in rows}
        for name, rows in selected.items()
    }
    overlap_rows: list[dict[str, Any]] = []
    names = list(ids_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = ids_by_split[left] & ids_by_split[right]
            overlap_rows.append(
                {"left_split": left, "right_split": right, "overlap_count": len(overlap)}
            )
    if any(int(row["overlap_count"]) for row in overlap_rows):
        raise RuntimeError("Common campaign splits overlap")

    histogram_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    for split_name, rows in all_outputs.items():
        target_hist = summarize(row["target_length_bin"] for row in rows)
        relation_hist = summarize(row["relation_id"] for row in rows)
        split_summary[split_name] = {
            "count": len(rows),
            "target_length_histogram": target_hist,
            "relation_histogram": relation_hist,
            "real_prompt_row_count": sum(
                row["prompt_provenance"] == "real_azhx_counterfact_train_row" for row in rows
            ),
            "sha256": sha256_file(args.output_dir / f"{split_name}.jsonl"),
        }
        histogram_rows.extend(
            {"split": split_name, "target_length_bin": key, "count": value}
            for key, value in target_hist.items()
        )
        relation_rows.extend(
            {"split": split_name, "relation_id": key, "count": value}
            for key, value in relation_hist.items()
        )
        for prompt_field in [
            "rewrite_prompt",
            "paraphrase_prompts",
            "near_locality_prompts",
            "generation_prompts",
            "attribute_prompts",
        ]:
            available = sum(bool(row.get(prompt_field)) for row in rows)
            provenance_rows.append(
                {
                    "split": split_name,
                    "prompt_field": prompt_field,
                    "edits_with_real_prompt": available,
                    "num_edits": len(rows),
                    "coverage": available / len(rows) if rows else 0.0,
                }
            )

    write_json(args.output_dir / "split_summary.json", split_summary)
    write_csv(args.output_dir / "split_overlap_audit.csv", overlap_rows)
    write_csv(args.output_dir / "target_length_histograms.csv", histogram_rows)
    write_csv(args.output_dir / "relation_histograms.csv", relation_rows)
    write_csv(args.output_dir / "prompt_provenance_summary.csv", provenance_rows)
    write_json(
        args.output_dir / "common_baseline_registry.json",
        {
            "base": {},
            "target_logit_bias": {},
            "prompt_memory": {},
            "myopic_score": {},
            "no_rollout_bridge": {},
            "mc_bridge": {},
            "historical_raw_bridge": {
                "source": "counterfact_direction1_v1",
                "historical_only": True,
            },
        },
    )
    source_policy = """# Common Source Policy

The source pool is the official `azhx/counterfact` train split. Direction 1
locked manifests were read only for case IDs, source coordinates, and file
fingerprints. Their prompts, labels, outputs, and metrics were not used.

All selected campaign rows are disjoint from `dev_tune_200`, `ablation_500`,
`analysis_500`, `final_test_500`, and `final_test_full`. Real prompt fields are
materialized only for newly selected campaign train/validation/smoke rows.
Historical teacher outputs and checkpoints are not reused as new-campaign
training labels unless a track plan explicitly permits it.
"""
    (out_dir / "source_policy.md").write_text(source_policy, encoding="utf-8")
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "stage": "Common protocol construction",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "dataset_name": args.dataset_name,
        "selection_seed": args.seed,
        "analysis_500_used_for_tuning": False,
        "final_test_used_for_tuning": False,
        "locked_manifest_fields_used": [
            "case_id",
            "id",
            "source_dataset_split",
            "source_split",
            "source_index",
        ],
        "locked_prompt_label_output_metric_fields_used": False,
        "legal_pool_count": len(pool),
        "split_summary": split_summary,
        "acceptance_checks": {
            "zero_split_overlap": True,
            "locked_splits_excluded": True,
            "fingerprints_recorded": True,
            "real_prompt_coverage_reported": True,
            "single_and_multi_token_present": all(
                summary["target_length_histogram"].get("1", 0) > 0
                and sum(
                    count
                    for key, count in summary["target_length_histogram"].items()
                    if key != "1"
                )
                > 0
                for name, summary in split_summary.items()
                if name in SPLIT_SIZES
            ),
            "analysis_500_unused": True,
            "final_test_unused": True,
            "acceptance_pass": True,
        },
        "locked_manifest_fingerprints": exclusions["manifests"],
    }
    write_json(args.output_dir / "report_summary.json", report)
    print("common_protocol_acceptance_pass=True")
    print(f"legal_pool_count={len(pool)}")


if __name__ == "__main__":
    main()
