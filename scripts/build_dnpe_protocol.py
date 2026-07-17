#!/usr/bin/env python3
"""Build fresh CounterFact and KAMEL manifests for the DNPE campaign."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_mask_pattern_publication_protocol import (
    _balanced_select as kamel_balanced_select,
    _candidate_pool as kamel_candidate_pool,
)
from scripts.build_mdm_memit_protocol import (
    contextual_target_ids,
    fingerprint_row,
    load_kamel_sources,
    round_robin_stratified,
)
from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    collect_historical_exclusions,
    git_commit,
    histogram,
    now_utc,
    record_stage,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


CF_COUNTS = {
    "dnpe_smoke_20": 20,
    "dnpe_pilot_100": 100,
    "dnpe_dev_200": 200,
    "dnpe_anchor_train_500": 500,
}
KAMEL_COUNTS = {
    "smoke": 20,
    "dev": 100,
    "locked": 200,
}
SEED = 260717101


def _counterfact_candidates(
    tokenizer: Any,
    *,
    dataset_name: str,
    exclusions: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train")
    excluded_ids = set(map(str, exclusions["case_ids"]))
    excluded_sources = set(map(str, exclusions["source_keys"]))
    excluded_fingerprints = set(map(str, exclusions["source_fingerprints"]))
    raw_templates: dict[str, str] = {}
    staged: list[tuple[int, Mapping[str, Any]]] = []
    counters = defaultdict(int)
    for source_index, raw in enumerate(dataset):
        rewrite = raw["requested_rewrite"]
        raw_templates.setdefault(str(rewrite["relation_id"]), str(rewrite["prompt"]))
        staged.append((source_index, raw))
    relations = sorted(raw_templates)
    legal: list[dict[str, Any]] = []
    for source_index, raw in staged:
        rewrite = raw["requested_rewrite"]
        subject = str(rewrite["subject"]).strip()
        relation = str(rewrite["relation_id"])
        template = str(rewrite["prompt"])
        prompt = template.format(subject)
        target_new = str(rewrite["target_new"]["str"]).strip()
        target_true = str(rewrite["target_true"]["str"]).strip()
        case_id = f"counterfact_train_{source_index}"
        source_fp = fingerprint_row(dataset_name, "train", source_index, subject, relation)
        if (
            case_id in excluded_ids
            or f"train:{source_index}" in excluded_sources
            or source_fp in excluded_fingerprints
        ):
            counters["historical_excluded"] += 1
            continue
        target_ids = contextual_target_ids(tokenizer, prompt, target_new)
        true_ids = contextual_target_ids(tokenizer, prompt, target_true)
        if len(target_ids) != 1 or not true_ids:
            counters["non_single_token_filtered"] += 1
            continue
        other_relations = [value for value in relations if value != relation]
        negative_relation = min(
            other_relations,
            key=lambda value: stable_hash(SEED, case_id, value),
        )
        same_subject_prompt = raw_templates[negative_relation].format(subject)
        legal.append(
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "protocol_version": CAMPAIGN_ID,
                "case_id": f"dnpe_cf_{source_index}",
                "source_dataset": dataset_name,
                "source_split": "train",
                "source_index": source_index,
                "source_fingerprint": source_fp,
                "counterfact_raw_case_id": int(raw["case_id"]),
                "relation_id": relation,
                "subject": subject,
                "rewrite_template": template,
                "rewrite_prompt": prompt,
                "target_new": target_new,
                "target_true": target_true,
                "target_new_token_ids": target_ids,
                "target_true_token_ids": true_ids,
                "target_length": 1,
                "target_length_bin": "1",
                "tokenizer_model_id": PRIMARY_MODEL_ID,
                "tokenizer_revision": PRIMARY_MODEL_REVISION,
                "paraphrase_prompts": list(raw.get("paraphrase_prompts") or []),
                "near_locality_prompts": list(raw.get("neighborhood_prompts") or []),
                "attribute_prompts": list(raw.get("attribute_prompts") or []),
                "generation_prompts": list(raw.get("generation_prompts") or []),
                "same_subject_prompts": [same_subject_prompt],
                "same_subject_negative_relation_id": negative_relation,
                "same_subject_provenance": "documented_cross_relation_template_construction",
                "prompt_provenance": "real_azhx_counterfact_train",
                "train_seen": {"rewrite": False, "paraphrase": False, "locality": False},
            }
        )
    counters["legal_single_token_candidates"] = len(legal)
    return legal, dict(counters)


def _select_counterfact(candidates: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    used: set[str] = set()
    result: dict[str, list[dict[str, Any]]] = {}
    for offset, (role, count) in enumerate(CF_COUNTS.items()):
        selected = round_robin_stratified(
            list(candidates),
            count,
            seed=SEED + offset,
            used=used,
            group_fields=("relation_id",),
        )
        for rank, source in enumerate(selected):
            row = dict(source)
            row["split_role"] = role
            row["selection_rank"] = rank
            row["train_seen"] = {
                "rewrite": role == "dnpe_anchor_train_500",
                "paraphrase": False,
                "locality": False,
            }
            selected[rank] = row
        result[role] = selected
    return result


def _select_kamel(
    raw_rows: Sequence[Mapping[str, Any]],
    templates: Mapping[str, str],
    tokenizer: Any,
    exclusions: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    candidates, counters = kamel_candidate_pool(
        raw_rows,
        templates,
        tokenizer,
        model_id=PRIMARY_MODEL_ID,
        lengths={2, 3, 4},
        exclusions=exclusions,
        seed=SEED,
    )
    normalized: list[dict[str, Any]] = []
    for source in candidates:
        row = dict(source)
        row["campaign_id"] = CAMPAIGN_ID
        row["protocol_version"] = CAMPAIGN_ID
        row["case_id"] = row["case_id"].replace("kamel_pub_", "dnpe_kamel_", 1)
        row["tokenizer_revision"] = PRIMARY_MODEL_REVISION
        normalized.append(row)
    splits: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    for length in (2, 3, 4):
        pool = [row for row in normalized if int(row["target_length"]) == length]
        for offset, (kind, count) in enumerate(KAMEL_COUNTS.items()):
            role = f"dnpe_kamel_{kind}_{count}_n{length}"
            selected = kamel_balanced_select(
                pool,
                count,
                role=role,
                used=used,
                seed=SEED + length * 10 + offset,
            )
            splits[role] = selected
    return splits, counters


def _summarize(path: Path, rows: Sequence[Mapping[str, Any]], *, locked: bool) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "count": len(rows),
        "unique_case_ids": len({str(row["case_id"]) for row in rows}),
        "unique_source_fingerprints": len({str(row["source_fingerprint"]) for row in rows}),
        "target_length_histogram": histogram(row["target_length"] for row in rows),
        "relation_histogram": histogram(row["relation_id"] for row in rows),
        "locked": locked,
        "opened_for_method_selection": False,
    }


def _overlap_audit(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    names = list(splits)
    audit = []
    for index, left_name in enumerate(names):
        left = {str(row["source_fingerprint"]) for row in splits[left_name]}
        for right_name in names[index + 1 :]:
            right = {str(row["source_fingerprint"]) for row in splits[right_name]}
            overlap = left & right
            audit.append({"left": left_name, "right": right_name, "overlap_count": len(overlap)})
            if overlap:
                raise RuntimeError(f"Protocol overlap: {left_name} vs {right_name}: {len(overlap)}")
    return audit


def _locked_registry() -> dict[str, Any]:
    base = ROOT / "runs" / "counterfact_direction1_v1" / "protocol"
    entries = {}
    for name in ("analysis_500", "final_test_500"):
        path = base / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        count = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
        entries[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "count": count,
            "access_policy": "identity_fingerprint_exclusion_only_until_lock",
            "opened_for_tuning": False,
        }
    return {
        "campaign_id": CAMPAIGN_ID,
        "analysis_500_used_for_tuning": False,
        "final_test_used_for_tuning": False,
        "locked_manifests": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=CAMPAIGN_ROOT / "source_cache")
    parser.add_argument("--dataset", default="azhx/counterfact")
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    started = now_utc()
    report_path = args.output_dir / "report_summary.json"
    if report_path.exists() and not args.allow_overwrite:
        raise FileExistsError(report_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_MODEL_ID,
        revision=PRIMARY_MODEL_REVISION,
        trust_remote_code=True,
    )
    exclusions = collect_historical_exclusions()
    cf_candidates, cf_filter = _counterfact_candidates(
        tokenizer,
        dataset_name=args.dataset,
        exclusions=exclusions,
    )
    cf_splits = _select_counterfact(cf_candidates)
    raw_kamel, templates, kamel_source = load_kamel_sources(args.cache_dir)
    kamel_splits, kamel_filter = _select_kamel(raw_kamel, templates, tokenizer, exclusions)
    splits = {**cf_splits, **kamel_splits}
    summaries = {}
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summaries[name] = _summarize(path, rows, locked="_locked_" in name)
    overlap = _overlap_audit(splits)
    write_csv(args.output_dir / "split_overlap_audit.csv", overlap)
    write_csv(args.output_dir / "historical_exclusion_audit.csv", exclusions["audit"])
    write_json(
        args.output_dir / "historical_exclusion_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "case_id_count": len(exclusions["case_ids"]),
            "source_key_count": len(exclusions["source_keys"]),
            "source_fingerprint_count": len(exclusions["source_fingerprints"]),
            "fact_fingerprint_count": len(exclusions["fact_fingerprints"]),
            "fact_target_fingerprint_count": len(exclusions["fact_target_fingerprints"]),
            "fields_used": [
                "case_id",
                "source_split",
                "source_index",
                "source_fingerprint",
                "fact_fingerprint",
                "fact_target_fingerprint",
            ],
            "prompt_label_output_metric_fields_used": False,
        },
    )
    locked_registry = _locked_registry()
    write_json(args.output_dir / "locked_manifest_registry.json", locked_registry)
    write_json(
        args.output_dir / "source_registry.json",
        {
            "counterfact": {"dataset": args.dataset, "split": "train"},
            "kamel": kamel_source,
            "tokenizer_model_id": PRIMARY_MODEL_ID,
            "tokenizer_revision": PRIMARY_MODEL_REVISION,
            "context_aware_tokenization": True,
        },
    )
    write_json(args.output_dir / "split_summary.json", {"splits": summaries})
    required = list(CF_COUNTS) + [
        f"dnpe_kamel_{kind}_{count}_n{length}"
        for length in (2, 3, 4)
        for kind, count in KAMEL_COUNTS.items()
    ]
    acceptance = {
        "all_required_manifests_exist": all(name in summaries for name in required),
        "zero_cross_split_overlap": all(row["overlap_count"] == 0 for row in overlap),
        "required_target_lengths_present": all(
            summaries[f"dnpe_kamel_dev_100_n{length}"]["target_length_histogram"] == {str(length): 100}
            for length in (2, 3, 4)
        ),
        "relation_histograms_written": all(bool(summary["relation_histogram"]) for summary in summaries.values()),
        "locked_manifests_opened_for_tuning": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = (
        acceptance["all_required_manifests_exist"]
        and acceptance["zero_cross_split_overlap"]
        and acceptance["required_target_lengths_present"]
        and acceptance["relation_histograms_written"]
        and not acceptance["locked_manifests_opened_for_tuning"]
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "A2_fresh_protocol",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "split_summaries": summaries,
        "counterfact_filter": cf_filter,
        "kamel_filter": kamel_filter,
        "locked_registry": locked_registry,
        "acceptance": acceptance,
        "acceptance_pass": passed,
    }
    write_json(report_path, report)
    write_json(args.output_dir / "validation_report.json", acceptance)
    record_stage(
        "A2_fresh_protocol",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes="Fresh CounterFact and KAMEL manifests with historical fingerprint exclusion.",
        next_stage="B1_mdm_memit_reproduction" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(f"A2 fresh protocol passed: {args.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
