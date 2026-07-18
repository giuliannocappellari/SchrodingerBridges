#!/usr/bin/env python3
"""Build fresh CounterFact, KAMEL, and protection-anchor manifests for TRM."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    collect_historical_exclusions,
    git_commit,
    now_utc,
    record_stage,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


CF_COUNTS = {
    "cf_trm_localize_50": 50,
    "cf_trm_smoke_20": 20,
    "cf_trm_pilot_100": 100,
    "cf_trm_dev_200": 200,
    "cf_trm_locked_500": 500,
    "cf_trm_scaling_100": 100,
    "cf_trm_anchor_train_500": 500,
    "cf_trm_locality_pool_300": 300,
}
KAMEL_COUNTS = {"dev": 50, "pilot": 100, "locked": 200}
SEED = 260718101


def target_length_bin(length: int) -> str:
    return ">=4" if length >= 4 else str(length)


def prompt_fingerprint(prompt: str) -> str:
    return stable_hash("prompt", " ".join(str(prompt).casefold().split()))


def _counterfact_candidates(
    tokenizer: Any, dataset_name: str, exclusions: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train")
    raw_templates: dict[str, str] = {}
    staged = []
    for source_index, raw in enumerate(dataset):
        rewrite = raw["requested_rewrite"]
        raw_templates.setdefault(str(rewrite["relation_id"]), str(rewrite["prompt"]))
        staged.append((source_index, raw))
    excluded_ids = set(map(str, exclusions["case_ids"]))
    excluded_sources = set(map(str, exclusions["source_keys"]))
    excluded_source_fps = set(map(str, exclusions["source_fingerprints"]))
    excluded_fact_fps = set(map(str, exclusions["fact_fingerprints"]))
    excluded_target_fps = set(map(str, exclusions["fact_target_fingerprints"]))
    excluded_prompt_fps = set(map(str, exclusions["prompt_fingerprints"]))
    relations = sorted(raw_templates)
    counters: Counter[str] = Counter()
    legal = []
    for source_index, raw in staged:
        rewrite = raw["requested_rewrite"]
        relation = str(rewrite["relation_id"])
        subject = str(rewrite["subject"]).strip()
        template = str(rewrite["prompt"])
        prompt = template.format(subject)
        target_new = str(rewrite["target_new"]["str"]).strip()
        target_true = str(rewrite["target_true"]["str"]).strip()
        source_fp = fingerprint_row(dataset_name, "train", source_index, subject, relation)
        fact_fp = stable_hash(relation, subject.casefold(), target_true.casefold())
        fact_target_fp = stable_hash(relation, subject.casefold(), target_new.casefold())
        case_names = {f"counterfact_train_{source_index}", str(raw.get("case_id", ""))}
        if (
            case_names & excluded_ids
            or f"train:{source_index}" in excluded_sources
            or source_fp in excluded_source_fps
            or fact_fp in excluded_fact_fps
            or fact_target_fp in excluded_target_fps
            or prompt_fingerprint(prompt) in excluded_prompt_fps
        ):
            counters["historical_excluded"] += 1
            continue
        target_ids = contextual_target_ids(tokenizer, prompt, target_new)
        true_ids = contextual_target_ids(tokenizer, prompt, target_true)
        if not target_ids or not true_ids or len(target_ids) > 6:
            counters["invalid_or_long_target"] += 1
            continue
        negative_relation = min(
            (value for value in relations if value != relation),
            key=lambda value: stable_hash(SEED, source_index, value),
        )
        same_subject = raw_templates[negative_relation].format(subject)
        legal.append(
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "protocol_version": CAMPAIGN_ID,
                "case_id": f"trm_cf_{source_index}",
                "source_dataset": dataset_name,
                "source_split": "train",
                "source_index": source_index,
                "source_fingerprint": source_fp,
                "fact_fingerprint": fact_fp,
                "fact_target_fingerprint": fact_target_fp,
                "prompt_fingerprint": prompt_fingerprint(prompt),
                "counterfact_raw_case_id": int(raw["case_id"]),
                "relation_id": relation,
                "subject": subject,
                "rewrite_template": template,
                "rewrite_prompt": prompt,
                "target_new": target_new,
                "target_true": target_true,
                "target_new_token_ids": list(map(int, target_ids)),
                "target_true_token_ids": list(map(int, true_ids)),
                "target_length": len(target_ids),
                "target_length_bin": target_length_bin(len(target_ids)),
                "tokenizer_model_id": PRIMARY_MODEL_ID,
                "tokenizer_revision": PRIMARY_MODEL_REVISION,
                "paraphrase_prompts": list(raw.get("paraphrase_prompts") or []),
                "near_locality_prompts": list(raw.get("neighborhood_prompts") or []),
                "attribute_prompts": list(raw.get("attribute_prompts") or []),
                "generation_prompts": list(raw.get("generation_prompts") or []),
                "same_subject_prompts": [same_subject],
                "same_subject_negative_relation_id": negative_relation,
                "same_subject_provenance": "fresh_cross_relation_template",
                "prompt_provenance": "real_azhx_counterfact_train",
                "train_seen": {"rewrite": True, "paraphrase": False, "locality": False},
            }
        )
    counters["legal_candidates"] = len(legal)
    return legal, dict(counters)


def _select_counterfact(candidates: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    used: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {}
    for offset, (role, count) in enumerate(CF_COUNTS.items()):
        selected = round_robin_stratified(
            list(candidates),
            count,
            seed=SEED + offset,
            used=used,
            group_fields=("target_length_bin", "relation_id"),
        )
        normalized = []
        for rank, source in enumerate(selected):
            row = dict(source)
            row["split_role"] = role
            row["selection_rank"] = rank
            row["role_access"] = "locked_confirmation_only" if "_locked_" in role else "development"
            normalized.append(row)
        splits[role] = normalized
    far_pool = splits["cf_trm_locality_pool_300"]
    for role in ("cf_trm_smoke_20", "cf_trm_pilot_100", "cf_trm_dev_200", "cf_trm_locked_500"):
        for row in splits[role]:
            selected = sorted(
                far_pool,
                key=lambda candidate: stable_hash(SEED, role, row["case_id"], candidate["case_id"]),
            )[:3]
            row["far_locality_cases"] = [
                {
                    "case_id": item["case_id"],
                    "prompt": item["rewrite_prompt"],
                    "target": item["target_true"],
                    "source_index": item["source_index"],
                    "prompt_provenance": "fresh_disjoint_counterfact_pool",
                }
                for item in selected
            ]
    return splits


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
    normalized = []
    for source in candidates:
        row = dict(source)
        row["campaign_id"] = CAMPAIGN_ID
        row["protocol_version"] = CAMPAIGN_ID
        row["case_id"] = row["case_id"].replace("kamel_pub_", "trm_kamel_", 1)
        row["tokenizer_revision"] = PRIMARY_MODEL_REVISION
        row["prompt_fingerprint"] = prompt_fingerprint(row["rewrite_prompt"])
        row["train_seen"] = {"rewrite": True, "paraphrase": False, "locality": False}
        normalized.append(row)
    splits: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    for length in (2, 3, 4):
        pool = [row for row in normalized if int(row["target_length"]) == length]
        for offset, (kind, count) in enumerate(KAMEL_COUNTS.items()):
            role = f"kamel_trm_{kind}_{count}_n{length}"
            selected = kamel_balanced_select(pool, count, role=role, used=used, seed=SEED + length * 10 + offset)
            for row in selected:
                row["role_access"] = "locked_confirmation_only" if kind == "locked" else "development"
            splits[role] = selected
    return splits, counters


def _summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = Counter(str(row["target_length"]) for row in rows)
    relations = Counter(str(row["relation_id"]) for row in rows)
    prompts = []
    for row in rows:
        prompts.append(prompt_fingerprint(str(row["rewrite_prompt"])))
        for field in ("paraphrase_prompts", "near_locality_prompts", "attribute_prompts", "generation_prompts", "same_subject_prompts"):
            prompts.extend(prompt_fingerprint(str(value)) for value in row.get(field, []))
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "count": len(rows),
        "unique_case_ids": len({str(row["case_id"]) for row in rows}),
        "unique_source_fingerprints": len({str(row["source_fingerprint"]) for row in rows}),
        "unique_prompt_fingerprints": len(set(prompts)),
        "target_length_histogram": dict(sorted(lengths.items())),
        "relation_histogram": dict(sorted(relations.items())),
        "locked": any(row.get("role_access") == "locked_confirmation_only" for row in rows),
        "opened_for_tuning": False,
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
                raise RuntimeError(f"Fresh protocol overlap: {left_name} vs {right_name}")
    return audit


def _prompt_overlap_audit(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    prompt_sets = {}
    for name, rows in splits.items():
        values = set()
        for row in rows:
            values.add(prompt_fingerprint(str(row["rewrite_prompt"])))
            for field in ("paraphrase_prompts", "near_locality_prompts", "attribute_prompts", "generation_prompts", "same_subject_prompts"):
                values.update(prompt_fingerprint(str(value)) for value in row.get(field, []))
        prompt_sets[name] = values
    names = list(splits)
    audit = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = prompt_sets[left] & prompt_sets[right]
            audit.append({"left": left, "right": right, "prompt_overlap_count": len(overlap)})
            if overlap:
                raise RuntimeError(f"Prompt overlap: {left} vs {right}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=CAMPAIGN_ROOT / "source_cache")
    parser.add_argument("--dataset", default="azhx/counterfact")
    args = parser.parse_args()
    started = now_utc()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_MODEL_ID,
        revision=PRIMARY_MODEL_REVISION,
        trust_remote_code=True,
    )
    exclusions = collect_historical_exclusions()
    counterfact_candidates, counterfact_filter = _counterfact_candidates(tokenizer, args.dataset, exclusions)
    cf_splits = _select_counterfact(counterfact_candidates)
    kamel_rows, templates, kamel_source = load_kamel_sources(args.cache_dir)
    kamel_splits, kamel_filter = _select_kamel(kamel_rows, templates, tokenizer, exclusions)
    splits = {**cf_splits, **kamel_splits}
    summaries = {}
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summaries[name] = _summary(path, rows)
    overlap = _overlap_audit(splits)
    prompt_overlap = _prompt_overlap_audit(splits)
    write_csv(args.output_dir / "split_overlap_audit.csv", overlap)
    write_csv(args.output_dir / "prompt_overlap_audit.csv", prompt_overlap)
    write_csv(args.output_dir / "historical_exclusion_audit.csv", exclusions["audit"])
    write_json(
        args.output_dir / "historical_exclusion_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "case_id_count": len(exclusions["case_ids"]),
            "source_key_count": len(exclusions["source_keys"]),
            "source_fingerprint_count": len(exclusions["source_fingerprints"]),
            "fact_fingerprint_count": len(exclusions["fact_fingerprints"]),
            "fact_target_fingerprint_count": len(exclusions["fact_target_fingerprints"]),
            "prompt_fingerprint_count": len(exclusions["prompt_fingerprints"]),
            "historical_locked_files_opened": False,
            "prompt_label_output_metric_fields_used_for_training": False,
        },
    )
    write_json(args.output_dir / "split_summary.json", {"splits": summaries})
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
    locked = {name: summary for name, summary in summaries.items() if summary["locked"]}
    write_json(
        args.output_dir / "locked_manifest_registry.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "manifests": locked,
            "access_policy": "fresh_locked_confirmation_only",
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    required = list(CF_COUNTS) + [
        f"kamel_trm_{kind}_{count}_n{length}"
        for length in (2, 3, 4)
        for kind, count in KAMEL_COUNTS.items()
    ]
    counterfact_multilength = all(
        "1" in summaries[role]["target_length_histogram"]
        and "2" in summaries[role]["target_length_histogram"]
        for role in ("cf_trm_pilot_100", "cf_trm_dev_200", "cf_trm_locked_500")
    )
    acceptance = {
        "all_required_manifests_exist": all(name in summaries for name in required),
        "zero_source_overlap": all(row["overlap_count"] == 0 for row in overlap),
        "zero_prompt_overlap": all(row["prompt_overlap_count"] == 0 for row in prompt_overlap),
        "counterfact_length_1_and_2_present": counterfact_multilength,
        "kamel_lengths_2_3_4_present": all(
            summaries[f"kamel_trm_pilot_100_n{length}"]["target_length_histogram"] == {str(length): 100}
            for length in (2, 3, 4)
        ),
        "training_anchor_families_present": all(
            any(row.get(field) for row in cf_splits["cf_trm_anchor_train_500"])
            for field in ("same_subject_prompts", "near_locality_prompts", "attribute_prompts", "generation_prompts")
        ),
        "historical_locked_files_opened": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = all(value for key, value in acceptance.items() if key not in {"analysis_500_used", "final_test_used", "historical_locked_files_opened"})
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "B0_fresh_protocol",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "split_summaries": summaries,
        "counterfact_filter": counterfact_filter,
        "kamel_filter": kamel_filter,
        "acceptance": acceptance,
        "acceptance_pass": passed,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "run_config.json", {"dataset": args.dataset, "seed": SEED})
    write_json(args.output_dir / "validation_report.json", acceptance)
    record_stage(
        "B0_fresh_protocol",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes="Fresh disjoint CounterFact/KAMEL manifests and train-only protection anchors.",
        next_stage="C0_timerome_source_reproduction" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(f"B0 fresh protocol passed: {args.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
