#!/usr/bin/env python3
"""Build fresh KAMEL dev/locked manifests with historical fingerprint exclusion."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_mdm_memit_protocol import load_kamel_sources
from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    collect_historical_kamel_exclusions,
    contextual_target_ids,
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


LLADA_DEV_PER_LENGTH = 200
LLADA_LOCKED_PER_LENGTH = {2: 300, 3: 500, 4: 500, 5: 300, 6: 300}
DREAM_DEV_PER_LENGTH = 100
DREAM_LOCKED_PER_LENGTH = 300
SEED = 260717011


def _render(template: str, subject: str) -> str:
    return template.replace("[S]", subject).strip()


def _paraphrase(template: str, subject: str) -> str:
    body = template.replace("[S]", subject).strip()
    if body:
        body = body[0].lower() + body[1:]
    return f"Regarding {subject}, {body}"


def _source_fingerprint(row: Mapping[str, Any]) -> str:
    return stable_hash(
        "KAMEL",
        "train",
        int(row["source_index"]),
        str(row["subject"]),
        str(row["relation_id"]),
    )


def _balanced_select(
    rows: Sequence[dict[str, Any]], count: int, *, role: str, used: set[str], seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["source_fingerprint"] not in used:
            groups[str(row["relation_id"])].append(row)
    for relation, values in groups.items():
        values.sort(key=lambda row: stable_hash(seed, role, relation, row["case_id"]))
    relations = sorted(groups, key=lambda value: stable_hash(seed, role, value))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        made_progress = False
        for relation in relations:
            values = groups[relation]
            while values and values[0]["source_fingerprint"] in used:
                values.pop(0)
            if not values:
                continue
            item = dict(values.pop(0))
            used.add(item["source_fingerprint"])
            item["split_role"] = role
            item["selection_rank"] = len(selected)
            selected.append(item)
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            break
    if len(selected) != count:
        available = sum(
            row["source_fingerprint"] not in used for row in rows
        ) + len(selected)
        raise RuntimeError(f"{role}: selected {len(selected)}/{count}; available before selection={available}")
    return selected


def _candidate_pool(
    raw_rows: Sequence[Mapping[str, Any]],
    templates: Mapping[str, str],
    tokenizer: Any,
    *,
    model_id: str,
    lengths: set[int],
    exclusions: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded_sources = set(exclusions["source_fingerprints"])
    excluded_facts = set(exclusions["fact_fingerprints"])
    excluded_fact_targets = set(exclusions["fact_target_fingerprints"])
    source_labels: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(dict)
    staged: list[dict[str, Any]] = []
    counters = defaultdict(int)
    for raw in raw_rows:
        relation = str(raw["relation_id"])
        if relation not in templates:
            continue
        source_fp = _source_fingerprint(raw)
        if source_fp in excluded_sources:
            counters["historical_source_excluded"] += 1
            continue
        subject = str(raw["subject"]).strip()
        target_true = str(raw["target_true"]).strip()
        fact_fp = stable_hash(relation, subject.casefold(), target_true.casefold())
        if fact_fp in excluded_facts:
            counters["historical_fact_excluded"] += 1
            continue
        prompt = _render(templates[relation], subject)
        true_ids = contextual_target_ids(tokenizer, prompt, target_true)
        if len(true_ids) not in lengths:
            counters["target_length_filtered"] += 1
            continue
        row = dict(raw)
        row.update(
            {
                "subject": subject,
                "target_true": target_true,
                "prompt": prompt,
                "target_true_token_ids": true_ids,
                "target_length": len(true_ids),
                "source_fingerprint": source_fp,
                "fact_fingerprint": fact_fp,
            }
        )
        staged.append(row)
        for label in raw.get("object_labels", []):
            label = str(label).strip()
            if not label:
                continue
            ids = contextual_target_ids(tokenizer, prompt, label)
            if len(ids) == len(true_ids):
                source_labels[(relation, len(true_ids))][label] = ids

    candidates: list[dict[str, Any]] = []
    all_relations = sorted(templates)
    for raw in staged:
        relation = str(raw["relation_id"])
        target_options = []
        for label, ids in source_labels[(relation, int(raw["target_length"]))].items():
            if label.casefold() == str(raw["target_true"]).casefold():
                continue
            pair_fp = stable_hash(relation, str(raw["subject"]).casefold(), label.casefold())
            if pair_fp in excluded_fact_targets:
                continue
            target_options.append((stable_hash(seed, raw["source_fingerprint"], label), label, ids, pair_fp))
        if not target_options:
            counters["no_legal_counterfactual_target"] += 1
            continue
        _, target_new, target_ids, pair_fp = min(target_options)
        negative_relation = next(
            candidate
            for candidate in sorted(
                (value for value in all_relations if value != relation),
                key=lambda value: stable_hash(seed, raw["source_fingerprint"], value),
            )
            if candidate in templates
        )
        same_subject_prompt = _render(templates[negative_relation], str(raw["subject"]))
        source_index = int(raw["source_index"])
        case_id = (
            f"kamel_pub_{model_id.split('/')[-1].lower()}_{relation}_{source_index}_"
            f"{stable_hash(raw['subject'])[:10]}"
        )
        candidates.append(
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "case_id": case_id,
                "source_dataset": "JanKalo/KAMEL",
                "source_split": "train",
                "source_index": source_index,
                "source_record_index": raw.get("source_record_index"),
                "source_fingerprint": raw["source_fingerprint"],
                "fact_fingerprint": raw["fact_fingerprint"],
                "fact_target_fingerprint": pair_fp,
                "relation_id": relation,
                "subject": raw["subject"],
                "rewrite_template": templates[relation].replace("[S]", "{}"),
                "rewrite_prompt": raw["prompt"],
                "target_new": target_new,
                "target_true": raw["target_true"],
                "target_new_token_ids": list(map(int, target_ids)),
                "target_true_token_ids": list(map(int, raw["target_true_token_ids"])),
                "target_length": int(raw["target_length"]),
                "target_length_bin": str(raw["target_length"]),
                "tokenizer_model_id": model_id,
                "paraphrase_prompts": [_paraphrase(templates[relation], str(raw["subject"]))],
                "paraphrase_provenance": "deterministic_held_out_relation_template_rewrite",
                "same_subject_prompts": [same_subject_prompt],
                "same_subject_negative_relation_id": negative_relation,
                "same_subject_provenance": "documented_cross_relation_template_construction",
                "prompt_provenance": "real_KAMEL_question_template",
                "counterfactual_target_policy": "same_relation_same_contextual_length",
                "train_seen": {"rewrite": True, "paraphrase": False, "locality": False},
            }
        )
    counters["candidate_count"] = len(candidates)
    return candidates, dict(counters)


def _summarize(path: Path, rows: Sequence[Mapping[str, Any]], *, locked: bool) -> dict[str, Any]:
    relation_hist = histogram(row["relation_id"] for row in rows)
    maximum_share = max(relation_hist.values(), default=0) / max(len(rows), 1)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "count": len(rows),
        "unique_source_fingerprints": len({row["source_fingerprint"] for row in rows}),
        "target_length_histogram": histogram(row["target_length"] for row in rows),
        "relation_histogram": relation_hist,
        "relation_count": len(relation_hist),
        "maximum_relation_share": maximum_share,
        "locked": locked,
        "opened_for_method_selection": False,
    }


def _assert_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    names = list(splits)
    audit = []
    for index, left_name in enumerate(names):
        left = {str(row["source_fingerprint"]) for row in splits[left_name]}
        for right_name in names[index + 1 :]:
            right = {str(row["source_fingerprint"]) for row in splits[right_name]}
            overlap = left & right
            audit.append(
                {"left": left_name, "right": right_name, "overlap_count": len(overlap)}
            )
            if overlap:
                raise RuntimeError(f"Fresh split overlap: {left_name} vs {right_name}: {len(overlap)}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=CAMPAIGN_ROOT / "source_cache")
    parser.add_argument("--llada_model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--llada_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--dream_model_id", default=SECONDARY_MODEL_ID)
    parser.add_argument("--dream_revision", default=SECONDARY_MODEL_REVISION)
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    started = now_utc()
    report_path = args.output_dir / "report_summary.json"
    if report_path.exists() and not args.allow_overwrite:
        raise FileExistsError(report_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    llada = AutoTokenizer.from_pretrained(
        args.llada_model_id, revision=args.llada_revision, trust_remote_code=True
    )
    dream = AutoTokenizer.from_pretrained(
        args.dream_model_id, revision=args.dream_revision, trust_remote_code=True
    )
    raw_rows, templates, source = load_kamel_sources(args.cache_dir)
    exclusions = collect_historical_kamel_exclusions()
    llada_candidates, llada_filter = _candidate_pool(
        raw_rows,
        templates,
        llada,
        model_id=args.llada_model_id,
        lengths={2, 3, 4, 5, 6},
        exclusions=exclusions,
        seed=SEED,
    )
    dream_candidates, dream_filter = _candidate_pool(
        raw_rows,
        templates,
        dream,
        model_id=args.dream_model_id,
        lengths={3, 4, 5},
        exclusions=exclusions,
        seed=SEED + 1,
    )

    splits: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    for length in (2, 3, 4, 5, 6):
        pool = [row for row in llada_candidates if int(row["target_length"]) == length]
        dev_name = f"kamel_pub_dev_n{length}"
        splits[dev_name] = _balanced_select(
            pool, LLADA_DEV_PER_LENGTH, role=dev_name, used=used, seed=SEED + length
        )
        locked_name = f"kamel_pub_locked_n{length}"
        splits[locked_name] = _balanced_select(
            pool,
            LLADA_LOCKED_PER_LENGTH[length],
            role=locked_name,
            used=used,
            seed=SEED + 100 + length,
        )
    for length in (3, 4, 5):
        pool = [row for row in dream_candidates if int(row["target_length"]) == length]
        dev_name = f"dream_pub_dev_n{length}"
        splits[dev_name] = _balanced_select(
            pool, DREAM_DEV_PER_LENGTH, role=dev_name, used=used, seed=SEED + 200 + length
        )
        locked_name = f"dream_pub_locked_n{length}"
        splits[locked_name] = _balanced_select(
            pool,
            DREAM_LOCKED_PER_LENGTH,
            role=locked_name,
            used=used,
            seed=SEED + 300 + length,
        )

    summaries: dict[str, Any] = {}
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summaries[name] = _summarize(path, rows, locked="_locked_" in name)
    overlaps = _assert_disjoint(splits)
    write_csv(args.output_dir / "split_overlap_audit.csv", overlaps)
    write_json(
        args.output_dir / "historical_exclusion_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "historical_protocol": "masked_diffusion_memit_sb_positive_result_v1",
            "excluded_case_id_count": len(exclusions["case_ids"]),
            "excluded_source_fingerprint_count": len(exclusions["source_fingerprints"]),
            "excluded_fact_fingerprint_count": len(exclusions["fact_fingerprints"]),
            "excluded_fact_target_fingerprint_count": len(exclusions["fact_target_fingerprints"]),
            "source_manifest_audit": exclusions["audit"],
            "fields_used_for_exclusion_only": True,
            "historical_prompt_or_metric_fields_used": False,
        },
    )
    write_json(args.output_dir / "split_summary.json", summaries)
    write_json(
        args.output_dir / "locked_manifest_registry.json",
        {
            name: value
            for name, value in summaries.items()
            if bool(value["locked"])
        },
    )
    write_json(
        args.output_dir / "source_registry.json",
        {
            **source,
            "llada_tokenizer": {
                "model_id": args.llada_model_id,
                "revision": args.llada_revision,
            },
            "dream_tokenizer": {
                "model_id": args.dream_model_id,
                "revision": args.dream_revision,
            },
            "llada_filter_counts": llada_filter,
            "dream_filter_counts": dream_filter,
        },
    )
    primary_counts_ok = all(
        summaries[f"kamel_pub_dev_n{length}"]["count"] == LLADA_DEV_PER_LENGTH
        and summaries[f"kamel_pub_locked_n{length}"]["count"]
        == LLADA_LOCKED_PER_LENGTH[length]
        for length in (2, 3, 4, 5, 6)
    )
    dream_counts_ok = all(
        summaries[f"dream_pub_dev_n{length}"]["count"] == DREAM_DEV_PER_LENGTH
        and summaries[f"dream_pub_locked_n{length}"]["count"] == DREAM_LOCKED_PER_LENGTH
        for length in (3, 4, 5)
    )
    primary_relation_ok = all(
        summaries[f"kamel_pub_locked_n{length}"]["relation_count"] >= 20
        and summaries[f"kamel_pub_locked_n{length}"]["maximum_relation_share"] <= 0.20
        for length in (3, 4)
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P0",
        "stage": "P0_fresh_data_protocol",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "selection_seed": SEED,
        "llada_counts_exact": primary_counts_ok,
        "dream_counts_exact": dream_counts_ok,
        "primary_relation_coverage_pass": primary_relation_ok,
        "zero_cross_split_source_overlap": not any(row["overlap_count"] for row in overlaps),
        "historical_source_fingerprint_overlap": 0,
        "historical_fact_fingerprint_overlap": 0,
        "locked_manifests_opened_for_method_selection": False,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "split_summaries": summaries,
        "acceptance_pass": primary_counts_ok and dream_counts_ok and primary_relation_ok,
    }
    write_json(report_path, report)
    if not report["acceptance_pass"]:
        write_json(
            args.output_dir / "data_feasibility_report.json",
            {
                "primary_counts_ok": primary_counts_ok,
                "dream_counts_ok": dream_counts_ok,
                "primary_relation_ok": primary_relation_ok,
                "classification_cap": "narrow_method_ready",
            },
        )
    record_stage(
        stage="P0_fresh_data_protocol",
        track="P0",
        status="passed" if report["acceptance_pass"] else "data_feasibility_limited",
        output_dir=args.output_dir,
        acceptance_pass=bool(report["acceptance_pass"]),
        started_at_utc=started,
        notes=(
            f"llada_counts={primary_counts_ok}; dream_counts={dream_counts_ok}; "
            f"primary_relation_coverage={primary_relation_ok}; locked sets remain unopened"
        ),
        next_stage="P1_partial_state_memit_discrepancy",
    )
    print(json.dumps({"acceptance_pass": report["acceptance_pass"], "splits": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
