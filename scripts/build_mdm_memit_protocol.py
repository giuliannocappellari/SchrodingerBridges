#!/usr/bin/env python3
"""Build fresh CounterFact and KAMEL manifests for the MDM-MEMIT campaign."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ID,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_ROOT,
    collect_historical_exclusions,
    git_commit,
    histogram,
    now_utc,
    read_jsonl,
    record_stage,
    repo_path,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


CF_SPLITS = {
    "cf_memit_smoke_20": 20,
    "cf_layer_select_500": 500,
    "cf_repro_main_500": 500,
    "cf_sb_dev_200": 200,
    "cf_sb_analysis_200": 200,
}
KAMEL_COUNTS = {
    "kamel_smoke_20_per_length": 20,
    "kamel_dev_50_per_length": 50,
    "kamel_repro_200_per_length": 200,
}
KAMEL_ARCHIVE_URL = "https://github.com/JanKalo/KAMEL/raw/master/data/kamel.zip"
KAMEL_TEMPLATES_URL = "https://raw.githubusercontent.com/JanKalo/KAMEL/21625baba6439faea03e61c28ce29475dc4996f6/question-templates.csv"


def contextual_target_ids(tokenizer: Any, prompt: str, target: str) -> list[int]:
    """Tokenize an answer in prompt context, with a stable leading-space fallback."""

    prefix = str(prompt).rstrip()
    combined = tokenizer(prefix + " " + str(target).strip(), add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if len(combined) > len(prefix_ids) and combined[: len(prefix_ids)] == prefix_ids:
        return list(map(int, combined[len(prefix_ids) :]))
    return list(map(int, tokenizer(" " + str(target).strip(), add_special_tokens=False)["input_ids"]))


def fingerprint_row(source: str, split: str, index: int, subject: str, relation: str) -> str:
    return stable_hash(source, split, index, subject, relation)


def round_robin_stratified(
    rows: Sequence[dict[str, Any]],
    count: int,
    *,
    seed: int,
    used: set[str],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["case_id"] in used:
            continue
        key = tuple(str(row.get(field, "")) for field in group_fields)
        groups[key].append(row)
    for key, values in groups.items():
        values.sort(key=lambda item: stable_hash(seed, key, item["case_id"]))
    keys = sorted(groups, key=lambda key: stable_hash(seed, key))
    selected: list[dict[str, Any]] = []
    cursor = 0
    empty_passes = 0
    while len(selected) < count and keys:
        key = keys[cursor % len(keys)]
        cursor += 1
        values = groups[key]
        while values and values[0]["case_id"] in used:
            values.pop(0)
        if values:
            item = values.pop(0)
            selected.append(item)
            used.add(item["case_id"])
            empty_passes = 0
        else:
            empty_passes += 1
            if empty_passes >= len(keys) * 2:
                keys = [candidate for candidate in keys if groups[candidate]]
                empty_passes = 0
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} of {count} requested rows")
    return selected


def build_counterfact(tokenizer: Any, dataset_name: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    exclusions = collect_historical_exclusions()
    excluded_ids = set(exclusions["case_ids"])
    excluded_sources = set(exclusions["source_keys"])
    dataset = load_dataset(dataset_name, split="train")
    legal: list[dict[str, Any]] = []
    for source_index, raw in enumerate(dataset):
        rewrite = raw["requested_rewrite"]
        subject = str(rewrite["subject"])
        relation = str(rewrite["relation_id"])
        template = str(rewrite["prompt"])
        prompt = template.format(subject)
        target_new = str(rewrite["target_new"]["str"]).strip()
        target_true = str(rewrite["target_true"]["str"]).strip()
        case_id = f"counterfact_train_{source_index}"
        if case_id in excluded_ids or f"train:{source_index}" in excluded_sources:
            continue
        target_ids = contextual_target_ids(tokenizer, prompt, target_new)
        true_ids = contextual_target_ids(tokenizer, prompt, target_true)
        if not target_ids or not true_ids:
            continue
        legal.append(
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "case_id": case_id,
                "source_dataset": dataset_name,
                "source_split": "train",
                "source_index": source_index,
                "source_fingerprint": fingerprint_row(dataset_name, "train", source_index, subject, relation),
                "counterfact_raw_case_id": int(raw["case_id"]),
                "relation_id": relation,
                "subject": subject,
                "rewrite_template": template,
                "rewrite_prompt": prompt,
                "target_new": target_new,
                "target_true": target_true,
                "target_new_token_ids": target_ids,
                "target_true_token_ids": true_ids,
                "target_length": len(target_ids),
                "target_length_bin": ">=4" if len(target_ids) >= 4 else str(len(target_ids)),
                "paraphrase_prompts": list(raw.get("paraphrase_prompts") or []),
                "neighborhood_prompts": list(raw.get("neighborhood_prompts") or []),
                "attribute_prompts": list(raw.get("attribute_prompts") or []),
                "generation_prompts": list(raw.get("generation_prompts") or []),
                "prompt_provenance": "real_azhx_counterfact_train",
                "train_seen": {"rewrite": True, "paraphrase": False, "locality": False},
            }
        )
    # The paper reproduction is dominated by valid single-token targets. Keep
    # multi-token rows available for the SB dev strata but prioritize length 1.
    legal.sort(key=lambda row: (row["target_length"] != 1, stable_hash(seed, row["case_id"])))
    used: set[str] = set()
    result: dict[str, list[dict[str, Any]]] = {}
    for offset, (name, count) in enumerate(CF_SPLITS.items()):
        candidate_pool = legal
        if name in {"cf_memit_smoke_20", "cf_layer_select_500", "cf_repro_main_500"}:
            singles = [row for row in legal if row["target_length"] == 1]
            if sum(row["case_id"] not in used for row in singles) >= count:
                candidate_pool = singles
        selected = round_robin_stratified(
            candidate_pool,
            count,
            seed=seed + offset,
            used=used,
            group_fields=("target_length_bin", "relation_id"),
        )
        for rank, row in enumerate(selected):
            row["split_role"] = name
            row["selection_rank"] = rank
        result[name] = selected
    write_json(
        PROTOCOL_ROOT / "historical_exclusion_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "case_ids": exclusions["case_ids"],
            "source_keys": exclusions["source_keys"],
            "source_files": exclusions["audit"],
            "prompt_label_output_metric_fields_used": False,
        },
    )
    write_csv(PROTOCOL_ROOT / "historical_exclusion_audit.csv", exclusions["audit"])
    return result


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, path.open("wb") as handle:
            handle.write(response.read())
    return path


def load_kamel_sources(cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    archive = _download(KAMEL_ARCHIVE_URL, cache_dir / "kamel.zip")
    templates_path = _download(KAMEL_TEMPLATES_URL, cache_dir / "question-templates.csv")
    templates: dict[str, str] = {}
    with templates_path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                templates[row[0].strip()] = ",".join(row[1:]).strip()
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        names = sorted(name for name in zf.namelist() if name.endswith("/train.jsonl") and not name.startswith("__MACOSX"))
        for name in names:
            relation = name.split("/")[0]
            if relation not in templates:
                continue
            with zf.open(name) as raw_handle:
                for line_no, line in enumerate(io.TextIOWrapper(raw_handle, encoding="utf-8")):
                    payload = json.loads(line)
                    labels = [str(value).strip() for value in payload.get("obj_label", []) if str(value).strip()]
                    if not labels:
                        continue
                    rows.append(
                        {
                            "relation_id": relation,
                            "source_index": line_no,
                            "source_record_index": payload.get("index"),
                            "subject": str(payload["sub_label"]).strip(),
                            "object_labels": labels,
                            "target_true": labels[0],
                            "question_template": templates[relation],
                        }
                    )
    source = {
        "dataset": "LeandraFichtel/KAMEL from JanKalo/KAMEL source archive",
        "archive_url": KAMEL_ARCHIVE_URL,
        "archive_sha256": sha256_file(archive),
        "templates_url": KAMEL_TEMPLATES_URL,
        "templates_sha256": sha256_file(templates_path),
        "source_commit": "21625baba6439faea03e61c28ce29475dc4996f6",
    }
    return rows, templates, source


def _render_kamel_prompt(template: str, subject: str) -> str:
    return template.replace("[S]", subject).strip()


def _paraphrase_template(template: str) -> str:
    body = template.replace("[S]", "this subject").strip()
    if body:
        body = body[0].lower() + body[1:]
    return "Regarding [S], " + body


def build_kamel(
    tokenizer: Any,
    *,
    seed: int,
    cache_dir: Path,
    dual_tokenizer: Any | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw_rows, templates, source = load_kamel_sources(cache_dir)
    candidates: list[dict[str, Any]] = []
    by_relation_length: dict[tuple[str, int], list[tuple[str, list[int]]]] = defaultdict(list)
    staged: list[dict[str, Any]] = []
    for raw in raw_rows:
        prompt = _render_kamel_prompt(raw["question_template"], raw["subject"])
        true_ids = contextual_target_ids(tokenizer, prompt, raw["target_true"])
        if len(true_ids) not in {1, 2, 3, 4}:
            continue
        if dual_tokenizer is not None and len(contextual_target_ids(dual_tokenizer, prompt, raw["target_true"])) != len(true_ids):
            continue
        item = dict(raw)
        item["prompt"] = prompt
        item["target_true_token_ids"] = true_ids
        item["target_length"] = len(true_ids)
        staged.append(item)
        for label in raw["object_labels"]:
            label_ids = contextual_target_ids(tokenizer, prompt, label)
            if len(label_ids) == len(true_ids):
                if dual_tokenizer is None or len(contextual_target_ids(dual_tokenizer, prompt, label)) == len(true_ids):
                    by_relation_length[(raw["relation_id"], len(true_ids))].append((label, label_ids))
    for item in staged:
        pool = sorted(
            {
                label: ids
                for label, ids in by_relation_length[(item["relation_id"], item["target_length"])]
                if label.casefold() != item["target_true"].casefold()
            }.items(),
            key=lambda pair: stable_hash(seed, item["relation_id"], item["subject"], pair[0]),
        )
        if not pool:
            continue
        target_new, target_ids = pool[0]
        source_index = int(item["source_index"])
        relation = item["relation_id"]
        case_id = f"kamel_train_{relation}_{source_index}_{stable_hash(item['subject'])[:8]}"
        paraphrase_template = _paraphrase_template(item["question_template"])
        candidates.append(
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "case_id": case_id,
                "source_dataset": "LeandraFichtel/KAMEL",
                "source_split": "train",
                "source_index": source_index,
                "source_record_index": item["source_record_index"],
                "source_fingerprint": fingerprint_row("KAMEL", "train", source_index, item["subject"], relation),
                "relation_id": relation,
                "subject": item["subject"],
                "rewrite_template": item["question_template"].replace("[S]", "{}"),
                "rewrite_prompt": item["prompt"],
                "target_new": target_new,
                "target_true": item["target_true"],
                "target_new_token_ids": list(map(int, target_ids)),
                "target_true_token_ids": item["target_true_token_ids"],
                "target_length": item["target_length"],
                "target_length_bin": str(item["target_length"]),
                "paraphrase_prompts": [paraphrase_template.replace("[S]", item["subject"])],
                "paraphrase_provenance": "deterministic_held_out_relation_question_rewrite",
                "prompt_provenance": "real_KAMEL_question_template",
                "counterfactual_target_policy": "same_relation_same_contextual_length",
                "train_seen": {"rewrite": True, "paraphrase": False, "locality": False},
            }
        )
    used: set[str] = set()
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in KAMEL_COUNTS}
    for length in (1, 2, 3, 4):
        length_rows = [row for row in candidates if row["target_length"] == length]
        for offset, (name, per_length) in enumerate(KAMEL_COUNTS.items()):
            selected = round_robin_stratified(
                length_rows,
                per_length,
                seed=seed + 100 * length + offset,
                used=used,
                group_fields=("relation_id",),
            )
            for row in selected:
                row["split_role"] = name
                row["selection_rank"] = len(result[name])
            result[name].extend(selected)
    source.update(
        {
            "dual_tokenizer_filter_applied": dual_tokenizer is not None,
            "primary_tokenizer": MODEL_ID,
            "secondary_tokenizer": getattr(dual_tokenizer, "name_or_path", None),
            "candidate_count": len(candidates),
            "relation_template_count": len(templates),
        }
    )
    return result, source


def overlap_audit(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = list(splits)
    for index, left in enumerate(names):
        left_ids = {str(row["case_id"]) for row in splits[left]}
        for right in names[index + 1 :]:
            right_ids = {str(row["case_id"]) for row in splits[right]}
            rows.append({"left": left, "right": right, "overlap_count": len(left_ids & right_ids)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--counterfact_dataset", default="azhx/counterfact")
    parser.add_argument("--seed", type=int, default=260603924)
    parser.add_argument("--build_kamel", type=int, choices=[0, 1], default=1)
    parser.add_argument("--dual_tokenizer_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    started = now_utc()
    out = repo_path(args.output_dir)
    report_path = out / "report_summary.json"
    if report_path.exists() and not args.allow_overwrite:
        raise FileExistsError(report_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True
    )
    cf_splits = build_counterfact(tokenizer, args.counterfact_dataset, args.seed)
    all_splits: dict[str, list[dict[str, Any]]] = dict(cf_splits)
    kamel_source: dict[str, Any] = {"built": False}
    if args.build_kamel:
        dual = None
        dual_error = None
        try:
            dual = AutoTokenizer.from_pretrained(args.dual_tokenizer_id)
        except Exception as exc:  # gated LLaMA access is optional under the plan
            dual_error = f"{type(exc).__name__}: {exc}"
        kamel_splits, kamel_source = build_kamel(
            tokenizer,
            seed=args.seed,
            cache_dir=out / "source_cache",
            dual_tokenizer=dual,
        )
        kamel_source["dual_tokenizer_unavailable_reason"] = dual_error
        kamel_source["built"] = True
        all_splits.update(kamel_splits)

    for name, rows in all_splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    overlaps = overlap_audit(all_splits)
    if any(row["overlap_count"] for row in overlaps):
        raise RuntimeError("Fresh campaign manifests overlap")
    write_csv(args.output_dir / "split_overlap_audit.csv", overlaps)

    summaries: dict[str, Any] = {}
    for name, rows in all_splits.items():
        path = args.output_dir / f"{name}.jsonl"
        summaries[name] = {
            "count": len(rows),
            "unique_edits": len({row["case_id"] for row in rows}),
            "target_length_histogram": histogram(row["target_length"] for row in rows),
            "relation_histogram": histogram(row["relation_id"] for row in rows),
            "relation_count": len({row["relation_id"] for row in rows}),
            "sha256": sha256_file(path),
        }
    write_json(args.output_dir / "split_summary.json", summaries)
    write_json(args.output_dir / "kamel_source_registry.json", kamel_source)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "B_fresh_data_protocol",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id_for_tokenization": args.model_id,
        "model_revision_for_tokenization": args.model_revision,
        "counterfact_dataset": args.counterfact_dataset,
        "selection_seed": args.seed,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "historical_locked_fields_used_for_exclusion_only": True,
        "locked_prompt_label_output_metric_fields_used": False,
        "counterfact_counts_exact": all(len(cf_splits[name]) == count for name, count in CF_SPLITS.items()),
        "kamel_counts_exact": (
            not args.build_kamel
            or all(len(all_splits[name]) == count * 4 for name, count in KAMEL_COUNTS.items())
        ),
        "zero_overlap": not any(row["overlap_count"] for row in overlaps),
        "split_summary": summaries,
        "kamel_source": kamel_source,
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="B_fresh_data_protocol",
        status="passed",
        output_dir=args.output_dir,
        acceptance_pass=True,
        started_at_utc=started,
        notes="Fresh CounterFact and KAMEL manifests created without historical prompt reuse.",
    )
    print(f"counterfact_counts_exact={report['counterfact_counts_exact']}")
    print(f"kamel_counts_exact={report['kamel_counts_exact']}")


if __name__ == "__main__":
    main()
