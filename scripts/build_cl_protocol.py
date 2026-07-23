#!/usr/bin/env python3
"""Build fresh, disjoint sequential CounterFact and KAMEL streams."""

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
from scripts.build_mdm_memit_protocol import load_kamel_sources, round_robin_stratified
from scripts.build_trm_protocol import _counterfact_candidates
from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    SEED,
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
    "cf_cl_smoke_20": 20,
    "cf_cl_pilot_100": 100,
    "cf_cl_confirmation_200": 200,
    "cf_cl_scale_500": 500,
    "base_denoising_retention_500": 500,
}
CF_BLOCK_SIZES = {
    "cf_cl_smoke_20": 5,
    "cf_cl_pilot_100": 10,
    "cf_cl_confirmation_200": 10,
    "cf_cl_scale_500": 50,
}
KAMEL_COUNTS = {"pilot": 30, "confirmation": 60}


def prompt_fingerprint(prompt: str) -> str:
    return stable_hash("prompt", " ".join(str(prompt).casefold().split()))


def normalize_counterfact(source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    index = int(row["source_index"])
    row.update(
        {
            "campaign_id": CAMPAIGN_ID,
            "protocol_version": CAMPAIGN_ID,
            "case_id": f"cl_cf_{index}",
            "tokenizer_model_id": PRIMARY_MODEL_ID,
            "tokenizer_revision": PRIMARY_MODEL_REVISION,
            "prompt_fingerprint": prompt_fingerprint(str(row["rewrite_prompt"])),
        }
    )
    return row


def _deduplicate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = prompt_fingerprint(str(row["rewrite_prompt"]))
        current = chosen.get(key)
        if current is None or stable_hash(SEED, row["case_id"]) < stable_hash(SEED, current["case_id"]):
            chosen[key] = row
    return list(chosen.values())


def _choose_unique(
    prompts: Sequence[str], used: set[str], *, count: int
) -> list[str]:
    output = []
    for prompt in prompts:
        value = str(prompt).strip()
        if not value:
            continue
        fingerprint = prompt_fingerprint(value)
        if fingerprint in used:
            continue
        used.add(fingerprint)
        output.append(value)
        if len(output) == count:
            break
    return output


def _prepare_edit_rows(
    role: str,
    rows: Sequence[dict[str, Any]],
    *,
    block_size: int,
    used_prompts: set[str],
    far_donors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    anchors = []
    far_cursor = 0
    for rank, source in enumerate(rows):
        row = dict(source)
        rewrite_fp = prompt_fingerprint(str(row["rewrite_prompt"]))
        # All selected primary rewrites are reserved before auxiliary allocation.
        used_prompts.add(rewrite_fp)
        paraphrases = _choose_unique(row.get("paraphrase_prompts", []), used_prompts, count=2)
        near_candidates = list(row.get("near_locality_prompts", []))
        near_eval = _choose_unique(near_candidates, used_prompts, count=2)
        near_train = _choose_unique(near_candidates[len(near_eval) :], used_prompts, count=2)
        same_candidates = [
            str(item.get("prompt", ""))
            for item in row.get("same_subject_prompt_candidates", [])
        ]
        same_eval = _choose_unique(same_candidates, used_prompts, count=1)
        same_train = _choose_unique(same_candidates[len(same_eval) :], used_prompts, count=1)
        if not paraphrases or not near_eval or not same_eval:
            raise RuntimeError(f"Incomplete held-out prompt coverage for {row['case_id']}")
        far = None
        while far_cursor < len(far_donors):
            donor = far_donors[far_cursor]
            far_cursor += 1
            fingerprint = prompt_fingerprint(str(donor["rewrite_prompt"]))
            if fingerprint in used_prompts:
                continue
            used_prompts.add(fingerprint)
            far = {
                "case_id": donor["case_id"],
                "prompt": donor["rewrite_prompt"],
                "target": donor["target_true"],
                "source_index": donor["source_index"],
                "source_fingerprint": donor["source_fingerprint"],
            }
            break
        if far is None:
            raise RuntimeError("Insufficient far-locality donors")
        row.update(
            {
                "split_role": role,
                "selection_rank": rank,
                "block_index": rank // block_size,
                "position_in_block": rank % block_size,
                "stream_order": rank,
                "role_access": "fresh_confirmation_only" if "confirmation" in role else "development",
                "paraphrase_prompts": paraphrases,
                "near_locality_prompts": near_eval,
                "same_subject_prompts": same_eval,
                "far_locality_cases": [far],
                "attribute_prompts": [],
                "generation_prompts": [],
                "train_seen": {"rewrite": True, "paraphrase": False, "locality": False},
            }
        )
        output.append(row)
        for category, prompts in (
            ("same_subject", same_train),
            ("near_locality", near_train),
        ):
            for offset, prompt in enumerate(prompts):
                anchors.append(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "split_role": f"{role}_training_anchors",
                        "case_id": row["case_id"],
                        "edit_block": row["block_index"],
                        "anchor_id": f"{row['case_id']}_{category}_{offset}",
                        "anchor_category": category,
                        "prompt": prompt,
                        "prompt_fingerprint": prompt_fingerprint(prompt),
                        "subject": row["subject"],
                        "relation_id": row["relation_id"],
                        "target_new": row["target_new"],
                        "target_true": row["target_true"],
                        "evaluation_prompt": False,
                    }
                )
    return output, anchors


def _source_fingerprints(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    output = {str(row["source_fingerprint"]) for row in rows}
    for row in rows:
        output.update(str(item["source_fingerprint"]) for item in row.get("far_locality_cases", []))
    return output


def _prompt_fingerprints(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    output = set()
    for row in rows:
        output.add(prompt_fingerprint(str(row["rewrite_prompt"])))
        for field in ("paraphrase_prompts", "near_locality_prompts", "same_subject_prompts"):
            output.update(prompt_fingerprint(str(prompt)) for prompt in row.get(field, []))
        output.update(prompt_fingerprint(str(item["prompt"])) for item in row.get("far_locality_cases", []))
    return output


def _manifest_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "count": len(rows),
        "unique_case_ids": len({str(row["case_id"]) for row in rows}),
        "target_length_histogram": dict(sorted(Counter(str(row.get("target_length")) for row in rows).items())),
        "relation_histogram": dict(sorted(Counter(str(row.get("relation_id")) for row in rows).items())),
        "block_histogram": dict(sorted(Counter(str(row.get("block_index")) for row in rows).items())),
        "locked": any(row.get("role_access") == "fresh_confirmation_only" for row in rows),
        "opened_for_tuning": False,
    }


def build_counterfact(tokenizer: Any, exclusions: Mapping[str, Any], dataset: str):
    raw, filter_report = _counterfact_candidates(tokenizer, dataset, exclusions)
    candidates = _deduplicate(
        [normalize_counterfact(row) for row in raw if int(row["target_length"]) == 1]
    )
    used_ids: set[str] = set()
    selected: dict[str, list[dict[str, Any]]] = {}
    for offset, (role, count) in enumerate(CF_COUNTS.items()):
        selected[role] = round_robin_stratified(
            candidates,
            count,
            seed=SEED + offset,
            used=used_ids,
            group_fields=("relation_id",),
        )
    donor_pool = [row for row in candidates if row["case_id"] not in used_ids]
    donor_pool.sort(key=lambda row: stable_hash(SEED, "far", row["case_id"]))
    primary_rows = [row for rows in selected.values() for row in rows]
    rewrite_fps = [prompt_fingerprint(str(row["rewrite_prompt"])) for row in primary_rows]
    if len(rewrite_fps) != len(set(rewrite_fps)):
        raise RuntimeError("Fresh CounterFact selections contain duplicate rewrite prompts")
    used_prompts: set[str] = set(rewrite_fps)
    anchors: list[dict[str, Any]] = []
    streams: dict[str, list[dict[str, Any]]] = {}
    for role in ("cf_cl_smoke_20", "cf_cl_pilot_100", "cf_cl_confirmation_200", "cf_cl_scale_500"):
        prepared, role_anchors = _prepare_edit_rows(
            role,
            selected[role],
            block_size=CF_BLOCK_SIZES[role],
            used_prompts=used_prompts,
            far_donors=donor_pool,
        )
        streams[role] = prepared
        anchors.extend(role_anchors)
    retention = []
    for rank, source in enumerate(selected["base_denoising_retention_500"]):
        row = dict(source)
        row.update(
            {
                "split_role": "base_denoising_retention_500",
                "selection_rank": rank,
                "role_access": "heldout_retention",
                "expected_target": row["target_true"],
                "paraphrase_prompts": [],
                "near_locality_prompts": [],
                "same_subject_prompts": [],
                "far_locality_cases": [],
                "attribute_prompts": [],
                "generation_prompts": [],
            }
        )
        retention.append(row)
    streams["base_denoising_retention_500"] = retention
    return streams, anchors, filter_report


def build_kamel(
    tokenizer: Any, exclusions: Mapping[str, Any], cache_dir: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw_rows, templates, source = load_kamel_sources(cache_dir)
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
    for source_row in candidates:
        row = dict(source_row)
        row.update(
            {
                "campaign_id": CAMPAIGN_ID,
                "protocol_version": CAMPAIGN_ID,
                "case_id": str(row["case_id"]).replace("kamel_pub_", "cl_kamel_", 1),
                "tokenizer_revision": PRIMARY_MODEL_REVISION,
            }
        )
        normalized.append(row)
    used: set[str] = set()
    output: dict[str, list[dict[str, Any]]] = {}
    for kind, per_length in KAMEL_COUNTS.items():
        role = f"kamel_cl_{kind}_{per_length * 3}"
        rows = []
        for length in (2, 3, 4):
            pool = [row for row in normalized if int(row["target_length"]) == length]
            chosen = kamel_balanced_select(
                pool,
                per_length,
                role=f"{role}_n{length}",
                used=used,
                seed=SEED + length + (0 if kind == "pilot" else 10),
            )
            rows.extend(chosen)
        rows.sort(key=lambda row: (int(row["target_length"]), stable_hash(SEED, role, row["case_id"])))
        for rank, row in enumerate(rows):
            row["split_role"] = role
            row["selection_rank"] = rank
            row["stream_order"] = rank
            row["block_index"] = rank // 10
            row["role_access"] = "fresh_confirmation_only" if kind == "confirmation" else "development"
        output[role] = rows
    return output, {"source": source, "filter": counters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=CAMPAIGN_ROOT / "source_cache")
    parser.add_argument("--counterfact_dataset", default="azhx/counterfact")
    args = parser.parse_args()
    started = now_utc()
    if args.output_dir.exists():
        report = args.output_dir / "report_summary.json"
        if report.is_file():
            existing = __import__("json").loads(report.read_text(encoding="utf-8"))
            if existing.get("acceptance_pass") is True:
                print(f"B0 fresh streams already exist: {args.output_dir}")
                return
        elif any(args.output_dir.iterdir()):
            raise FileExistsError(args.output_dir)
    else:
        args.output_dir.mkdir(parents=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_MODEL_ID,
        revision=PRIMARY_MODEL_REVISION,
        trust_remote_code=True,
    )
    exclusions = collect_historical_exclusions()
    counterfact, anchors, cf_filter = build_counterfact(
        tokenizer, exclusions, args.counterfact_dataset
    )
    kamel, kamel_report = build_kamel(tokenizer, exclusions, args.cache_dir)
    manifests = {**counterfact, **kamel}
    summaries = {}
    for name, rows in manifests.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summaries[name] = _manifest_summary(path, rows)
    anchor_path = args.output_dir / "training_anchors.jsonl"
    write_jsonl(anchor_path, anchors)

    names = list(manifests)
    source_overlap = []
    prompt_overlap = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            sources = _source_fingerprints(manifests[left]) & _source_fingerprints(manifests[right])
            prompts = _prompt_fingerprints(manifests[left]) & _prompt_fingerprints(manifests[right])
            source_overlap.append({"left": left, "right": right, "overlap_count": len(sources)})
            prompt_overlap.append({"left": left, "right": right, "overlap_count": len(prompts)})
    anchor_fps = {str(row["prompt_fingerprint"]) for row in anchors}
    eval_fps = set().union(*(_prompt_fingerprints(rows) for rows in manifests.values()))
    checks = {
        "all_counterfact_counts_exact": all(len(counterfact[name]) == count for name, count in CF_COUNTS.items()),
        "all_counterfact_edit_targets_single_token": all(
            int(row["target_length"]) == 1
            for name, rows in counterfact.items()
            if name.startswith("cf_cl_")
            for row in rows
        ),
        "kamel_pilot_length_balance": Counter(int(row["target_length"]) for row in kamel["kamel_cl_pilot_90"]) == Counter({2: 30, 3: 30, 4: 30}),
        "kamel_confirmation_length_balance": Counter(int(row["target_length"]) for row in kamel["kamel_cl_confirmation_180"]) == Counter({2: 60, 3: 60, 4: 60}),
        "zero_source_overlap": all(row["overlap_count"] == 0 for row in source_overlap),
        "zero_prompt_overlap": all(row["overlap_count"] == 0 for row in prompt_overlap),
        "training_eval_prompt_disjoint": not (anchor_fps & eval_fps),
        "same_subject_heldout_available": all(
            row.get("same_subject_prompts")
            for name, rows in counterfact.items()
            if name.startswith("cf_cl_")
            for row in rows
        ),
        "stream_orders_frozen": all(
            [row["stream_order"] for row in rows] == list(range(len(rows)))
            for name, rows in manifests.items()
            if name.startswith(("cf_cl_", "kamel_cl_"))
        ),
        "historical_locked_content_fields_used": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    acceptance = all(
        value for key, value in checks.items()
        if key not in {"historical_locked_content_fields_used", "analysis_500_used", "final_test_used"}
    )
    write_csv(args.output_dir / "source_overlap_audit.csv", source_overlap)
    write_csv(args.output_dir / "prompt_overlap_audit.csv", prompt_overlap)
    write_csv(args.output_dir / "historical_exclusion_audit.csv", exclusions["audit"])
    write_json(args.output_dir / "split_summary.json", {"splits": summaries})
    write_json(
        args.output_dir / "fresh_confirmation_registry.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "access_policy": "open_only_after_pilot_candidate_freeze",
            "opened_for_tuning": False,
            "manifests": {
                name: summary for name, summary in summaries.items() if "confirmation" in name
            },
        },
    )
    write_json(
        args.output_dir / "historical_exclusion_manifest.json",
        {
            "case_id_count": len(exclusions.get("case_ids", [])),
            "source_key_count": len(exclusions.get("source_keys", [])),
            "source_fingerprint_count": len(exclusions.get("source_fingerprints", [])),
            "prompt_fingerprint_count": len(exclusions.get("prompt_fingerprints", [])),
            "locked_content_fields_used": False,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "B0_fresh_streams",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "counterfact_filter": cf_filter,
        "kamel_report": kamel_report,
        "split_summaries": summaries,
        "training_anchor_count": len(anchors),
        "checks": checks,
        "acceptance_pass": acceptance,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", checks)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "seed": SEED,
            "primary_model_id": PRIMARY_MODEL_ID,
            "primary_model_revision": PRIMARY_MODEL_REVISION,
            "context_aware_target_tokenization": True,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    record_stage(
        "B0_fresh_streams",
        status="passed" if acceptance else "failed",
        acceptance_pass=acceptance,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes="Fresh sequential CounterFact/KAMEL streams and disjoint train-only anchors.",
        next_stage="B1_sequential_harness" if acceptance else None,
        exit_code=0 if acceptance else 2,
    )
    if not acceptance:
        raise SystemExit(2)
    print(f"B0 fresh streams passed: {args.output_dir}")


if __name__ == "__main__":
    main()
