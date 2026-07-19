#!/usr/bin/env python3
"""Build fresh CounterFact and KAMEL manifests for direction selection."""

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
from scripts.nds_common import (
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
    "cf_nds_statistics_train_500": 500,
    "cf_nds_calibration_200": 200,
    "cf_nds_smoke_20": 20,
    "cf_nds_pilot_100": 100,
    "cf_nds_confirmation_200": 200,
}
KAMEL_COUNTS = {
    "train": 200,
    "calibration": 100,
    "pilot": 100,
    "confirmation": 200,
}
SEED = 260719101


def prompt_fingerprint(prompt: str) -> str:
    return stable_hash("prompt", " ".join(str(prompt).casefold().split()))


def normalize_counterfact_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        if int(source["target_length"]) != 1:
            continue
        row = dict(source)
        source_index = int(row["source_index"])
        row.update(
            {
                "campaign_id": CAMPAIGN_ID,
                "protocol_version": CAMPAIGN_ID,
                "case_id": f"nds_cf_{source_index}",
                "tokenizer_model_id": PRIMARY_MODEL_ID,
                "tokenizer_revision": PRIMARY_MODEL_REVISION,
                "target_length_bin": "1",
            }
        )
        output.append(row)
    return output


def _primary_prompt_fingerprints(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> set[str]:
    values = set()
    for rows in splits.values():
        for row in rows:
            values.add(prompt_fingerprint(str(row["rewrite_prompt"])))
            values.update(
                prompt_fingerprint(str(prompt))
                for prompt in row.get("paraphrase_prompts", [])
                if str(prompt).strip()
            )
    return values


def allocate_auxiliary_prompts(
    splits: dict[str, list[dict[str, Any]]],
    candidates: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
) -> None:
    """Allocate auxiliary prompts once across all fresh split roles."""

    used = _primary_prompt_fingerprints(splits)
    priority = (
        "cf_nds_confirmation_200",
        "cf_nds_pilot_100",
        "cf_nds_smoke_20",
        "cf_nds_calibration_200",
        "cf_nds_statistics_train_500",
    )
    for role in priority:
        for row in splits[role]:
            chosen = None
            for candidate in row.pop("same_subject_prompt_candidates", []):
                fingerprint = prompt_fingerprint(str(candidate["prompt"]))
                if fingerprint not in used:
                    chosen = candidate
                    used.add(fingerprint)
                    break
            if chosen is None:
                raise RuntimeError(f"No unique same-subject prompt for {row['case_id']}")
            row["same_subject_prompts"] = [chosen["prompt"]]
            row["same_subject_negative_relation_id"] = chosen["relation_id"]
            for field in (
                "near_locality_prompts",
                "attribute_prompts",
                "generation_prompts",
            ):
                kept = []
                for prompt in row.get(field, []):
                    if not str(prompt).strip():
                        continue
                    fingerprint = prompt_fingerprint(str(prompt))
                    if fingerprint in used:
                        continue
                    used.add(fingerprint)
                    kept.append(prompt)
                row[field] = kept

    donors = [row for row in candidates if str(row["case_id"]) not in selected_ids]
    donors.sort(key=lambda row: stable_hash(SEED, "far", row["case_id"]))
    cursor = 0
    for role in priority:
        for row in splits[role]:
            while cursor < len(donors):
                donor = donors[cursor]
                cursor += 1
                fingerprint = prompt_fingerprint(str(donor["rewrite_prompt"]))
                if fingerprint in used:
                    continue
                used.add(fingerprint)
                row["far_locality_cases"] = [
                    {
                        "case_id": donor["case_id"],
                        "prompt": donor["rewrite_prompt"],
                        "target": donor["target_true"],
                        "source_index": donor["source_index"],
                        "source_fingerprint": donor["source_fingerprint"],
                        "prompt_provenance": "fresh_disjoint_counterfact_fact",
                    }
                ]
                break
            else:
                raise RuntimeError("Insufficient unique far-locality donors")


def select_counterfact(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    used: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {}
    for offset, (role, count) in enumerate(CF_COUNTS.items()):
        selected = round_robin_stratified(
            list(candidates),
            count,
            seed=SEED + offset,
            used=used,
            group_fields=("relation_id",),
        )
        normalized = []
        for rank, source in enumerate(selected):
            row = dict(source)
            row["split_role"] = role
            row["selection_rank"] = rank
            row["role_access"] = (
                "fresh_confirmation_only" if "confirmation" in role else "development"
            )
            normalized.append(row)
        splits[role] = normalized
    allocate_auxiliary_prompts(splits, candidates, used)
    return splits


def select_kamel(
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
        row["case_id"] = row["case_id"].replace("kamel_pub_", "nds_kamel_", 1)
        row["tokenizer_revision"] = PRIMARY_MODEL_REVISION
        row["prompt_fingerprint"] = prompt_fingerprint(str(row["rewrite_prompt"]))
        normalized.append(row)
    splits: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    for length in (2, 3, 4):
        pool = [row for row in normalized if int(row["target_length"]) == length]
        for offset, (kind, count) in enumerate(KAMEL_COUNTS.items()):
            role = f"kamel_nds_{kind}_{count}_n{length}"
            selected = kamel_balanced_select(
                pool,
                count,
                role=role,
                used=used,
                seed=SEED + length * 10 + offset,
            )
            for row in selected:
                row["role_access"] = (
                    "fresh_confirmation_only" if kind == "confirmation" else "development"
                )
            splits[role] = selected
    return splits, counters


def source_fingerprints(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    values = {str(row["source_fingerprint"]) for row in rows}
    for row in rows:
        values.update(
            str(item["source_fingerprint"])
            for item in row.get("far_locality_cases", [])
        )
    return values


def all_prompt_fingerprints(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    values = set()
    for row in rows:
        values.add(prompt_fingerprint(str(row["rewrite_prompt"])))
        for field in (
            "paraphrase_prompts",
            "near_locality_prompts",
            "attribute_prompts",
            "generation_prompts",
            "same_subject_prompts",
        ):
            values.update(
                prompt_fingerprint(str(prompt))
                for prompt in row.get(field, [])
                if str(prompt).strip()
            )
        values.update(
            prompt_fingerprint(str(item["prompt"]))
            for item in row.get("far_locality_cases", [])
        )
    return values


def overlap_audit(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = []
    prompt_rows = []
    names = list(splits)
    for index, left in enumerate(names):
        left_sources = source_fingerprints(splits[left])
        left_prompts = all_prompt_fingerprints(splits[left])
        for right in names[index + 1 :]:
            source_overlap = left_sources & source_fingerprints(splits[right])
            prompt_overlap = left_prompts & all_prompt_fingerprints(splits[right])
            source_rows.append(
                {"left": left, "right": right, "overlap_count": len(source_overlap)}
            )
            prompt_rows.append(
                {
                    "left": left,
                    "right": right,
                    "prompt_overlap_count": len(prompt_overlap),
                }
            )
            if source_overlap or prompt_overlap:
                raise RuntimeError(f"Fresh split overlap: {left} vs {right}")
    return source_rows, prompt_rows


def summarize_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "count": len(rows),
        "unique_case_ids": len({str(row["case_id"]) for row in rows}),
        "unique_source_fingerprints": len(
            {str(row["source_fingerprint"]) for row in rows}
        ),
        "unique_prompt_fingerprints": len(all_prompt_fingerprints(rows)),
        "target_length_histogram": dict(
            sorted(Counter(str(row["target_length"]) for row in rows).items())
        ),
        "relation_histogram": dict(
            sorted(Counter(str(row["relation_id"]) for row in rows).items())
        ),
        "locked": any(row.get("role_access") == "fresh_confirmation_only" for row in rows),
        "opened_for_tuning": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=CAMPAIGN_ROOT / "source_cache")
    parser.add_argument("--counterfact_dataset", default="azhx/counterfact")
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
    raw_cf, counterfact_filter = _counterfact_candidates(
        tokenizer, args.counterfact_dataset, exclusions
    )
    cf_candidates = normalize_counterfact_candidates(raw_cf)
    cf_splits = select_counterfact(cf_candidates)
    kamel_rows, templates, kamel_source = load_kamel_sources(args.cache_dir)
    kamel_splits, kamel_filter = select_kamel(
        kamel_rows, templates, tokenizer, exclusions
    )
    splits = {**cf_splits, **kamel_splits}
    source_overlap, prompt_overlap = overlap_audit(splits)
    summaries = {}
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summaries[name] = summarize_manifest(path, rows)

    write_csv(args.output_dir / "source_overlap_audit.csv", source_overlap)
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
            "historical_locked_content_fields_used": False,
            "allowed_fields_only": [
                "case_id",
                "source_split",
                "source_index",
                "source_fingerprint",
                "fact_fingerprint",
                "fact_target_fingerprint",
                "prompt_fingerprint",
            ],
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    write_json(args.output_dir / "split_summary.json", {"splits": summaries})
    write_json(
        args.output_dir / "source_registry.json",
        {
            "counterfact": {"dataset": args.counterfact_dataset, "split": "train"},
            "kamel": kamel_source,
            "tokenizer_model_id": PRIMARY_MODEL_ID,
            "tokenizer_revision": PRIMARY_MODEL_REVISION,
            "context_aware_tokenization": True,
        },
    )
    confirmations = {
        name: summary for name, summary in summaries.items() if "confirmation" in name
    }
    write_json(
        args.output_dir / "fresh_confirmation_registry.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "manifests": confirmations,
            "access_policy": "fresh_confirmation_only_after_candidate_freeze",
            "opened_for_tuning": False,
        },
    )
    required = set(CF_COUNTS)
    required.update(
        f"kamel_nds_{kind}_{count}_n{length}"
        for length in (2, 3, 4)
        for kind, count in KAMEL_COUNTS.items()
    )
    checks = {
        "all_required_manifests_exist": required <= set(summaries),
        "all_manifest_counts_exact": all(
            summaries[name]["count"] == count for name, count in CF_COUNTS.items()
        )
        and all(
            summaries[f"kamel_nds_{kind}_{count}_n{length}"]["count"] == count
            for length in (2, 3, 4)
            for kind, count in KAMEL_COUNTS.items()
        ),
        "counterfact_single_token_scope": all(
            summary["target_length_histogram"] == {"1": CF_COUNTS[name]}
            for name, summary in summaries.items()
            if name.startswith("cf_nds_")
        ),
        "kamel_exact_length_strata": all(
            summaries[f"kamel_nds_{kind}_{count}_n{length}"]["target_length_histogram"]
            == {str(length): count}
            for length in (2, 3, 4)
            for kind, count in KAMEL_COUNTS.items()
        ),
        "zero_source_overlap": all(row["overlap_count"] == 0 for row in source_overlap),
        "zero_prompt_overlap": all(
            row["prompt_overlap_count"] == 0 for row in prompt_overlap
        ),
        "historical_locked_content_fields_used": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    acceptance = all(
        value
        for key, value in checks.items()
        if key not in {
            "historical_locked_content_fields_used",
            "analysis_500_used",
            "final_test_used",
        }
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "S0_fresh_manifests",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "split_summaries": summaries,
        "counterfact_filter": counterfact_filter,
        "counterfact_single_token_candidates": len(cf_candidates),
        "kamel_filter": kamel_filter,
        "checks": checks,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": acceptance,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", checks)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "seed": SEED,
            "counterfact_target_length_scope": [1],
            "kamel_target_length_scope": [2, 3, 4],
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    record_stage(
        "S0_fresh_manifests",
        status="passed" if acceptance else "failed",
        acceptance_pass=acceptance,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes="Fresh CounterFact and KAMEL manifests built with identity-only historical exclusion.",
        next_stage="S1_common_baselines" if acceptance else None,
        exit_code=0 if acceptance else 2,
    )
    if not acceptance:
        raise SystemExit(2)
    print(f"S0 fresh manifests passed: {args.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
