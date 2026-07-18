#!/usr/bin/env python3
"""Run the single legal D2 relation-conditioned protection rescue."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_editor import get_module, infer_mask_id, resolved_key_module_name
from scripts.run_dnpe_editor import align_base, build_eval_tasks, evaluate_tasks
from scripts.run_mdm_memit_stage import load_model
from scripts.run_trm_state_conditioned_protection import paired_tfpr_bootstrap
from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    record_stage,
    record_stage_cost,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.trm_editor import (
    fit_residual_memory_for_requests,
    install_state_bucketed_residual_memories,
    summarize_editor_rows,
)
from scripts.trm_protection import (
    REQUIRED_PROTECTION_FAMILIES,
    build_protection_prompt_records,
    extract_protection_keys,
)


BUCKET_SCHEDULES = {
    "early": "fewer_revealed",
    "middle": "uniform",
    "late": "more_revealed",
}


def relation_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["relation_id"])].append(dict(row))
    return dict(sorted(grouped.items()))


def relation_protection_records(
    records: Sequence[Mapping[str, Any]],
    relation_id: str,
    *,
    maximum_per_family: int,
    minimum_per_family: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_family[str(row["family"])].append(row)
    selected: list[dict[str, Any]] = []
    exact_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    for family in REQUIRED_PROTECTION_FAMILIES:
        family_rows = by_family[family]
        exact = [row for row in family_rows if str(row["relation_id"]) == relation_id]
        chosen = list(exact[: int(maximum_per_family)])
        used = {str(row["anchor_id"]) for row in chosen}
        for row in family_rows:
            if len(chosen) >= int(minimum_per_family):
                break
            if str(row["anchor_id"]) not in used:
                chosen.append(row)
                used.add(str(row["anchor_id"]))
        exact_counts[family] = sum(
            str(row["relation_id"]) == relation_id for row in chosen
        )
        fallback_counts[family] = len(chosen) - exact_counts[family]
        selected.extend(dict(row) for row in chosen)
    return selected, {
        "relation_id": relation_id,
        "num_rows": len(selected),
        "exact_relation_counts": exact_counts,
        "fallback_counts": fallback_counts,
        "all_families_present": all(
            any(row["family"] == family for row in selected)
            for family in REQUIRED_PROTECTION_FAMILIES
        ),
        "cluster_source": "fresh_train_only_relation_id_with_deterministic_family_backoff",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=PROTOCOL_ROOT / "cf_trm_smoke_20.jsonl"
    )
    parser.add_argument(
        "--anchor_manifest",
        type=Path,
        default=PROTOCOL_ROOT / "cf_trm_anchor_train_500.jsonl",
    )
    parser.add_argument(
        "--d2_dir",
        type=Path,
        default=CAMPAIGN_ROOT / "D2_state_conditioned_protection_v1",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CAMPAIGN_ROOT / "D2_relation_conditioned_rescue_v1",
    )
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--preservation_strength", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top_q", type=int, default=256)
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--maximum_per_family", type=int, default=16)
    parser.add_argument("--minimum_per_family", type=int, default=4)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260718701)
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    d2 = read_json(args.d2_dir / "report_summary.json")
    if not d2.get("acceptance_pass"):
        raise RuntimeError("D2 integrity did not pass")
    if not d2.get("relation_rescue_triggered"):
        raise RuntimeError("The frozen D2 relation-rescue trigger did not fire")
    if any(
        token in str(path).casefold()
        for path in (args.manifest, args.anchor_manifest)
        for token in ("analysis_500", "final_test", "locked")
    ):
        raise RuntimeError("Relation rescue cannot open locked evaluation data")
    args.output_dir.mkdir(parents=True)
    requests = read_jsonl(args.manifest)
    anchors = read_jsonl(args.anchor_manifest)
    if len(requests) != 20 or len(anchors) != 500:
        raise RuntimeError("Relation rescue requires fresh smoke20 and anchor500")
    groups = relation_groups(requests)
    all_records, anchor_summary = build_protection_prompt_records(
        anchors, max_per_family=len(anchors)
    )
    cluster_records: dict[str, list[dict[str, Any]]] = {}
    cluster_reports = []
    for relation_id in groups:
        selected, summary = relation_protection_records(
            all_records,
            relation_id,
            maximum_per_family=args.maximum_per_family,
            minimum_per_family=args.minimum_per_family,
        )
        if not summary["all_families_present"]:
            raise RuntimeError(f"Missing relation protection family for {relation_id}")
        cluster_records[relation_id] = selected
        cluster_reports.append(summary)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "D2_relation_conditioned_rescue",
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "anchor_manifest": str(args.anchor_manifest),
            "anchor_manifest_sha256": sha256_file(args.anchor_manifest),
            "relation_cluster": "exact_training_metadata_relation_id",
            "maximum_per_family": args.maximum_per_family,
            "minimum_per_family": args.minimum_per_family,
            "ridge": args.ridge,
            "preservation_strength": args.preservation_strength,
            "alpha": args.alpha,
            "top_q": args.top_q,
            "runtime_feature_schema": [
                "current_hidden_state",
                "active_mask_count",
                "answer_span_length",
                "edit_relation_id",
                "fitted_residual_memory",
            ],
            "evaluation_bucket_used_as_runtime_input": False,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    write_json(args.output_dir / "relation_cluster_summary.json", cluster_reports)
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    tasks = build_eval_tasks(tokenizer, requests, include_locality=True)
    base_rows = evaluate_tasks(
        model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
    )
    d2_config = read_json(args.d2_dir / "run_config.json")
    layer = int(d2_config["layer"])
    module = get_module(
        model,
        resolved_key_module_name(model, layer),
    )
    raw_rows: list[dict[str, Any]] = []
    memory_rows = []
    for relation_id, relation_requests in groups.items():
        relation_dir = args.output_dir / "relation_memories" / relation_id
        relation_dir.mkdir(parents=True)
        memories = {}
        for bucket, schedule in BUCKET_SCHEDULES.items():
            keys, metadata = extract_protection_keys(
                model,
                tokenizer,
                cluster_records[relation_id],
                layer=layer,
                state_bucket=bucket,
                span_length=3,
                seed=args.seed,
            )
            memory, diagnostics, fit_runtime = fit_residual_memory_for_requests(
                model,
                tokenizer,
                relation_requests,
                layer=layer,
                ridge=args.ridge,
                target_optimization_steps=args.target_optimization_steps,
                learning_rate=args.learning_rate,
                partial_mask_schedule=schedule,
                reveal_policy="random",
                state_consistency_weight=0.1,
                old_target_suppression_weight=0.25,
                seed=args.seed,
                cache_dir=relation_dir / f"target_value_cache_{bucket}",
                protect_keys=keys.to("cuda"),
                preservation_strength=args.preservation_strength,
            )
            memories[bucket] = memory
            torch.save(memory.cpu_payload(), relation_dir / f"residual_memory_{bucket}.pt")
            write_json(relation_dir / f"target_value_diagnostics_{bucket}.json", diagnostics)
            write_csv(relation_dir / f"protection_metadata_{bucket}.csv", metadata)
            memory_rows.append(
                {
                    "relation_id": relation_id,
                    "state_bucket": bucket,
                    "num_edits": len(relation_requests),
                    "protect_rows": memory.protect_row_count,
                    "storage_bytes": memory.storage_bytes,
                    "fit_runtime_seconds": fit_runtime,
                }
            )
        relation_cases = {str(row["case_id"]) for row in relation_requests}
        relation_tasks = [
            task for task in tasks if str(task["case_id"]) in relation_cases
        ]
        with install_state_bucketed_residual_memories(
            model,
            module,
            memories,
            mask_id=infer_mask_id(model),
            alpha=args.alpha,
            top_q=args.top_q,
        ):
            raw_rows.extend(
                evaluate_tasks(
                    model,
                    tokenizer,
                    relation_tasks,
                    decode_batch_size=args.decode_batch_size,
                    steps=None,
                )
            )
        del memories
        torch.cuda.empty_cache()
    edited = align_base(base_rows, raw_rows)
    metrics = summarize_editor_rows(base_rows, edited)
    write_csv(args.output_dir / "base_per_prompt.csv", base_rows)
    write_csv(args.output_dir / "edited_per_prompt.csv", edited)
    write_csv(args.output_dir / "relation_memory_summary.csv", memory_rows)
    state_rows_path = args.d2_dir / "state_conditioned_preservation" / "edited_per_prompt.csv"
    import csv

    with state_rows_path.open(newline="", encoding="utf-8") as handle:
        state_rows = []
        for source in csv.DictReader(handle):
            row = dict(source)
            for key in ("target_new_hit", "target_true_hit", "malformed", "base_agreement"):
                if key in row:
                    row[key] = str(row[key]).casefold() == "true"
            state_rows.append(row)
    bootstrap = paired_tfpr_bootstrap(edited, state_rows, seed=args.seed)
    full_budget = float(d2["same_subject_full_editor_budget"])
    rescue_full_editor_pass = bool(
        metrics["rewrite_exact"] >= 0.85
        and metrics["declarative_paraphrase_exact"] >= 0.40
        and metrics["same_subject_tfpr"] <= full_budget
        and metrics["near_tfpr"] <= float(d2["near_full_editor_budget"])
        and metrics["far_tfpr"] <= float(d2["far_full_editor_budget"])
        and metrics["malformed_rate"] <= 0.05
    )
    state_summary = read_json(
        args.d2_dir / "state_conditioned_preservation" / "report_summary.json"
    )
    rescue_improved = bool(
        metrics["rewrite_exact"] >= float(state_summary["rewrite_exact"]) - 0.02
        and metrics["declarative_paraphrase_exact"]
        >= float(state_summary["declarative_paraphrase_exact"]) - 0.02
        and metrics["same_subject_tfpr"] < float(state_summary["same_subject_tfpr"])
    )
    integrity = {
        "trigger_was_legally_satisfied": True,
        "all_relation_groups_evaluated": len(raw_rows) == len(base_rows),
        "all_required_anchor_families_present": anchor_summary[
            "all_required_families_present"
        ],
        "all_metrics_finite": metrics["all_metrics_finite"],
        "runtime_inputs_deployable": True,
        "evaluation_bucket_used_as_runtime_input": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = all(
        value
        for key, value in integrity.items()
        if key
        not in {
            "evaluation_bucket_used_as_runtime_input",
            "analysis_500_used",
            "final_test_used",
        }
    ) and not any(
        integrity[key]
        for key in (
            "evaluation_bucket_used_as_runtime_input",
            "analysis_500_used",
            "final_test_used",
        )
    )
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D2_relation_conditioned_rescue",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "method": "timerome_partial_state_state_relation_protected",
        "num_relation_clusters": len(groups),
        "relation_cluster_source": "fresh_train_only_exact_relation_id",
        "rescue_full_editor_pass": rescue_full_editor_pass,
        "rescue_improved_over_state_conditioned": rescue_improved,
        "paired_same_subject_bootstrap_vs_state_conditioned": bootstrap,
        "integrity": integrity,
        "runtime_seconds": runtime,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        **metrics,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": passed,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {"integrity": integrity, "acceptance_pass": passed},
    )
    record_stage_cost(
        "D2_relation_conditioned_rescue",
        runtime_seconds=runtime,
        notes="Single frozen relation-conditioned protection rescue",
    )
    record_stage(
        "D2_relation_conditioned_rescue",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=(
            f"full_editor={rescue_full_editor_pass}; "
            f"improved_vs_state={rescue_improved}"
        ),
        next_stage="E1_smoke20" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(
        json.dumps(
            {
                "acceptance_pass": True,
                "rescue_full_editor_pass": rescue_full_editor_pass,
                "rescue_improved": rescue_improved,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
