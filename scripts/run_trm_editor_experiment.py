#!/usr/bin/env python3
"""Run one reusable temporal-residual editor experiment on an allowed manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from scripts.run_trm_state_conditioned_protection import anchor_logits, distribution_kl
from scripts.trm_common import (
    CAMPAIGN_ID,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.trm_editor import (
    build_input_protection_basis,
    fit_residual_memory_for_requests,
    install_factorized_residual_memory,
    install_state_bucketed_residual_memories,
    summarize_editor_rows,
)
from scripts.trm_protection import build_protection_prompt_records, extract_protection_keys


BUCKET_SCHEDULES = {
    "early": "fewer_revealed",
    "middle": "uniform",
    "late": "more_revealed",
}
PROTECTION_MODES = {"none", "static", "shared", "state"}
STATE_MODES = {"shared", "bucketed"}
ALLOWED_ANCHOR_ROLES = {
    "cf_trm_anchor_train_500",
    "cf_nds_statistics_train_500",
}


def validate_mode(state_mode: str, protection_mode: str) -> None:
    if state_mode not in STATE_MODES:
        raise ValueError(f"unknown state mode: {state_mode}")
    if protection_mode not in PROTECTION_MODES:
        raise ValueError(f"unknown protection mode: {protection_mode}")
    if state_mode == "shared" and protection_mode == "state":
        raise ValueError("state protection requires bucketed state mode")
    if state_mode == "bucketed" and protection_mode in {"static", "shared"}:
        raise ValueError("bucketed mode supports none or state protection")


def validate_manifest_access(path: Path) -> None:
    lower = str(path).casefold()
    if "analysis_500" in lower or "final_test" in lower or "final_test_full" in lower:
        raise PermissionError("Historical analysis/final manifests are forbidden")
    if "locked" in path.name.casefold() and os.environ.get("PS_TRM_DEV_METHOD_LOCKED") != "1":
        raise PermissionError("Fresh TRM locked confirmation requires a validated dev lock")


def validate_anchor_role(
    anchors: Sequence[Mapping[str, Any]], expected_role: str
) -> None:
    if expected_role not in ALLOWED_ANCHOR_ROLES:
        raise PermissionError(f"Anchor role is not training-only: {expected_role}")
    roles = {str(row.get("split_role") or "") for row in anchors}
    if not anchors or roles != {expected_role}:
        raise RuntimeError(
            f"Temporal protection diagnostics require {expected_role}; got {sorted(roles)}"
        )


def grouped_diagnostics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    length_groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    relation_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        length_groups[(int(row.get("target_length") or 0), str(row["bucket"]))].append(row)
        relation_groups[(str(row.get("relation_id") or ""), str(row["bucket"]))].append(row)

    def summarize(groups: Mapping[tuple[Any, str], Sequence[Mapping[str, Any]]], name: str) -> list[dict[str, Any]]:
        output = []
        for (group, bucket), values in sorted(groups.items()):
            expected = [row for row in values if row.get("expected_hit") is not None]
            output.append(
                {
                    name: group,
                    "bucket": bucket,
                    "num_rows": len(values),
                    "num_edits": len({str(row["case_id"]) for row in values}),
                    "expected_exact": (
                        sum(bool(row["expected_hit"]) for row in expected) / len(expected)
                        if expected
                        else None
                    ),
                    "target_new_rate": sum(bool(row["target_new_hit"]) for row in values)
                    / len(values),
                    "malformed_rate": sum(bool(row["malformed"]) for row in values)
                    / len(values),
                }
            )
        return output

    return summarize(length_groups, "target_length"), summarize(
        relation_groups, "relation_id"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--state_mode", choices=sorted(STATE_MODES), default="shared")
    parser.add_argument(
        "--protection_mode", choices=sorted(PROTECTION_MODES), default="none"
    )
    parser.add_argument(
        "--anchor_manifest",
        type=Path,
        default=PROTOCOL_ROOT / "cf_trm_anchor_train_500.jsonl",
    )
    parser.add_argument(
        "--anchor_role",
        choices=sorted(ALLOWED_ANCHOR_ROLES),
        default="cf_trm_anchor_train_500",
    )
    parser.add_argument("--anchor_per_family", type=int, default=32)
    parser.add_argument("--partial_mask_schedule", default="uniform")
    parser.add_argument("--reveal_policy", default="random")
    parser.add_argument("--state_consistency_weight", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--preservation_strength", type=float, default=1.0)
    parser.add_argument("--protected_variance", type=float, default=0.95)
    parser.add_argument("--maximum_basis_rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top_q", type=int, default=0)
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260718801)
    args = parser.parse_args()
    validate_mode(args.state_mode, args.protection_mode)
    validate_manifest_access(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    begin = time.monotonic()
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No edit rows selected")
    anchors = read_jsonl(args.anchor_manifest)
    validate_anchor_role(anchors, args.anchor_role)
    records, anchor_summary = build_protection_prompt_records(
        anchors, max_per_family=args.anchor_per_family
    )
    if not anchor_summary["all_required_families_present"]:
        raise RuntimeError("Protection anchor families are incomplete")
    config = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "temporal_residual_editor_experiment",
        "method": args.method,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "num_edits": len(rows),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layer": args.layer,
        "state_mode": args.state_mode,
        "protection_mode": args.protection_mode,
        "partial_mask_schedule": args.partial_mask_schedule,
        "reveal_policy": args.reveal_policy,
        "state_consistency_weight": args.state_consistency_weight,
        "ridge": args.ridge,
        "preservation_strength": args.preservation_strength,
        "protected_variance": args.protected_variance,
        "maximum_basis_rank": args.maximum_basis_rank,
        "alpha": args.alpha,
        "top_q": args.top_q,
        "anchor_manifest": str(args.anchor_manifest),
        "anchor_role": args.anchor_role,
        "anchor_manifest_sha256": sha256_file(args.anchor_manifest),
        "runtime_feature_schema": [
            "current_hidden_state",
            "active_mask_count" if args.state_mode == "bucketed" else "shared_state",
            "answer_span_length",
            "fitted_residual_memory",
        ],
        "evaluation_bucket_used_as_runtime_input": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "run_config.json", config)
    write_csv(args.output_dir / "protection_anchor_manifest.csv", records)
    write_json(args.output_dir / "protection_anchor_summary.json", anchor_summary)
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    tasks = build_eval_tasks(tokenizer, rows, include_locality=True)
    base_rows = evaluate_tasks(
        model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
    )
    module = get_module(model, resolved_key_module_name(model, args.layer))
    protection_keys = {}
    protection_metadata = []
    if args.protection_mode != "none":
        for bucket in ("early", "middle", "late"):
            keys, metadata = extract_protection_keys(
                model,
                tokenizer,
                records,
                layer=args.layer,
                state_bucket=bucket,
                span_length=3,
                seed=args.seed,
            )
            protection_keys[bucket] = keys
            protection_metadata.extend(metadata)
    write_csv(args.output_dir / "protection_key_metadata.csv", protection_metadata)
    combined = (
        torch.cat(
            [protection_keys[bucket] for bucket in ("early", "middle", "late")],
            dim=0,
        ).to("cuda")
        if protection_keys
        else None
    )
    kl_records = records[: min(64, len(records))]
    base_anchor_logits = anchor_logits(model, tokenizer, kl_records)
    fit_runtime = 0.0
    memory_storage = 0
    memory_rank = 0
    protect_rows = 0
    memory_finite = True
    activation: Mapping[str, Any]
    if args.state_mode == "shared":
        projection = None
        protect = None
        strength = 0.0
        if args.protection_mode == "static":
            assert combined is not None
            projection, basis_report = build_input_protection_basis(
                combined,
                explained_variance=args.protected_variance,
                maximum_rank=args.maximum_basis_rank,
            )
            torch.save(
                {"basis": projection.cpu(), "report": basis_report},
                args.output_dir / "static_protection_basis.pt",
            )
        elif args.protection_mode == "shared":
            assert combined is not None
            protect = combined
            strength = args.preservation_strength
        memory, diagnostics, fit_runtime = fit_residual_memory_for_requests(
            model,
            tokenizer,
            rows,
            layer=args.layer,
            ridge=args.ridge,
            target_optimization_steps=args.target_optimization_steps,
            learning_rate=args.learning_rate,
            partial_mask_schedule=args.partial_mask_schedule,
            reveal_policy=args.reveal_policy,
            state_consistency_weight=args.state_consistency_weight,
            old_target_suppression_weight=0.25,
            seed=args.seed,
            cache_dir=args.output_dir / "target_value_cache",
            protect_keys=protect,
            preservation_strength=strength,
            input_projection_basis=projection,
        )
        with install_factorized_residual_memory(
            module, memory, alpha=args.alpha, top_q=args.top_q
        ) as state:
            raw = evaluate_tasks(
                model,
                tokenizer,
                tasks,
                decode_batch_size=args.decode_batch_size,
                steps=None,
            )
            edited_anchor_logits = anchor_logits(model, tokenizer, kl_records)
        activation = state
        torch.save(memory.cpu_payload(), args.output_dir / "residual_memory.pt")
        write_json(args.output_dir / "target_value_diagnostics.json", diagnostics)
        memory_storage = memory.storage_bytes
        memory_rank = memory.rank_bound
        protect_rows = memory.protect_row_count
        memory_finite = all(
            torch.isfinite(value).all()
            for value in (memory.keys, memory.dual, memory.residuals)
        )
    else:
        memories = {}
        diagnostics_by_bucket = {}
        for bucket, schedule in BUCKET_SCHEDULES.items():
            protect = (
                protection_keys[bucket].to("cuda")
                if args.protection_mode == "state"
                else None
            )
            memory, diagnostics, runtime = fit_residual_memory_for_requests(
                model,
                tokenizer,
                rows,
                layer=args.layer,
                ridge=args.ridge,
                target_optimization_steps=args.target_optimization_steps,
                learning_rate=args.learning_rate,
                partial_mask_schedule=schedule,
                reveal_policy="random",
                state_consistency_weight=0.1,
                old_target_suppression_weight=0.25,
                seed=args.seed,
                cache_dir=args.output_dir / f"target_value_cache_{bucket}",
                protect_keys=protect,
                preservation_strength=(
                    args.preservation_strength if protect is not None else 0.0
                ),
            )
            memories[bucket] = memory
            diagnostics_by_bucket[bucket] = diagnostics
            fit_runtime += runtime
            torch.save(
                memory.cpu_payload(), args.output_dir / f"residual_memory_{bucket}.pt"
            )
        with install_state_bucketed_residual_memories(
            model,
            module,
            memories,
            mask_id=infer_mask_id(model),
            alpha=args.alpha,
            top_q=args.top_q,
        ) as state:
            raw = evaluate_tasks(
                model,
                tokenizer,
                tasks,
                decode_batch_size=args.decode_batch_size,
                steps=None,
            )
            edited_anchor_logits = anchor_logits(model, tokenizer, kl_records)
        activation = state
        write_json(
            args.output_dir / "target_value_diagnostics.json",
            [
                {"state_bucket": bucket, **row}
                for bucket, values in diagnostics_by_bucket.items()
                for row in values
            ],
        )
        memory_storage = sum(memory.storage_bytes for memory in memories.values())
        memory_rank = sum(memory.rank_bound for memory in memories.values())
        protect_rows = sum(memory.protect_row_count for memory in memories.values())
        memory_finite = all(
            torch.isfinite(value).all()
            for memory in memories.values()
            for value in (memory.keys, memory.dual, memory.residuals)
        )
    edited = align_base(base_rows, raw)
    metrics = summarize_editor_rows(base_rows, edited)
    length_rows, relation_rows = grouped_diagnostics(edited)
    runtime = time.monotonic() - begin
    write_csv(args.output_dir / "base_per_prompt.csv", base_rows)
    write_csv(args.output_dir / "edited_per_prompt.csv", edited)
    write_csv(args.output_dir / "target_length_breakdown.csv", length_rows)
    write_csv(args.output_dir / "relation_breakdown.csv", relation_rows)
    report = {
        **config,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "memory_storage_bytes": memory_storage,
        "memory_rank_bound": memory_rank,
        "protect_row_count": protect_rows,
        "memory_finite": memory_finite,
        "activation_diagnostics": activation,
        "retain_distribution_kl": distribution_kl(
            base_anchor_logits, edited_anchor_logits
        ),
        "fit_runtime_seconds": fit_runtime,
        "runtime_seconds": runtime,
        "gpu_minutes_per_edit": runtime / 60.0 / len(rows),
        "model_eval_count": sum(int(row["model_eval_count"]) for row in edited),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        **metrics,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": bool(memory_finite and metrics["all_metrics_finite"]),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "memory_finite": memory_finite,
            "all_metrics_finite": metrics["all_metrics_finite"],
            "runtime_inputs_deployable": True,
            "evaluation_bucket_used_as_runtime_input": False,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": report["acceptance_pass"],
        },
    )
    if not report["acceptance_pass"]:
        raise SystemExit(2)
    print(
        json.dumps(
            {
                "method": args.method,
                "rewrite_exact": report["rewrite_exact"],
                "paraphrase_exact": report["declarative_paraphrase_exact"],
                "same_subject_tfpr": report["same_subject_tfpr"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
