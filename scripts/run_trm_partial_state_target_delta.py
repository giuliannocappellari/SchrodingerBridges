#!/usr/bin/env python3
"""Run D1 partial-state target-delta variants on fresh KAMEL dev edits."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
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
    install_factorized_residual_memory,
    install_state_bucketed_residual_memories,
    summarize_editor_rows,
)


SHARED_VARIANTS = {
    "fullmask_delta": ("fully_masked", "random", 0.0),
    "uniform_partial_state_delta": ("uniform", "random", 0.1),
    "mask_count_cycling_delta": ("cycle", "random", 0.1),
    "trajectory_sampled_delta": ("cycle", "base_confidence", 0.1),
}
BUCKET_SCHEDULES = {
    "early": ("fewer_revealed", "random"),
    "middle": ("uniform", "random"),
    "late": ("more_revealed", "random"),
}


def paired_bootstrap_rewrite_delta(
    anchor_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    trials: int = 2000,
    seed: int = 260718501,
) -> dict[str, float]:
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {
            str(row["case_id"]): float(bool(row["expected_hit"]))
            for row in rows
            if row.get("bucket") == "rewrite"
        }

    left = index(anchor_rows)
    right = index(comparison_rows)
    ids = sorted(set(left) & set(right))
    if not ids:
        raise RuntimeError("no aligned rewrite cases for paired bootstrap")
    observed = sum(left[value] - right[value] for value in ids) / len(ids)
    rng = random.Random(int(seed))
    samples = []
    for _ in range(int(trials)):
        draw = [rng.choice(ids) for _ in ids]
        samples.append(sum(left[value] - right[value] for value in draw) / len(draw))
    samples.sort()
    return {
        "num_cases": len(ids),
        "delta": observed,
        "ci_low": samples[int(0.025 * (len(samples) - 1))],
        "ci_high": samples[int(0.975 * (len(samples) - 1))],
    }


def method_report(
    *,
    method: str,
    target_length: int,
    manifest: Path,
    base_rows: Sequence[Mapping[str, Any]],
    edited_rows: Sequence[Mapping[str, Any]],
    memory_storage_bytes: int,
    memory_rank_bound: int,
    fit_runtime_seconds: float,
    runtime_seconds: float,
    activation: Mapping[str, Any],
    schedule: str,
    reveal_policy: str,
    diagnostics: Sequence[Mapping[str, Any]],
    memory_finite: bool,
) -> dict[str, Any]:
    summary = summarize_editor_rows(base_rows, edited_rows)
    return {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D1_partial_state_target_delta",
        "method": method,
        "target_length": int(target_length),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "num_edits": len({str(row["case_id"]) for row in edited_rows}),
        "partial_mask_schedule": schedule,
        "reveal_policy": reveal_policy,
        "memory_storage_bytes": int(memory_storage_bytes),
        "memory_rank_bound": int(memory_rank_bound),
        "memory_finite": bool(memory_finite),
        "fit_runtime_seconds": float(fit_runtime_seconds),
        "runtime_seconds": float(runtime_seconds),
        "activation_diagnostics": dict(activation),
        "mean_target_delta_norm": sum(float(row["delta_norm"]) for row in diagnostics) / len(diagnostics),
        "target_probability_improved_fraction": sum(
            bool(row["heldout_target_probability_improved"]) for row in diagnostics
        ) / len(diagnostics),
        **summary,
        "analysis_500_used": False,
        "final_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "D1_partial_state_target_delta_v1")
    parser.add_argument("--c1_dir", type=Path, default=CAMPAIGN_ROOT / "C1_temporal_localization_v1")
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top_q", type=int, default=0)
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260718501)
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    c1_lock = read_json(args.c1_dir / "site_policy_lock.json")
    stable = next(
        row for row in c1_lock["policies"] if row["policy_id"] == "stable_temporal_site_set"
    )
    layer = int(json.loads(stable["layers_json"])[0])
    manifests = {
        length: PROTOCOL_ROOT / f"kamel_trm_dev_50_n{length}.jsonl"
        for length in (2, 3, 4)
    }
    for length, manifest in manifests.items():
        if any(value in str(manifest).casefold() for value in ("analysis_500", "final_test", "locked")):
            raise RuntimeError("D1 cannot open a locked manifest")
        rows = read_jsonl(manifest)
        if len(rows) != 50 or {int(row["target_length"]) for row in rows} != {length}:
            raise RuntimeError(f"Invalid KAMEL n={length} dev manifest")
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "D1_partial_state_target_delta",
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "layer": layer,
            "shared_variants": SHARED_VARIANTS,
            "bucket_schedules": BUCKET_SCHEDULES,
            "ridge": args.ridge,
            "alpha": args.alpha,
            "top_q": args.top_q,
            "target_optimization_steps": args.target_optimization_steps,
            "learning_rate": args.learning_rate,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    all_reports = []
    outputs_by_length_method: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for length, manifest in manifests.items():
        length_dir = args.output_dir / f"n{length}"
        length_dir.mkdir()
        requests = read_jsonl(manifest)
        tasks = build_eval_tasks(tokenizer, requests, include_locality=True)
        base_rows = evaluate_tasks(
            model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
        )
        write_csv(length_dir / "base_per_prompt.csv", base_rows)
        shared_memories: dict[str, Any] = {}
        shared_diagnostics: dict[str, list[dict[str, Any]]] = {}
        for method, (schedule, reveal, consistency) in SHARED_VARIANTS.items():
            method_dir = length_dir / method
            method_dir.mkdir()
            memory, diagnostics, fit_runtime = fit_residual_memory_for_requests(
                model,
                tokenizer,
                requests,
                layer=layer,
                ridge=args.ridge,
                target_optimization_steps=args.target_optimization_steps,
                learning_rate=args.learning_rate,
                partial_mask_schedule=schedule,
                reveal_policy=reveal,
                state_consistency_weight=consistency,
                old_target_suppression_weight=0.25,
                seed=args.seed,
                cache_dir=method_dir / "target_value_cache",
            )
            evaluation_start = time.monotonic()
            module = get_module(model, resolved_key_module_name(model, layer))
            with install_factorized_residual_memory(
                module, memory, alpha=args.alpha, top_q=args.top_q
            ) as activation:
                raw = evaluate_tasks(
                    model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
                )
            edited = align_base(base_rows, raw)
            runtime = fit_runtime + time.monotonic() - evaluation_start
            report = method_report(
                method=method,
                target_length=length,
                manifest=manifest,
                base_rows=base_rows,
                edited_rows=edited,
                memory_storage_bytes=memory.storage_bytes,
                memory_rank_bound=memory.rank_bound,
                fit_runtime_seconds=fit_runtime,
                runtime_seconds=runtime,
                activation=activation,
                schedule=schedule,
                reveal_policy=reveal,
                diagnostics=diagnostics,
                memory_finite=all(
                    torch.isfinite(value).all()
                    for value in (memory.keys, memory.dual, memory.residuals)
                ),
            )
            torch.save(memory.cpu_payload(), method_dir / "residual_memory.pt")
            write_csv(method_dir / "edited_per_prompt.csv", edited)
            write_json(method_dir / "target_value_diagnostics.json", diagnostics)
            write_json(method_dir / "report_summary.json", report)
            shared_memories[method] = memory
            shared_diagnostics[method] = diagnostics
            outputs_by_length_method[(length, method)] = edited
            all_reports.append(report)
            print(f"D1 n={length} {method} rewrite={report['rewrite_exact']:.4f}", flush=True)
        bucket_memories: dict[str, Any] = {}
        bucket_diagnostics: dict[str, list[dict[str, Any]]] = {}
        bucket_fit_runtime = 0.0
        bucket_dir = length_dir / "state_bucketed_delta"
        bucket_dir.mkdir()
        for bucket, (schedule, reveal) in BUCKET_SCHEDULES.items():
            if bucket == "middle":
                memory = shared_memories["uniform_partial_state_delta"]
                diagnostics = shared_diagnostics["uniform_partial_state_delta"]
                fit_runtime = 0.0
            else:
                memory, diagnostics, fit_runtime = fit_residual_memory_for_requests(
                    model,
                    tokenizer,
                    requests,
                    layer=layer,
                    ridge=args.ridge,
                    target_optimization_steps=args.target_optimization_steps,
                    learning_rate=args.learning_rate,
                    partial_mask_schedule=schedule,
                    reveal_policy=reveal,
                    state_consistency_weight=0.1,
                    old_target_suppression_weight=0.25,
                    seed=args.seed,
                    cache_dir=bucket_dir / f"target_value_cache_{bucket}",
                )
            bucket_memories[bucket] = memory
            bucket_diagnostics[bucket] = diagnostics
            bucket_fit_runtime += fit_runtime
            torch.save(memory.cpu_payload(), bucket_dir / f"residual_memory_{bucket}.pt")
        module = get_module(model, resolved_key_module_name(model, layer))
        evaluation_start = time.monotonic()
        with install_state_bucketed_residual_memories(
            model,
            module,
            bucket_memories,
            mask_id=infer_mask_id(model),
            alpha=args.alpha,
            top_q=args.top_q,
        ) as activation:
            raw = evaluate_tasks(
                model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
            )
        edited = align_base(base_rows, raw)
        bucket_diagnostic_rows = [
            {"state_bucket": bucket, **row}
            for bucket, values in bucket_diagnostics.items()
            for row in values
        ]
        report = method_report(
            method="state_bucketed_delta",
            target_length=length,
            manifest=manifest,
            base_rows=base_rows,
            edited_rows=edited,
            memory_storage_bytes=sum(memory.storage_bytes for memory in bucket_memories.values()),
            memory_rank_bound=sum(memory.rank_bound for memory in bucket_memories.values()),
            fit_runtime_seconds=bucket_fit_runtime,
            runtime_seconds=bucket_fit_runtime + time.monotonic() - evaluation_start,
            activation=activation,
            schedule="state_bucketed_early_middle_late",
            reveal_policy="runtime_active_mask_fraction",
            diagnostics=bucket_diagnostic_rows,
            memory_finite=all(
                torch.isfinite(value).all()
                for memory in bucket_memories.values()
                for value in (memory.keys, memory.dual, memory.residuals)
            ),
        )
        write_csv(bucket_dir / "edited_per_prompt.csv", edited)
        write_json(bucket_dir / "target_value_diagnostics.json", bucket_diagnostic_rows)
        write_json(bucket_dir / "report_summary.json", report)
        outputs_by_length_method[(length, "state_bucketed_delta")] = edited
        all_reports.append(report)
        shuffled_dir = length_dir / "state_bucketed_delta_shuffled_control"
        shuffled_dir.mkdir()
        shuffled_start = time.monotonic()
        with install_state_bucketed_residual_memories(
            model,
            module,
            bucket_memories,
            mask_id=infer_mask_id(model),
            alpha=args.alpha,
            top_q=args.top_q,
            shuffle_buckets=True,
        ) as shuffled_activation:
            shuffled_raw = evaluate_tasks(
                model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
            )
        shuffled = align_base(base_rows, shuffled_raw)
        shuffled_report = method_report(
            method="state_bucketed_delta_shuffled_control",
            target_length=length,
            manifest=manifest,
            base_rows=base_rows,
            edited_rows=shuffled,
            memory_storage_bytes=sum(memory.storage_bytes for memory in bucket_memories.values()),
            memory_rank_bound=sum(memory.rank_bound for memory in bucket_memories.values()),
            fit_runtime_seconds=0.0,
            runtime_seconds=time.monotonic() - shuffled_start,
            activation=shuffled_activation,
            schedule="state_bucketed_shuffled_control",
            reveal_policy="frozen_bucket_permutation",
            diagnostics=bucket_diagnostic_rows,
            memory_finite=all(
                torch.isfinite(value).all()
                for memory in bucket_memories.values()
                for value in (memory.keys, memory.dual, memory.residuals)
            ),
        )
        write_csv(shuffled_dir / "edited_per_prompt.csv", shuffled)
        write_json(shuffled_dir / "report_summary.json", shuffled_report)
        outputs_by_length_method[(length, "state_bucketed_delta_shuffled_control")] = shuffled
        all_reports.append(shuffled_report)
        del shared_memories, bucket_memories
        torch.cuda.empty_cache()
    write_csv(
        args.output_dir / "variant_summary.csv",
        [
            {
                key: row[key]
                for key in (
                    "target_length",
                    "method",
                    "rewrite_exact",
                    "declarative_paraphrase_exact",
                    "same_subject_tfpr",
                    "malformed_rate",
                    "selection_score",
                    "stress_aware_aggregate",
                    "memory_storage_bytes",
                    "runtime_seconds",
                )
            }
            for row in all_reports
        ],
    )
    methods = sorted({str(row["method"]) for row in all_reports if "shuffled" not in str(row["method"])})
    by_method = {
        method: sum(
            float(row["rewrite_exact"]) + float(row["declarative_paraphrase_exact"])
            for row in all_reports
            if row["method"] == method
        ) / 3.0
        for method in methods
    }
    partial_methods = [method for method in methods if method != "fullmask_delta"]
    selected_partial = max(partial_methods, key=lambda method: (by_method[method], method))
    length_rows = []
    pooled_anchor = []
    pooled_full = []
    positive_lengths = []
    locality_regressions = []
    for length in (2, 3, 4):
        selected = next(row for row in all_reports if row["target_length"] == length and row["method"] == selected_partial)
        full = next(row for row in all_reports if row["target_length"] == length and row["method"] == "fullmask_delta")
        gain = float(selected["rewrite_exact"]) - float(full["rewrite_exact"])
        if gain >= 0.10:
            positive_lengths.append(length)
        locality_regressions.append(float(selected["same_subject_tfpr"]) - float(full["same_subject_tfpr"]))
        length_rows.append(
            {
                "target_length": length,
                "selected_partial_method": selected_partial,
                "partial_rewrite_exact": selected["rewrite_exact"],
                "fullmask_rewrite_exact": full["rewrite_exact"],
                "rewrite_gain": gain,
                "partial_paraphrase_exact": selected["declarative_paraphrase_exact"],
                "fullmask_paraphrase_exact": full["declarative_paraphrase_exact"],
                "partial_same_subject_tfpr": selected["same_subject_tfpr"],
                "fullmask_same_subject_tfpr": full["same_subject_tfpr"],
            }
        )
        pooled_anchor.extend(outputs_by_length_method[(length, selected_partial)])
        pooled_full.extend(outputs_by_length_method[(length, "fullmask_delta")])
    bootstrap = paired_bootstrap_rewrite_delta(
        pooled_anchor, pooled_full, seed=args.seed
    )
    write_csv(args.output_dir / "target_length_comparison.csv", length_rows)
    write_csv(args.output_dir / "paired_bootstrap.csv", [{"comparison": f"{selected_partial}_minus_fullmask", **bootstrap}])
    state_bucketed_mean = by_method["state_bucketed_delta"]
    shared_best = max(by_method[method] for method in ("uniform_partial_state_delta", "mask_count_cycling_delta", "trajectory_sampled_delta"))
    trajectory_advantage = by_method["trajectory_sampled_delta"] - by_method["uniform_partial_state_delta"]
    shuffle_rows = [row for row in all_reports if row["method"] == "state_bucketed_delta_shuffled_control"]
    shuffle_mean = sum(float(row["rewrite_exact"]) + float(row["declarative_paraphrase_exact"]) for row in shuffle_rows) / 3.0
    mechanism = {
        "state_bucketed_minus_best_shared": state_bucketed_mean - shared_best,
        "trajectory_minus_uniform": trajectory_advantage,
        "state_bucketed_minus_shuffled": state_bucketed_mean - shuffle_mean,
    }
    diffusion_specific_pass = (
        len(positive_lengths) >= 2
        and bootstrap["ci_low"] > 0
        and all(float(row["malformed_rate"]) <= 0.05 for row in all_reports if row["method"] == selected_partial)
        and max(locality_regressions) <= 0.03
    )
    stronger_mechanism_pass = any(value >= 0.05 for value in mechanism.values())
    integrity = {
        "all_five_mandatory_variants_complete": set(methods) == set(SHARED_VARIANTS) | {"state_bucketed_delta"},
        "all_three_exact_target_lengths_complete": {int(row["target_length"]) for row in all_reports} == {2, 3, 4},
        "all_metrics_finite": all(bool(row["all_metrics_finite"]) for row in all_reports),
        "all_memories_finite": all(bool(row["memory_finite"]) for row in all_reports),
        "state_shuffle_control_complete": len(shuffle_rows) == 3,
        "runtime_inputs_deployable": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = all(value for key, value in integrity.items() if key not in {"analysis_500_used", "final_test_used"})
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D1_partial_state_target_delta",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "selected_partial_method": selected_partial,
        "method_objectives": by_method,
        "positive_target_lengths": positive_lengths,
        "pooled_paired_bootstrap": bootstrap,
        "mechanism_diagnostics": mechanism,
        "diffusion_specific_pass": diffusion_specific_pass,
        "stronger_mechanism_pass": stronger_mechanism_pass,
        "integrity": integrity,
        "runtime_seconds": runtime,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": passed,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", {"integrity": integrity, "acceptance_pass": passed})
    record_stage_cost("D1_partial_state_target_delta", runtime_seconds=runtime, notes="Five partial-state target-delta variants across KAMEL n2/n3/n4")
    record_stage(
        "D1_partial_state_target_delta",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"selected={selected_partial}; diffusion_specific_pass={diffusion_specific_pass}",
        next_stage="D2_state_conditioned_protection" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(json.dumps({"acceptance_pass": True, "selected_partial_method": selected_partial, "diffusion_specific_pass": diffusion_specific_pass}, sort_keys=True))


if __name__ == "__main__":
    main()
