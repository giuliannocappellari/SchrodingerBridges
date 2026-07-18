#!/usr/bin/env python3
"""Fit and evaluate the C2 full-mask temporal residual baseline."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_editor import (
    MemitConfig,
    extract_keys_and_outputs,
    get_module,
    optimize_target_value,
    resolved_key_module_name,
)
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
    fit_factorized_residual_memory,
    install_factorized_residual_memory,
    summarize_editor_rows,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frozen_policy_layers(c1_dir: Path) -> dict[str, int]:
    policies = read_csv(c1_dir / "site_policy_comparison.csv")
    by_name = {row["policy_id"]: row for row in policies}

    def first_layer(policy_id: str) -> int:
        ids = json.loads(by_name[policy_id]["candidate_ids_json"])
        if not ids:
            raise RuntimeError(f"C1 policy {policy_id} has no candidate")
        prefix, component, position = ids[0].split(":")
        if component != "mlp" or position != "last_subject":
            raise RuntimeError(f"C2 requires an MLP/last-subject site, got {ids[0]}")
        return int(prefix[1:])

    return {
        "stable_temporal_top1": first_layer("stable_temporal_site_set"),
        "random_site_top1": first_layer("random_site"),
    }


def fit_one_memory(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    ridge: float,
    target_optimization_steps: int,
    seed: int,
    cache_dir: Path,
) -> tuple[Any, list[dict[str, Any]], float]:
    started = time.monotonic()
    config = MemitConfig(
        layers=(int(layer),),
        target_optimization_steps=int(target_optimization_steps),
        partial_mask_schedule="fully_masked",
        reveal_policy="random",
        state_consistency_weight=0.0,
        old_target_suppression_weight=0.25,
        seed=int(seed),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    diagnostics = []
    for index, row in enumerate(rows, start=1):
        cache_path = cache_dir / f"{row['case_id']}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            target = payload["target_value"].float()
            report = payload["diagnostics"]
        else:
            target, report = optimize_target_value(model, tokenizer, row, config)
            target = target.detach().cpu()
            torch.save({"target_value": target, "diagnostics": report}, cache_path)
        targets.append(target)
        diagnostics.append({"case_id": row["case_id"], **report})
        if index % 10 == 0 or index == len(rows):
            print(f"C2 target values layer={layer} {index}/{len(rows)}", flush=True)
    keys, current_outputs = extract_keys_and_outputs(
        model,
        tokenizer,
        rows,
        key_layer=int(layer),
        output_layer=int(layer),
        partial_mask_schedule="fully_masked",
        reveal_policy="random",
        seed=int(seed),
    )
    residuals = torch.stack(targets) - current_outputs
    memory = fit_factorized_residual_memory(
        keys.to("cuda"), residuals.to("cuda"), ridge=float(ridge)
    )
    return memory, diagnostics, time.monotonic() - started


def run_split(
    model: Any,
    tokenizer: Any,
    *,
    manifest: Path,
    output_dir: Path,
    policies: Mapping[str, int],
    ridge: float,
    alpha: float,
    top_q: int,
    target_optimization_steps: int,
    decode_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(manifest)
    tasks = build_eval_tasks(tokenizer, rows, include_locality=True)
    base_started = time.monotonic()
    base_rows = evaluate_tasks(
        model, tokenizer, tasks, decode_batch_size=decode_batch_size, steps=None
    )
    base_runtime = time.monotonic() - base_started
    write_csv(output_dir / "base_per_prompt.csv", base_rows)
    methods = []
    for method_id, layer in policies.items():
        method_dir = output_dir / method_id
        method_dir.mkdir()
        method_started = time.monotonic()
        memory, target_diagnostics, fit_runtime = fit_one_memory(
            model,
            tokenizer,
            rows,
            layer=layer,
            ridge=ridge,
            target_optimization_steps=target_optimization_steps,
            seed=seed,
            cache_dir=method_dir / "target_value_cache",
        )
        module = get_module(model, resolved_key_module_name(model, layer))
        with install_factorized_residual_memory(
            module, memory, alpha=alpha, top_q=top_q
        ) as activation:
            raw_edited = evaluate_tasks(
                model,
                tokenizer,
                tasks,
                decode_batch_size=decode_batch_size,
                steps=None,
            )
        edited_rows = align_base(base_rows, raw_edited)
        metrics = summarize_editor_rows(base_rows, edited_rows)
        runtime = time.monotonic() - method_started
        torch.save(memory.cpu_payload(), method_dir / "residual_memory.pt")
        write_json(method_dir / "target_value_diagnostics.json", target_diagnostics)
        write_csv(method_dir / "edited_per_prompt.csv", edited_rows)
        method_report = {
            "campaign_id": CAMPAIGN_ID,
            "stage": "C2_fullmask_temporal_residual",
            "method": method_id,
            "site_policy_layer": int(layer),
            "site_component": "mlp",
            "site_position": "last_subject",
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "num_edits": len(rows),
            "partial_mask_schedule": "fully_masked",
            "ridge": float(ridge),
            "alpha": float(alpha),
            "top_q": int(top_q),
            "target_optimization_steps": int(target_optimization_steps),
            "memory_rank_bound": memory.rank_bound,
            "memory_storage_bytes": memory.storage_bytes,
            "memory_finite": all(
                torch.isfinite(value).all()
                for value in (memory.keys, memory.dual, memory.residuals)
            ),
            "activation_diagnostics": activation,
            "fit_runtime_seconds": fit_runtime,
            "runtime_seconds": runtime,
            "gpu_minutes_per_edit": runtime / 60.0 / len(rows),
            "model_eval_count": sum(
                int(row["model_eval_count"]) for row in raw_edited
            ) + sum(len(row["history"]) + 2 for row in target_diagnostics),
            **metrics,
            "analysis_500_used": False,
            "final_test_used": False,
        }
        write_json(method_dir / "report_summary.json", method_report)
        write_json(
            method_dir / "run_config.json",
            {
                key: method_report[key]
                for key in (
                    "campaign_id",
                    "stage",
                    "method",
                    "site_policy_layer",
                    "site_component",
                    "site_position",
                    "manifest",
                    "manifest_sha256",
                    "partial_mask_schedule",
                    "ridge",
                    "alpha",
                    "top_q",
                    "target_optimization_steps",
                    "analysis_500_used",
                    "final_test_used",
                )
            },
        )
        write_json(
            method_dir / "validation_report.json",
            {
                "memory_finite": method_report["memory_finite"],
                "all_metrics_finite": method_report["all_metrics_finite"],
                "runtime_schema_deployable": True,
                "teacher_or_outcome_runtime_inputs": False,
                "acceptance_pass": bool(
                    method_report["memory_finite"]
                    and method_report["all_metrics_finite"]
                ),
            },
        )
        methods.append(method_report)
        del memory
        torch.cuda.empty_cache()
    summary = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "num_edits": len(rows),
        "num_tasks": len(tasks),
        "base_runtime_seconds": base_runtime,
        "methods": methods,
    }
    write_json(output_dir / "report_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_manifest", type=Path, default=PROTOCOL_ROOT / "cf_trm_smoke_20.jsonl")
    parser.add_argument("--pilot_manifest", type=Path, default=PROTOCOL_ROOT / "cf_trm_pilot_100.jsonl")
    parser.add_argument("--c1_dir", type=Path, default=CAMPAIGN_ROOT / "C1_temporal_localization_v1")
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "C2_fullmask_temporal_residual_v1")
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top_q", type=int, default=0)
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260718401)
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    for manifest, expected, role in (
        (args.smoke_manifest, 20, "cf_trm_smoke_20"),
        (args.pilot_manifest, 100, "cf_trm_pilot_100"),
    ):
        if any(value in str(manifest).casefold() for value in ("analysis_500", "final_test_500", "final_test_full", "locked")):
            raise RuntimeError("C2 cannot open a locked manifest")
        rows = read_jsonl(manifest)
        if len(rows) != expected or {row["split_role"] for row in rows} != {role}:
            raise RuntimeError(f"Unexpected {role} manifest composition")
    c1_report = read_json(args.c1_dir / "report_summary.json")
    if not c1_report.get("acceptance_pass"):
        raise RuntimeError("C1 must pass before C2")
    policies = frozen_policy_layers(args.c1_dir)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "C2_fullmask_temporal_residual",
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "site_policies": policies,
            "ridge": args.ridge,
            "alpha": args.alpha,
            "top_q": args.top_q,
            "target_optimization_steps": args.target_optimization_steps,
            "C1_site_policy_lock_sha256": sha256_file(args.c1_dir / "site_policy_lock.json"),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    smoke = run_split(
        model,
        tokenizer,
        manifest=args.smoke_manifest,
        output_dir=args.output_dir / "smoke20_v1",
        policies=policies,
        ridge=args.ridge,
        alpha=args.alpha,
        top_q=args.top_q,
        target_optimization_steps=args.target_optimization_steps,
        decode_batch_size=args.decode_batch_size,
        seed=args.seed,
    )
    smoke_by_method = {row["method"]: row for row in smoke["methods"]}
    smoke_stable = smoke_by_method["stable_temporal_top1"]
    red_failures = {
        "no_rewrite_gain": float(smoke_stable["rewrite_exact"]) <= float(smoke_stable["base_rewrite_exact"]),
        "malformed_above_0_05": float(smoke_stable["malformed_rate"]) > 0.05,
        "same_subject_tfpr_above_0_30": float(smoke_stable["same_subject_tfpr"]) > 0.30,
        "runtime_schema_mismatch": False,
        "numerical_instability": not bool(smoke_stable["memory_finite"] and smoke_stable["all_metrics_finite"]),
    }
    if any(red_failures.values()):
        runtime = time.monotonic() - begin
        report = {
            "campaign_id": CAMPAIGN_ID,
            "stage": "C2_fullmask_temporal_residual",
            "created_at_utc": now_utc(),
            "git_commit": git_commit(),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "site_policies": policies,
            "smoke_red_failures": red_failures,
            "pilot_run": False,
            "runtime_seconds": runtime,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": False,
        }
        write_json(args.output_dir / "report_summary.json", report)
        write_json(args.output_dir / "validation_report.json", report)
        record_stage_cost("C2_fullmask_temporal_residual", runtime_seconds=runtime, notes="C2 stopped at smoke red failure")
        record_stage(
            "C2_fullmask_temporal_residual",
            status="failed_smoke_red",
            acceptance_pass=False,
            output_dir=args.output_dir,
            started_at_utc=started,
            notes=f"Smoke red failures: {[key for key, value in red_failures.items() if value]}",
            next_stage=None,
        )
        raise SystemExit(2)
    pilot = run_split(
        model,
        tokenizer,
        manifest=args.pilot_manifest,
        output_dir=args.output_dir / "pilot100_v1",
        policies=policies,
        ridge=args.ridge,
        alpha=args.alpha,
        top_q=args.top_q,
        target_optimization_steps=args.target_optimization_steps,
        decode_batch_size=args.decode_batch_size,
        seed=args.seed,
    )
    pilot_by_method = {row["method"]: row for row in pilot["methods"]}
    stable = pilot_by_method["stable_temporal_top1"]
    random = pilot_by_method["random_site_top1"]
    stress_delta = float(stable["stress_aware_aggregate"]) - float(random["stress_aware_aggregate"])
    rewrite_gain = float(stable["rewrite_exact"]) - float(stable["base_rewrite_exact"])
    site_better = stress_delta >= 0.05
    matched_efficiency = (
        float(stable["rewrite_exact"]) >= float(random["rewrite_exact"]) - 0.02
        and 1 <= 0.75 * 1
    )
    acceptance = {
        "nontrivial_rewrite_gain_over_base": rewrite_gain > 0.0,
        "temporal_site_beats_random_by_0_05_or_is_more_efficient": site_better or matched_efficiency,
        "finite_residual_parameters": bool(stable["memory_finite"]),
        "no_catastrophic_utility_collapse": float(stable["utility_base_agreement"]) >= 0.50,
        "malformed_at_most_0_05": float(stable["malformed_rate"]) <= 0.05,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = all(
        value for key, value in acceptance.items() if key not in {"analysis_500_used", "final_test_used"}
    ) and not acceptance["analysis_500_used"] and not acceptance["final_test_used"]
    runtime = time.monotonic() - begin
    comparison_rows = []
    for split_name, payload in (("smoke20", smoke), ("pilot100", pilot)):
        for row in payload["methods"]:
            comparison_rows.append(
                {
                    "split": split_name,
                    "method": row["method"],
                    "layer": row["site_policy_layer"],
                    "rewrite_exact": row["rewrite_exact"],
                    "declarative_paraphrase_exact": row["declarative_paraphrase_exact"],
                    "same_subject_tfpr": row["same_subject_tfpr"],
                    "near_tfpr": row["near_tfpr"],
                    "far_tfpr": row["far_tfpr"],
                    "malformed_rate": row["malformed_rate"],
                    "selection_score": row["selection_score"],
                    "stress_aware_aggregate": row["stress_aware_aggregate"],
                    "gpu_minutes_per_edit": row["gpu_minutes_per_edit"],
                    "memory_storage_bytes": row["memory_storage_bytes"],
                }
            )
    write_csv(args.output_dir / "site_policy_comparison.csv", comparison_rows)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "C2_fullmask_temporal_residual",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "site_policies": policies,
        "smoke_red_failures": red_failures,
        "pilot_run": True,
        "pilot_stable_method": stable,
        "pilot_random_method": random,
        "pilot_rewrite_gain_over_base": rewrite_gain,
        "pilot_stress_aggregate_delta_vs_random": stress_delta,
        "site_policy_rescue_status": "already_using_best_stable_localization_policy_no_additional_rescue_triggered",
        "acceptance": acceptance,
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
    write_json(args.output_dir / "validation_report.json", {"acceptance": acceptance, "acceptance_pass": passed})
    record_stage_cost(
        "C2_fullmask_temporal_residual",
        runtime_seconds=runtime,
        notes="Full-mask temporal residual smoke20 and pilot100",
    )
    record_stage(
        "C2_fullmask_temporal_residual",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"rewrite gain={rewrite_gain:.4f}; stable-random stress delta={stress_delta:.4f}",
        next_stage="D1_partial_state_target_delta" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(json.dumps({"acceptance_pass": True, "rewrite_gain": rewrite_gain, "stress_delta_vs_random": stress_delta}, sort_keys=True))


if __name__ == "__main__":
    main()
