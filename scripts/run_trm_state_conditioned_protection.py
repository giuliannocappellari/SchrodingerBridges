#!/usr/bin/env python3
"""Run D2 state-conditioned locality-preservation variants on smoke20."""

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
from typing import Any, Callable, ContextManager, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_editor import (
    get_module,
    infer_mask_id,
    model_device,
    pad_batch,
    resolved_key_module_name,
)
from scripts.run_dnpe_editor import align_base, build_eval_tasks, evaluate_tasks
from scripts.run_mdm_memit_stage import load_model
from scripts.run_trm_partial_state_target_delta import SHARED_VARIANTS
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
    FactorizedResidualMemory,
    build_input_protection_basis,
    fit_residual_memory_for_requests,
    install_factorized_residual_memory,
    install_state_bucketed_residual_memories,
    summarize_editor_rows,
)
from scripts.trm_protection import build_protection_prompt_records, extract_protection_keys


def selected_shared_policy(d1_report: Mapping[str, Any]) -> tuple[str, str, float]:
    method = str(d1_report["selected_partial_method"])
    if method == "state_bucketed_delta":
        return "cycle", "base_confidence", 0.1
    if method not in SHARED_VARIANTS:
        raise RuntimeError(f"Unknown D1 selected method: {method}")
    return SHARED_VARIANTS[method]


@torch.no_grad()
def anchor_logits(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 16,
) -> torch.Tensor:
    device = model_device(model)
    mask_id = infer_mask_id(model)
    outputs = []
    for start in range(0, len(records), int(batch_size)):
        subset = records[start : start + int(batch_size)]
        prompt_ids = [
            list(map(int, tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"]))
            for row in subset
        ]
        batch = pad_batch(
            [{"input_ids": values + [mask_id]} for values in prompt_ids],
            int(tokenizer.pad_token_id),
            device,
        )
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits.float()
        offsets = batch["left_offsets"].tolist()
        for row_index, values in enumerate(prompt_ids):
            outputs.append(logits[row_index, int(offsets[row_index]) + len(values)].cpu())
    return torch.stack(outputs)


def distribution_kl(base_logits: torch.Tensor, edited_logits: torch.Tensor) -> float:
    return float(
        F.kl_div(
            F.log_softmax(edited_logits.float(), dim=-1),
            F.softmax(base_logits.float(), dim=-1),
            reduction="batchmean",
        )
    )


def paired_tfpr_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    trials: int = 2000,
    seed: int = 260718601,
) -> dict[str, float]:
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("bucket") == "same_subject":
                grouped[str(row["case_id"])].append(
                    float(bool(row["target_new_hit"]))
                )
        return {
            case_id: sum(values) / len(values)
            for case_id, values in grouped.items()
        }

    a = index(left)
    b = index(right)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise RuntimeError("same-subject bootstrap has no aligned edits")
    observed = sum(a[value] - b[value] for value in ids) / len(ids)
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(trials)):
        sample = [rng.choice(ids) for _ in ids]
        draws.append(sum(a[value] - b[value] for value in sample) / len(sample))
    draws.sort()
    return {
        "num_cases": len(ids),
        "delta": observed,
        "ci_low": draws[int(0.025 * (len(draws) - 1))],
        "ci_high": draws[int(0.975 * (len(draws) - 1))],
    }


def memory_drift(
    memory: FactorizedResidualMemory,
    keys: torch.Tensor,
    *,
    alpha: float,
    top_q: int,
) -> float:
    with torch.no_grad():
        return float(
            memory.predict(keys.to(memory.dual.device), alpha=alpha, top_q=top_q)
            .float()
            .norm(dim=1)
            .mean()
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROTOCOL_ROOT / "cf_trm_smoke_20.jsonl")
    parser.add_argument("--anchor_manifest", type=Path, default=PROTOCOL_ROOT / "cf_trm_anchor_train_500.jsonl")
    parser.add_argument("--d1_dir", type=Path, default=CAMPAIGN_ROOT / "D1_partial_state_target_delta_v1")
    parser.add_argument("--c1_dir", type=Path, default=CAMPAIGN_ROOT / "C1_temporal_localization_v1")
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "D2_state_conditioned_protection_v1")
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--preservation_strength", type=float, default=1.0)
    parser.add_argument("--protected_variance", type=float, default=0.95)
    parser.add_argument("--maximum_basis_rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sparse_top_q", type=int, default=256)
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--anchor_per_family", type=int, default=32)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260718601)
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    requests = read_jsonl(args.manifest)
    anchors = read_jsonl(args.anchor_manifest)
    if len(requests) != 20 or {row["split_role"] for row in requests} != {"cf_trm_smoke_20"}:
        raise RuntimeError("D2 requires fresh CounterFact smoke20")
    if len(anchors) != 500 or {row["split_role"] for row in anchors} != {"cf_trm_anchor_train_500"}:
        raise RuntimeError("D2 protection anchors must be the fresh train-only 500")
    for path in (args.manifest, args.anchor_manifest):
        if any(value in str(path).casefold() for value in ("analysis_500", "final_test", "locked")):
            raise RuntimeError("D2 cannot open locked evaluation data")
    d1 = read_json(args.d1_dir / "report_summary.json")
    if not d1.get("acceptance_pass"):
        raise RuntimeError("D1 integrity must pass before D2")
    schedule, reveal, consistency = selected_shared_policy(d1)
    c1_lock = read_json(args.c1_dir / "site_policy_lock.json")
    stable = next(row for row in c1_lock["policies"] if row["policy_id"] == "stable_temporal_site_set")
    layer = int(json.loads(stable["layers_json"])[0])
    records, anchor_summary = build_protection_prompt_records(
        anchors, max_per_family=args.anchor_per_family
    )
    if not anchor_summary["all_required_families_present"]:
        raise RuntimeError("train-only protection data lacks a required anchor family")
    write_csv(args.output_dir / "protection_anchor_manifest.csv", records)
    write_json(args.output_dir / "protection_anchor_summary.json", anchor_summary)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "D2_state_conditioned_protection",
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "layer": layer,
            "D1_selected_partial_method": d1["selected_partial_method"],
            "shared_schedule": schedule,
            "shared_reveal_policy": reveal,
            "ridge": args.ridge,
            "preservation_strength": args.preservation_strength,
            "protected_variance": args.protected_variance,
            "maximum_basis_rank": args.maximum_basis_rank,
            "alpha": args.alpha,
            "sparse_top_q": args.sparse_top_q,
            "anchor_manifest": str(args.anchor_manifest),
            "anchor_manifest_sha256": sha256_file(args.anchor_manifest),
            "evaluation_prompts_used_as_anchors": False,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    protection_keys = {}
    protection_metadata = []
    for bucket in ("early", "middle", "late"):
        keys, metadata = extract_protection_keys(
            model,
            tokenizer,
            records,
            layer=layer,
            state_bucket=bucket,
            span_length=3,
            seed=args.seed,
        )
        protection_keys[bucket] = keys
        protection_metadata.extend(metadata)
        torch.save(keys, args.output_dir / f"protection_keys_{bucket}.pt")
    write_csv(args.output_dir / "protection_key_metadata.csv", protection_metadata)
    combined_protection = torch.cat(
        [protection_keys[bucket] for bucket in ("early", "middle", "late")], dim=0
    ).to("cuda")
    basis, basis_report = build_input_protection_basis(
        combined_protection,
        explained_variance=args.protected_variance,
        maximum_rank=args.maximum_basis_rank,
    )
    torch.save({"basis": basis.cpu(), "report": basis_report}, args.output_dir / "static_protection_basis.pt")
    tasks = build_eval_tasks(tokenizer, requests, include_locality=True)
    base_rows = evaluate_tasks(
        model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
    )
    write_csv(args.output_dir / "base_per_prompt.csv", base_rows)
    kl_records = records[: min(64, len(records))]
    base_anchor_logits = anchor_logits(model, tokenizer, kl_records)
    shared_cache = args.output_dir / "shared_target_value_cache"
    shared_specs = (
        ("unprotected_temporal_residual", None, 0.0, None),
        ("static_global_nullspace", None, 0.0, basis),
        (
            "shared_soft_preservation",
            combined_protection,
            args.preservation_strength,
            None,
        ),
    )
    reports = []
    edited_outputs: dict[str, list[dict[str, Any]]] = {}
    module = get_module(model, resolved_key_module_name(model, layer))
    for method, protect, strength, projection in shared_specs:
        method_dir = args.output_dir / method
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
            cache_dir=shared_cache,
            protect_keys=protect,
            preservation_strength=strength,
            input_projection_basis=projection,
        )
        evaluation_start = time.monotonic()
        with install_factorized_residual_memory(
            module, memory, alpha=args.alpha, top_q=0
        ) as activation:
            raw = evaluate_tasks(
                model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
            )
            edited_anchor_logits = anchor_logits(model, tokenizer, kl_records)
        edited = align_base(base_rows, raw)
        metrics = summarize_editor_rows(base_rows, edited)
        report = {
            "method": method,
            "protection_type": method,
            "state_conditioned": False,
            "sparse_top_q": 0,
            "memory_finite": all(
                torch.isfinite(value).all()
                for value in (memory.keys, memory.dual, memory.residuals)
            ),
            "memory_storage_bytes": memory.storage_bytes,
            "memory_rank_bound": memory.rank_bound,
            "protect_row_count": memory.protect_row_count,
            "preservation_key_drift": memory_drift(
                memory, combined_protection, alpha=args.alpha, top_q=0
            ),
            "retain_distribution_kl": distribution_kl(
                base_anchor_logits, edited_anchor_logits
            ),
            "activation_diagnostics": activation,
            "fit_runtime_seconds": fit_runtime,
            "runtime_seconds": fit_runtime + time.monotonic() - evaluation_start,
            **metrics,
            "analysis_500_used": False,
            "final_test_used": False,
        }
        torch.save(memory.cpu_payload(), method_dir / "residual_memory.pt")
        write_csv(method_dir / "edited_per_prompt.csv", edited)
        write_json(method_dir / "target_value_diagnostics.json", diagnostics)
        write_json(method_dir / "report_summary.json", report)
        reports.append(report)
        edited_outputs[method] = edited
        del memory
        torch.cuda.empty_cache()
    bucket_memories = {}
    bucket_diagnostics = {}
    bucket_fit_runtime = 0.0
    for bucket, bucket_schedule in (
        ("early", "fewer_revealed"),
        ("middle", "uniform"),
        ("late", "more_revealed"),
    ):
        memory, diagnostics, fit_runtime = fit_residual_memory_for_requests(
            model,
            tokenizer,
            requests,
            layer=layer,
            ridge=args.ridge,
            target_optimization_steps=args.target_optimization_steps,
            learning_rate=args.learning_rate,
            partial_mask_schedule=bucket_schedule,
            reveal_policy="random",
            state_consistency_weight=0.1,
            old_target_suppression_weight=0.25,
            seed=args.seed,
            cache_dir=args.output_dir / f"bucket_target_value_cache_{bucket}",
            protect_keys=protection_keys[bucket].to("cuda"),
            preservation_strength=args.preservation_strength,
        )
        bucket_memories[bucket] = memory
        bucket_diagnostics[bucket] = diagnostics
        bucket_fit_runtime += fit_runtime
    for method, top_q in (
        ("state_conditioned_preservation", 0),
        ("state_conditioned_sparsification", args.sparse_top_q),
    ):
        method_dir = args.output_dir / method
        method_dir.mkdir()
        evaluation_start = time.monotonic()
        with install_state_bucketed_residual_memories(
            model,
            module,
            bucket_memories,
            mask_id=infer_mask_id(model),
            alpha=args.alpha,
            top_q=top_q,
        ) as activation:
            raw = evaluate_tasks(
                model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None
            )
            edited_anchor_logits = anchor_logits(model, tokenizer, kl_records)
        edited = align_base(base_rows, raw)
        metrics = summarize_editor_rows(base_rows, edited)
        report = {
            "method": method,
            "protection_type": method,
            "state_conditioned": True,
            "sparse_top_q": int(top_q),
            "memory_finite": all(
                torch.isfinite(value).all()
                for memory in bucket_memories.values()
                for value in (memory.keys, memory.dual, memory.residuals)
            ),
            "memory_storage_bytes": sum(memory.storage_bytes for memory in bucket_memories.values()),
            "memory_rank_bound": sum(memory.rank_bound for memory in bucket_memories.values()),
            "protect_row_count": sum(memory.protect_row_count for memory in bucket_memories.values()),
            "preservation_key_drift": sum(
                memory_drift(
                    bucket_memories[bucket],
                    protection_keys[bucket],
                    alpha=args.alpha,
                    top_q=top_q,
                )
                for bucket in ("early", "middle", "late")
            )
            / 3.0,
            "retain_distribution_kl": distribution_kl(
                base_anchor_logits, edited_anchor_logits
            ),
            "activation_diagnostics": activation,
            "fit_runtime_seconds": bucket_fit_runtime if method == "state_conditioned_preservation" else 0.0,
            "runtime_seconds": (
                bucket_fit_runtime if method == "state_conditioned_preservation" else 0.0
            )
            + time.monotonic()
            - evaluation_start,
            **metrics,
            "analysis_500_used": False,
            "final_test_used": False,
        }
        for bucket, memory in bucket_memories.items():
            torch.save(memory.cpu_payload(), method_dir / f"residual_memory_{bucket}.pt")
        write_csv(method_dir / "edited_per_prompt.csv", edited)
        write_json(
            method_dir / "target_value_diagnostics.json",
            [
                {"state_bucket": bucket, **row}
                for bucket, values in bucket_diagnostics.items()
                for row in values
            ],
        )
        write_json(method_dir / "report_summary.json", report)
        reports.append(report)
        edited_outputs[method] = edited
    write_csv(
        args.output_dir / "protection_variant_summary.csv",
        [
            {
                key: row[key]
                for key in (
                    "method",
                    "rewrite_exact",
                    "declarative_paraphrase_exact",
                    "same_subject_tfpr",
                    "near_tfpr",
                    "far_tfpr",
                    "generation_tfpr",
                    "malformed_rate",
                    "selection_score",
                    "stress_aware_aggregate",
                    "preservation_key_drift",
                    "retain_distribution_kl",
                    "memory_storage_bytes",
                )
            }
            for row in reports
        ],
    )
    by_method = {str(row["method"]): row for row in reports}
    unprotected = by_method["unprotected_temporal_residual"]
    shared = by_method["shared_soft_preservation"]
    state = by_method["state_conditioned_preservation"]
    sparse = by_method["state_conditioned_sparsification"]
    full_budget = float(
        sum(bool(row["target_new_hit"]) for row in base_rows if row["bucket"] == "same_subject")
        / max(sum(row["bucket"] == "same_subject" for row in base_rows), 1)
        + 0.03
    )
    near_budget = float(
        sum(bool(row["target_new_hit"]) for row in base_rows if row["bucket"] == "near_locality")
        / max(sum(row["bucket"] == "near_locality" for row in base_rows), 1)
        + 0.03
    )
    far_budget = float(
        sum(bool(row["target_new_hit"]) for row in base_rows if row["bucket"] == "far_locality")
        / max(sum(row["bucket"] == "far_locality" for row in base_rows), 1)
        + 0.03
    )
    state_relative_reduction = (
        (float(shared["same_subject_tfpr"]) - float(state["same_subject_tfpr"]))
        / max(float(shared["same_subject_tfpr"]), 1e-8)
    )
    state_matched = (
        float(state["rewrite_exact"]) >= float(shared["rewrite_exact"]) - 0.02
        and float(state["declarative_paraphrase_exact"])
        >= float(shared["declarative_paraphrase_exact"]) - 0.02
    )
    state_conditioning_pass = (
        state_matched and state_relative_reduction >= 0.20
    ) or (
        float(state["stress_aware_aggregate"])
        - float(shared["stress_aware_aggregate"])
        >= 0.05
    )
    relation_rescue_triggered = (
        float(state["rewrite_exact"]) >= float(unprotected["rewrite_exact"]) - 0.02
        and float(state["same_subject_tfpr"])
        <= 0.90 * float(unprotected["same_subject_tfpr"])
        and float(state["same_subject_tfpr"]) > full_budget
    )
    bootstrap = paired_tfpr_bootstrap(
        edited_outputs["state_conditioned_preservation"],
        edited_outputs["shared_soft_preservation"],
        seed=args.seed,
    )
    write_csv(
        args.output_dir / "paired_bootstrap.csv",
        [{"comparison": "state_conditioned_minus_shared_same_subject_tfpr", **bootstrap}],
    )
    positive_classes = {
        "full_editor": (
            float(state["rewrite_exact"]) >= 0.85
            and float(state["declarative_paraphrase_exact"]) >= 0.40
            and float(state["same_subject_tfpr"]) <= full_budget
            and float(state["near_tfpr"]) <= near_budget
            and float(state["far_tfpr"]) <= far_budget
            and float(state["malformed_rate"]) <= 0.05
        ),
        "state_conditioning": state_conditioning_pass,
        "sparsification_pareto": (
            float(sparse["rewrite_exact"]) >= float(state["rewrite_exact"]) - 0.02
            and float(sparse["declarative_paraphrase_exact"])
            >= float(state["declarative_paraphrase_exact"]) - 0.02
            and float(sparse["same_subject_tfpr"])
            <= float(state["same_subject_tfpr"])
        ),
    }
    integrity = {
        "all_five_mandatory_variants_complete": {row["method"] for row in reports}
        == {
            "unprotected_temporal_residual",
            "static_global_nullspace",
            "shared_soft_preservation",
            "state_conditioned_preservation",
            "state_conditioned_sparsification",
        },
        "all_required_train_only_anchor_families_present": anchor_summary[
            "all_required_families_present"
        ],
        "all_memories_finite": all(bool(row["memory_finite"]) for row in reports),
        "all_metrics_finite": all(bool(row["all_metrics_finite"]) for row in reports),
        "runtime_inputs_deployable": True,
        "evaluation_prompts_used_as_anchors": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = all(
        value
        for key, value in integrity.items()
        if key
        not in {
            "evaluation_prompts_used_as_anchors",
            "analysis_500_used",
            "final_test_used",
        }
    ) and not any(
        integrity[key]
        for key in (
            "evaluation_prompts_used_as_anchors",
            "analysis_500_used",
            "final_test_used",
        )
    )
    selected = max(
        reports,
        key=lambda row: (
            bool(row["same_subject_tfpr"] <= full_budget),
            float(row["stress_aware_aggregate"]),
            -float(row["memory_storage_bytes"]),
            str(row["method"]),
        ),
    )
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D2_state_conditioned_protection",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "selected_protection_method_for_E1": selected["method"],
        "same_subject_full_editor_budget": full_budget,
        "near_full_editor_budget": near_budget,
        "far_full_editor_budget": far_budget,
        "state_conditioned_relative_tfpr_reduction_vs_shared": state_relative_reduction,
        "state_conditioned_matched_efficacy": state_matched,
        "state_conditioning_pass": state_conditioning_pass,
        "relation_rescue_triggered": relation_rescue_triggered,
        "relation_rescue_status": (
            "required_before_E1" if relation_rescue_triggered else "not_triggered"
        ),
        "positive_classes_on_smoke": positive_classes,
        "paired_bootstrap": bootstrap,
        "basis_report": basis_report,
        "anchor_summary": anchor_summary,
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
    record_stage_cost(
        "D2_state_conditioned_protection",
        runtime_seconds=runtime,
        notes="Train-only state-conditioned protection variants on CounterFact smoke20",
    )
    record_stage(
        "D2_state_conditioned_protection",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"selected={selected['method']}; state_conditioning_pass={state_conditioning_pass}; rescue_triggered={relation_rescue_triggered}",
        next_stage=(
            "D2_relation_conditioned_rescue"
            if passed and relation_rescue_triggered
            else ("E1_smoke20" if passed else None)
        ),
    )
    if not passed:
        raise SystemExit(2)
    print(
        json.dumps(
            {
                "acceptance_pass": True,
                "selected_method": selected["method"],
                "state_conditioning_pass": state_conditioning_pass,
                "relation_rescue_triggered": relation_rescue_triggered,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
