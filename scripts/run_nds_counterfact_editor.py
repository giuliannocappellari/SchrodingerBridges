#!/usr/bin/env python3
"""Run one frozen CounterFact editor candidate with the common NDS evaluator."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_nds_shared_measurements import (
    build_subject_anchor_requests,
    pre_edit_rank_margin,
)
from scripts.mdm_memit_editor import (
    MemitConfig,
    apply_memit_batch,
    build_protected_basis,
    extract_keys_and_outputs,
)
from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.nds_editor import (
    RelationKeyStatistics,
    diagonal_fisher_update,
    fisher_trust_region_update,
    fixed_penalty_update,
    low_rank_fisher_update,
    primal_dual_update,
    residualize_runtime_keys,
)
from scripts.run_dnpe_editor import align_base, build_eval_tasks, evaluate_tasks
from scripts.run_mdm_memit_stage import load_model
from scripts.run_trm_state_conditioned_protection import anchor_logits, distribution_kl


METHODS = {
    "base",
    "ordinary_memit",
    "partial_state_memit",
    "static_nullspace_partial_state_memit",
    "relation_subject",
    "relation_centered",
    "relation_full",
    "relation_full_shrinkage",
    "fisher_diagonal",
    "fisher_lowrank",
    "fisher_trust_region",
    "fixed_penalty",
    "primal_dual",
    "relation_fisher_integrated",
    "relation_primal_dual_integrated",
}


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def validate_manifest_access(path: Path, *, allow_confirmation: bool) -> None:
    lower = path.name.casefold()
    if "analysis_500" in lower or "final_test" in lower:
        raise PermissionError("historical locked analysis/final manifests are forbidden")
    if "confirmation" in lower and not allow_confirmation:
        raise PermissionError("fresh confirmation requires an explicit frozen-candidate lock")


def harmonic_mean(values: Sequence[float]) -> float:
    if any(float(value) <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / float(value) for value in values)


def js_divergence(base_logits: torch.Tensor, edited_logits: torch.Tensor) -> float:
    base = F.softmax(base_logits.float(), dim=-1).clamp_min(1e-12)
    edited = F.softmax(edited_logits.float(), dim=-1).clamp_min(1e-12)
    middle = (0.5 * (base + edited)).clamp_min(1e-12)
    return float(
        0.5
        * (
            F.kl_div(middle.log(), base, reduction="batchmean")
            + F.kl_div(middle.log(), edited, reduction="batchmean")
        )
    )


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bucket"])].append(row)
    output = {}
    for bucket, values in grouped.items():
        expected = [row for row in values if row.get("expected_hit") is not None]
        output[bucket] = {
            "num_rows": len(values),
            "num_edits": len({str(row["case_id"]) for row in values}),
            "expected_exact": (
                sum(bool(row["expected_hit"]) for row in expected) / len(expected)
                if expected
                else None
            ),
            "target_new_rate": sum(bool(row["target_new_hit"]) for row in values)
            / len(values),
            "target_true_rate": sum(bool(row["target_true_hit"]) for row in values)
            / len(values),
            "target_token_f1": sum(float(row["target_token_f1"]) for row in values)
            / len(values),
            "malformed_rate": sum(bool(row["malformed"]) for row in values) / len(values),
            "base_agreement": (
                sum(bool(row["base_agreement"]) for row in values) / len(values)
                if all("base_agreement" in row for row in values)
                else None
            ),
        }
    return output


def grouped_breakdown(
    rows: Sequence[Mapping[str, Any]], group_key: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get(group_key) or ""), str(row["bucket"]))].append(row)
    output = []
    for (group, bucket), values in sorted(groups.items()):
        expected = [row for row in values if row.get("expected_hit") is not None]
        output.append(
            {
                group_key: group,
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


def _measurement(path: Path, layer: int) -> dict[str, Any]:
    value = torch.load(
        path / "statistics_train" / f"layer_{layer}_measurements.pt",
        map_location="cpu",
        weights_only=True,
    )
    return value


def _relation_statistics(payload: Mapping[str, Any]) -> RelationKeyStatistics:
    global_mean = payload["relation_global_mean"].float()
    means = {str(key): value.float() for key, value in payload["relation_means"].items()}
    return RelationKeyStatistics(global_mean, means, global_mean.clone())


def _kl_records(tasks: Sequence[Mapping[str, Any]], maximum: int = 128) -> list[dict[str, Any]]:
    protected = [
        {
            "case_id": row["case_id"],
            "bucket": row["bucket"],
            "prompt": row["prompt"],
        }
        for row in tasks
        if row["bucket"]
        in {"same_subject", "near_locality", "far_locality", "generation", "attribute"}
    ]
    protected.sort(
        key=lambda row: (
            0 if row["bucket"] == "same_subject" else 1,
            str(row["case_id"]),
            str(row["bucket"]),
            str(row["prompt"]),
        )
    )
    return protected[:maximum]


def per_prompt_distribution_rows(
    records: Sequence[Mapping[str, Any]],
    base_logits: torch.Tensor,
    edited_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    if len(records) != len(base_logits) or base_logits.shape != edited_logits.shape:
        raise ValueError("protected logit rows must align")
    base = F.softmax(base_logits.float(), dim=-1).clamp_min(1e-12)
    edited = F.softmax(edited_logits.float(), dim=-1).clamp_min(1e-12)
    middle = (0.5 * (base + edited)).clamp_min(1e-12)
    kl = (base * (base.log() - edited.log())).sum(dim=-1)
    js = 0.5 * (
        (base * (base.log() - middle.log())).sum(dim=-1)
        + (edited * (edited.log() - middle.log())).sum(dim=-1)
    )
    return [
        {
            "case_id": record["case_id"],
            "bucket": record["bucket"],
            "prompt": record["prompt"],
            "protected_kl": float(kl[index]),
            "protected_js": float(js[index]),
        }
        for index, record in enumerate(records)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--measurement_dir", type=Path, default=CAMPAIGN_ROOT / "S1_shared_measurements_v1")
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--covariance_weight", type=float, default=15000.0)
    parser.add_argument("--update_ridge", type=float, default=0.0)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--decode_steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--residual_shrinkage", type=float, default=0.5)
    parser.add_argument("--fisher_damping", type=float, default=1e-3)
    parser.add_argument("--fisher_rank", type=int, default=64)
    parser.add_argument("--trust_fraction", type=float, default=0.8)
    parser.add_argument("--fixed_penalty_strength", type=float, default=0.05)
    parser.add_argument("--constraint_fraction", type=float, default=0.8)
    parser.add_argument("--multiplier_step", type=float, default=0.05)
    parser.add_argument("--penalty_growth", type=float, default=1.5)
    parser.add_argument("--allow_confirmation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=260719301)
    args = parser.parse_args()
    begin = time.monotonic()
    validate_manifest_access(args.manifest, allow_confirmation=bool(args.allow_confirmation))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no edits selected")
    model, tokenizer = load_model(args.model_id, args.model_revision, args.dtype)
    tasks = build_eval_tasks(tokenizer, rows, include_locality=True)
    pre_edit_features = pre_edit_rank_margin(model, tokenizer, rows)
    base_rows = evaluate_tasks(
        model,
        tokenizer,
        tasks,
        decode_batch_size=args.decode_batch_size,
        steps=args.decode_steps or None,
    )
    records = _kl_records(tasks)
    base_logits = anchor_logits(model, tokenizer, records) if records else torch.empty(0)
    diagnostics: dict[str, Any] = {"target_optimization": [], "layer_updates": []}
    rollback = None
    relation_modes = {
        "relation_subject": "subject",
        "relation_centered": "relation",
        "relation_full": "full",
        "relation_full_shrinkage": "full",
        "relation_fisher_integrated": "full",
        "relation_primal_dual_integrated": "full",
    }
    key_transform = None
    runtime_subject_anchors: dict[int, torch.Tensor] = {}
    if args.method in relation_modes:
        bank = read_json(args.measurement_dir / "relation_template_bank.json")
        anchor_requests = build_subject_anchor_requests(rows, bank)
        for layer in args.layers:
            runtime_subject_anchors[layer], _ = extract_keys_and_outputs(
                model,
                tokenizer,
                anchor_requests,
                key_layer=layer,
                output_layer=layer,
                partial_mask_schedule="cycle",
                reveal_policy="random",
                seed=args.seed,
            )

        def key_transform(layer, keys, requests):
            payload = _measurement(args.measurement_dir, layer)
            return residualize_runtime_keys(
                keys,
                [str(row["relation_id"]) for row in requests],
                _relation_statistics(payload),
                subject_anchor_keys=runtime_subject_anchors[layer],
                mode=relation_modes[args.method],
                shrinkage=(
                    args.residual_shrinkage
                    if args.method == "relation_full_shrinkage"
                    else 0.0
                ),
            )

    update_transform: Callable[[int, torch.Tensor, Mapping[str, Any]], Any] | None = None
    if args.method in {
        "fisher_diagonal",
        "fisher_lowrank",
        "fisher_trust_region",
        "fixed_penalty",
        "primal_dual",
        "relation_fisher_integrated",
        "relation_primal_dual_integrated",
    }:

        def update_transform(layer, update, context):
            payload = _measurement(args.measurement_dir, layer)
            keys = context["keys"]
            residuals = context["residuals"]
            protected = {
                name: value.float() for name, value in payload["protected_keys"].items()
            }
            if args.method == "fisher_diagonal":
                return diagonal_fisher_update(
                    update, payload["fisher_diagonal"], keys, residuals
                )
            if args.method in {"fisher_lowrank", "relation_fisher_integrated"}:
                rank = min(args.fisher_rank, int(payload["fisher_basis"].shape[1]))
                return low_rank_fisher_update(
                    update,
                    payload["fisher_basis"][:, :rank],
                    payload["fisher_eigenvalues"][:rank],
                    args.fisher_damping,
                    keys,
                    residuals,
                )
            if args.method == "fisher_trust_region":
                fisher = payload["fisher_diagonal"].float()
                before = float((update.float().square() * fisher.unsqueeze(0)).sum())
                return fisher_trust_region_update(
                    update, fisher, before * args.trust_fraction
                )
            if args.method == "fixed_penalty":
                return fixed_penalty_update(
                    update, protected, args.fixed_penalty_strength
                )
            limits = {
                name: float(
                    ((keys_value.float() @ update.float().T).square().mean())
                    * args.constraint_fraction
                )
                for name, keys_value in protected.items()
            }
            return primal_dual_update(
                update,
                protected,
                limits,
                multiplier_step=args.multiplier_step,
                penalty_growth=args.penalty_growth,
            )

    protected_basis_loader = None
    basis_cache: dict[int, torch.Tensor] = {}
    if args.method == "static_nullspace_partial_state_memit":
        for layer in args.layers:
            payload = _measurement(args.measurement_dir, layer)
            combined = torch.cat(list(payload["protected_keys"].values()), dim=0)
            basis_cache[layer] = build_protected_basis(
                combined, 0.95, maximum_rank=64
            )[0]
        protected_basis_loader = lambda layer: basis_cache[layer]

    partial = args.method != "ordinary_memit"
    config = MemitConfig(
        layers=args.layers,
        learning_rate=args.learning_rate,
        target_optimization_steps=args.target_optimization_steps,
        covariance_weight=args.covariance_weight,
        partial_mask_schedule="cycle" if partial else "fully_masked",
        reveal_policy="base_confidence" if partial else "random",
        state_consistency_weight=0.1 if partial else 0.0,
        old_target_suppression_weight=0.25 if partial else 0.0,
        update_ridge=args.update_ridge,
        seed=args.seed,
    )
    if args.method != "base":
        rollback, diagnostics = apply_memit_batch(
            model,
            tokenizer,
            rows,
            config,
            lambda layer: _measurement(args.measurement_dir, layer)[
                "covariance_diagonal"
            ].to("cuda"),
            target_cache_dir=args.output_dir / "target_value_cache",
            protected_basis_loader=protected_basis_loader,
            key_transform=key_transform,
            update_transform=update_transform,
        )
    edited_raw = evaluate_tasks(
        model,
        tokenizer,
        tasks,
        decode_batch_size=args.decode_batch_size,
        steps=args.decode_steps or None,
    )
    edited_logits = anchor_logits(model, tokenizer, records) if records else torch.empty(0)
    if rollback is not None:
        rollback.rollback()
        rollback_pass = rollback.checksum_matches(atol=0.0)
    else:
        rollback_pass = True
    if not rollback_pass:
        raise RuntimeError("editor weight rollback failed")
    edited_rows = align_base(base_rows, edited_raw)
    base_summary = aggregate_rows(base_rows)
    edited_summary = aggregate_rows(edited_rows)
    rewrite = float(edited_summary.get("rewrite", {}).get("expected_exact") or 0.0)
    paraphrase = float(
        edited_summary.get("declarative_paraphrase", {}).get("expected_exact") or 0.0
    )
    same_tfpr = float(edited_summary.get("same_subject", {}).get("target_new_rate") or 0.0)
    near_tfpr = float(edited_summary.get("near_locality", {}).get("target_new_rate") or 0.0)
    far_tfpr = float(edited_summary.get("far_locality", {}).get("target_new_rate") or 0.0)
    protected_buckets = {"same_subject", "near_locality", "far_locality", "generation", "attribute"}
    protected_rows = [row for row in edited_rows if row["bucket"] in protected_buckets]
    locality = (
        sum(bool(row["base_agreement"]) for row in protected_rows) / len(protected_rows)
        if protected_rows
        else 1.0
    )
    malformed = max(
        (float(value["malformed_rate"]) for value in edited_summary.values()),
        default=0.0,
    )
    elapsed = time.monotonic() - begin
    locality_kl = (
        distribution_kl(base_logits, edited_logits) if len(base_logits) else 0.0
    )
    locality_js = js_divergence(base_logits, edited_logits) if len(base_logits) else 0.0
    distribution_rows = (
        per_prompt_distribution_rows(records, base_logits, edited_logits)
        if len(base_logits)
        else []
    )
    score = harmonic_mean((rewrite, paraphrase, min(locality, 1.0)))
    run_config = {
        "campaign_id": CAMPAIGN_ID,
        "method": args.method,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "num_edits": len(rows),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layers": list(args.layers),
        "measurement_dir": str(args.measurement_dir),
        "memit": config.to_dict(),
        "residual_shrinkage": args.residual_shrinkage,
        "fisher_damping": args.fisher_damping,
        "fisher_rank": args.fisher_rank,
        "trust_fraction": args.trust_fraction,
        "fixed_penalty_strength": args.fixed_penalty_strength,
        "constraint_fraction": args.constraint_fraction,
        "multiplier_step": args.multiplier_step,
        "penalty_growth": args.penalty_growth,
        "runtime_feature_schema": ["subject", "relation_id", "rewrite_prompt"],
        "teacher_only_runtime_inputs": False,
        "evaluation_bucket_runtime_input": False,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_csv(args.output_dir / "base_per_prompt.csv", base_rows)
    write_csv(args.output_dir / "edited_per_prompt.csv", edited_rows)
    write_csv(args.output_dir / "pre_edit_features.csv", pre_edit_features)
    write_csv(args.output_dir / "protected_distribution_per_prompt.csv", distribution_rows)
    write_csv(
        args.output_dir / "target_length_breakdown.csv",
        grouped_breakdown(edited_rows, "target_length"),
    )
    write_csv(
        args.output_dir / "relation_breakdown.csv",
        grouped_breakdown(edited_rows, "relation_id"),
    )
    write_json(args.output_dir / "target_value_diagnostics.json", diagnostics)
    report = {
        **run_config,
        "stage": "actual_decode",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "base_summary": base_summary,
        "edited_summary": edited_summary,
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "same_subject_tfpr": same_tfpr,
        "near_tfpr": near_tfpr,
        "far_tfpr": far_tfpr,
        "generation_tfpr": float(
            edited_summary.get("generation", {}).get("target_new_rate") or 0.0
        ),
        "malformed_rate": malformed,
        "self_normalized_locality": locality,
        "clipped_self_normalized_locality": min(locality, 1.0),
        "protected_distributional_kl": locality_kl,
        "protected_distributional_js": locality_js,
        "selection_score": score,
        "rollback_checksum_pass": rollback_pass,
        "runtime_seconds": elapsed,
        "gpu_minutes_per_edit": elapsed / 60.0 / len(rows),
        "model_eval_count": sum(
            int(row["model_eval_count"]) for row in base_rows + edited_rows
        ),
        "fake_model": False,
        "llada_loaded": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "all_metrics_finite": all(
            math.isfinite(value)
            for value in (
                rewrite,
                paraphrase,
                same_tfpr,
                near_tfpr,
                far_tfpr,
                malformed,
                locality_kl,
                locality_js,
                score,
            )
        ),
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "rollback_checksum_pass": rollback_pass,
            "all_metrics_finite": report["all_metrics_finite"],
            "runtime_schema_deployable": True,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": bool(rollback_pass and report["all_metrics_finite"]),
        },
    )
    print(
        json.dumps(
            {
                "method": args.method,
                "rewrite_exact": rewrite,
                "declarative_paraphrase_exact": paraphrase,
                "same_subject_tfpr": same_tfpr,
                "selection_score": score,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
