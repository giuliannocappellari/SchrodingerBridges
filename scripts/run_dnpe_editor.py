#!/usr/bin/env python3
"""Run one actual-decode DNPE editor configuration on an allowed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    git_commit,
    now_utc,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mdm_memit_editor import (
    MemitConfig,
    apply_memit_batch,
    contextual_target_ids,
    denoise_answer_spans_batch,
    normalized_hit,
)
from scripts.run_mdm_memit_stage import load_model


HISTORICAL_COVARIANCE = (
    ROOT
    / "runs"
    / "masked_diffusion_memit_sb_positive_result_v1"
    / "covariance_cache_v1"
)


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return layers


def _forbid_locked_manifest(path: Path) -> None:
    name = path.name.lower()
    if "analysis_500" in name and __import__("os").environ.get("DEV_METHOD_LOCKED") != "1":
        raise PermissionError("analysis_500 requires a validated dev method lock")
    if "final_test_500" in name and __import__("os").environ.get("FINAL_METHOD_LOCKED") != "1":
        raise PermissionError("final_test_500 requires a validated final method lock")
    if "_locked_" in name and __import__("os").environ.get("DNPE_KAMEL_LOCKED") != "1":
        raise PermissionError("KAMEL locked confirmation manifest is not open")


def _answer_length(tokenizer: Any, prompt: str, *targets: str) -> int:
    return max(1, *(len(contextual_target_ids(tokenizer, prompt, target)) for target in targets if target))


def build_eval_tasks(
    tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, include_locality: bool
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in rows:
        common = {
            "case_id": row["case_id"],
            "target_new": str(row["target_new"]),
            "target_true": str(row["target_true"]),
            "target_length": int(row["target_length"]),
            "relation_id": str(row["relation_id"]),
        }
        prompts = [("rewrite", str(row["rewrite_prompt"]), str(row["target_new"]))]
        prompts += [
            ("declarative_paraphrase", str(prompt), str(row["target_new"]))
            for prompt in list(row.get("paraphrase_prompts") or [])
        ]
        if include_locality:
            near_cases = list(row.get("near_locality_cases") or [])[:10]
            if near_cases:
                prompts += [
                    ("near_locality", str(case["prompt"]), str(case["target"]))
                    for case in near_cases
                ]
            else:
                prompts += [
                    ("near_locality", str(prompt), str(row["target_true"]))
                    for prompt in list(
                        row.get("near_locality_prompts")
                        or row.get("neighborhood_prompts")
                        or []
                    )[:10]
                ]
            prompts += [
                ("far_locality", str(case["prompt"]), str(case["target"]))
                for case in list(row.get("far_locality_cases") or [])
            ]
            prompts += [
                ("same_subject", str(prompt), "")
                for prompt in list(row.get("same_subject_prompts") or [])
            ]
            prompts += [
                ("generation", str(prompt), "")
                for prompt in list(row.get("generation_prompts") or [])[:3]
            ]
            prompts += [
                ("attribute", str(prompt), "")
                for prompt in list(row.get("attribute_prompts") or [])[:3]
            ]
        for bucket, prompt, expected in prompts:
            tasks.append(
                {
                    **common,
                    "bucket": bucket,
                    "prompt": prompt,
                    "expected": expected,
                    "answer_length": _answer_length(
                        tokenizer,
                        prompt,
                        expected or str(row["target_new"]),
                        str(row["target_true"]),
                    ),
                }
            )
    return tasks


def evaluate_tasks(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[Mapping[str, Any]],
    *,
    decode_batch_size: int,
    steps: int | None,
) -> list[dict[str, Any]]:
    decoded = denoise_answer_spans_batch(
        model,
        tokenizer,
        [str(task["prompt"]) for task in tasks],
        [int(task["answer_length"]) for task in tasks],
        steps=steps,
        batch_size=decode_batch_size,
    )
    results = []
    for task, result in zip(tasks, decoded):
        target_ids = contextual_target_ids(tokenizer, str(task["prompt"]), str(task["target_new"]))
        predicted = list(map(int, result["output_token_ids"]))
        common = set(predicted) & set(target_ids)
        precision = len(common) / max(len(predicted), 1)
        recall = len(common) / max(len(target_ids), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        expected = str(task["expected"])
        results.append(
            {
                **dict(task),
                "prompt_fingerprint": hashlib.sha256(
                    f"{task['case_id']}::{task['bucket']}::{task['prompt']}".encode()
                ).hexdigest(),
                "output_text": result["output_text"],
                "output_token_ids": json.dumps(predicted),
                "target_new_hit": normalized_hit(result["output_text"], str(task["target_new"])),
                "target_true_hit": normalized_hit(result["output_text"], str(task["target_true"])),
                "expected_hit": normalized_hit(result["output_text"], expected) if expected else None,
                "target_token_f1": f1,
                "malformed": bool(result["malformed"]),
                "model_eval_count": int(result["model_eval_count"]),
            }
        )
    return results


def align_base(
    base: Sequence[Mapping[str, Any]], edited: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    index = {str(row["prompt_fingerprint"]): row for row in base}
    enriched = []
    for source in edited:
        row = dict(source)
        baseline = index.get(str(row["prompt_fingerprint"]))
        if baseline is None:
            raise RuntimeError(f"Missing base alignment for {row['prompt_fingerprint']}")
        row["base_output_text"] = baseline["output_text"]
        row["base_agreement"] = " ".join(str(row["output_text"]).casefold().split()) == " ".join(
            str(baseline["output_text"]).casefold().split()
        )
        enriched.append(row)
    return enriched


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["bucket"])].append(row)
    summary = {}
    for bucket, values in groups.items():
        expected = [row for row in values if row.get("expected_hit") is not None]
        summary[bucket] = {
            "num_rows": len(values),
            "num_edits": len({str(row["case_id"]) for row in values}),
            "expected_exact": sum(bool(row["expected_hit"]) for row in expected) / max(len(expected), 1),
            "target_new_tfpr_or_exact": sum(bool(row["target_new_hit"]) for row in values) / len(values),
            "target_true_exact": sum(bool(row["target_true_hit"]) for row in values) / len(values),
            "target_token_f1": sum(float(row["target_token_f1"]) for row in values) / len(values),
            "malformed_rate": sum(bool(row["malformed"]) for row in values) / len(values),
            "base_agreement": (
                sum(bool(row.get("base_agreement")) for row in values) / len(values)
                if all("base_agreement" in row for row in values)
                else None
            ),
            "model_eval_count": sum(int(row["model_eval_count"]) for row in values),
        }
    return summary


def _load_covariance(path: Path, layer: int):
    import torch

    value = torch.load(path / f"layer_{layer}_covariance.pt", map_location="cpu", weights_only=True)
    return value.to("cuda")


def _basis_loader(path: Path | None, variance: float):
    if path is None:
        return None

    def load(layer: int):
        import torch

        basis_path = path / f"layer_{layer}_variance_{variance:.2f}_basis.pt"
        if not basis_path.exists():
            raise FileNotFoundError(basis_path)
        return torch.load(basis_path, map_location="cpu", weights_only=True)["basis"].to("cuda")

    return load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--campaign_id", default=CAMPAIGN_ID)
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--covariance_dir", type=Path, default=HISTORICAL_COVARIANCE)
    parser.add_argument("--protected_basis_dir", type=Path)
    parser.add_argument("--protected_variance", type=float, default=0.95)
    parser.add_argument("--partial_mask_schedule", default="fully_masked")
    parser.add_argument("--reveal_policy", default="random")
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--state_consistency_weight", type=float, default=0.0)
    parser.add_argument("--old_target_suppression_weight", type=float, default=0.0)
    parser.add_argument("--kl_factor", type=float, default=0.0625)
    parser.add_argument("--lambda_path", type=float, default=0.0)
    parser.add_argument("--lambda_identity", type=float, default=0.0)
    parser.add_argument("--covariance_weight", type=float, default=15000.0)
    parser.add_argument("--update_ridge", type=float, default=0.0)
    parser.add_argument("--update_scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include_locality", type=int, choices=(0, 1), default=1)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--decode_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260717101)
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    _forbid_locked_manifest(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows selected")
    model, tokenizer = load_model(args.model_id, args.model_revision, args.dtype)
    tasks = build_eval_tasks(tokenizer, rows, include_locality=bool(args.include_locality))
    base_rows = evaluate_tasks(
        model,
        tokenizer,
        tasks,
        decode_batch_size=args.decode_batch_size,
        steps=args.decode_steps or None,
    )
    config = MemitConfig(
        layers=args.layers,
        learning_rate=args.learning_rate,
        target_optimization_steps=args.target_optimization_steps,
        kl_factor=args.kl_factor,
        covariance_weight=args.covariance_weight / max(args.update_scale, 1e-8),
        partial_mask_schedule=args.partial_mask_schedule,
        reveal_policy=args.reveal_policy,
        lambda_path=args.lambda_path,
        lambda_identity=args.lambda_identity,
        state_consistency_weight=args.state_consistency_weight,
        old_target_suppression_weight=args.old_target_suppression_weight,
        update_ridge=args.update_ridge,
        seed=args.seed,
    )
    rollback, diagnostics = apply_memit_batch(
        model,
        tokenizer,
        rows,
        config,
        lambda layer: _load_covariance(args.covariance_dir, layer),
        target_cache_dir=args.output_dir / "target_value_cache",
        protected_basis_loader=_basis_loader(args.protected_basis_dir, args.protected_variance),
    )
    edited_rows = evaluate_tasks(
        model,
        tokenizer,
        tasks,
        decode_batch_size=args.decode_batch_size,
        steps=args.decode_steps or None,
    )
    rollback.rollback()
    rollback_pass = rollback.checksum_matches(atol=0.0)
    if not rollback_pass:
        raise RuntimeError("Weight rollback checksum failed")
    edited_rows = align_base(base_rows, edited_rows)
    base_summary = aggregate(base_rows)
    edited_summary = aggregate(edited_rows)
    rewrite = edited_summary.get("rewrite", {}).get("expected_exact", 0.0)
    paraphrase = edited_summary.get("declarative_paraphrase", {}).get("expected_exact", 0.0)
    malformed = max((item["malformed_rate"] for item in edited_summary.values()), default=0.0)
    pre_edit_target = base_summary.get("rewrite", {}).get("target_new_tfpr_or_exact", 0.0)
    elapsed = time.monotonic() - begin
    write_csv(args.output_dir / "base_per_prompt.csv", base_rows)
    write_csv(args.output_dir / "edited_per_prompt.csv", edited_rows)
    write_json(args.output_dir / "target_value_diagnostics.json", diagnostics)
    run_config = {
        "campaign_id": args.campaign_id,
        "method": args.method,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "num_edits": len(rows),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "quantized": False,
        "layers": list(args.layers),
        "memit": config.to_dict(),
        "protected_basis_dir": str(args.protected_basis_dir or ""),
        "protected_variance": args.protected_variance,
        "covariance_dir": str(args.covariance_dir),
        "covariance_source": "training_only_wikipedia_uncentered_mlp_key_covariance",
        "analysis_500_used": "analysis_500" in args.manifest.name,
        "final_test_used": "final_test_500" in args.manifest.name,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    report = {
        **run_config,
        "stage": "actual_decode",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "base_summary": base_summary,
        "edited_summary": edited_summary,
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "pre_edit_target_new_rewrite_exact": pre_edit_target,
        "target_token_f1": edited_summary.get("rewrite", {}).get("target_token_f1", 0.0),
        "old_target_suppression": 1.0 - edited_summary.get("rewrite", {}).get("target_true_exact", 0.0),
        "same_subject_tfpr": edited_summary.get("same_subject", {}).get("target_new_tfpr_or_exact", 0.0),
        "near_tfpr": edited_summary.get("near_locality", {}).get("target_new_tfpr_or_exact", 0.0),
        "far_tfpr": edited_summary.get("far_locality", {}).get("target_new_tfpr_or_exact", 0.0),
        "generation_tfpr": edited_summary.get("generation", {}).get("target_new_tfpr_or_exact", 0.0),
        "malformed_rate": malformed,
        "rollback_checksum_pass": rollback_pass,
        "runtime_seconds": elapsed,
        "gpu_minutes_per_edit": elapsed / 60.0 / len(rows),
        "model_eval_count": sum(item["model_eval_count"] for item in edited_summary.values()),
        "fake_model": False,
        "llada_loaded": True,
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": __import__("torch").version.cuda,
            "gpu": __import__("torch").cuda.get_device_name(0),
        },
        "acceptance_pass": bool(
            rewrite >= 0.75
            and paraphrase >= 0.40
            and pre_edit_target <= 0.10
            and malformed <= 0.05
        ),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "all_numeric_finite": all(
                math.isfinite(float(report[key]))
                for key in (
                    "rewrite_exact",
                    "declarative_paraphrase_exact",
                    "same_subject_tfpr",
                    "near_tfpr",
                    "far_tfpr",
                    "malformed_rate",
                )
            ),
            "rollback_checksum_pass": rollback_pass,
            "locked_split_access_legal": not report["analysis_500_used"] and not report["final_test_used"],
        },
    )
    print(json.dumps({key: report[key] for key in ("method", "rewrite_exact", "declarative_paraphrase_exact", "same_subject_tfpr", "acceptance_pass")}, sort_keys=True))


if __name__ == "__main__":
    main()
