#!/usr/bin/env python3
"""Run one bounded MDM-MEMIT reproduction stage on RunPod."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    MODEL_ID,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mdm_memit_editor import (
    MemitConfig,
    apply_memit_batch,
    contextual_target_ids,
    denoise_answer_span,
    infer_mask_id,
    normalized_hit,
    render_masked_input,
)


M1_ROOT = CAMPAIGN_ROOT / "M1_mdm_memit_reproduction_v1"


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    if layers != tuple(range(layers[0], layers[-1] + 1)):
        raise argparse.ArgumentTypeError("MEMIT layers must be contiguous")
    return layers


def load_model(model_id: str, dtype_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == infer_mask_id(model):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_covariance(cache_dir: Path, layer: int):
    import torch

    path = cache_dir / f"layer_{layer}_covariance.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    covariance = torch.load(path, map_location="cpu", weights_only=True)
    return covariance.to("cuda")


def target_probability(model: Any, tokenizer: Any, row: Mapping[str, Any]) -> float:
    import torch
    import torch.nn.functional as F

    prompt = str(row["rewrite_prompt"])
    target_ids = list(row.get("target_new_token_ids") or contextual_target_ids(tokenizer, prompt, row["target_new"]))
    rendered = render_masked_input(tokenizer, prompt, target_ids, infer_mask_id(model))
    tensor = torch.tensor([rendered["input_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids=tensor).logits[0].float()
    log_probability = 0.0
    for position, token_id in zip(rendered["answer_positions"], target_ids):
        log_probability += float(F.log_softmax(logits[position], dim=-1)[int(token_id)])
    return math.exp(log_probability / len(target_ids))


def answer_length(tokenizer: Any, prompt: str, row: Mapping[str, Any]) -> int:
    new_len = len(contextual_target_ids(tokenizer, prompt, str(row["target_new"])))
    true_len = len(contextual_target_ids(tokenizer, prompt, str(row["target_true"])))
    return max(1, new_len, true_len)


def evaluate_prompt(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    prompt: str,
    bucket: str,
    expected: str,
    *,
    fixed_length: int | None = None,
    fixed_steps: int | None = None,
) -> dict[str, Any]:
    length = fixed_length or answer_length(tokenizer, prompt, row)
    decoded = denoise_answer_span(model, tokenizer, prompt, length, steps=fixed_steps)
    return {
        "case_id": row["case_id"],
        "bucket": bucket,
        "prompt": prompt,
        "target_new": row["target_new"],
        "target_true": row["target_true"],
        "expected_target": expected,
        "output_text": decoded["output_text"],
        "target_new_hit": normalized_hit(decoded["output_text"], row["target_new"]),
        "target_true_hit": normalized_hit(decoded["output_text"], row["target_true"]),
        "expected_hit": normalized_hit(decoded["output_text"], expected),
        "malformed": decoded["malformed"],
        "model_eval_count": decoded["model_eval_count"],
        "target_length": row.get("target_length"),
        "relation_id": row.get("relation_id"),
    }


def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    include_locality: bool,
    fixed_length: int | None = None,
    fixed_steps: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            evaluate_prompt(
                model,
                tokenizer,
                row,
                str(row["rewrite_prompt"]),
                "rewrite",
                str(row["target_new"]),
                fixed_length=fixed_length,
                fixed_steps=fixed_steps,
            )
        )
        for prompt in list(row.get("paraphrase_prompts") or []):
            results.append(
                evaluate_prompt(
                    model,
                    tokenizer,
                    row,
                    str(prompt),
                    "paraphrase",
                    str(row["target_new"]),
                    fixed_length=fixed_length,
                    fixed_steps=fixed_steps,
                )
            )
        if not include_locality:
            continue
        for prompt in list(row.get("neighborhood_prompts") or [])[:10]:
            results.append(
                evaluate_prompt(
                    model,
                    tokenizer,
                    row,
                    str(prompt),
                    "classic_specificity",
                    str(row["target_true"]),
                    fixed_length=fixed_length,
                    fixed_steps=fixed_steps,
                )
            )
        same_subject_prompts = list(row.get("generation_prompts") or [])[:2] + list(row.get("attribute_prompts") or [])[:2]
        for prompt in same_subject_prompts:
            if str(row["subject"]).casefold() not in str(prompt).casefold():
                continue
            results.append(
                evaluate_prompt(
                    model,
                    tokenizer,
                    row,
                    str(prompt),
                    "same_subject_stress",
                    str(row["target_true"]),
                    fixed_length=fixed_length,
                    fixed_steps=fixed_steps,
                )
            )
    return results


def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["bucket"])].append(row)
    buckets: dict[str, Any] = {}
    for bucket, values in grouped.items():
        buckets[bucket] = {
            "num_rows": len(values),
            "num_edits": len({row["case_id"] for row in values}),
            "expected_exact": sum(bool(row["expected_hit"]) for row in values) / len(values),
            "target_new_exact": sum(bool(row["target_new_hit"]) for row in values) / len(values),
            "target_true_exact": sum(bool(row["target_true_hit"]) for row in values) / len(values),
            "malformed_rate": sum(bool(row["malformed"]) for row in values) / len(values),
            "model_eval_count": sum(int(row["model_eval_count"]) for row in values),
        }
    return buckets


def write_result_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["one_edit", "smoke20", "batch"], required=True)
    parser.add_argument("--manifest", type=Path, default=PROTOCOL_ROOT / "cf_memit_smoke_20.jsonl")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--covariance_dir", type=Path, default=CAMPAIGN_ROOT / "covariance_cache_v1")
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--clamp_norm_factor", type=float, default=0.75)
    parser.add_argument("--kl_factor", type=float, default=0.0625)
    parser.add_argument("--weight_decay", type=float, default=0.5)
    parser.add_argument("--covariance_weight", type=float, default=15000.0)
    parser.add_argument("--partial_mask_schedule", default="fully_masked")
    parser.add_argument("--reveal_policy", default="random")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include_locality", type=int, choices=[0, 1], default=1)
    parser.add_argument("--fixed_length", type=int, default=0)
    parser.add_argument("--fixed_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260603924)
    args = parser.parse_args()
    started_at = now_utc()
    start = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = read_jsonl(args.manifest)
    if args.stage == "one_edit":
        rows = rows[:1]
    elif args.stage == "smoke20":
        rows = rows[:20]
    elif args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No edit rows selected")

    import torch
    import transformers

    model, tokenizer = load_model(args.model_id, args.dtype)
    config = MemitConfig(
        layers=args.layers,
        learning_rate=args.learning_rate,
        target_optimization_steps=args.target_optimization_steps,
        clamp_norm_factor=args.clamp_norm_factor,
        kl_factor=args.kl_factor,
        weight_decay=args.weight_decay,
        covariance_weight=args.covariance_weight,
        partial_mask_schedule=args.partial_mask_schedule,
        reveal_policy=args.reveal_policy,
        seed=args.seed,
    )
    before_probability = target_probability(model, tokenizer, rows[0])
    base_results = evaluate_rows(
        model,
        tokenizer,
        rows,
        include_locality=bool(args.include_locality),
        fixed_length=args.fixed_length or None,
        fixed_steps=args.fixed_steps or None,
    )
    target_cache = args.output_dir / "target_value_cache"
    rollback, diagnostics = apply_memit_batch(
        model,
        tokenizer,
        rows,
        config,
        lambda layer: load_covariance(args.covariance_dir, layer),
        target_cache_dir=target_cache,
    )
    post_probability = target_probability(model, tokenizer, rows[0])
    edited_results = evaluate_rows(
        model,
        tokenizer,
        rows,
        include_locality=bool(args.include_locality),
        fixed_length=args.fixed_length or None,
        fixed_steps=args.fixed_steps or None,
    )
    rollback.rollback()
    rollback_pass = rollback.checksum_matches(atol=0.0)
    if not rollback_pass:
        raise RuntimeError("MEMIT rollback checksum failed")

    write_result_csv(args.output_dir / "base_per_prompt.csv", base_results)
    write_result_csv(args.output_dir / "edited_per_prompt.csv", edited_results)
    write_json(args.output_dir / "target_value_diagnostics.json", diagnostics)
    base_summary = aggregate(base_results)
    edited_summary = aggregate(edited_results)
    rewrite = edited_summary.get("rewrite", {}).get("expected_exact", 0.0)
    paraphrase = edited_summary.get("paraphrase", {}).get("expected_exact", 0.0)
    malformed = max(
        (bucket.get("malformed_rate", 0.0) for bucket in edited_summary.values()),
        default=0.0,
    )
    first_history = diagnostics["target_optimization"][0]["history"]
    target_loss_decreased = bool(
        first_history and first_history[-1]["nll_loss"] < first_history[0]["nll_loss"]
    )
    probability_increased = post_probability > before_probability + 1e-6
    if args.stage == "one_edit":
        acceptance = target_loss_decreased and probability_increased and rollback_pass and malformed <= 0.05
    elif args.stage == "smoke20":
        acceptance = rewrite >= 0.50 and paraphrase >= 0.20 and malformed <= 0.05 and rollback_pass
    else:
        acceptance = rollback_pass and malformed <= 0.05
    elapsed = time.monotonic() - start
    run_config = {
        "campaign_id": CAMPAIGN_ID,
        "stage": args.stage,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "num_edits": len(rows),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "use_4bit": False,
        "layers": list(args.layers),
        "memit": config.to_dict(),
        "include_locality": bool(args.include_locality),
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "M1",
        "stage": args.stage,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "num_edits": len(rows),
        "base_target_new_probability_first_edit": before_probability,
        "post_edit_target_new_probability_first_edit": post_probability,
        "target_probability_increased": probability_increased,
        "target_loss_decreased": target_loss_decreased,
        "rollback_checksum_pass": rollback_pass,
        "base_summary": base_summary,
        "edited_summary": edited_summary,
        "rewrite_exact": rewrite,
        "paraphrase_exact": paraphrase,
        "malformed_rate": malformed,
        "runtime_seconds": elapsed,
        "gpu_minutes_per_edit": elapsed / 60.0 / len(rows),
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "fake_model": False,
        "llada_loaded": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "acceptance_pass": acceptance,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage=f"M1_{args.stage}",
        track="M1",
        status="passed" if acceptance else "failed",
        output_dir=args.output_dir,
        acceptance_pass=acceptance,
        started_at_utc=started_at,
        notes=f"rewrite={rewrite:.4f}; paraphrase={paraphrase:.4f}; malformed={malformed:.4f}",
    )
    print(json.dumps({"acceptance_pass": acceptance, "rewrite_exact": rewrite, "paraphrase_exact": paraphrase}, sort_keys=True))


if __name__ == "__main__":
    main()
