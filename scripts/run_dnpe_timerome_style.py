#!/usr/bin/env python3
"""Run a clearly labelled TimeROME-DLM-style temporal residual memory baseline."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

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
    extract_keys_and_outputs,
    get_module,
    optimize_target_value,
    resolved_key_module_name,
)
from scripts.run_dnpe_editor import (
    _forbid_locked_manifest,
    aggregate,
    align_base,
    build_eval_tasks,
    evaluate_tasks,
)
from scripts.run_mdm_memit_stage import load_model


def build_residual_memory(keys, residuals, ridge: float):
    import torch

    keys = keys.float().cuda()
    residuals = residuals.float().cuda()
    gram = keys @ keys.T
    system = gram + torch.eye(len(keys), device=keys.device) * float(ridge)
    dual = torch.linalg.solve(system, keys)
    if not torch.isfinite(dual).all() or not torch.isfinite(residuals).all():
        raise FloatingPointError("Residual memory contains non-finite values")
    return {
        "keys": keys,
        "normalized_keys": keys / keys.norm(dim=1, keepdim=True).clamp_min(1e-8),
        "dual": dual,
        "residuals": residuals,
    }


def install_residual_memory(module, memory, *, similarity_threshold: float, top_k: int):
    import torch

    def pre_hook(_module, inputs):
        memory["current_input"] = inputs[0]

    def hook(_module, _inputs, output):
        x = memory.pop("current_input").float()
        normalized = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        similarities = normalized @ memory["normalized_keys"].T
        k = min(int(top_k), similarities.shape[-1])
        top_values, top_indices = similarities.topk(k, dim=-1)
        sparse_mask = torch.zeros_like(similarities)
        sparse_mask.scatter_(-1, top_indices, (top_values >= float(similarity_threshold)).to(similarities.dtype))
        coefficients = (x @ memory["dual"].T) * sparse_mask
        delta = coefficients @ memory["residuals"]
        return output + delta.to(dtype=output.dtype)

    return module.register_forward_pre_hook(pre_hook), module.register_forward_hook(hook)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    parser.add_argument("--top_k_memory", type=int, default=4)
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--partial_mask_schedule", default="cycle")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260717101)
    args = parser.parse_args()
    _forbid_locked_manifest(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    begin = time.monotonic()
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    tasks = build_eval_tasks(tokenizer, rows, include_locality=True)
    base_rows = evaluate_tasks(model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None)
    config = MemitConfig(
        layers=(args.layer,),
        target_optimization_steps=args.target_optimization_steps,
        partial_mask_schedule=args.partial_mask_schedule,
        state_consistency_weight=0.1,
        old_target_suppression_weight=0.25,
        seed=args.seed,
    )
    targets = []
    target_reports = []
    for row in rows:
        target, report = optimize_target_value(model, tokenizer, row, config)
        targets.append(target.cpu())
        target_reports.append({"case_id": row["case_id"], **report})
    keys, outputs = extract_keys_and_outputs(
        model,
        tokenizer,
        rows,
        key_layer=args.layer,
        output_layer=args.layer,
        partial_mask_schedule=args.partial_mask_schedule,
        seed=args.seed,
    )
    import torch

    target_matrix = torch.stack(targets)
    residuals = target_matrix - outputs
    memory = build_residual_memory(keys, residuals, args.ridge)
    module = get_module(model, resolved_key_module_name(model, args.layer))
    handles = install_residual_memory(
        module,
        memory,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k_memory,
    )
    try:
        edited_rows = evaluate_tasks(model, tokenizer, tasks, decode_batch_size=args.decode_batch_size, steps=None)
    finally:
        for handle in handles:
            handle.remove()
    edited_rows = align_base(base_rows, edited_rows)
    base_summary = aggregate(base_rows)
    edited_summary = aggregate(edited_rows)
    rewrite = edited_summary.get("rewrite", {}).get("expected_exact", 0.0)
    paraphrase = edited_summary.get("declarative_paraphrase", {}).get("expected_exact", 0.0)
    malformed = max((value["malformed_rate"] for value in edited_summary.values()), default=0.0)
    runtime = time.monotonic() - begin
    storage_bytes = sum(
        tensor.numel() * tensor.element_size()
        for key, tensor in memory.items()
        if key != "normalized_keys"
    )
    write_csv(args.output_dir / "base_per_prompt.csv", base_rows)
    write_csv(args.output_dir / "edited_per_prompt.csv", edited_rows)
    write_json(args.output_dir / "target_value_diagnostics.json", target_reports)
    config_payload = {
        "campaign_id": CAMPAIGN_ID,
        "method": "timerome_dlm_style_residual_memory",
        "reproduction_claim": "style_baseline_not_exact_reproduction",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layer": args.layer,
        "ridge": args.ridge,
        "similarity_threshold": args.similarity_threshold,
        "top_k_memory": args.top_k_memory,
        "partial_mask_schedule": args.partial_mask_schedule,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "run_config.json", config_payload)
    report = {
        **config_payload,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "num_edits": len(rows),
        "base_summary": base_summary,
        "edited_summary": edited_summary,
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "same_subject_tfpr": edited_summary.get("same_subject", {}).get("target_new_tfpr_or_exact", 0.0),
        "near_tfpr": edited_summary.get("near_locality", {}).get("target_new_tfpr_or_exact", 0.0),
        "far_tfpr": edited_summary.get("far_locality", {}).get("target_new_tfpr_or_exact", 0.0),
        "malformed_rate": malformed,
        "residual_memory_rank": min(len(rows), int(torch.linalg.matrix_rank(residuals.float()))),
        "residual_memory_finite": bool(torch.isfinite(memory["dual"]).all() and torch.isfinite(memory["residuals"]).all()),
        "storage_bytes": storage_bytes,
        "runtime_seconds": runtime,
        "gpu_minutes_per_edit": runtime / 60.0 / len(rows),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "acceptance_pass": bool(math.isfinite(rewrite) and math.isfinite(paraphrase) and malformed <= 0.05),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", {"temporal_localization_source": "frozen C1/C2 site or source-aligned layer", "residual_memory_finite": report["residual_memory_finite"], "metrics_complete": True, "runtime_and_storage_reported": True, "acceptance_pass": report["acceptance_pass"]})
    print(json.dumps({"acceptance_pass": report["acceptance_pass"], "rewrite_exact": rewrite, "paraphrase_exact": paraphrase}, sort_keys=True))


if __name__ == "__main__":
    main()
