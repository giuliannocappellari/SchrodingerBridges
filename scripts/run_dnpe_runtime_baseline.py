#!/usr/bin/env python3
"""Run prompt-memory or target-logit-bias baselines on DNPE manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    git_commit,
    now_utc,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mdm_memit_editor import contextual_target_ids, infer_mask_id, normalized_hit
from scripts.run_dnpe_editor import _forbid_locked_manifest, aggregate, build_eval_tasks, evaluate_tasks
from scripts.run_mdm_memit_stage import load_model


def prompt_memory_tasks(
    tasks: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_case = {str(row["case_id"]): row for row in rows}
    output = []
    for source in tasks:
        row = by_case[str(source["case_id"])]
        statement = f"The following fact is true: {row['rewrite_prompt']} {row['target_new']}."
        task = dict(source)
        task["original_prompt"] = task["prompt"]
        task["prompt"] = f"{statement} {task['prompt']}"
        output.append(task)
    return output


def _biased_decode_one(
    model: Any,
    tokenizer: Any,
    task: Mapping[str, Any],
    *,
    guidance_scale: float,
) -> dict[str, Any]:
    import torch

    prompt = str(task["prompt"])
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    n = int(task["answer_length"])
    mask_id = infer_mask_id(model)
    state = torch.tensor([prompt_ids + [mask_id] * n], dtype=torch.long, device=next(model.parameters()).device)
    positions = list(range(len(prompt_ids), len(prompt_ids) + n))
    target_ids = contextual_target_ids(tokenizer, prompt, str(task["target_new"]))
    target_ids = (target_ids + [target_ids[-1]] * n)[:n]
    evaluations = 0
    with torch.no_grad():
        while any(int(state[0, position]) == mask_id for position in positions):
            logits = model(input_ids=state).logits[0].float()
            evaluations += 1
            candidates = []
            for relative, position in enumerate(positions):
                if int(state[0, position]) != mask_id:
                    continue
                adjusted = logits[position].clone()
                adjusted[int(target_ids[relative])] += float(guidance_scale)
                probabilities = torch.softmax(adjusted, dim=-1)
                confidence, token = probabilities.max(dim=-1)
                candidates.append((float(confidence), position, int(token)))
            _confidence, selected_position, selected_token = max(candidates)
            state[0, selected_position] = selected_token
    output_ids = [int(state[0, position]) for position in positions]
    return {
        "output_text": tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
        "output_token_ids": output_ids,
        "malformed": any(token == mask_id for token in output_ids),
        "model_eval_count": evaluations,
    }


def biased_results(model: Any, tokenizer: Any, tasks: Sequence[Mapping[str, Any]], scale: float) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        decoded = _biased_decode_one(model, tokenizer, task, guidance_scale=scale)
        target_ids = contextual_target_ids(tokenizer, str(task["prompt"]), str(task["target_new"]))
        predicted = decoded["output_token_ids"]
        overlap = len(set(predicted) & set(target_ids))
        precision = overlap / max(len(predicted), 1)
        recall = overlap / max(len(target_ids), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        expected = str(task["expected"])
        rows.append(
            {
                **dict(task),
                "prompt_fingerprint": hashlib.sha256(
                    f"{task['case_id']}::{task['bucket']}::{task['prompt']}".encode()
                ).hexdigest(),
                "output_text": decoded["output_text"],
                "output_token_ids": json.dumps(predicted),
                "target_new_hit": normalized_hit(decoded["output_text"], str(task["target_new"])),
                "target_true_hit": normalized_hit(decoded["output_text"], str(task["target_true"])),
                "expected_hit": normalized_hit(decoded["output_text"], expected) if expected else None,
                "target_token_f1": f1,
                "malformed": decoded["malformed"],
                "model_eval_count": decoded["model_eval_count"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("prompt_memory", "target_logit_bias"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--decode_batch_size", type=int, default=16)
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
    if args.method == "prompt_memory":
        memory_tasks = prompt_memory_tasks(tasks, rows)
        raw = evaluate_tasks(model, tokenizer, memory_tasks, decode_batch_size=args.decode_batch_size, steps=None)
        # Restore original prompt identity for bucket-aligned reporting.
        edited_rows = []
        for task, result in zip(tasks, raw):
            value = dict(result)
            value["prompt"] = task["prompt"]
            value["prompt_fingerprint"] = hashlib.sha256(
                f"{task['case_id']}::{task['bucket']}::{task['prompt']}".encode()
            ).hexdigest()
            edited_rows.append(value)
    else:
        edited_rows = biased_results(model, tokenizer, tasks, args.guidance_scale)
    base_index = {row["prompt_fingerprint"]: row for row in base_rows}
    for row in edited_rows:
        base = base_index[row["prompt_fingerprint"]]
        row["base_output_text"] = base["output_text"]
        row["base_agreement"] = " ".join(str(row["output_text"]).casefold().split()) == " ".join(str(base["output_text"]).casefold().split())
    base_summary = aggregate(base_rows)
    edited_summary = aggregate(edited_rows)
    rewrite = edited_summary.get("rewrite", {}).get("expected_exact", 0.0)
    paraphrase = edited_summary.get("declarative_paraphrase", {}).get("expected_exact", 0.0)
    malformed = max((value["malformed_rate"] for value in edited_summary.values()), default=0.0)
    runtime = time.monotonic() - begin
    write_csv(args.output_dir / "base_per_prompt.csv", base_rows)
    write_csv(args.output_dir / "edited_per_prompt.csv", edited_rows)
    config = {
        "campaign_id": CAMPAIGN_ID,
        "method": args.method,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "guidance_scale": args.guidance_scale if args.method == "target_logit_bias" else 0.0,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "run_config.json", config)
    report = {
        **config,
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
        "runtime_seconds": runtime,
        "gpu_minutes_per_edit": runtime / 60.0 / len(rows),
        "environment": {"python": platform.python_version(), "torch": __import__("torch").__version__, "transformers": __import__("transformers").__version__, "gpu": __import__("torch").cuda.get_device_name(0)},
        "acceptance_pass": malformed <= 0.05,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", {"metrics_complete": True, "malformed_pass": malformed <= 0.05, "acceptance_pass": report["acceptance_pass"]})
    print(json.dumps({"method": args.method, "rewrite_exact": rewrite, "paraphrase_exact": paraphrase, "same_subject_tfpr": report["same_subject_tfpr"]}, sort_keys=True))


if __name__ == "__main__":
    main()
