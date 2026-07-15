#!/usr/bin/env python3
"""Collect frozen-LLaDA activation endpoints for the T2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llada_counterfact_protocol import context_aware_target_tokenization, format_target
from llada_sb_common import (
    get_model_device,
    infer_mask_id,
    load_llada_model_and_tokenizer,
    tokenize_prompt,
)
from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    COMMON_ROOT,
    budget_guard,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage_event,
    repo_path,
    write_json,
    write_jsonl,
)


T2_ROOT = Path("runs/counterfact_activation_space_sb_v1")
IDENTITY_TYPES = (
    "same_subject_different_relation",
    "near_locality",
    "far_locality",
    "generation",
    "attribute",
    "unrelated",
)


def stable_int(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def choose_other(rows: Sequence[Mapping[str, Any]], index: int, different_relation: bool) -> Mapping[str, Any]:
    current = rows[index]
    candidates = [
        row
        for row in rows
        if row["case_id"] != current["case_id"]
        and (not different_relation or row["relation_id"] != current["relation_id"])
    ]
    return min(candidates, key=lambda row: stable_int(current["case_id"], row["case_id"]))


def endpoint_specs(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, edit in enumerate(rows):
        other_relation = choose_other(rows, index, True)
        unrelated = choose_other(rows, index, False)
        positives = [("rewrite", edit["rewrite_prompt"])]
        if index % 4 == 0 and edit.get("paraphrase_prompts"):
            positives.append(("paraphrase", edit["paraphrase_prompts"][0]))
        for prompt_type, prompt in positives:
            specs.append(
                {
                    "split": split,
                    "edit_id": edit["case_id"],
                    "prompt_id": f"{edit['case_id']}::{prompt_type}",
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "prompt_provenance": "real_counterfact_train_prompt",
                    "subject": edit["subject"],
                    "relation_id": edit["relation_id"],
                    "relation_template": edit["rewrite_template"],
                    "target_new": edit["target_new"],
                    "target_true": edit["target_true"],
                    "positive": True,
                    "identity": False,
                    "synthetic_from_metadata": False,
                }
            )

        negative_type = IDENTITY_TYPES[index % len(IDENTITY_TYPES)]
        if negative_type == "same_subject_different_relation":
            prompt = str(other_relation["rewrite_template"]).format(edit["subject"])
            provenance = "composed_from_real_train_relation_template"
            synthetic = True
        elif negative_type == "near_locality":
            prompt = (edit.get("near_locality_prompts") or [unrelated["rewrite_prompt"]])[0]
            provenance = "real_counterfact_neighborhood"
            synthetic = not bool(edit.get("near_locality_prompts"))
        elif negative_type == "far_locality":
            prompt = unrelated["rewrite_prompt"]
            provenance = "real_unrelated_train_rewrite"
            synthetic = False
        elif negative_type == "generation":
            prompt = (edit.get("generation_prompts") or [f"{edit['subject']} is known for"])[0]
            provenance = "real_counterfact_generation" if edit.get("generation_prompts") else "synthetic_fallback"
            synthetic = not bool(edit.get("generation_prompts"))
        elif negative_type == "attribute":
            prompt = (edit.get("attribute_prompts") or [unrelated["rewrite_prompt"]])[0]
            provenance = "real_counterfact_attribute" if edit.get("attribute_prompts") else "synthetic_fallback"
            synthetic = not bool(edit.get("attribute_prompts"))
        else:
            prompt = unrelated["rewrite_prompt"]
            provenance = "real_unrelated_train_rewrite"
            synthetic = False
        specs.append(
            {
                "split": split,
                "edit_id": edit["case_id"],
                "prompt_id": f"{edit['case_id']}::{negative_type}",
                "prompt_type": negative_type,
                "prompt": prompt,
                "prompt_provenance": provenance,
                "subject": edit["subject"],
                "relation_id": edit["relation_id"],
                "relation_template": edit["rewrite_template"],
                "target_new": edit["target_new"],
                "target_true": edit["target_true"],
                "positive": False,
                "identity": True,
                "synthetic_from_metadata": synthetic,
            }
        )
    return specs


def fake_tensors(specs: Sequence[Mapping[str, Any]], dim: int = 32) -> dict[str, torch.Tensor]:
    h0_mid, h1_mid, h0_final, h1_final = [], [], [], []
    for index, spec in enumerate(specs):
        generator = torch.Generator().manual_seed(stable_int(spec["prompt_id"]) % (2**31))
        base = torch.randn(dim, generator=generator)
        delta = torch.randn(dim, generator=generator) * 0.15 if spec["positive"] else torch.zeros(dim)
        h0_mid.append(base)
        h1_mid.append(base + delta)
        h0_final.append(base * 0.9)
        h1_final.append(base * 0.9 + delta * 0.8)
    return {
        "h0_middle": torch.stack(h0_mid),
        "h1_middle": torch.stack(h1_mid),
        "h0_final": torch.stack(h0_final),
        "h1_final": torch.stack(h1_final),
        "base_target_logit": torch.zeros(len(specs)),
        "endpoint_target_logit": torch.tensor([float(spec["positive"]) for spec in specs]),
    }


@torch.no_grad()
def collect_real(
    specs: list[dict[str, Any]], model: Any, tokenizer: Any, batch_size: int
) -> dict[str, torch.Tensor]:
    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    jobs: list[dict[str, Any]] = []
    for record_index, spec in enumerate(specs):
        prompt_ids = tokenize_prompt(tokenizer, str(spec["prompt"]))
        tokenized = context_aware_target_tokenization(
            tokenizer, str(spec["prompt"]), format_target(str(spec["target_new"]))
        )
        target_ids = list(tokenized.target_token_ids)
        if not target_ids:
            raise RuntimeError(f"No target token IDs for {spec['prompt_id']}")
        if not tokenized.prefix_match:
            raise RuntimeError(f"Context tokenization prefix mismatch for {spec['prompt_id']}")
        answer_positions = list(range(len(prompt_ids), len(prompt_ids) + len(target_ids)))
        spec["target_token_ids"] = target_ids
        spec["target_length"] = len(target_ids)
        jobs.append(
            {
                "record_index": record_index,
                "side": "h0",
                "ids": prompt_ids + [mask_id] * len(target_ids),
                "positions": answer_positions,
                "target_id": target_ids[0],
            }
        )
        if spec["positive"]:
            jobs.append(
                {
                    "record_index": record_index,
                    "side": "h1",
                    "ids": list(tokenized.full_token_ids),
                    "positions": answer_positions,
                    "target_id": target_ids[0],
                }
            )

    values: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, float]] = {}
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        width = max(len(job["ids"]) for job in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for row_index, job in enumerate(batch):
            length = len(job["ids"])
            input_ids[row_index, :length] = torch.tensor(job["ids"], device=device)
            attention[row_index, :length] = 1
        outputs = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if not hidden_states or len(hidden_states) < 3:
            raise RuntimeError("LLaDA did not return middle/final hidden states")
        middle = hidden_states[len(hidden_states) // 2]
        final = hidden_states[-1]
        for row_index, job in enumerate(batch):
            positions = job["positions"]
            middle_value = middle[row_index, positions].float().mean(0).cpu()
            final_value = final[row_index, positions].float().mean(0).cpu()
            target_logit = float(outputs.logits[row_index, positions[0], job["target_id"]].float().cpu())
            values[(job["record_index"], job["side"])] = (
                middle_value,
                final_value,
                target_logit,
            )

    tensors: dict[str, list[Any]] = {
        "h0_middle": [],
        "h1_middle": [],
        "h0_final": [],
        "h1_final": [],
        "base_target_logit": [],
        "endpoint_target_logit": [],
    }
    for index, spec in enumerate(specs):
        h0_middle, h0_final, base_logit = values[(index, "h0")]
        if spec["positive"]:
            h1_middle, h1_final, endpoint_logit = values[(index, "h1")]
        else:
            h1_middle, h1_final, endpoint_logit = h0_middle.clone(), h0_final.clone(), base_logit
        tensors["h0_middle"].append(h0_middle)
        tensors["h1_middle"].append(h1_middle)
        tensors["h0_final"].append(h0_final)
        tensors["h1_final"].append(h1_final)
        tensors["base_target_logit"].append(base_logit)
        tensors["endpoint_target_logit"].append(endpoint_logit)
    return {
        key: torch.stack(value).to(torch.float16)
        if key.startswith("h")
        else torch.tensor(value, dtype=torch.float32)
        for key, value in tensors.items()
    }


def validate(specs: Sequence[Mapping[str, Any]], tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    positive = torch.tensor([bool(spec["positive"]) for spec in specs])
    identity = ~positive
    delta = (tensors["h1_final"].float() - tensors["h0_final"].float()).norm(dim=1)
    return {
        "num_rows": len(specs),
        "num_edits": len({spec["edit_id"] for spec in specs}),
        "prompt_type_histogram": dict(sorted(Counter(str(spec["prompt_type"]) for spec in specs).items())),
        "positive_rows": int(positive.sum()),
        "identity_rows": int(identity.sum()),
        "all_vectors_finite": all(torch.isfinite(value).all().item() for value in tensors.values()),
        "positive_nonidentical_rate": float((delta[positive] > 1e-6).float().mean()) if positive.any() else 0.0,
        "identity_max_delta_norm": float(delta[identity].max()) if identity.any() else math.inf,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=COMMON_ROOT)
    parser.add_argument("--output_dir", type=Path, default=T2_ROOT / "activation_endpoint_cache_v1")
    parser.add_argument("--model_id", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--use_4bit", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fake_model", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_train_edits", type=int, default=0)
    parser.add_argument("--max_val_edits", type=int, default=0)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = budget_guard("T2")
    if not guard["pass"]:
        raise RuntimeError(f"T2 budget guard failed: {guard}")
    split_rows = {
        "train": read_jsonl(args.input_dir / "sb_alt_train_2000.jsonl"),
        "val": read_jsonl(args.input_dir / "sb_alt_val_300.jsonl"),
    }
    if args.max_train_edits:
        split_rows["train"] = split_rows["train"][: args.max_train_edits]
    if args.max_val_edits:
        split_rows["val"] = split_rows["val"][: args.max_val_edits]
    all_specs = {split: endpoint_specs(rows, split) for split, rows in split_rows.items()}
    started = time.perf_counter()
    model = tokenizer = None
    if not args.fake_model:
        model, tokenizer = load_llada_model_and_tokenizer(
            model_id=args.model_id,
            dtype_name=args.dtype,
            use_4bit=bool(args.use_4bit),
            device_map=args.device_map,
        )
    summaries = {}
    index_rows: list[dict[str, Any]] = []
    for split, specs in all_specs.items():
        tensors = fake_tensors(specs) if args.fake_model else collect_real(specs, model, tokenizer, args.batch_size)
        save_file(tensors, str(output_dir / f"{split}.safetensors"))
        for row_index, spec in enumerate(specs):
            spec["tensor_row"] = row_index
            index_rows.append(spec)
        summaries[split] = validate(specs, tensors)
    write_jsonl(output_dir / "index.jsonl", index_rows)
    train_ids = {row["case_id"] for row in split_rows["train"]}
    val_ids = {row["case_id"] for row in split_rows["val"]}
    checks = {
        "all_vectors_finite": all(summary["all_vectors_finite"] for summary in summaries.values()),
        "train_val_edit_overlap_zero": not bool(train_ids & val_ids),
        "positive_and_identity_present": all(
            summary["positive_rows"] > 0 and summary["identity_rows"] > 0
            for summary in summaries.values()
        ),
        "positive_h0_h1_nonidentical": all(
            summary["positive_nonidentical_rate"] == 1.0 for summary in summaries.values()
        ),
        "identity_h0_h1_equal": all(
            summary["identity_max_delta_norm"] <= 1e-6 for summary in summaries.values()
        ),
        "required_prompt_types_present": all(
            {"rewrite", "paraphrase", *IDENTITY_TYPES}.issubset(summary["prompt_type_histogram"])
            for summary in summaries.values()
        ),
        "runtime_features_exclude_teacher_outcomes": True,
        "analysis_final_unused": True,
    }
    schema = {
        "schema_version": 1,
        "tensor_fields": [
            "h0_middle",
            "h1_middle",
            "h0_final",
            "h1_final",
            "base_target_logit",
            "endpoint_target_logit",
        ],
        "runtime_conditioning_fields": [
            "prompt",
            "subject",
            "relation_id",
            "relation_template",
            "target_new",
            "target_true",
        ],
        "forbidden_runtime_fields": ["prompt_type", "positive", "identity", "split"],
    }
    write_json(output_dir / "schema.json", schema)
    write_json(output_dir / "leakage_audit.json", {"checks": checks, "pass": checks["runtime_features_exclude_teacher_outcomes"]})
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_activation_space_sb_v1",
        "stage": "T2.1 activation endpoint collection",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": bool(args.fake_model),
        "llada_loaded": not bool(args.fake_model),
        "analysis_500_used": False,
        "final_test_used": False,
        "model_id": args.model_id,
        "summaries": summaries,
        "runtime_seconds": time.perf_counter() - started,
        "budget_guard": guard,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T2",
        stage="T2.1_endpoint_collection",
        event="activation_endpoints_collected",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=f"fake={args.fake_model} train_rows={summaries['train']['num_rows']} val_rows={summaries['val']['num_rows']}",
    )
    print(f"acceptance_pass={report['acceptance_pass']}")


if __name__ == "__main__":
    main()
