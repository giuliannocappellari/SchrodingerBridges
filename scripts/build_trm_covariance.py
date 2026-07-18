#!/usr/bin/env python3
"""Build a train-only diagonal MLP-key second moment for fresh TRM baselines."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_editor import get_module, resolved_key_module_name
from scripts.run_mdm_memit_stage import load_model
from scripts.trm_common import (
    CAMPAIGN_ID,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    sha256_file,
    write_json,
)


def finalize_diagonal_covariance(sum_squares: torch.Tensor, count: int) -> torch.Tensor:
    if sum_squares.ndim != 1 or int(count) <= 0:
        raise ValueError("sum_squares must be a vector with a positive count")
    covariance = sum_squares.float() / float(count)
    covariance = covariance.clamp_min(1e-8)
    if not torch.isfinite(covariance).all():
        raise FloatingPointError("non-finite diagonal covariance")
    return covariance


def training_prompt_texts(rows: Sequence[dict[str, Any]]) -> list[str]:
    texts = []
    for row in rows:
        texts.append(str(row["rewrite_prompt"]))
        for key in (
            "same_subject_prompts",
            "generation_prompts",
            "attribute_prompts",
        ):
            texts.extend(str(value) for value in list(row.get(key) or [])[:1])
    return [text for text in texts if text.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor_manifest",
        type=Path,
        default=PROTOCOL_ROOT / "cf_trm_anchor_train_500.jsonl",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--maximum_tokens", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if any(
        token in str(args.anchor_manifest).casefold()
        for token in ("analysis_500", "final_test", "locked")
    ):
        raise RuntimeError("Covariance cannot use locked evaluation data")
    args.output_dir.mkdir(parents=True)
    begin = time.monotonic()
    rows = read_jsonl(args.anchor_manifest)
    if len(rows) != 500 or {str(row["split_role"]) for row in rows} != {
        "cf_trm_anchor_train_500"
    }:
        raise RuntimeError("Covariance requires fresh train-only anchor500")
    texts = training_prompt_texts(rows)
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    module = get_module(model, resolved_key_module_name(model, args.layer))
    width = int(module.weight.shape[1])
    sum_squares = torch.zeros(width, dtype=torch.float64, device="cuda")
    count = 0
    current_attention: list[torch.Tensor | None] = [None]

    def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        nonlocal count
        attention = current_attention[0]
        if attention is None:
            raise RuntimeError("Missing covariance attention mask")
        flat = inputs[0].detach().float()[attention.bool()]
        remaining = int(args.maximum_tokens) - count
        if remaining <= 0:
            return
        flat = flat[:remaining]
        if flat.numel():
            sum_squares.add_(flat.double().square().sum(dim=0))
            count += int(flat.shape[0])

    handle = module.register_forward_pre_hook(pre_hook)
    try:
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            current_attention[0] = batch["attention_mask"].to("cuda")
            with torch.no_grad():
                model(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=current_attention[0],
                )
            if count >= args.maximum_tokens:
                break
    finally:
        handle.remove()
    covariance = finalize_diagonal_covariance(sum_squares, count).cpu()
    path = args.output_dir / f"layer_{args.layer}_covariance.pt"
    torch.save(covariance, path)
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "E1_train_only_covariance",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layer": args.layer,
        "definition": "train_only_anchor_diagonal_uncentered_mlp_key_second_moment",
        "anchor_manifest": str(args.anchor_manifest),
        "anchor_manifest_sha256": sha256_file(args.anchor_manifest),
        "num_anchor_edits": len(rows),
        "num_prompt_texts": len(texts),
        "sample_count": count,
        "shape": list(covariance.shape),
        "finite": bool(torch.isfinite(covariance).all()),
        "minimum": float(covariance.min()),
        "mean": float(covariance.mean()),
        "runtime_seconds": runtime,
        "analysis_500_used": False,
        "final_test_used": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "acceptance_pass": bool(
            count > 0
            and covariance.shape == (width,)
            and torch.isfinite(covariance).all()
            and bool((covariance > 0).all())
        ),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "run_config.json",
        {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "layer": args.layer,
            "maximum_tokens": args.maximum_tokens,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "source_prompt_fields": [
                "rewrite_prompt",
                "same_subject_prompts[0]",
                "generation_prompts[0]",
                "attribute_prompts[0]",
            ],
        },
    )
    if not report["acceptance_pass"]:
        raise SystemExit(2)
    print(json.dumps({"acceptance_pass": True, "sample_count": count}))


if __name__ == "__main__":
    main()
