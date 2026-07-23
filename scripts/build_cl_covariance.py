#!/usr/bin/env python3
"""Estimate fresh uncentered key covariance for the pinned LLaDA model."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    git_commit,
    now_utc,
    sha256_file,
    write_json,
)
from scripts.mdm_memit_editor import get_module, resolved_key_module_name


DEFAULT_OUTPUT = CAMPAIGN_ROOT / "B1_covariance_cache_v1"


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return layers


def wikipedia_texts(dataset_name: str, config_name: str) -> Iterable[str]:
    from datasets import load_dataset

    for row in load_dataset(dataset_name, config_name, split="train", streaming=True):
        text = str(row.get("text") or "").strip()
        if text:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--sample_size", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dataset_name", default="wikimedia/wikipedia")
    parser.add_argument("--dataset_config", default="20231101.en")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        report = args.output_dir / "report_summary.json"
        if report.is_file():
            print(f"Covariance cache already exists: {args.output_dir}")
            return
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == model.config.mask_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    modules = {layer: get_module(model, resolved_key_module_name(model, layer)) for layer in args.layers}
    widths = {layer: int(module.weight.shape[1]) for layer, module in modules.items()}
    accumulators = {
        layer: torch.zeros((widths[layer], widths[layer]), dtype=torch.float32, device="cuda")
        for layer in args.layers
    }
    counts = {layer: 0 for layer in args.layers}
    attention_box: list[Any] = [None]
    handles = []
    for layer, module in modules.items():
        def make_hook(layer_index: int):
            def hook(_module, inputs):
                values = inputs[0].detach().float()
                attention = attention_box[0]
                if attention is None:
                    raise RuntimeError("Missing covariance attention mask")
                flat = values[attention.bool()]
                remaining = args.sample_size - counts[layer_index]
                flat = flat[: max(remaining, 0)]
                if flat.numel():
                    accumulators[layer_index].addmm_(flat.T, flat)
                    counts[layer_index] += int(flat.shape[0])
            return hook
        handles.append(module.register_forward_pre_hook(make_hook(layer)))

    batch_texts: list[str] = []
    try:
        for text in wikipedia_texts(args.dataset_name, args.dataset_config):
            batch_texts.append(text)
            if len(batch_texts) < args.batch_size:
                continue
            batch = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=args.max_length,
                return_tensors="pt",
            )
            batch_texts.clear()
            attention_box[0] = batch["attention_mask"].to("cuda")
            with torch.no_grad():
                model(input_ids=batch["input_ids"].to("cuda"), attention_mask=attention_box[0])
            if min(counts.values()) >= args.sample_size:
                break
    finally:
        for handle in handles:
            handle.remove()
    if min(counts.values()) < args.sample_size:
        raise RuntimeError(f"Wikipedia stream ended before target: {counts}")

    summaries = {}
    for layer in args.layers:
        covariance = (accumulators[layer] / counts[layer]).cpu()
        covariance = (covariance + covariance.T) * 0.5
        path = args.output_dir / f"layer_{layer}_covariance.pt"
        torch.save(covariance, path)
        summaries[str(layer)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "shape": list(covariance.shape),
            "sample_count": counts[layer],
            "finite": bool(torch.isfinite(covariance).all()),
            "diagonal_min": float(covariance.diagonal().min()),
            "symmetry_max_error": float((covariance - covariance.T).abs().max()),
        }
    acceptance = all(
        row["finite"] and row["sample_count"] == args.sample_size
        and row["symmetry_max_error"] <= 1e-3
        for row in summaries.values()
    )
    write_json(
        args.output_dir / "report_summary.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "B1_covariance_cache",
            "created_at_utc": now_utc(),
            "git_commit": git_commit(),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "sample_size_per_layer": args.sample_size,
            "layers": args.layers,
            "layer_summaries": summaries,
            "runtime_seconds": time.monotonic() - started,
            "analysis_500_used": False,
            "final_test_used": False,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "acceptance_pass": acceptance,
        },
    )
    if not acceptance:
        raise SystemExit(2)
    print(f"B1 covariance cache passed: {args.output_dir}")


if __name__ == "__main__":
    main()
