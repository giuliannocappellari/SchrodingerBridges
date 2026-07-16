#!/usr/bin/env python3
"""Estimate official-style uncentered MLP-key covariance for LLaDA MEMIT."""

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

from scripts.mdm_memit_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    MODEL_ID,
    MODEL_REVISION,
    git_commit,
    now_utc,
    record_stage,
    sha256_file,
    write_json,
)
from scripts.mdm_memit_editor import get_module, key_module_name


DEFAULT_OUTPUT = CAMPAIGN_ROOT / "covariance_cache_v1"


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return layers


def wikipedia_texts(dataset_name: str, config_name: str) -> Iterable[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
    for row in dataset:
        text = str(row.get("text") or "").strip()
        if text:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--sample_size", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dataset_name", default="wikimedia/wikipedia")
    parser.add_argument("--dataset_config", default="20231101.en")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    started_at = now_utc()
    start = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = [args.output_dir / f"layer_{layer}_covariance.pt" for layer in args.layers]
    existing = [path for path in expected if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(f"Covariance outputs already exist: {existing}")

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for covariance estimation")
    model = model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == model.config.mask_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    key_width = int(model.config.mlp_hidden_size)
    accumulators = {
        layer: torch.zeros((key_width, key_width), dtype=torch.float32, device="cuda")
        for layer in args.layers
    }
    counts = {layer: 0 for layer in args.layers}
    current_attention: list[torch.Tensor | None] = [None]
    handles = []

    for layer in args.layers:
        module = get_module(model, key_module_name(layer))

        def make_hook(layer_index: int):
            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                activations = inputs[0].detach().float()
                attention = current_attention[0]
                if attention is None:
                    raise RuntimeError("Missing attention mask for covariance hook")
                flat = activations[attention.bool()]
                remaining = args.sample_size - counts[layer_index]
                if flat.shape[0] > remaining:
                    flat = flat[:remaining]
                if flat.numel():
                    accumulators[layer_index].addmm_(flat.T, flat)
                    counts[layer_index] += int(flat.shape[0])
            return hook

        handles.append(module.register_forward_pre_hook(make_hook(layer)))

    source = wikipedia_texts(args.dataset_name, args.dataset_config)
    texts: list[str] = []
    try:
        for text in source:
            texts.append(text)
            if len(texts) < args.batch_size:
                continue
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            texts.clear()
            current_attention[0] = batch["attention_mask"].to("cuda")
            with torch.no_grad():
                model(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=current_attention[0],
                )
            if min(counts.values()) >= args.sample_size:
                break
    finally:
        for handle in handles:
            handle.remove()
    if min(counts.values()) < args.sample_size:
        raise RuntimeError(f"Wikipedia stream ended before sample target: {counts}")

    layer_summaries: dict[str, Any] = {}
    for layer in args.layers:
        covariance = (accumulators[layer] / float(counts[layer])).cpu()
        covariance = (covariance + covariance.T) * 0.5
        path = args.output_dir / f"layer_{layer}_covariance.pt"
        torch.save(covariance, path)
        layer_summaries[str(layer)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "shape": list(covariance.shape),
            "sample_count": counts[layer],
            "finite": bool(torch.isfinite(covariance).all()),
            "diagonal_min": float(covariance.diagonal().min()),
            "diagonal_mean": float(covariance.diagonal().mean()),
            "symmetry_max_error": float((covariance - covariance.T).abs().max()),
        }
        del covariance
    elapsed = time.monotonic() - start
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "M1_covariance_cache",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_dtype": "float16",
        "quantized": False,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "mom2_definition": "uncentered_mean_outer_product_of_ff_out_inputs",
        "sample_size_per_layer": args.sample_size,
        "layers": args.layers,
        "layer_summaries": layer_summaries,
        "runtime_seconds": elapsed,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "acceptance_pass": all(
            item["finite"]
            and item["shape"] == [key_width, key_width]
            and item["symmetry_max_error"] <= 1e-3
            for item in layer_summaries.values()
        ),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "run_config.json",
        {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "layers": args.layers,
            "sample_size": args.sample_size,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
        },
    )
    record_stage(
        stage="M1_covariance_cache",
        track="M1",
        status="passed" if report["acceptance_pass"] else "failed",
        output_dir=args.output_dir,
        acceptance_pass=report["acceptance_pass"],
        started_at_utc=started_at,
        notes=f"Estimated exact uncentered covariance for layers {args.layers}.",
    )
    print(f"acceptance_pass={report['acceptance_pass']}")


if __name__ == "__main__":
    main()
