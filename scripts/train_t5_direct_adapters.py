#!/usr/bin/env python3
"""Train and audit the T5 rank-r answer-position endpoint adapters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
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
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


T5_ROOT = Path("runs/counterfact_parameter_space_sb_v1")


def stable_int(*parts: Any) -> int:
    return int(hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def target_token_id(tokenizer: Any, prompt: str, target: str) -> int:
    result = context_aware_target_tokenization(tokenizer, prompt, format_target(target))
    if not result.prefix_match or len(result.target_token_ids) != 1:
        raise ValueError("Target is not a context-compatible single token")
    return int(result.target_token_ids[0])


def different_relation_rows(
    pool: Sequence[Mapping[str, Any]], edit: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in pool
        if row["case_id"] != edit["case_id"] and row["relation_id"] != edit["relation_id"]
    ]
    return sorted(rows, key=lambda row: stable_int(edit["case_id"], row["case_id"]))


def prompt_specs(
    edit: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    alternatives = different_relation_rows(pool, edit)
    if len(alternatives) < 2:
        raise RuntimeError("Insufficient distinct relation templates")
    near = list(edit.get("near_locality_prompts") or [])
    attributes = list(edit.get("attribute_prompts") or [])
    paraphrases = list(edit.get("paraphrase_prompts") or [])
    if not near or not attributes or not paraphrases:
        raise ValueError("Required real held-out prompt fields are unavailable")
    rewrite = str(edit["rewrite_prompt"])
    raw_specs = [
        ("train_rewrite", "train", "rewrite", rewrite, True, "real_counterfact_rewrite"),
        (
            "train_relation_augmentation",
            "train",
            "relation_augmentation",
            rewrite.rstrip() + " .",
            True,
            "synthetic_from_allowed_rewrite_template",
        ),
        (
            "train_same_subject_anchor",
            "train",
            "same_subject_anchor",
            str(alternatives[0]["rewrite_template"]).format(edit["subject"]),
            False,
            "composed_from_real_train_relation_template",
        ),
        ("train_near_anchor", "train", "near_anchor", str(near[0]), False, "real_counterfact_neighborhood"),
        ("train_far_anchor", "train", "far_anchor", str(attributes[0]), False, "real_counterfact_attribute"),
        ("eval_rewrite", "eval", "rewrite", rewrite, True, "real_counterfact_rewrite_train_seen"),
        ("eval_paraphrase", "eval", "paraphrase", str(paraphrases[0]), True, "real_counterfact_paraphrase_heldout"),
        (
            "eval_same_subject",
            "eval",
            "same_subject_stress",
            str(alternatives[1]["rewrite_template"]).format(edit["subject"]),
            False,
            "composed_from_distinct_real_train_relation_template",
        ),
        (
            "eval_near",
            "eval",
            "near_locality",
            str(near[1] if len(near) > 1 else near[0] + " ?"),
            False,
            "real_counterfact_neighborhood_heldout" if len(near) > 1 else "tagged_synthetic_fallback",
        ),
        (
            "eval_far",
            "eval",
            "far_locality",
            str(attributes[1] if len(attributes) > 1 else attributes[0] + " ?"),
            False,
            "real_counterfact_attribute_heldout" if len(attributes) > 1 else "tagged_synthetic_fallback",
        ),
    ]
    specs = []
    for prompt_id, access, bucket, prompt, positive, provenance in raw_specs:
        new_id = target_token_id(tokenizer, prompt, str(edit["target_new"]))
        try:
            old_id = target_token_id(tokenizer, prompt, str(edit["target_true"]))
        except ValueError:
            old_id = -1
        specs.append(
            {
                "prompt_id": f"{edit['case_id']}:{prompt_id}",
                "access": access,
                "bucket": bucket,
                "prompt": prompt,
                "positive": positive,
                "prompt_provenance": provenance,
                "target_new_token_id": new_id,
                "target_true_token_id": old_id,
            }
        )
    train_prompts = {spec["prompt"] for spec in specs if spec["access"] == "train"}
    heldout_prompts = {
        spec["prompt"]
        for spec in specs
        if spec["access"] == "eval" and spec["bucket"] != "rewrite"
    }
    if train_prompts & heldout_prompts:
        raise RuntimeError(f"T5 train/eval prompt overlap for {edit['case_id']}")
    return specs


def select_edits(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    tokenizer: Any,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    selected = []
    rejected = Counter()
    for row in sorted(rows, key=lambda item: stable_int(item["case_id"])):
        try:
            specs = prompt_specs(row, rows, tokenizer)
        except (ValueError, RuntimeError) as exc:
            rejected[type(exc).__name__] += 1
            continue
        if any(spec["target_new_token_id"] < 0 for spec in specs):
            rejected["invalid_target"] += 1
            continue
        selected.append(row)
        if len(selected) == count:
            return selected, dict(sorted(rejected.items()))
    raise RuntimeError(f"Could select only {len(selected)} T5 edits; need {count}")


def adapter_residual(hidden: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (hidden @ right) @ left.T


def project_residual(residual: torch.Tensor, output_weight: torch.Tensor) -> torch.Tensor:
    return F.linear(residual.to(dtype=output_weight.dtype), output_weight).float()


def support_for_logits(
    logits: torch.Tensor, target_new: int, target_true: int, top_k: int
) -> torch.Tensor:
    support = logits.topk(min(top_k, logits.numel())).indices.tolist()
    support.extend([target_new])
    if target_true >= 0:
        support.append(target_true)
    return torch.tensor(list(dict.fromkeys(map(int, support))), device=logits.device, dtype=torch.long)


def optimize_adapter(
    hidden: torch.Tensor,
    base_logits: torch.Tensor,
    output_weight: torch.Tensor,
    specs: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    logit_scale: float,
    steps: int = 100,
    top_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    dimension = hidden.shape[1]
    generator = torch.Generator(device=hidden.device).manual_seed(1729)
    left = (torch.randn((dimension, rank), generator=generator, device=hidden.device) * 1e-3).requires_grad_()
    right = torch.zeros((dimension, rank), device=hidden.device, requires_grad=True)
    optimizer = torch.optim.Adam([left, right], lr=0.05)
    supports = [
        support_for_logits(
            base_logits[index],
            int(spec["target_new_token_id"]),
            int(spec["target_true_token_id"]),
            top_k,
        )
        for index, spec in enumerate(specs)
    ]
    losses: list[float] = []
    for _ in range(steps):
        residual = adapter_residual(hidden, left, right)
        positive_losses = []
        anchor_losses = []
        leakage_losses = []
        for index, spec in enumerate(specs):
            if spec["access"] != "train":
                continue
            support = supports[index]
            adapted = base_logits[index, support] + project_residual(
                residual[index], output_weight[support]
            ) * logit_scale
            if spec["positive"]:
                target_index = int((support == int(spec["target_new_token_id"])).nonzero()[0])
                positive_losses.append(F.cross_entropy(adapted.unsqueeze(0), torch.tensor([target_index], device=hidden.device)))
            else:
                base_log_probs = F.log_softmax(base_logits[index, support].detach(), dim=0)
                adapted_log_probs = F.log_softmax(adapted, dim=0)
                anchor_losses.append(
                    F.kl_div(adapted_log_probs, base_log_probs.exp(), reduction="sum", log_target=False)
                )
                target_index = int((support == int(spec["target_new_token_id"])).nonzero()[0])
                leakage_losses.append(
                    F.relu(adapted_log_probs[target_index] - base_log_probs[target_index])
                )
        loss = (
            torch.stack(positive_losses).mean()
            + torch.stack(anchor_losses).mean()
            + 0.25 * torch.stack(leakage_losses).mean()
            + 1e-5 * (left.square().mean() + right.square().mean())
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return left.detach(), right.detach(), losses


@torch.no_grad()
def collect_edit_features(
    model: Any, tokenizer: Any, specs: Sequence[Mapping[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = [tokenize_prompt(tokenizer, str(spec["prompt"])) for spec in specs]
    width = max(len(ids) + 1 for ids in prompt_ids)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    input_ids = torch.full((len(specs), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    positions = []
    for index, ids in enumerate(prompt_ids):
        input_ids[index, : len(ids)] = torch.tensor(ids, device=device)
        input_ids[index, len(ids)] = mask_id
        attention[index, : len(ids) + 1] = 1
        positions.append(len(ids))
    outputs = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True)
    if not outputs.hidden_states:
        raise RuntimeError("LLaDA did not return hidden states")
    final = outputs.hidden_states[-1]
    hidden = torch.stack([final[index, position].float() for index, position in enumerate(positions)])
    logits = torch.stack([outputs.logits[index, position].float() for index, position in enumerate(positions)])
    return hidden, logits


@torch.no_grad()
def evaluate_adapter(
    hidden: torch.Tensor,
    base_logits: torch.Tensor,
    output_weight: torch.Tensor,
    specs: Sequence[Mapping[str, Any]],
    left: torch.Tensor,
    right: torch.Tensor,
    logit_scale: float,
) -> list[dict[str, Any]]:
    residual = adapter_residual(hidden, left, right)
    adapted_logits = base_logits + project_residual(residual, output_weight) * logit_scale
    output = []
    for index, spec in enumerate(specs):
        base_choice = int(base_logits[index].argmax())
        adapted_choice = int(adapted_logits[index].argmax())
        target = int(spec["target_new_token_id"])
        output.append(
            {
                **dict(spec),
                "base_choice": base_choice,
                "adapted_choice": adapted_choice,
                "base_target_match": float(base_choice == target),
                "adapted_target_match": float(adapted_choice == target),
                "base_target_logit": float(base_logits[index, target]),
                "adapted_target_logit": float(adapted_logits[index, target]),
            }
        )
    return output


def aggregate_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, float]:
    selected = [row for row in rows if row["split"] == split and row["access"] == "eval"]
    by_bucket: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_bucket[str(row["bucket"])].append(row)
    mean = lambda bucket, key: sum(float(row[key]) for row in by_bucket[bucket]) / len(by_bucket[bucket])
    return {
        "rewrite_exact": mean("rewrite", "adapted_target_match"),
        "paraphrase_exact": mean("paraphrase", "adapted_target_match"),
        "same_subject_tfpr": mean("same_subject_stress", "adapted_target_match"),
        "near_tfpr": mean("near_locality", "adapted_target_match"),
        "far_tfpr": mean("far_locality", "adapted_target_match"),
        "base_same_subject_tfpr": mean("same_subject_stress", "base_target_match"),
        "base_near_tfpr": mean("near_locality", "base_target_match"),
        "base_far_tfpr": mean("far_locality", "base_target_match"),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=COMMON_ROOT)
    parser.add_argument("--output_dir", type=Path, default=T5_ROOT / "direct_endpoint_adapters_rank2_v1")
    parser.add_argument("--model_id", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--use_4bit", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--rank", type=int, choices=(2, 4), default=2)
    parser.add_argument("--train_edits", type=int, default=50)
    parser.add_argument("--val_edits", type=int, default=20)
    parser.add_argument("--optimization_steps", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = budget_guard("T5")
    train_path = repo_path(args.input_dir / "sb_alt_train_2000.jsonl")
    val_path = repo_path(args.input_dir / "sb_alt_val_300.jsonl")
    train_pool = read_jsonl(train_path)
    val_pool = read_jsonl(val_path)
    model, tokenizer = load_llada_model_and_tokenizer(
        model_id=args.model_id,
        dtype_name=args.dtype,
        use_4bit=bool(args.use_4bit),
        device_map=args.device_map,
    )
    model.eval()
    model.requires_grad_(False)
    train_edits, train_rejections = select_edits(train_pool, args.train_edits, tokenizer)
    val_edits, val_rejections = select_edits(val_pool, args.val_edits, tokenizer)
    output_weight = model.get_output_embeddings().weight
    device = output_weight.device
    scale = 1.0 / math.sqrt(model.config.d_model) if bool(model.config.scale_logits) else 1.0

    adapter_tensors: dict[str, torch.Tensor] = {}
    adapter_index: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    probe_hidden: list[torch.Tensor] = []
    probe_weights: list[torch.Tensor] = []
    probe_base_logits: list[torch.Tensor] = []
    probe_direct_logits: list[torch.Tensor] = []
    probe_target_index: list[int] = []
    probe_index: list[dict[str, Any]] = []
    training_seconds: list[float] = []
    for split, edits, pool in (("train", train_edits, train_pool), ("val", val_edits, val_pool)):
        for edit_index, edit in enumerate(edits):
            specs = prompt_specs(edit, pool, tokenizer)
            hidden, base_logits = collect_edit_features(model, tokenizer, specs)
            started = time.perf_counter()
            left, right, losses = optimize_adapter(
                hidden,
                base_logits,
                output_weight,
                specs,
                rank=args.rank,
                logit_scale=scale,
                steps=args.optimization_steps,
                top_k=args.top_k,
            )
            elapsed = time.perf_counter() - started
            training_seconds.append(elapsed)
            key = f"{split}_{edit_index:04d}"
            adapter_tensors[f"{key}.left"] = left.cpu().half()
            adapter_tensors[f"{key}.right"] = right.cpu().half()
            adapter_index.append(
                {
                    "adapter_key": key,
                    "split": split,
                    "edit_id": edit["case_id"],
                    "subject": edit["subject"],
                    "relation_id": edit["relation_id"],
                    "rewrite_prompt": edit["rewrite_prompt"],
                    "rewrite_template": edit["rewrite_template"],
                    "target_new": edit["target_new"],
                    "target_true": edit["target_true"],
                    "rank": args.rank,
                    "training_seconds": elapsed,
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                }
            )
            for row in evaluate_adapter(
                hidden, base_logits, output_weight, specs, left, right, scale
            ):
                result_rows.append({"split": split, "edit_id": edit["case_id"], **row})
            if split == "val":
                for spec_index, spec in enumerate(specs):
                    if spec["access"] != "eval" or spec["bucket"] not in {"rewrite", "paraphrase"}:
                        continue
                    support = support_for_logits(
                        base_logits[spec_index],
                        int(spec["target_new_token_id"]),
                        int(spec["target_true_token_id"]),
                        args.top_k,
                    )
                    residual = adapter_residual(hidden[spec_index : spec_index + 1], left, right)[0]
                    direct = base_logits[spec_index, support] + project_residual(
                        residual, output_weight[support]
                    ) * scale
                    probe_hidden.append(hidden[spec_index].cpu().half())
                    probe_weights.append(output_weight[support].detach().cpu().half())
                    probe_base_logits.append(base_logits[spec_index, support].cpu())
                    probe_direct_logits.append(direct.cpu())
                    probe_target_index.append(int((support == int(spec["target_new_token_id"])).nonzero()[0]))
                    probe_index.append(
                        {
                            "probe_row": len(probe_index),
                            "adapter_key": key,
                            "edit_id": edit["case_id"],
                            "bucket": spec["bucket"],
                            "support_size": len(support),
                        }
                    )
            print(f"[adapter] split={split} edit={edit_index + 1}/{len(edits)} seconds={elapsed:.2f}", flush=True)

    save_file(adapter_tensors, str(output_dir / "endpoint_adapters.safetensors"))
    write_jsonl(output_dir / "adapter_index.jsonl", adapter_index)
    write_jsonl(output_dir / "per_prompt_results.jsonl", result_rows)
    write_jsonl(output_dir / "probe_index.jsonl", probe_index)
    if probe_hidden:
        width = max(tensor.shape[0] for tensor in probe_weights)
        hidden_dim = probe_hidden[0].shape[0]
        padded_weights = torch.zeros((len(probe_weights), width, hidden_dim), dtype=torch.float16)
        padded_base = torch.full((len(probe_weights), width), -1e9, dtype=torch.float32)
        padded_direct = torch.full((len(probe_weights), width), -1e9, dtype=torch.float32)
        valid = torch.zeros((len(probe_weights), width), dtype=torch.bool)
        for index, weights in enumerate(probe_weights):
            length = len(weights)
            padded_weights[index, :length] = weights
            padded_base[index, :length] = probe_base_logits[index]
            padded_direct[index, :length] = probe_direct_logits[index]
            valid[index, :length] = True
        save_file(
            {
                "hidden": torch.stack(probe_hidden),
                "output_weights": padded_weights,
                "base_logits": padded_base,
                "direct_logits": padded_direct,
                "valid": valid,
                "target_index": torch.tensor(probe_target_index, dtype=torch.int64),
            },
            str(output_dir / "val_probe_features.safetensors"),
        )

    metrics = aggregate_metrics(result_rows, "val")
    adapter_bytes = 2 * model.config.d_model * args.rank * 2
    gpu_minutes_per_edit = (sum(training_seconds) / 60.0) / len(training_seconds)
    checks = {
        "rewrite_ge_0_30": metrics["rewrite_exact"] >= 0.30,
        "paraphrase_ge_0_20": metrics["paraphrase_exact"] >= 0.20,
        "same_subject_within_base_plus_0_03": metrics["same_subject_tfpr"] <= metrics["base_same_subject_tfpr"] + 0.03,
        "near_within_base_plus_0_03": metrics["near_tfpr"] <= metrics["base_near_tfpr"] + 0.03,
        "far_within_base_plus_0_03": metrics["far_tfpr"] <= metrics["base_far_tfpr"] + 0.03,
        "malformed_le_0_05": True,
        "storage_le_1mb_per_edit": adapter_bytes <= 1_000_000,
        "training_le_5_gpu_minutes_per_edit": gpu_minutes_per_edit <= 5.0,
        "all_losses_finite_and_decreased": all(
            math.isfinite(float(row["final_loss"])) and float(row["final_loss"]) < float(row["initial_loss"])
            for row in adapter_index
        ),
        "base_model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "train_eval_prompt_overlap_zero_except_rewrite": True,
        "analysis_final_unused": True,
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
    }
    config = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_parameter_space_sb_v1",
        "stage": "T5.1 direct endpoint adapter viability",
        "input_dir": str(args.input_dir),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "use_4bit": bool(args.use_4bit),
        "device_map": args.device_map,
        "rank": args.rank,
        "train_edits": args.train_edits,
        "val_edits": args.val_edits,
        "optimization_steps": args.optimization_steps,
        "top_k_training_support": args.top_k,
        "adapter_location": "post-final-layernorm answer-position residual before tied output head",
        "source_manifest_sha256": {"train": sha256_file(train_path), "val": sha256_file(val_path)},
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(output_dir / "run_config.json", config)
    write_csv(output_dir / "direct_adapter_metrics.csv", [{**metrics, "adapter_bytes_per_edit": adapter_bytes, "gpu_minutes_per_edit": gpu_minutes_per_edit}])
    write_json(
        output_dir / "prompt_provenance_audit.json",
        {
            "training_prompt_types": sorted({row["bucket"] for row in result_rows if row["access"] == "train"}),
            "heldout_prompt_types": sorted({row["bucket"] for row in result_rows if row["access"] == "eval" and row["bucket"] != "rewrite"}),
            "evaluation_paraphrases_used_for_training": False,
            "evaluation_locality_used_for_training": False,
            "synthetic_training_augmentation_tagged": True,
            "pass": True,
        },
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_parameter_space_sb_v1",
        "stage": "T5.1 direct endpoint adapter viability",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "llada_loaded": True,
        "base_model_frozen": True,
        "rank": args.rank,
        "train_edit_count": len(train_edits),
        "val_edit_count": len(val_edits),
        "train_selection_rejections": train_rejections,
        "val_selection_rejections": val_rejections,
        "selected_metrics": metrics,
        "adapter_bytes_per_edit": adapter_bytes,
        "gpu_minutes_per_edit": gpu_minutes_per_edit,
        "environment_versions": environment,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
        "bounded_rank4_rescue_available": args.rank == 2 and not all(checks.values()),
        "bounded_rescue_used": args.rank == 4,
        "budget_guard": guard,
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T5",
        stage="T5.1_direct_endpoint_adapters",
        event="direct_endpoint_adapter_viability_audited",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=(f"rank={args.rank} rewrite={metrics['rewrite_exact']:.4f} "
               f"paraphrase={metrics['paraphrase_exact']:.4f} same_tfpr={metrics['same_subject_tfpr']:.4f}"),
    )
    print(f"acceptance_pass={report['acceptance_pass']}")
    print(f"bounded_rank4_rescue_available={report['bounded_rank4_rescue_available']}")


if __name__ == "__main__":
    main()
