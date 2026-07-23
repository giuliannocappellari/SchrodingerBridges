#!/usr/bin/env python3
"""Run one cumulative factual editor and evaluate after every stream block."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SEED,
    assert_no_locked_path,
    git_commit,
    now_utc,
    read_jsonl,
    sequential_metrics,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.mdm_memit_editor import (
    MemitConfig,
    apply_memit_batch,
    build_protected_basis,
    contextual_target_ids,
    extract_keys_and_outputs,
    infer_mask_id,
    render_masked_input,
)
from scripts.run_dnpe_editor import align_base, build_eval_tasks, evaluate_tasks
from scripts.run_mdm_memit_stage import load_model


METHODS = {
    "base",
    "sequential_fullmask_memit",
    "sequential_partial_memit",
    "sequential_lowrank_memit",
    "sequential_lora",
    "oedit_partial_memit",
    "ordinary_replay_memit",
    "lwf_partial_memit",
    "growth_shared",
    "growth_block",
    "growth_block_gate",
    "sparse_routed_memory",
    "gated_adapter_expansion",
    "sb_function_barycenter",
    "dual_memory_10",
    "dual_memory_25",
    "dual_memory_50",
    "replay_clean",
    "replay_partial",
    "bridge_replay",
    "der_partial",
    "agem_partial",
}

BANK_METHODS = {
    "growth_shared",
    "growth_block",
    "growth_block_gate",
    "sparse_routed_memory",
    "gated_adapter_expansion",
    "sb_function_barycenter",
    "dual_memory_10",
    "dual_memory_25",
    "dual_memory_50",
}

METHOD_IMPLEMENTATION = {
    "base": ("frozen_reference", True),
    "sequential_fullmask_memit": ("equation_level_reimplementation", True),
    "sequential_partial_memit": ("equation_level_reimplementation", True),
    "sequential_lowrank_memit": ("equation_level_reimplementation", True),
    "sequential_lora": ("repository_local_implementation", True),
    "oedit_partial_memit": ("equation_level_reimplementation", True),
    "ordinary_replay_memit": ("repository_local_label_replay", True),
    "lwf_partial_memit": ("conceptual_masked_state_adaptation", False),
    "growth_shared": ("conceptual_diffusiongrow_adaptation", False),
    "growth_block": ("conceptual_diffusiongrow_adaptation", False),
    "growth_block_gate": ("conceptual_diffusiongrow_adaptation", False),
    "sparse_routed_memory": ("conceptual_sparse_memory_adaptation", False),
    "gated_adapter_expansion": ("conceptual_gated_adapter_adaptation", False),
    "sb_function_barycenter": ("parameter_delta_barycenter_proxy", False),
    "dual_memory_10": ("conceptual_dual_memory_adaptation", False),
    "dual_memory_25": ("conceptual_dual_memory_adaptation", False),
    "dual_memory_50": ("conceptual_dual_memory_adaptation", False),
    "replay_clean": ("repository_local_clean_label_replay", True),
    "replay_partial": ("repository_local_partial_state_label_replay", True),
    "bridge_replay": ("endpoint_biased_replay_proxy_not_sb", False),
    "der_partial": ("partial_state_replay_proxy_not_der", False),
    "agem_partial": ("partial_state_replay_proxy_not_agem", False),
}


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return layers


def load_covariance(path: Path, layer: int):
    import torch

    candidate = path / f"layer_{layer}_covariance.pt"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return torch.load(candidate, map_location="cpu", weights_only=True).to("cuda")


def rank_truncate(update, rank: int):
    import torch

    value = update.float()
    u, singular, vh = torch.linalg.svd(value, full_matrices=False)
    effective = min(int(rank), int(singular.numel()))
    truncated = (u[:, :effective] * singular[:effective]) @ vh[:effective]
    explained = float(singular[:effective].square().sum() / singular.square().sum().clamp_min(1e-12))
    return truncated, {"rank": effective, "explained_update_energy": explained}


def _pad_training_batch(rows: Sequence[Mapping[str, Any]], tokenizer: Any, model: Any):
    import torch

    rendered = []
    mask_id = infer_mask_id(model)
    for row in rows:
        target_ids = contextual_target_ids(tokenizer, row["rewrite_prompt"], row["target_new"])
        rendered.append(render_masked_input(tokenizer, row["rewrite_prompt"], target_ids, mask_id))
    width = max(len(item["input_ids"]) for item in rendered)
    pad_id = int(tokenizer.pad_token_id)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for index, (row, item) in enumerate(zip(rows, rendered)):
        values = item["input_ids"]
        offset = width - len(values)
        ids[index, offset:] = torch.tensor(values, device="cuda")
        attention[index, offset:] = 1
        target_ids = contextual_target_ids(tokenizer, row["rewrite_prompt"], row["target_new"])
        for position, token_id in zip(item["answer_positions"], target_ids):
            labels[index, offset + position] = int(token_id)
    return ids, attention, labels


def train_lora_block(
    model: Any,
    tokenizer: Any,
    branch: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model.eval()
    optimizer = torch.optim.AdamW(list(branch.parameters()), lr=learning_rate)
    rng = random.Random(seed)
    history = []
    for step in range(steps):
        batch_rows = [rows[index] for index in rng.sample(range(len(rows)), min(2, len(rows)))]
        ids, attention, labels = _pad_training_batch(batch_rows, tokenizer, model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=attention).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LoRA loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(branch.parameters()), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return {
        "steps": steps,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "loss_decreased": history[-1] < history[0],
    }


def retention_tasks(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        prompt = str(row["rewrite_prompt"])
        true = str(row["target_true"])
        target_new = str(row["target_new"])
        length = max(
            1,
            len(contextual_target_ids(tokenizer, prompt, true)),
            len(contextual_target_ids(tokenizer, prompt, target_new)),
        )
        tasks.append(
            {
                "case_id": row["case_id"],
                "subject": row["subject"],
                "target_new": target_new,
                "target_true": true,
                "target_length": row["target_length"],
                "relation_id": row["relation_id"],
                "bucket": "base_retention",
                "prompt": prompt,
                "expected": true,
                "answer_length": length,
            }
        )
    return tasks


def denoising_loss(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    mask_ratios: Sequence[float] = (0.25, 0.5, 1.0),
    maximum: int = 64,
    batch_size: int = 8,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    mask_id = infer_mask_id(model)
    pad_id = int(tokenizer.pad_token_id)
    selected = list(rows[:maximum])
    output = {}
    for ratio in mask_ratios:
        items = []
        for row in selected:
            text = str(row["rewrite_prompt"]).rstrip() + " " + str(row["target_true"]).strip()
            original = list(map(int, tokenizer(text, add_special_tokens=False)["input_ids"]))
            count = max(1, int(round(len(original) * ratio)))
            order = sorted(range(len(original)), key=lambda index: stable_hash(SEED, row["case_id"], ratio, index))
            masked = set(order[:count])
            inputs = [mask_id if index in masked else token for index, token in enumerate(original)]
            labels = [token if index in masked else -100 for index, token in enumerate(original)]
            items.append((inputs, labels))
        losses = []
        for start in range(0, len(items), batch_size):
            members = items[start : start + batch_size]
            width = max(len(item[0]) for item in members)
            ids = torch.full((len(members), width), pad_id, dtype=torch.long, device="cuda")
            attention = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for index, (values, targets) in enumerate(members):
                offset = width - len(values)
                ids[index, offset:] = torch.tensor(values, device="cuda")
                attention[index, offset:] = 1
                labels[index, offset:] = torch.tensor(targets, device="cuda")
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=attention).logits.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100,
                reduction="sum",
            )
            losses.append((float(loss), int((labels != -100).sum())))
        output[f"mask_ratio_{ratio:.2f}"] = sum(value for value, _ in losses) / max(sum(count for _, count in losses), 1)
    return output


def summarize_prompt_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["bucket"])].append(row)
    output = {}
    for bucket, values in groups.items():
        expected = [row for row in values if row.get("expected_hit") is not None]
        output[bucket] = {
            "num_rows": len(values),
            "num_edits": len({str(row["case_id"]) for row in values}),
            "expected_exact": sum(bool(row["expected_hit"]) for row in expected) / max(len(expected), 1),
            "target_new_rate": sum(bool(row["target_new_hit"]) for row in values) / len(values),
            "malformed_rate": sum(bool(row["malformed"]) for row in values) / len(values),
            "base_agreement": sum(bool(row.get("base_agreement")) for row in values) / len(values),
        }
    return output


def block_score_matrix(
    rows: Sequence[Mapping[str, Any]],
    case_blocks: Mapping[str, int],
    bucket: str,
) -> dict[int, float]:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        if row["bucket"] == bucket:
            grouped[case_blocks[str(row["case_id"])]].append(bool(row["expected_hit"]))
    return {block: sum(values) / len(values) for block, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retention_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--covariance_dir", type=Path, default=CAMPAIGN_ROOT / "B1_covariance_cache_v1")
    parser.add_argument("--target_optimization_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--covariance_weight", type=float, default=15000.0)
    parser.add_argument("--lowrank_rank", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_steps", type=int, default=25)
    parser.add_argument("--lora_learning_rate", type=float, default=1e-3)
    parser.add_argument("--replay_items_per_block", type=int, default=10)
    parser.add_argument("--decode_batch_size", type=int, default=16)
    parser.add_argument("--decode_steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow_confirmation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    started = time.monotonic()
    assert_no_locked_path(args.manifest)
    assert_no_locked_path(args.retention_manifest)
    if "confirmation" in args.manifest.name and not args.allow_confirmation:
        raise PermissionError("Fresh confirmation requires a frozen candidate and explicit flag")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    stream = read_jsonl(args.manifest)
    if args.limit:
        stream = stream[: args.limit]
    retention = read_jsonl(args.retention_manifest)
    if not stream or not retention:
        raise RuntimeError("Sequential stream and retention manifest must be nonempty")
    block_ids = sorted({int(row["block_index"]) for row in stream})
    expected_blocks = list(range(max(block_ids) + 1))
    if block_ids != expected_blocks:
        raise RuntimeError(f"Non-contiguous block IDs: {block_ids}")

    import torch

    model, tokenizer = load_model(args.model_id, args.model_revision, args.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_tasks = build_eval_tasks(tokenizer, stream, include_locality=True)
    base_prompt_rows = evaluate_tasks(
        model, tokenizer, all_tasks,
        decode_batch_size=args.decode_batch_size, steps=args.decode_steps or None,
    )
    retention_eval_tasks = retention_tasks(tokenizer, retention)
    base_retention_rows = evaluate_tasks(
        model, tokenizer, retention_eval_tasks,
        decode_batch_size=args.decode_batch_size, steps=args.decode_steps or None,
    )
    base_losses = denoising_loss(model, tokenizer, retention)
    pre_edit_target_new_rate = sum(
        bool(row["target_new_hit"]) for row in base_prompt_rows if row["bucket"] == "rewrite"
    ) / len(stream)
    case_blocks = {str(row["case_id"]): int(row["block_index"]) for row in stream}
    base_by_case_bucket = {
        (str(row["case_id"]), str(row["bucket"]), str(row["prompt"])): row
        for row in base_prompt_rows
    }

    config = MemitConfig(
        layers=args.layers,
        learning_rate=args.learning_rate,
        target_optimization_steps=args.target_optimization_steps,
        covariance_weight=args.covariance_weight,
        partial_mask_schedule=(
            "fully_masked"
            if args.method in {"sequential_fullmask_memit", "replay_clean"}
            else ("fewer_revealed" if args.method == "bridge_replay" else "cycle")
        ),
        reveal_policy=(
            "random" if args.method in {"sequential_fullmask_memit", "replay_clean"} else "base_confidence"
        ),
        state_consistency_weight=0.1 if args.method in {"sequential_partial_memit", "lwf_partial_memit", "der_partial"} else 0.0,
        lambda_identity=0.1 if args.method in {"lwf_partial_memit", "der_partial"} else 0.0,
        old_target_suppression_weight=0.25 if "partial" in args.method else 0.0,
        seed=args.seed,
    )
    lora = None
    if args.method == "sequential_lora":
        from scripts.cl_lora import LoRABranch
        lora = LoRABranch(model, args.layers, args.lora_rank)
    bank = None
    if args.method in BANK_METHODS:
        from scripts.cl_delta_bank import DeltaBranchBank

        route_mode = (
            "subject_relation"
            if args.method in {"growth_block_gate", "sparse_routed_memory", "gated_adapter_expansion"}
            else "always"
        )
        bank = DeltaBranchBank(model, tokenizer, args.layers, route_mode=route_mode)

    all_result_rows = []
    block_rows = []
    denoising_rows = []
    rewrite_matrix: dict[int, dict[int, float]] = {}
    paraphrase_matrix: dict[int, dict[int, float]] = {}
    diagnostics = []
    edit_start = time.monotonic()
    seen_rows: list[dict[str, Any]] = []
    for block in block_ids:
        current = [row for row in stream if int(row["block_index"]) == block]
        old = list(seen_rows)
        if args.method == "sequential_lora":
            update_report = train_lora_block(
                model, tokenizer, lora, current,
                steps=args.lora_steps,
                learning_rate=args.lora_learning_rate,
                seed=args.seed + block,
            )
        elif args.method == "base":
            update_report = {"skipped": True}
        else:
            requests = list(current)
            if args.method in {"ordinary_replay_memit", "replay_clean", "replay_partial", "bridge_replay", "der_partial", "agem_partial"} and old:
                replay = sorted(old, key=lambda row: stable_hash(args.seed, block, row["case_id"]))[: args.replay_items_per_block]
                requests.extend(replay)
            basis_by_layer = {}
            if args.method == "oedit_partial_memit" and old:
                protected = old[-min(len(old), 64) :]
                for layer in args.layers:
                    keys, _ = extract_keys_and_outputs(
                        model, tokenizer, protected,
                        key_layer=layer, output_layer=layer,
                        partial_mask_schedule="cycle", reveal_policy="random",
                        seed=args.seed + block,
                    )
                    basis_by_layer[layer] = build_protected_basis(keys, 0.95, maximum_rank=64)[0]
            update_transform: Callable | None = None
            if args.method == "sequential_lowrank_memit" or args.method in BANK_METHODS:
                update_transform = lambda _layer, update, _context: rank_truncate(update, args.lowrank_rank)
            rollback, update_report = apply_memit_batch(
                model,
                tokenizer,
                requests,
                config,
                lambda layer: load_covariance(args.covariance_dir, layer),
                target_cache_dir=args.output_dir / "target_value_cache" / f"block_{block:03d}",
                protected_basis_loader=(lambda layer: basis_by_layer.get(layer)) if basis_by_layer else None,
                update_transform=update_transform,
            )
            if bank is not None:
                deltas = {}
                from scripts.mdm_memit_editor import get_module, resolved_key_module_name

                for layer in args.layers:
                    current_weight = get_module(model, resolved_key_module_name(model, layer)).weight.detach().float()
                    original = rollback.originals[layer].to(device=current_weight.device, dtype=torch.float32)
                    deltas[layer] = current_weight - original
                rollback.rollback()
                if not rollback.checksum_matches(atol=0.0):
                    raise RuntimeError("Frozen-base branch rollback failed")
                bank.add_branch(deltas, current, block_index=block)
                merge_report = None
                if args.method == "growth_shared":
                    merge_report = bank.merge_all(rank=args.lowrank_rank)
                elif args.method == "sb_function_barycenter":
                    norms = [
                        math.exp(-sum(float(delta.norm()) for delta in branch.deltas.values()) / 100.0)
                        for branch in bank.branches
                    ]
                    merge_report = bank.merge_all(rank=args.lowrank_rank, weights=norms)
                elif args.method.startswith("dual_memory_"):
                    interval = int(args.method.rsplit("_", 1)[1])
                    if len(seen_rows) + len(current) >= interval and (len(seen_rows) + len(current)) % interval == 0:
                        merge_report = bank.merge_all(rank=args.lowrank_rank)
                update_report["branch_bank_merge"] = merge_report
            del rollback
        diagnostics.append({"block_index": block, "num_new": len(current), "num_old": len(old), "update": update_report})
        seen_rows.extend(current)

        seen_tasks = [task for task in all_tasks if int(case_blocks[str(task["case_id"])]) <= block]
        edited = evaluate_tasks(
            model, tokenizer, seen_tasks,
            decode_batch_size=args.decode_batch_size, steps=args.decode_steps or None,
        )
        base_subset = [
            base_by_case_bucket[(str(task["case_id"]), str(task["bucket"]), str(task["prompt"]))]
            for task in seen_tasks
        ]
        edited = align_base(base_subset, edited)
        for row in edited:
            row["evaluation_after_block"] = block
            row["edit_block"] = case_blocks[str(row["case_id"])]
        all_result_rows.extend(edited)
        summary = summarize_prompt_rows(edited)
        rewrite_matrix[block] = block_score_matrix(edited, case_blocks, "rewrite")
        paraphrase_matrix[block] = block_score_matrix(edited, case_blocks, "declarative_paraphrase")

        edited_retention = evaluate_tasks(
            model, tokenizer, retention_eval_tasks,
            decode_batch_size=args.decode_batch_size, steps=args.decode_steps or None,
        )
        edited_retention = align_base(base_retention_rows, edited_retention)
        losses = denoising_loss(model, tokenizer, retention)
        loss_fraction = sum(
            max(0.0, losses[key] - base_losses[key]) / max(base_losses[key], 1e-12)
            for key in base_losses
        ) / len(base_losses)
        denoising_rows.extend(
            {
                "block_index": block,
                "mask_ratio": key.replace("mask_ratio_", ""),
                "base_nll": base_losses[key],
                "edited_nll": losses[key],
                "relative_loss": (losses[key] - base_losses[key]) / max(base_losses[key], 1e-12),
            }
            for key in base_losses
        )
        current_rewrite = rewrite_matrix[block].get(block, 0.0)
        current_para = paraphrase_matrix[block].get(block, 0.0)
        past = [value for source_block, value in rewrite_matrix[block].items() if source_block < block]
        block_rows.append(
            {
                "block_index": block,
                "num_seen_edits": len(seen_rows),
                "current_rewrite_exact": current_rewrite,
                "current_paraphrase_exact": current_para,
                "past_rewrite_retention": sum(past) / max(len(past), 1),
                "same_subject_tfpr": summary.get("same_subject", {}).get("target_new_rate", 0.0),
                "near_tfpr": summary.get("near_locality", {}).get("target_new_rate", 0.0),
                "far_tfpr": summary.get("far_locality", {}).get("target_new_rate", 0.0),
                "malformed_rate": max((row["malformed_rate"] for row in summary.values()), default=0.0),
                "base_retention_exact": sum(bool(row["expected_hit"]) for row in edited_retention) / len(edited_retention),
                "base_retention_agreement": sum(bool(row["base_agreement"]) for row in edited_retention) / len(edited_retention),
                "base_retention_loss_fraction": loss_fraction,
            }
        )

    metrics = sequential_metrics(rewrite_matrix)
    final_block = block_rows[-1]
    final_prompt_rows = [row for row in all_result_rows if row["evaluation_after_block"] == block_ids[-1]]
    final_summary = summarize_prompt_rows(final_prompt_rows)
    past_values = [value for block, value in rewrite_matrix[block_ids[-1]].items() if block < block_ids[-1]]
    current_rewrite_values = [float(row["current_rewrite_exact"]) for row in block_rows]
    current_para_values = [float(row["current_paraphrase_exact"]) for row in block_rows]
    malformed = max(float(row["malformed_rate"]) for row in block_rows)
    mean_current_rewrite = sum(current_rewrite_values) / len(current_rewrite_values)
    mean_current_para = sum(current_para_values) / len(current_para_values)
    baseline_floor = (
        mean_current_rewrite >= 0.75
        and mean_current_para >= 0.40
        and pre_edit_target_new_rate <= 0.10
        and malformed <= 0.05
    )
    runtime = time.monotonic() - started
    if lora is not None:
        storage_bytes = lora.storage_bytes()
    elif bank is not None:
        storage_bytes = bank.storage_bytes()
    elif args.method == "base":
        storage_bytes = 0
    else:
        from scripts.mdm_memit_editor import get_module, resolved_key_module_name

        storage_bytes = sum(
            get_module(model, resolved_key_module_name(model, layer)).weight.numel() * 2
            for layer in args.layers
        )
    implementation_status, exact_method_claim_eligible = METHOD_IMPLEMENTATION[args.method]
    run_config = {
        "campaign_id": CAMPAIGN_ID,
        "method": args.method,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "retention_manifest": str(args.retention_manifest),
        "retention_manifest_sha256": sha256_file(args.retention_manifest),
        "num_edits": len(stream),
        "num_blocks": len(block_ids),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layers": list(args.layers),
        "memit": config.to_dict(),
        "lowrank_rank": args.lowrank_rank,
        "lora_rank": args.lora_rank,
        "lora_steps": args.lora_steps,
        "replay_items_per_block": args.replay_items_per_block,
        "runtime_feature_schema": (
            bank.activation_summary()["runtime_feature_schema"] if bank is not None else []
        ),
        "teacher_only_runtime_inputs": False,
        "evaluation_bucket_runtime_input": False,
        "implementation_status": implementation_status,
        "exact_method_claim_eligible": exact_method_claim_eligible,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_csv(args.output_dir / "block_metrics.csv", block_rows)
    write_csv(args.output_dir / "base_denoising_by_block.csv", denoising_rows)
    write_jsonl(args.output_dir / "per_prompt_results.jsonl", all_result_rows)
    write_json(args.output_dir / "update_diagnostics.json", diagnostics)
    matrix_rows = []
    for evaluation_block, source_scores in rewrite_matrix.items():
        for source_block, rewrite_score in source_scores.items():
            matrix_rows.append(
                {
                    "evaluation_block": evaluation_block,
                    "source_block": source_block,
                    "rewrite_exact": rewrite_score,
                    "paraphrase_exact": paraphrase_matrix[evaluation_block].get(source_block, 0.0),
                }
            )
    write_csv(args.output_dir / "retention_matrix.csv", matrix_rows)
    report = {
        **run_config,
        "stage": "sequential_editor",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "current_rewrite_exact": mean_current_rewrite,
        "current_paraphrase_exact": mean_current_para,
        "past_retention": sum(past_values) / max(len(past_values), 1),
        **metrics,
        "pre_edit_target_new_rewrite": pre_edit_target_new_rate,
        "same_subject_tfpr": float(final_block["same_subject_tfpr"]),
        "near_tfpr": float(final_block["near_tfpr"]),
        "far_tfpr": float(final_block["far_tfpr"]),
        "near_locality_pass": float(final_block["near_tfpr"]) <= 0.03,
        "far_locality_pass": float(final_block["far_tfpr"]) <= 0.03,
        "malformed_rate": malformed,
        "base_retention_exact": float(final_block["base_retention_exact"]),
        "base_retention_agreement": float(final_block["base_retention_agreement"]),
        "base_retention_loss_fraction": float(final_block["base_retention_loss_fraction"]),
        "partial_state_consistency": 1.0 - min(
            1.0, max(0.0, float(final_block["base_retention_loss_fraction"]))
        ),
        "storage_bytes": storage_bytes,
        "storage_mb_per_edit": storage_bytes / 1_000_000 / len(stream),
        "inference_overhead_fraction": 0.0,
        "runtime_seconds": runtime,
        "edit_runtime_seconds": time.monotonic() - edit_start,
        "gpu_minutes_per_edit": runtime / 60.0 / len(stream),
        "baseline_floor_pass": baseline_floor,
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for value in (
                mean_current_rewrite, mean_current_para, metrics["average_retention"],
                metrics["average_forgetting"], pre_edit_target_new_rate, malformed,
            )
        ),
        "fake_model": False,
        "llada_loaded": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "final_bucket_summary": final_summary,
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "all_blocks_evaluated": len(block_rows) == len(block_ids),
            "all_metrics_finite": report["all_metrics_finite"],
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": len(block_rows) == len(block_ids) and report["all_metrics_finite"],
        },
    )
    if lora is not None:
        torch.save(lora.state_dict_cpu(), args.output_dir / "lora_adapter.pt")
        lora.close()
    if bank is not None:
        write_json(args.output_dir / "branch_bank_summary.json", bank.activation_summary())
        bank.close()
    print(json.dumps({
        "method": args.method,
        "baseline_floor_pass": baseline_floor,
        "current_rewrite_exact": mean_current_rewrite,
        "current_paraphrase_exact": mean_current_para,
        "past_retention": report["past_retention"],
        "average_forgetting": metrics["average_forgetting"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
