#!/usr/bin/env python3
"""Causal tracing and preservation-subspace primitives for DNPE."""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from scripts.mdm_memit_editor import (
    find_subject_token_span,
    get_module,
    infer_mask_id,
    model_device,
    output_hidden,
    pad_batch,
    replace_output_hidden,
    resolved_block_name,
    resolved_key_module_name,
)


def explicit_partial_state(
    target_ids: Sequence[int],
    revealed_positions: Sequence[int],
    mask_id: int,
) -> tuple[list[int], list[int]]:
    revealed = set(map(int, revealed_positions))
    if any(index < 0 or index >= len(target_ids) for index in revealed):
        raise ValueError("revealed position is outside the target span")
    state = [int(token) if index in revealed else int(mask_id) for index, token in enumerate(target_ids)]
    supervised = [index for index in range(len(target_ids)) if index not in revealed]
    return state, supervised


def state_bank(
    target_ids: Sequence[int],
    *,
    policy: str,
    seed: int,
) -> list[dict[str, Any]]:
    n = len(target_ids)
    if n <= 0:
        raise ValueError("target_ids must be non-empty")
    rng = random.Random(seed)
    if policy == "fully_masked_only":
        subsets = [()]
    elif policy in {"all_mask_counts_random_positions", "uniform_mask_count_states"}:
        subsets = [tuple(sorted(rng.sample(range(n), k))) for k in range(n)]
    elif policy == "confidence_trajectory_states":
        # Deterministic stand-in ordering; real runs replace ordering with base
        # confidence while preserving one state for every mask count.
        order = list(range(n))
        rng.shuffle(order)
        subsets = [tuple(sorted(order[:k])) for k in range(n)]
    elif policy == "three_bucket_states":
        subsets = [()]
        if n > 2:
            subsets.append(tuple(sorted(rng.sample(range(n), max(1, (n - 1) // 2)))))
        if n > 1:
            subsets.append(tuple(sorted(rng.sample(range(n), n - 1))))
    else:
        raise ValueError(f"Unsupported state policy: {policy}")
    unique = []
    seen = set()
    for subset in subsets:
        if subset in seen:
            continue
        seen.add(subset)
        unique.append(
            {
                "revealed_positions": list(subset),
                "revealed_count": len(subset),
                "masked_count": n - len(subset),
            }
        )
    return unique


def geometric_target_probability(
    logits: torch.Tensor,
    answer_positions: Sequence[int],
    target_ids: Sequence[int],
) -> float:
    terms = []
    for position, token_id in zip(answer_positions, target_ids):
        terms.append(F.log_softmax(logits[int(position)].float(), dim=-1)[int(token_id)])
    if not terms:
        raise ValueError("No target probability terms")
    return float(torch.exp(torch.stack(terms).mean()))


def normalized_aie(clean: float, corrupted: float, restored: float, eps: float = 1e-8) -> float:
    values = (clean, corrupted, restored)
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("AIE inputs must be finite")
    denominator = max(clean - corrupted, eps)
    value = float((restored - corrupted) / denominator)
    return max(-1.0, min(1.0, value))


@contextlib.contextmanager
def corrupt_subject_embeddings(
    model: torch.nn.Module,
    subject_positions: Sequence[int],
    *,
    noise_scale: float,
    seed: int,
) -> Iterator[None]:
    embedding = model.get_input_embeddings()

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        edited = output.clone()
        generator = torch.Generator(device=edited.device)
        generator.manual_seed(int(seed))
        reference = edited[:, list(subject_positions)].float()
        std = reference.std().clamp_min(1e-6)
        noise = torch.randn(
            reference.shape,
            generator=generator,
            device=edited.device,
            dtype=torch.float32,
        ) * (float(noise_scale) * std)
        edited[:, list(subject_positions)] = (
            reference + noise
        ).to(dtype=edited.dtype)
        return edited

    handle = embedding.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def capture_module_output(module: torch.nn.Module, box: list[torch.Tensor]) -> Iterator[None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        box.append(output_hidden(output).detach().clone())

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def restore_module_position(
    module: torch.nn.Module,
    clean_output: torch.Tensor,
    position: int,
) -> Iterator[None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output_hidden(output)
        restored = hidden.clone()
        restored[:, int(position)] = clean_output[:, int(position)].to(restored)
        return replace_output_hidden(output, restored)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def component_module_name(model: torch.nn.Module, layer: int, component: str) -> str:
    if component == "hidden":
        return resolved_block_name(model, layer)
    block = resolved_block_name(model, layer)
    if component == "mlp":
        return f"{block}.ff_out"
    if component == "attention":
        return f"{block}.attn_out"
    raise ValueError(f"Unknown component: {component}")


@torch.no_grad()
def trace_single_coordinate(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    layer: int,
    component: str,
    position_name: str,
    noise_scale: float = 3.0,
    seed: int = 0,
) -> dict[str, Any]:
    prompt = str(row["rewrite_prompt"])
    target_ids = list(map(int, row["target_true_token_ids"]))
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    mask_id = infer_mask_id(model)
    ids = torch.tensor([prompt_ids + [mask_id] * len(target_ids)], dtype=torch.long, device=model_device(model))
    answer_positions = list(range(len(prompt_ids), len(prompt_ids) + len(target_ids)))
    first_subject, last_subject = find_subject_token_span(tokenizer, prompt, str(row["subject"]))
    positions = {
        "first_subject": first_subject,
        "last_subject": last_subject,
        "relation_cue": max(last_subject + 1, len(prompt_ids) - 1),
        "first_answer_mask": answer_positions[0],
    }
    if position_name not in positions:
        raise ValueError(f"Unknown position: {position_name}")
    subject_positions = list(range(first_subject, last_subject + 1))
    module = get_module(model, component_module_name(model, layer, component))
    clean_box: list[torch.Tensor] = []
    with capture_module_output(module, clean_box):
        clean_logits = model(input_ids=ids).logits[0]
    clean = geometric_target_probability(clean_logits, answer_positions, target_ids)
    with corrupt_subject_embeddings(model, subject_positions, noise_scale=noise_scale, seed=seed):
        corrupted_logits = model(input_ids=ids).logits[0]
    corrupted = geometric_target_probability(corrupted_logits, answer_positions, target_ids)
    with corrupt_subject_embeddings(model, subject_positions, noise_scale=noise_scale, seed=seed):
        with restore_module_position(module, clean_box[0], positions[position_name]):
            restored_logits = model(input_ids=ids).logits[0]
    restored = geometric_target_probability(restored_logits, answer_positions, target_ids)
    return {
        "case_id": row["case_id"],
        "layer": int(layer),
        "component": component,
        "position": position_name,
        "clean_probability": clean,
        "corrupted_probability": corrupted,
        "restored_probability": restored,
        "normalized_aie": normalized_aie(clean, corrupted, restored),
        "noise_scale_rule": f"{noise_scale}x_subject_embedding_std",
    }


def protected_key_drift(update: torch.Tensor, protected_keys: torch.Tensor) -> float:
    if update.ndim != 2 or protected_keys.ndim != 2:
        raise ValueError("update and protected_keys must be matrices")
    if update.shape[1] != protected_keys.shape[1]:
        raise ValueError("key dimensions do not align")
    return float((update.float() @ protected_keys.float().T).norm(dim=0).mean())


@torch.no_grad()
def trace_case_grid(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    layers: Sequence[int],
    components: Sequence[str],
    position_names: Sequence[str],
    noise_scale: float = 3.0,
    seed: int = 0,
    revealed_positions: Sequence[int] = (),
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Trace a full coordinate grid with one shared clean/corrupted pass."""

    prompt = str(row["rewrite_prompt"])
    target_ids = list(map(int, row["target_true_token_ids"]))
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    mask_id = infer_mask_id(model)
    state, supervised_relative = explicit_partial_state(target_ids, revealed_positions, mask_id)
    ids = torch.tensor([prompt_ids + state], dtype=torch.long, device=model_device(model))
    answer_positions = [len(prompt_ids) + index for index in supervised_relative]
    supervised_targets = [target_ids[index] for index in supervised_relative]
    first_subject, last_subject = find_subject_token_span(tokenizer, prompt, str(row["subject"]))
    positions = {
        "first_subject": first_subject,
        "last_subject": last_subject,
        "relation_cue": max(last_subject + 1, len(prompt_ids) - 1),
        "first_answer_mask": len(prompt_ids) + supervised_relative[0],
    }
    requested_positions = {name: positions[name] for name in position_names}
    subject_positions = list(range(first_subject, last_subject + 1))
    modules: dict[tuple[int, str], torch.nn.Module] = {}
    clean_boxes: dict[tuple[int, str], list[torch.Tensor]] = {}
    handles = []
    for layer in layers:
        for component in components:
            key = (int(layer), str(component))
            module = get_module(model, component_module_name(model, layer, component))
            modules[key] = module
            clean_boxes[key] = []

            def make_hook(box: list[torch.Tensor]):
                def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                    box.append(output_hidden(output).detach().clone())

                return hook

            handles.append(module.register_forward_hook(make_hook(clean_boxes[key])))
    try:
        clean_logits = model(input_ids=ids).logits[0]
    finally:
        for handle in handles:
            handle.remove()
    clean = geometric_target_probability(clean_logits, answer_positions, supervised_targets)
    with corrupt_subject_embeddings(model, subject_positions, noise_scale=noise_scale, seed=seed):
        corrupted_logits = model(input_ids=ids).logits[0]
    corrupted = geometric_target_probability(corrupted_logits, answer_positions, supervised_targets)
    rows = []
    for layer in layers:
        for component in components:
            key = (int(layer), str(component))
            for position_name, position in requested_positions.items():
                with corrupt_subject_embeddings(
                    model, subject_positions, noise_scale=noise_scale, seed=seed
                ):
                    with restore_module_position(
                        modules[key], clean_boxes[key][0], int(position)
                    ):
                        restored_logits = model(input_ids=ids).logits[0]
                restored = geometric_target_probability(
                    restored_logits, answer_positions, supervised_targets
                )
                rows.append(
                    {
                        "case_id": row["case_id"],
                        "relation_id": row.get("relation_id"),
                        "target_length": len(target_ids),
                        "revealed_count": len(revealed_positions),
                        "revealed_positions": json_list(revealed_positions),
                        "layer": int(layer),
                        "component": str(component),
                        "position": position_name,
                        "clean_probability": clean,
                        "corrupted_probability": corrupted,
                        "restored_probability": restored,
                        "normalized_aie": normalized_aie(clean, corrupted, restored),
                    }
                )
    return rows, {
        "clean_probability": clean,
        "corrupted_probability": corrupted,
        "corruption_drop": clean - corrupted,
    }


def json_list(values: Sequence[int]) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"
