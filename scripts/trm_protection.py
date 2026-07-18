#!/usr/bin/env python3
"""Training-only protection-anchor construction for temporal residual editing."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

from scripts.mdm_memit_editor import (
    find_last_subject_token,
    get_module,
    infer_mask_id,
    model_device,
    output_hidden,
    pad_batch,
    resolved_key_module_name,
)


REQUIRED_PROTECTION_FAMILIES = (
    "same_subject_different_relation",
    "near_locality",
    "far_locality",
    "attribute",
    "generation",
    "unrelated",
)


def build_protection_prompt_records(
    anchor_rows: Sequence[Mapping[str, Any]],
    *,
    max_per_family: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not anchor_rows:
        raise ValueError("anchor_rows must not be empty")
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        by_relation[str(row["relation_id"])].append(row)
    records: dict[str, list[dict[str, Any]]] = {
        family: [] for family in REQUIRED_PROTECTION_FAMILIES
    }

    def add(
        family: str,
        source: Mapping[str, Any],
        prompt: str,
        *,
        lookup_subject: str | None,
        provenance: str,
    ) -> None:
        if len(records[family]) >= int(max_per_family) or not str(prompt).strip():
            return
        records[family].append(
            {
                "anchor_id": f"{source['case_id']}::{family}::{len(records[family])}",
                "source_case_id": str(source["case_id"]),
                "source_split_role": str(source["split_role"]),
                "relation_id": str(source["relation_id"]),
                "family": family,
                "prompt": str(prompt),
                "lookup_subject": str(lookup_subject or ""),
                "lookup_mode": "last_subject" if lookup_subject else "last_prompt_token",
                "prompt_provenance": provenance,
                "synthetic": False,
            }
        )

    for index, row in enumerate(anchor_rows):
        subject = str(row["subject"])
        same = list(row.get("same_subject_prompts") or [])
        if same:
            add(
                "same_subject_different_relation",
                row,
                same[0],
                lookup_subject=subject,
                provenance="fresh_train_only_cross_relation_template",
            )
        generation = list(row.get("generation_prompts") or [])
        if generation:
            add(
                "generation",
                row,
                generation[0],
                lookup_subject=subject,
                provenance="real_train_only_generation_prompt",
            )
        attributes = list(row.get("attribute_prompts") or [])
        if attributes:
            add(
                "attribute",
                row,
                attributes[0],
                lookup_subject=None,
                provenance="real_train_only_attribute_prompt",
            )
        near_candidates = [
            candidate
            for candidate in by_relation[str(row["relation_id"])]
            if candidate["case_id"] != row["case_id"]
        ]
        if near_candidates:
            donor = near_candidates[0]
            add(
                "near_locality",
                row,
                str(donor["rewrite_prompt"]),
                lookup_subject=str(donor["subject"]),
                provenance="real_train_only_same_relation_fact",
            )
        far_donor = next(
            (
                candidate
                for offset in range(1, len(anchor_rows) + 1)
                for candidate in [anchor_rows[(index + offset) % len(anchor_rows)]]
                if candidate["relation_id"] != row["relation_id"]
            ),
            None,
        )
        if far_donor is not None:
            add(
                "far_locality",
                row,
                str(far_donor["rewrite_prompt"]),
                lookup_subject=str(far_donor["subject"]),
                provenance="real_train_only_different_relation_fact",
            )
        unrelated = anchor_rows[(index + len(anchor_rows) // 2) % len(anchor_rows)]
        add(
            "unrelated",
            row,
            str(unrelated["rewrite_prompt"]),
            lookup_subject=str(unrelated["subject"]),
            provenance="real_train_only_deterministic_unrelated_fact",
        )
    flattened = [record for family in REQUIRED_PROTECTION_FAMILIES for record in records[family]]
    summary = {
        "source_split_roles": sorted({str(row["split_role"]) for row in anchor_rows}),
        "family_counts": {family: len(records[family]) for family in REQUIRED_PROTECTION_FAMILIES},
        "all_required_families_present": all(records[family] for family in REQUIRED_PROTECTION_FAMILIES),
        "synthetic_rows": 0,
        "evaluation_prompts_used": False,
    }
    return flattened, summary


def state_revealed_count(bucket: str, span_length: int) -> int:
    n = max(int(span_length), 1)
    if bucket == "early":
        return 0
    if bucket == "middle":
        return min(n - 1, max(1, n // 2))
    if bucket == "late":
        return n - 1
    raise ValueError(f"unknown protection state bucket: {bucket}")


@torch.no_grad()
def extract_protection_keys(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    state_bucket: str,
    span_length: int = 3,
    batch_size: int = 16,
    seed: int = 260718601,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not records:
        raise ValueError("protection records must not be empty")
    device = model_device(model)
    mask_id = infer_mask_id(model)
    module = get_module(model, resolved_key_module_name(model, int(layer)))
    output_keys: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    revealed_count = state_revealed_count(state_bucket, span_length)
    rng = random.Random(int(seed))
    for start in range(0, len(records), int(batch_size)):
        subset = list(records[start : start + int(batch_size)])
        prompt_ids = [
            list(map(int, tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"]))
            for row in subset
        ]
        full_rows = [
            {"input_ids": values + [mask_id] * int(span_length)} for values in prompt_ids
        ]
        full_batch = pad_batch(full_rows, int(tokenizer.pad_token_id), device)
        logits = model(
            input_ids=full_batch["input_ids"], attention_mask=full_batch["attention_mask"]
        ).logits.float()
        states = []
        lookups = []
        revealed_patterns = []
        for row_index, (record, values) in enumerate(zip(subset, prompt_ids)):
            offset = int(full_batch["left_offsets"][row_index])
            answer_positions = [offset + len(values) + index for index in range(int(span_length))]
            predicted = [int(logits[row_index, position].argmax()) for position in answer_positions]
            positions = list(range(int(span_length)))
            revealed = sorted(rng.sample(positions, revealed_count)) if revealed_count else []
            state = [predicted[index] if index in revealed else mask_id for index in positions]
            states.append({"input_ids": values + state})
            subject = str(record.get("lookup_subject") or "")
            if subject and subject.casefold() in str(record["prompt"]).casefold():
                lookup = find_last_subject_token(tokenizer, str(record["prompt"]), subject)
            else:
                lookup = max(len(values) - 1, 0)
            lookups.append(lookup)
            revealed_patterns.append(revealed)
        batch = pad_batch(states, int(tokenizer.pad_token_id), device)
        offsets = batch["left_offsets"].tolist()
        padded_lookups = [int(offset) + int(lookup) for offset, lookup in zip(offsets, lookups)]
        box: list[torch.Tensor] = []

        def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            box.append(inputs[0].detach())

        handle = module.register_forward_pre_hook(pre_hook)
        try:
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        finally:
            handle.remove()
        keys = box[0]
        for row_index, record in enumerate(subset):
            output_keys.append(keys[row_index, padded_lookups[row_index]].float().cpu())
            metadata.append(
                {
                    "anchor_id": record["anchor_id"],
                    "family": record["family"],
                    "relation_id": record["relation_id"],
                    "state_bucket": state_bucket,
                    "span_length": int(span_length),
                    "revealed_positions": revealed_patterns[row_index],
                    "prompt_provenance": record["prompt_provenance"],
                }
            )
    result = torch.stack(output_keys)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite protection keys")
    return result, metadata
