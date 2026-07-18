#!/usr/bin/env python3
"""Temporal causal-localization primitives for the TRM campaign."""

from __future__ import annotations

import contextlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterator, Mapping, Sequence

import torch

from scripts.dnpe_editor import component_module_name
from scripts.mdm_memit_editor import (
    find_subject_token_span,
    get_module,
    infer_mask_id,
    model_device,
    output_hidden,
    replace_output_hidden,
)


@dataclass(frozen=True, order=True)
class TraceCandidate:
    layer: int
    component: str
    position: str

    @property
    def candidate_id(self) -> str:
        return f"L{self.layer:02d}:{self.component}:{self.position}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "layer": self.layer,
            "component": self.component,
            "position": self.position,
        }


def candidate_grid(
    layers: Sequence[int], components: Sequence[str], positions: Sequence[str]
) -> list[TraceCandidate]:
    return [
        TraceCandidate(int(layer), str(component), str(position))
        for layer in layers
        for component in components
        for position in positions
    ]


def temporal_state_specs(
    target_length: int,
    *,
    seed: int,
    confidence_order: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return the frozen full/early/middle/late/trajectory state family.

    Short answer spans cannot realize five distinct mask counts. We preserve
    every semantic state label and report the effective revealed subset rather
    than silently substituting a different target-length regime.
    """

    n = int(target_length)
    if n <= 0:
        raise ValueError("target_length must be positive")
    rng = random.Random(int(seed))
    random_order = list(range(n))
    rng.shuffle(random_order)
    confidence = list(map(int, confidence_order or random_order))
    if sorted(confidence) != list(range(n)):
        raise ValueError("confidence_order must be a permutation of target positions")

    def capped(count: int) -> int:
        return max(0, min(n - 1, int(count)))

    counts = {
        "fully_masked": 0,
        "early": capped(n // 4),
        "middle": capped(n // 2),
        "late": capped(n - 1),
        "actual_confidence_trajectory": capped(max(1, n // 2) if n > 1 else 0),
    }
    rows = []
    for label, count in counts.items():
        ordering = confidence if label == "actual_confidence_trajectory" else random_order
        revealed = sorted(ordering[:count])
        rows.append(
            {
                "state_label": label,
                "revealed_positions": revealed,
                "revealed_count": len(revealed),
                "masked_count": n - len(revealed),
                "effective_state_signature": ",".join(map(str, revealed)) or "all_masked",
            }
        )
    return rows


def target_support_metrics(
    logits: torch.Tensor,
    answer_positions: Sequence[int],
    target_ids: Sequence[int],
    supervised_indices: Sequence[int],
) -> list[dict[str, float]]:
    """Measure target probability, margin, and decoded support per batch row."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if not supervised_indices:
        raise ValueError("at least one target position must remain masked")
    outputs = []
    for batch_index in range(logits.shape[0]):
        log_probabilities = []
        margins = []
        hits = []
        for target_index in supervised_indices:
            position = int(answer_positions[int(target_index)])
            token_id = int(target_ids[int(target_index)])
            vector = logits[batch_index, position].float()
            target_logit = vector[token_id]
            top_values, top_ids = torch.topk(vector, k=2)
            alternative = top_values[0] if int(top_ids[0]) != token_id else top_values[1]
            log_probabilities.append(target_logit - torch.logsumexp(vector, dim=0))
            margins.append(target_logit - alternative)
            hits.append(float(int(vector.argmax()) == token_id))
        mean_log_probability = torch.stack(log_probabilities).mean()
        outputs.append(
            {
                "target_probability": float(torch.exp(mean_log_probability)),
                "target_margin": float(torch.stack(margins).mean()),
                "decoded_support": float(sum(hits) / len(hits)),
            }
        )
    return outputs


def causal_recovery_metrics(
    clean: Mapping[str, float],
    corrupted: Mapping[str, float],
    restored: Mapping[str, float],
    *,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in ("target_probability", "target_margin", "decoded_support"):
        clean_value = float(clean[name])
        corrupt_value = float(corrupted[name])
        restored_value = float(restored[name])
        corruption = clean_value - corrupt_value
        raw_delta = restored_value - corrupt_value
        if abs(corruption) < epsilon:
            recovery_fraction = 0.0
            distance_recovery = 0.0
        else:
            recovery_fraction = max(-2.0, min(2.0, raw_delta / corruption))
            distance_recovery = max(
                -1.0,
                min(1.0, 1.0 - abs(restored_value - clean_value) / abs(corruption)),
            )
        result[f"clean_{name}"] = clean_value
        result[f"corrupted_{name}"] = corrupt_value
        result[f"restored_{name}"] = restored_value
        result[f"corruption_effect_{name}"] = corruption
        result[f"restoration_delta_{name}"] = raw_delta
        result[f"recovery_fraction_{name}"] = recovery_fraction
        result[f"distance_recovery_{name}"] = distance_recovery
    return result


@contextlib.contextmanager
def shared_subject_corruption(
    model: torch.nn.Module,
    subject_positions: Sequence[int],
    *,
    noise_scale: float,
    seed: int,
) -> Iterator[None]:
    """Apply identical subject noise to every candidate in a trace batch."""

    embedding = model.get_input_embeddings()

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        edited = output.clone()
        reference = edited[:, list(subject_positions)].float()
        generator = torch.Generator(device=edited.device)
        generator.manual_seed(int(seed))
        std = reference[0].std().clamp_min(1e-6)
        noise = torch.randn(
            (1, reference.shape[1], reference.shape[2]),
            generator=generator,
            device=edited.device,
            dtype=torch.float32,
        ) * (float(noise_scale) * std)
        edited[:, list(subject_positions)] = (reference + noise).to(dtype=edited.dtype)
        return edited

    handle = embedding.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def confidence_order(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    target_ids: Sequence[int],
) -> list[int]:
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    mask_id = infer_mask_id(model)
    answer_positions = list(range(len(prompt_ids), len(prompt_ids) + len(target_ids)))
    input_ids = torch.tensor(
        [prompt_ids + [mask_id] * len(target_ids)],
        dtype=torch.long,
        device=model_device(model),
    )
    logits = model(input_ids=input_ids).logits[0].float()
    scores = [float(logits[position, int(token_id)]) for position, token_id in zip(answer_positions, target_ids)]
    return sorted(range(len(target_ids)), key=lambda index: (-scores[index], index))


@torch.no_grad()
def trace_candidates_batched(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    case_id: str,
    prompt: str,
    subject: str,
    target_ids: Sequence[int],
    candidates: Sequence[TraceCandidate],
    revealed_positions: Sequence[int] = (),
    noise_scale: float = 3.0,
    seed: int = 0,
    chunk_size: int = 16,
) -> list[dict[str, Any]]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    target_ids = list(map(int, target_ids))
    revealed = set(map(int, revealed_positions))
    supervised = [index for index in range(len(target_ids)) if index not in revealed]
    if not supervised:
        raise ValueError("at least one target position must remain masked")
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    mask_id = infer_mask_id(model)
    state = [token if index in revealed else mask_id for index, token in enumerate(target_ids)]
    sequence = prompt_ids + state
    answer_positions = list(range(len(prompt_ids), len(sequence)))
    first_subject, last_subject = find_subject_token_span(tokenizer, prompt, subject)
    named_positions = {
        "first_subject": first_subject,
        "last_subject": last_subject,
        "first_answer_mask": answer_positions[supervised[0]],
    }
    subject_positions = list(range(first_subject, last_subject + 1))
    unique_modules: dict[str, torch.nn.Module] = {}
    for candidate in candidates:
        module_name = component_module_name(model, candidate.layer, candidate.component)
        unique_modules[module_name] = get_module(model, module_name)
    clean_boxes: dict[str, list[torch.Tensor]] = {name: [] for name in unique_modules}
    handles = []
    for name, module in unique_modules.items():
        box = clean_boxes[name]

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, *, _box=box) -> None:
            _box.append(output_hidden(output).detach().clone())

        handles.append(module.register_forward_hook(capture))
    clean_ids = torch.tensor([sequence], dtype=torch.long, device=model_device(model))
    try:
        clean_logits = model(input_ids=clean_ids).logits
    finally:
        for handle in handles:
            handle.remove()
    clean_metrics = target_support_metrics(clean_logits, answer_positions, target_ids, supervised)[0]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(candidates), int(chunk_size)):
        chunk = list(candidates[start : start + int(chunk_size)])
        batch_ids = clean_ids.repeat(len(chunk), 1)
        with shared_subject_corruption(
            model, subject_positions, noise_scale=noise_scale, seed=seed
        ):
            corrupted_logits = model(input_ids=batch_ids).logits
        corrupted_metrics = target_support_metrics(
            corrupted_logits, answer_positions, target_ids, supervised
        )
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for batch_index, candidate in enumerate(chunk):
            module_name = component_module_name(model, candidate.layer, candidate.component)
            grouped[module_name].append((batch_index, int(named_positions[candidate.position])))
        with contextlib.ExitStack() as stack:
            for module_name, assignments in grouped.items():
                module = unique_modules[module_name]
                clean_hidden = clean_boxes[module_name][0]

                def restore(
                    _module: torch.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    _assignments=tuple(assignments),
                    _clean=clean_hidden,
                ) -> Any:
                    hidden = output_hidden(output)
                    edited = hidden.clone()
                    for batch_index, position in _assignments:
                        edited[int(batch_index), int(position)] = _clean[0, int(position)].to(edited)
                    return replace_output_hidden(output, edited)

                handle = module.register_forward_hook(restore)
                stack.callback(handle.remove)
            with shared_subject_corruption(
                model, subject_positions, noise_scale=noise_scale, seed=seed
            ):
                restored_logits = model(input_ids=batch_ids).logits
        restored_metrics = target_support_metrics(
            restored_logits, answer_positions, target_ids, supervised
        )
        for candidate, corrupt, restored in zip(chunk, corrupted_metrics, restored_metrics):
            metrics = causal_recovery_metrics(clean_metrics, corrupt, restored)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError(f"Non-finite trace metric for {candidate.candidate_id}")
            rows.append(
                {
                    "case_id": case_id,
                    **candidate.to_dict(),
                    "target_length": len(target_ids),
                    "revealed_count": len(revealed),
                    "revealed_positions": ",".join(map(str, sorted(revealed))),
                    "supervised_count": len(supervised),
                    "noise_seed": int(seed),
                    "noise_scale": float(noise_scale),
                    **metrics,
                }
            )
    return rows


def aggregate_coordinates(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = ("target_role", "candidate_id", "layer", "component", "position"),
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output = []
    for key, values in groups.items():
        weights = [abs(float(row["corruption_effect_target_margin"])) for row in values]
        recoveries = [float(row["distance_recovery_target_margin"]) for row in values]
        weighted = (
            sum(weight * recovery for weight, recovery in zip(weights, recoveries)) / sum(weights)
            if sum(weights) > 1e-8
            else 0.0
        )
        result = {field: value for field, value in zip(group_fields, key)}
        result.update(
            {
                "num_rows": len(values),
                "num_edits": len({str(row["case_id"]) for row in values}),
                "mean_margin_distance_recovery": mean(recoveries),
                "weighted_margin_distance_recovery": weighted,
                "mean_abs_corruption_margin": mean(weights),
                "mean_target_probability_delta": mean(
                    float(row["restoration_delta_target_probability"]) for row in values
                ),
                "mean_target_margin_delta": mean(
                    float(row["restoration_delta_target_margin"]) for row in values
                ),
                "mean_decoded_support_delta": mean(
                    float(row["restoration_delta_decoded_support"]) for row in values
                ),
                "positive_recovery_fraction": sum(value > 0 for value in recoveries) / len(recoveries),
                "all_finite": all(
                    math.isfinite(float(row[metric]))
                    for row in values
                    for metric in (
                        "distance_recovery_target_margin",
                        "restoration_delta_target_probability",
                        "restoration_delta_target_margin",
                    )
                ),
                "site_score": weighted,
            }
        )
        output.append(result)
    return sorted(output, key=lambda row: float(row["site_score"]), reverse=True)


def shortlist_candidates(
    aggregate: Sequence[Mapping[str, Any]],
    *,
    all_candidates: Sequence[TraceCandidate],
    limit: int,
    seed: int,
) -> list[TraceCandidate]:
    by_candidate: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    objects = {candidate.candidate_id: candidate for candidate in all_candidates}
    for row in aggregate:
        by_candidate[str(row["candidate_id"])][str(row["target_role"])] = row

    def combined(candidate: TraceCandidate) -> float:
        roles = by_candidate.get(candidate.candidate_id, {})
        destination = float(roles.get("target_new", {}).get("site_score", 0.0))
        source = float(roles.get("target_true", {}).get("site_score", 0.0))
        return 0.65 * destination + 0.35 * source

    ranked = sorted(all_candidates, key=lambda candidate: (-combined(candidate), candidate.candidate_id))
    chosen: list[TraceCandidate] = []

    def add(candidate: TraceCandidate) -> None:
        if candidate not in chosen and len(chosen) < int(limit):
            chosen.append(candidate)

    # Preserve the source-compatible fixed window before data-driven slots.
    for layer in (3, 4, 5, 6):
        candidate = TraceCandidate(layer, "mlp", "last_subject")
        if candidate.candidate_id in objects:
            add(candidate)
    for candidate in ranked[:4]:
        add(candidate)
    for candidate in [c for c in ranked if c.component == "mlp" and c.position == "last_subject"][:4]:
        add(candidate)
    for candidate in [c for c in ranked if c.position == "first_answer_mask"][:2]:
        add(candidate)
    rng = random.Random(int(seed))
    add(rng.choice(list(all_candidates)))
    for candidate in ranked:
        add(candidate)
        if len(chosen) >= int(limit):
            break
    return chosen


def stability_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["candidate_id"])].append(row)
    output = []
    for candidate_id, values in groups.items():
        def dimension_means(field: str) -> list[float]:
            nested: dict[str, list[float]] = defaultdict(list)
            for row in values:
                nested[str(row[field])].append(float(row["distance_recovery_target_margin"]))
            return [mean(items) for items in nested.values()]

        seed_means = dimension_means("noise_seed")
        prompt_means = dimension_means("prompt_type")
        state_means = dimension_means("state_label")
        aggregate = aggregate_coordinates(
            values,
            group_fields=("candidate_id", "layer", "component", "position"),
        )[0]
        instability = sum(
            pstdev(items) if len(items) > 1 else 0.0
            for items in (seed_means, prompt_means, state_means)
        ) / 3.0
        output.append(
            {
                **aggregate,
                "seed_std": pstdev(seed_means) if len(seed_means) > 1 else 0.0,
                "prompt_type_std": pstdev(prompt_means) if len(prompt_means) > 1 else 0.0,
                "state_label_std": pstdev(state_means) if len(state_means) > 1 else 0.0,
                "instability_penalty": instability,
                "stability_score": float(aggregate["site_score"]) - 0.25 * instability,
                "num_prompt_types": len({str(row["prompt_type"]) for row in values}),
                "num_state_labels": len({str(row["state_label"]) for row in values}),
                "num_seeds": len({int(row["noise_seed"]) for row in values}),
            }
        )
    return sorted(output, key=lambda row: float(row["stability_score"]), reverse=True)


def build_site_policy_rows(
    stability: Sequence[Mapping[str, Any]],
    fine_rows: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not stability:
        raise ValueError("stability rows must not be empty")
    by_id = {str(row["candidate_id"]): row for row in stability}

    def score(candidate_ids: Sequence[str]) -> float:
        values = [float(by_id[value]["stability_score"]) for value in candidate_ids if value in by_id]
        return mean(values) if values else float("nan")

    stable_mlp = [
        row
        for row in stability
        if row["component"] == "mlp" and row["position"] == "last_subject"
    ]
    stable_ids = [str(row["candidate_id"]) for row in stable_mlp[:4]]
    middle_mlp = [
        row
        for row in stable_mlp
        if num_layers // 4 <= int(row["layer"]) < 3 * num_layers // 4
    ]
    middle_ids = [str(row["candidate_id"]) for row in middle_mlp[:2]] or stable_ids[:2]
    answer_ids = [
        str(row["candidate_id"])
        for row in stability
        if row["position"] == "first_answer_mask"
    ][:2]
    fixed_ids = [f"L{layer:02d}:mlp:last_subject" for layer in (3, 4, 5, 6)]
    rng = random.Random(int(seed))
    random_id = str(rng.choice(list(stability))["candidate_id"])
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in fine_rows:
        by_case[str(row["case_id"])].append(float(row["distance_recovery_target_margin"]))
    per_edit_proxy = mean(max(values) for values in by_case.values())
    policies = [
        {
            "policy_id": "source_paper_compatible_fixed_site",
            "candidate_ids": fixed_ids,
            "layers": [3, 4, 5, 6],
            "component": "mlp",
            "position": "last_subject",
            "selection_source": "paper_compatible_window_exact_source_code_unavailable",
            "localization_proxy": score(fixed_ids),
        },
        {
            "policy_id": "per_edit_highest_tie_site",
            "candidate_ids": [str(row["candidate_id"]) for row in stability],
            "layers": sorted({int(row["layer"]) for row in stability}),
            "component": "frozen_shortlist",
            "position": "per_edit_argmax",
            "selection_source": "cf_trm_localize_50_only",
            "localization_proxy": per_edit_proxy,
        },
        {
            "policy_id": "stable_temporal_site_set",
            "candidate_ids": stable_ids,
            "layers": [int(by_id[value]["layer"]) for value in stable_ids],
            "component": "mlp",
            "position": "last_subject",
            "selection_source": "cross_prompt_state_seed_stability",
            "localization_proxy": score(stable_ids),
        },
        {
            "policy_id": "last_subject_early_mid_mlp_site",
            "candidate_ids": middle_ids,
            "layers": [int(by_id[value]["layer"]) for value in middle_ids],
            "component": "mlp",
            "position": "last_subject",
            "selection_source": "frozen_middle_layer_filter",
            "localization_proxy": score(middle_ids),
        },
        {
            "policy_id": "random_site",
            "candidate_ids": [random_id],
            "layers": [int(by_id[random_id]["layer"])],
            "component": str(by_id[random_id]["component"]),
            "position": str(by_id[random_id]["position"]),
            "selection_source": f"seeded_random_seed_{int(seed)}",
            "localization_proxy": score([random_id]),
        },
        {
            "policy_id": "late_answer_mask_site",
            "candidate_ids": answer_ids,
            "layers": [int(by_id[value]["layer"]) for value in answer_ids],
            "component": "best_shortlisted",
            "position": "first_answer_mask",
            "selection_source": "frozen_answer_position_filter",
            "localization_proxy": score(answer_ids),
        },
    ]
    for row in policies:
        row["candidate_ids_json"] = json.dumps(row.pop("candidate_ids"))
        row["layers_json"] = json.dumps(row.pop("layers"))
        row["pilot_acceptance_status"] = "pending_C2_site_policy_comparison"
    return policies
