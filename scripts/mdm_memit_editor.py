#!/usr/bin/env python3
"""Paper-matched MEMIT primitives for LLaDA masked prediction.

The implementation follows official MEMIT's key/value update while ensuring
all LLaDA forward-pass distributions share the same mask-augmented context.
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F


DEFAULT_CONTEXT_PREFIXES = (
    "{}",
    "The following is factual. {}",
    "In this context, {}",
    "As a matter of record, {}",
    "According to the source, {}",
)


@dataclass(frozen=True)
class MemitConfig:
    layers: tuple[int, ...] = (4, 5, 6, 7)
    learning_rate: float = 0.1
    target_optimization_steps: int = 25
    clamp_norm_factor: float = 0.75
    kl_factor: float = 0.0625
    weight_decay: float = 0.5
    covariance_weight: float = 15000.0
    partial_mask_schedule: str = "fully_masked"
    reveal_policy: str = "random"
    lambda_path: float = 0.0
    lambda_identity: float = 0.0
    lambda_weight: float = 0.0
    sparse_kl_top_k: int = 32
    state_consistency_weight: float = 0.0
    old_target_suppression_weight: float = 0.0
    seed: int = 260603924
    block_module_template: str | None = None
    key_module_template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        return payload


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    current: Any = model
    for part in name.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def block_name(layer: int) -> str:
    return f"model.transformer.blocks.{int(layer)}"


def key_module_name(layer: int) -> str:
    return f"model.transformer.blocks.{int(layer)}.ff_out"


def editable_weight_name(layer: int) -> str:
    return key_module_name(layer) + ".weight"


def resolved_block_name(
    model: torch.nn.Module, layer: int, template: str | None = None
) -> str:
    if template:
        return template.format(layer=int(layer))
    if hasattr(getattr(model, "model", None), "transformer"):
        return block_name(layer)
    if hasattr(getattr(model, "model", None), "layers"):
        return f"model.layers.{int(layer)}"
    raise AttributeError("Unsupported masked-diffusion block layout")


def resolved_key_module_name(
    model: torch.nn.Module, layer: int, template: str | None = None
) -> str:
    if template:
        return template.format(layer=int(layer))
    if hasattr(getattr(model, "model", None), "transformer"):
        return key_module_name(layer)
    if hasattr(getattr(model, "model", None), "layers"):
        return f"model.layers.{int(layer)}.self_attn.o_proj"
    raise AttributeError("Unsupported masked-diffusion editable-module layout")


def model_hidden_size(model: torch.nn.Module) -> int:
    for key in ("d_model", "hidden_size"):
        value = getattr(model.config, key, None)
        if value is not None:
            return int(value)
    raise AttributeError("Model config has neither d_model nor hidden_size")


def infer_mask_id(model: Any) -> int:
    return int(getattr(model.config, "mask_token_id", 126336))


def contextual_target_ids(tokenizer: Any, prompt: str, target: str) -> list[int]:
    prefix = str(prompt).rstrip()
    combined = tokenizer(prefix + " " + str(target).strip(), add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if len(combined) > len(prefix_ids) and combined[: len(prefix_ids)] == prefix_ids:
        return list(map(int, combined[len(prefix_ids) :]))
    return list(map(int, tokenizer(" " + str(target).strip(), add_special_tokens=False)["input_ids"]))


def find_subject_token_span(tokenizer: Any, prompt: str, subject: str) -> tuple[int, int]:
    """Return the inclusive token span for the subject in a rendered prompt."""

    prompt_text = str(prompt)
    subject_text = str(subject)
    start = prompt_text.casefold().find(subject_text.casefold())
    if start < 0:
        raise ValueError(f"Subject {subject_text!r} is absent from prompt {prompt_text!r}")
    end = start + len(subject_text)
    try:
        encoded = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        indices = [
            index
            for index, (left, right) in enumerate(offsets)
            if int(right) > start and int(left) < end
        ]
        if indices:
            return min(indices), max(indices)
    except (TypeError, ValueError, NotImplementedError, KeyError):
        pass
    prefix_ids = tokenizer(prompt_text[:start], add_special_tokens=False)["input_ids"]
    through_ids = tokenizer(prompt_text[:end], add_special_tokens=False)["input_ids"]
    if len(through_ids) <= len(prefix_ids):
        raise ValueError("Could not map subject characters to tokens")
    return len(prefix_ids), len(through_ids) - 1


def find_last_subject_token(tokenizer: Any, prompt: str, subject: str) -> int:
    return find_subject_token_span(tokenizer, prompt, subject)[1]


def partial_mask_state(
    target_ids: Sequence[int],
    *,
    step: int,
    mask_id: int,
    schedule: str,
    reveal_policy: str,
    rng: random.Random,
    confidence: Sequence[float] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return state IDs, supervised positions, and revealed positions."""

    n = len(target_ids)
    if n <= 0:
        raise ValueError("Target must contain at least one token")
    if schedule == "fully_masked":
        k = 0
    elif schedule == "cycle":
        k = int(step) % n
    elif schedule == "uniform":
        k = rng.randrange(n)
    elif schedule == "fewer_revealed":
        weights = [n - index for index in range(n)]
        k = rng.choices(range(n), weights=weights, k=1)[0]
    elif schedule == "more_revealed":
        weights = [index + 1 for index in range(n)]
        k = rng.choices(range(n), weights=weights, k=1)[0]
    else:
        raise ValueError(f"Unsupported partial-mask schedule: {schedule}")

    positions = list(range(n))
    if k == 0:
        revealed: list[int] = []
    elif reveal_policy == "left_to_right":
        revealed = positions[:k]
    elif reveal_policy == "base_confidence":
        if confidence is None or len(confidence) != n:
            raise ValueError("base_confidence reveal requires one score per target position")
        revealed = sorted(positions, key=lambda index: (-float(confidence[index]), index))[:k]
    elif reveal_policy == "random":
        revealed = sorted(rng.sample(positions, k))
    else:
        raise ValueError(f"Unsupported reveal policy: {reveal_policy}")
    revealed_set = set(revealed)
    state = [int(target_ids[index]) if index in revealed_set else int(mask_id) for index in positions]
    supervised = [index for index in positions if index not in revealed_set]
    return state, supervised, revealed


def render_masked_input(
    tokenizer: Any,
    prompt: str,
    target_ids: Sequence[int],
    mask_id: int,
    *,
    step: int = 0,
    schedule: str = "fully_masked",
    reveal_policy: str = "random",
    rng: random.Random | None = None,
    confidence: Sequence[float] | None = None,
) -> dict[str, Any]:
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    state, supervised, revealed = partial_mask_state(
        target_ids,
        step=step,
        mask_id=mask_id,
        schedule=schedule,
        reveal_policy=reveal_policy,
        rng=rng or random.Random(0),
        confidence=confidence,
    )
    return {
        "input_ids": prompt_ids + state,
        "prompt_len": len(prompt_ids),
        "answer_positions": [len(prompt_ids) + index for index in range(len(target_ids))],
        "supervised_positions": [len(prompt_ids) + index for index in supervised],
        "revealed_positions": [len(prompt_ids) + index for index in revealed],
    }


def pad_batch(rows: Sequence[Mapping[str, Any]], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    left_offsets: list[int] = []
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        offset = width - values.numel()
        ids[index, offset:] = values
        attention[index, offset:] = 1
        left_offsets.append(offset)
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "left_offsets": torch.tensor(left_offsets, dtype=torch.long, device=device),
    }


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def output_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_output_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


@contextlib.contextmanager
def intervene_block_output(
    module: torch.nn.Module,
    lookup_indices: Sequence[int],
    delta: torch.Tensor,
    target_init_box: list[torch.Tensor],
) -> Iterator[None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output_hidden(output)
        edited = hidden.clone()
        if not target_init_box:
            target_init_box.append(hidden[0, int(lookup_indices[0])].detach().clone())
        for batch_index, token_index in enumerate(lookup_indices):
            edited[batch_index, int(token_index)] = edited[batch_index, int(token_index)] + delta
        return replace_output_hidden(output, edited)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _masked_context_rows(
    tokenizer: Any,
    request: Mapping[str, Any],
    target_ids: Sequence[int],
    mask_id: int,
    *,
    config: MemitConfig,
    step: int,
    rng: random.Random,
    confidence: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], list[int], int]:
    subject = str(request["subject"])
    template = str(request.get("rewrite_template") or request.get("prompt_template") or "{}")
    base_prompt = str(request.get("rewrite_prompt") or template.format(subject))
    rows: list[dict[str, Any]] = []
    lookup_unpadded: list[int] = []
    for prefix in DEFAULT_CONTEXT_PREFIXES:
        prompt = prefix.format(base_prompt)
        row = render_masked_input(
            tokenizer,
            prompt,
            target_ids,
            mask_id,
            step=step,
            schedule=config.partial_mask_schedule,
            reveal_policy=config.reveal_policy,
            rng=rng,
            confidence=confidence,
        )
        rows.append(row)
        lookup_unpadded.append(find_last_subject_token(tokenizer, prompt, subject))
    anchor_template = str(request.get("kl_anchor_template") or "{} is a")
    anchor_prompt = anchor_template.format(subject)
    anchor_target = [mask_id]
    anchor_row = render_masked_input(tokenizer, anchor_prompt, anchor_target, mask_id)
    rows.append(anchor_row)
    lookup_unpadded.append(find_last_subject_token(tokenizer, anchor_prompt, subject))
    for identity_prompt in list(request.get("identity_prompts") or [])[:2]:
        identity_text = str(identity_prompt)
        if subject.casefold() not in identity_text.casefold():
            raise ValueError("Identity prompt must contain the edited subject")
        identity_row = render_masked_input(tokenizer, identity_text, [mask_id], mask_id)
        rows.append(identity_row)
        lookup_unpadded.append(find_last_subject_token(tokenizer, identity_text, subject))
    return rows, lookup_unpadded, len(DEFAULT_CONTEXT_PREFIXES)


def optimize_target_value(
    model: torch.nn.Module,
    tokenizer: Any,
    request: Mapping[str, Any],
    config: MemitConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Optimize MEMIT's target value at the last edited block."""

    device = model_device(model)
    mask_id = infer_mask_id(model)
    prompt = str(request.get("rewrite_prompt") or str(request["rewrite_template"]).format(request["subject"]))
    target_ids = list(request.get("target_new_token_ids") or contextual_target_ids(tokenizer, prompt, request["target_new"]))
    if not target_ids:
        raise ValueError("Target tokenization is empty")
    layer = config.layers[-1]
    module = get_module(
        model, resolved_block_name(model, layer, config.block_module_template)
    )
    hidden_size = model_hidden_size(model)
    delta = torch.zeros(hidden_size, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=config.learning_rate)
    target_init_box: list[torch.Tensor] = []
    initial_kl: torch.Tensor | None = None
    history: list[dict[str, float]] = []
    rng = random.Random(config.seed + int(stable_request_int(request)))
    confidence: list[float] | None = None
    if config.reveal_policy == "base_confidence":
        rendered = render_masked_input(tokenizer, prompt, target_ids, mask_id)
        ids = torch.tensor([rendered["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            base_logits = model(input_ids=ids).logits[0].float()
        confidence = [
            float(F.softmax(base_logits[position], dim=-1)[int(token_id)])
            for position, token_id in zip(rendered["answer_positions"], target_ids)
        ]
    for step in range(config.target_optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        rows, lookup_unpadded, rewrite_count = _masked_context_rows(
            tokenizer,
            request,
            target_ids,
            mask_id,
            config=config,
            step=step,
            rng=rng,
            confidence=confidence,
        )
        batch = pad_batch(rows, int(tokenizer.pad_token_id), device)
        offsets = batch["left_offsets"].tolist()
        lookup = [int(offsets[index]) + int(value) for index, value in enumerate(lookup_unpadded)]
        base_logits: torch.Tensor | None = None
        if config.lambda_path > 0 or config.lambda_identity > 0:
            with torch.no_grad():
                base_logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits.float()
        with intervene_block_output(module, lookup, delta, target_init_box):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()

        losses: list[torch.Tensor] = []
        row_target_support: list[torch.Tensor] = []
        old_suppression_terms: list[torch.Tensor] = []
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
        target_true_ids = list(map(int, request.get("target_true_token_ids") or []))
        for index, row in enumerate(rows[:rewrite_count]):
            offset = int(offsets[index])
            absolute_positions = [offset + int(pos) for pos in row["supervised_positions"]]
            relative_positions = [int(pos) - int(row["prompt_len"]) for pos in row["supervised_positions"]]
            if not absolute_positions:
                continue
            row_logits = logits[index, absolute_positions]
            row_targets = target_tensor[relative_positions]
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            losses.append(F.nll_loss(row_log_probs, row_targets))
            selected = row_log_probs.gather(1, row_targets[:, None]).squeeze(1)
            row_target_support.append(selected.mean())
            if config.old_target_suppression_weight > 0 and target_true_ids:
                old_ids = []
                new_ids = []
                logits_for_margin = []
                for local_index, relative in enumerate(relative_positions):
                    if relative < len(target_true_ids):
                        old_ids.append(int(target_true_ids[relative]))
                        new_ids.append(int(target_ids[relative]))
                        logits_for_margin.append(row_log_probs[local_index])
                if logits_for_margin:
                    margin_logits = torch.stack(logits_for_margin)
                    new_tensor = torch.tensor(new_ids, dtype=torch.long, device=device)
                    old_tensor = torch.tensor(old_ids, dtype=torch.long, device=device)
                    new_logp = margin_logits.gather(1, new_tensor[:, None]).squeeze(1)
                    old_logp = margin_logits.gather(1, old_tensor[:, None]).squeeze(1)
                    old_suppression_terms.append(F.relu(old_logp - new_logp + 0.5).mean())
        nll = torch.stack(losses).mean()
        state_consistency = (
            torch.stack(row_target_support).var(unbiased=False)
            if len(row_target_support) > 1
            else torch.zeros((), device=device)
        )
        old_suppression = (
            torch.stack(old_suppression_terms).mean()
            if old_suppression_terms
            else torch.zeros((), device=device)
        )

        anchor_index = rewrite_count
        anchor_offset = int(offsets[anchor_index])
        anchor_pos = anchor_offset + int(rows[anchor_index]["answer_positions"][0])
        anchor_log_probs = F.log_softmax(logits[anchor_index, anchor_pos], dim=-1)
        if initial_kl is None:
            initial_kl = anchor_log_probs.detach().clone()
        kl = F.kl_div(initial_kl, anchor_log_probs, log_target=True, reduction="sum")
        path_terms: list[torch.Tensor] = []
        if config.lambda_path > 0:
            if base_logits is None:
                raise RuntimeError("Path KL requested without frozen-base logits")
            target_true_ids = list(map(int, request.get("target_true_token_ids") or []))
            extra_ids = list(target_ids) + target_true_ids
            for index, row in enumerate(rows[:rewrite_count]):
                offset = int(offsets[index])
                for position in row["supervised_positions"]:
                    absolute = offset + int(position)
                    path_terms.append(
                        sparse_support_kl(
                            logits[index, absolute],
                            base_logits[index, absolute],
                            extra_ids=extra_ids,
                            top_k=config.sparse_kl_top_k,
                        )
                    )
        path_kl = (
            torch.stack(path_terms).mean()
            if path_terms
            else torch.zeros((), device=device)
        )
        identity_terms: list[torch.Tensor] = []
        if config.lambda_identity > 0:
            if base_logits is None:
                raise RuntimeError("Identity loss requested without frozen-base logits")
            identity_start = rewrite_count + 1
            for index in range(identity_start, len(rows)):
                offset = int(offsets[index])
                absolute = offset + int(rows[index]["answer_positions"][0])
                identity_kl = sparse_support_kl(
                    logits[index, absolute],
                    base_logits[index, absolute],
                    extra_ids=target_ids,
                    top_k=config.sparse_kl_top_k,
                )
                edited_probs = F.softmax(logits[index, absolute], dim=-1)
                target_pressure = edited_probs[
                    torch.tensor(target_ids, dtype=torch.long, device=device)
                ].sum()
                identity_terms.append(identity_kl + target_pressure)
        identity_loss = (
            torch.stack(identity_terms).mean()
            if identity_terms
            else torch.zeros((), device=device)
        )
        target_init = target_init_box[0].float()
        decay = config.weight_decay * delta.norm() / target_init.norm().clamp_min(1e-8).pow(2)
        delta_l2 = delta.float().pow(2).mean()
        total = (
            nll
            + config.kl_factor * kl
            + decay
            + config.lambda_path * path_kl
            + config.lambda_identity * identity_loss
            + config.lambda_weight * delta_l2
            + config.state_consistency_weight * state_consistency
            + config.old_target_suppression_weight * old_suppression
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite target-value loss")
        history.append(
            {
                "step": float(step),
                "total_loss": float(total.detach()),
                "nll_loss": float(nll.detach()),
                "kl_loss": float(kl.detach()),
                "weight_decay": float(decay.detach()),
                "path_kl_loss": float(path_kl.detach()),
                "identity_loss": float(identity_loss.detach()),
                "state_consistency_loss": float(state_consistency.detach()),
                "old_target_suppression_loss": float(old_suppression.detach()),
                "delta_l2": float(delta_l2.detach()),
                "delta_norm": float(delta.detach().norm()),
            }
        )
        if float(total.detach()) < 0.05 or step == config.target_optimization_steps - 1:
            break
        total.backward()
        optimizer.step()
        with torch.no_grad():
            max_norm = config.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                delta.mul_(max_norm / delta.norm().clamp_min(1e-8))
    target = target_init_box[0].float() + delta.detach()
    return target.detach(), {
        "target_ids": target_ids,
        "history": history,
        "initial_value_norm": float(target_init_box[0].float().norm()),
        "delta_norm": float(delta.detach().norm()),
        "target_value_norm": float(target.norm()),
        "mask_count": len(target_ids),
        "shared_masked_context": True,
    }


def stable_request_int(request: Mapping[str, Any]) -> int:
    value = str(request.get("case_id") or request.get("id") or request.get("subject"))
    return int.from_bytes(value.encode("utf-8")[:8].ljust(8, b"\0"), "little") % 1_000_003


@torch.no_grad()
def extract_keys_and_outputs(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    key_layer: int,
    output_layer: int,
    block_module_template: str | None = None,
    key_module_template: str | None = None,
    batch_size: int = 8,
    partial_mask_schedule: str = "fully_masked",
    reveal_policy: str = "random",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract context-averaged ff_out inputs and final edit-layer outputs."""

    device = model_device(model)
    mask_id = infer_mask_id(model)
    key_module = get_module(
        model, resolved_key_module_name(model, key_layer, key_module_template)
    )
    output_module = get_module(
        model, resolved_block_name(model, output_layer, block_module_template)
    )
    all_keys: list[torch.Tensor] = []
    all_outputs: list[torch.Tensor] = []
    for start in range(0, len(requests), batch_size):
        subset = requests[start : start + batch_size]
        rows: list[dict[str, Any]] = []
        lookups: list[int] = []
        request_indices: list[int] = []
        for request_index, request in enumerate(subset):
            subject = str(request["subject"])
            base_prompt = str(request.get("rewrite_prompt") or str(request["rewrite_template"]).format(subject))
            target_ids = list(request.get("target_new_token_ids") or contextual_target_ids(tokenizer, base_prompt, request["target_new"]))
            for prefix_index, prefix in enumerate(DEFAULT_CONTEXT_PREFIXES):
                prompt = prefix.format(base_prompt)
                rows.append(
                    render_masked_input(
                        tokenizer,
                        prompt,
                        target_ids,
                        mask_id,
                        step=prefix_index,
                        schedule=partial_mask_schedule,
                        reveal_policy=reveal_policy,
                        rng=random.Random(seed + stable_request_int(request) + prefix_index),
                    )
                )
                lookups.append(find_last_subject_token(tokenizer, prompt, subject))
                request_indices.append(request_index)
        batch = pad_batch(rows, int(tokenizer.pad_token_id), device)
        offsets = batch["left_offsets"].tolist()
        padded_lookups = [int(offsets[index]) + lookups[index] for index in range(len(rows))]
        key_box: list[torch.Tensor] = []
        output_box: list[torch.Tensor] = []

        def key_pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            key_box.append(inputs[0])

        def output_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            output_box.append(output_hidden(output))

        key_handle = key_module.register_forward_pre_hook(key_pre_hook)
        out_handle = output_module.register_forward_hook(output_hook)
        try:
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        finally:
            key_handle.remove()
            out_handle.remove()
        keys = key_box[0]
        outputs = output_box[0]
        for request_index in range(len(subset)):
            member_indices = [index for index, value in enumerate(request_indices) if value == request_index]
            member_keys = torch.stack([keys[index, padded_lookups[index]].float() for index in member_indices])
            member_outputs = torch.stack([outputs[index, padded_lookups[index]].float() for index in member_indices])
            all_keys.append(member_keys.mean(dim=0).cpu())
            all_outputs.append(member_outputs.mean(dim=0).cpu())
    return torch.stack(all_keys), torch.stack(all_outputs)


def validate_covariance(covariance: torch.Tensor, key_width: int) -> None:
    if covariance.ndim == 1:
        if covariance.shape != (key_width,):
            raise ValueError(f"Covariance diagonal shape {tuple(covariance.shape)} != {(key_width,)}")
        if not torch.isfinite(covariance).all() or bool((covariance <= 0).any()):
            raise FloatingPointError("Covariance diagonal must be finite and positive")
        return
    if covariance.shape != (key_width, key_width):
        raise ValueError(f"Covariance shape {tuple(covariance.shape)} != {(key_width, key_width)}")
    if not torch.isfinite(covariance).all():
        raise FloatingPointError("Covariance contains non-finite values")
    symmetry_error = float((covariance - covariance.T).abs().max())
    if symmetry_error > 1e-3:
        raise ValueError(f"Covariance is not symmetric: max error {symmetry_error}")


def solve_memit_update(
    keys: torch.Tensor,
    residuals: torch.Tensor,
    covariance: torch.Tensor,
    covariance_weight: float,
) -> torch.Tensor:
    """Solve MEMIT's closed-form update in a numerically stable dtype."""

    # keys: edits x key_width; residuals: edits x d_model.
    device = covariance.device
    key_matrix = keys.T.to(device=device, dtype=torch.float64)
    residual_matrix = residuals.T.to(device=device, dtype=torch.float64)
    cov = covariance.to(dtype=torch.float64)
    validate_covariance(cov, key_matrix.shape[0])
    if cov.ndim == 1:
        scaled_diagonal = float(covariance_weight) * cov
        inverse_diagonal_keys = key_matrix / scaled_diagonal[:, None]
        small_system = (
            torch.eye(key_matrix.shape[1], dtype=torch.float64, device=device)
            + key_matrix.T @ inverse_diagonal_keys
        )
        adjusted_keys = torch.linalg.solve(
            small_system, inverse_diagonal_keys.T
        ).T
        update = residual_matrix @ adjusted_keys.T
        if not torch.isfinite(update).all():
            raise FloatingPointError("MEMIT diagonal-covariance update contains non-finite values")
        return update.float()
    system = float(covariance_weight) * cov + key_matrix @ key_matrix.T
    jitter = max(1e-8, float(system.diagonal().mean().abs()) * 1e-10)
    system = system + torch.eye(system.shape[0], dtype=system.dtype, device=system.device) * jitter
    adjusted_keys = torch.linalg.solve(system, key_matrix)
    update = residual_matrix @ adjusted_keys.T
    if not torch.isfinite(update).all():
        raise FloatingPointError("MEMIT update contains non-finite values")
    return update.float()


def build_protected_basis(
    protected_keys: torch.Tensor,
    explained_variance: float,
    *,
    maximum_rank: int | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Return an orthonormal key-space basis covering protected variance.

    The basis is stored implicitly (key_width x rank), avoiding a dense
    key_width-square projector for LLaDA's 12,288-dimensional MLP keys.
    """

    if protected_keys.ndim != 2 or protected_keys.shape[0] < 2:
        raise ValueError("protected_keys must be a 2D tensor with at least two rows")
    if not 0.0 < float(explained_variance) <= 1.0:
        raise ValueError("explained_variance must be in (0, 1]")
    keys = protected_keys.float()
    if not torch.isfinite(keys).all():
        raise FloatingPointError("protected_keys contain non-finite values")
    centered = keys - keys.mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    total = energy.sum().clamp_min(1e-12)
    cumulative = torch.cumsum(energy, dim=0) / total
    rank = int(torch.searchsorted(cumulative, torch.tensor(float(explained_variance), device=cumulative.device)).item()) + 1
    if maximum_rank is not None:
        rank = min(rank, int(maximum_rank))
    basis = vh[:rank].T.contiguous()
    captured = float(cumulative[rank - 1])
    return basis, {
        "protected_rank": rank,
        "key_width": int(keys.shape[1]),
        "remaining_editable_dimension": int(keys.shape[1] - rank),
        "captured_variance": captured,
    }


def project_update_to_nullspace(
    update: torch.Tensor, protected_basis: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project a weight update away from the protected key subspace."""

    if update.ndim != 2 or protected_basis.ndim != 2:
        raise ValueError("update and protected_basis must be matrices")
    basis = protected_basis.to(device=update.device, dtype=torch.float32)
    value = update.float()
    if value.shape[1] != basis.shape[0]:
        raise ValueError(
            f"Update key width {value.shape[1]} != basis width {basis.shape[0]}"
        )
    projected = value - (value @ basis) @ basis.T
    protected_before = (value @ basis).norm()
    protected_after = (projected @ basis).norm()
    return projected, {
        "update_norm_before_projection": float(value.norm()),
        "update_norm_after_projection": float(projected.norm()),
        "protected_energy_before": float(protected_before),
        "protected_energy_after": float(protected_after),
        "projection_energy_ratio": float(projected.norm() / value.norm().clamp_min(1e-12)),
    }


class WeightRollback:
    def __init__(
        self,
        model: torch.nn.Module,
        layers: Sequence[int],
        *,
        key_module_template: str | None = None,
    ) -> None:
        self.model = model
        self.layers = list(map(int, layers))
        self.key_module_template = key_module_template
        self.originals = {
            layer: get_module(
                model, resolved_key_module_name(model, layer, key_module_template)
            )
            .weight.detach()
            .cpu()
            .clone()
            for layer in self.layers
        }

    def apply(self, layer: int, update: torch.Tensor) -> None:
        weight = get_module(
            self.model,
            resolved_key_module_name(self.model, layer, self.key_module_template),
        ).weight
        if update.shape != weight.shape:
            if update.T.shape == weight.shape:
                update = update.T
            else:
                raise ValueError(f"Update shape {tuple(update.shape)} does not match {tuple(weight.shape)}")
        with torch.no_grad():
            weight.add_(update.to(device=weight.device, dtype=weight.dtype))

    def rollback(self) -> None:
        with torch.no_grad():
            for layer, original in self.originals.items():
                weight = get_module(
                    self.model,
                    resolved_key_module_name(
                        self.model, layer, self.key_module_template
                    ),
                ).weight
                weight.copy_(original.to(device=weight.device, dtype=weight.dtype))

    def checksum_matches(self, atol: float = 0.0) -> bool:
        for layer, original in self.originals.items():
            current = (
                get_module(
                    self.model,
                    resolved_key_module_name(
                        self.model, layer, self.key_module_template
                    ),
                )
                .weight.detach()
                .cpu()
            )
            if not torch.allclose(current, original, atol=atol, rtol=0):
                return False
        return True


def apply_memit_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    config: MemitConfig,
    covariance_loader: Callable[[int], torch.Tensor],
    *,
    target_cache_dir: Path | None = None,
    protected_basis_loader: Callable[[int], torch.Tensor | None] | None = None,
) -> tuple[WeightRollback, dict[str, Any]]:
    """Optimize values and apply a multi-layer MEMIT update in place."""

    rollback = WeightRollback(
        model, config.layers, key_module_template=config.key_module_template
    )
    target_values: list[torch.Tensor] = []
    optimization: list[dict[str, Any]] = []
    for request in requests:
        cache_path = None
        if target_cache_dir is not None:
            target_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = target_cache_dir / f"{request['case_id']}.pt"
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            target = payload["target_value"].float()
            diagnostics = payload["diagnostics"]
        else:
            target, diagnostics = optimize_target_value(model, tokenizer, request, config)
            target = target.cpu()
            if cache_path is not None:
                torch.save({"target_value": target, "diagnostics": diagnostics}, cache_path)
        target_values.append(target)
        optimization.append({"case_id": request["case_id"], **diagnostics})
    target_matrix = torch.stack(target_values)
    layer_reports: list[dict[str, Any]] = []
    try:
        for index, layer in enumerate(config.layers):
            keys, current_outputs = extract_keys_and_outputs(
                model,
                tokenizer,
                requests,
                key_layer=layer,
                output_layer=config.layers[-1],
                block_module_template=config.block_module_template,
                key_module_template=config.key_module_template,
                partial_mask_schedule=config.partial_mask_schedule,
                reveal_policy=config.reveal_policy,
                seed=config.seed,
            )
            residual = (target_matrix - current_outputs) / float(len(config.layers) - index)
            covariance = covariance_loader(layer)
            update = solve_memit_update(keys, residual, covariance, config.covariance_weight)
            projection_report: dict[str, Any] | None = None
            if protected_basis_loader is not None:
                basis = protected_basis_loader(layer)
                if basis is not None:
                    update, projection_report = project_update_to_nullspace(update, basis)
            rollback.apply(layer, update)
            layer_reports.append(
                {
                    "layer": layer,
                    "num_keys": keys.shape[0],
                    "key_width": keys.shape[1],
                    "mean_residual_norm": float(residual.norm(dim=1).mean()),
                    "update_norm": float(update.norm()),
                    "projection": projection_report,
                    "weight_norm": float(
                        get_module(
                            model,
                            resolved_key_module_name(
                                model, layer, config.key_module_template
                            ),
                        )
                        .weight.float()
                        .norm()
                    ),
                }
            )
    except Exception:
        rollback.rollback()
        raise
    return rollback, {"target_optimization": optimization, "layer_updates": layer_reports}


@torch.no_grad()
def denoise_answer_span(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    answer_length: int,
    *,
    steps: int | None = None,
) -> dict[str, Any]:
    if answer_length <= 0:
        raise ValueError("answer_length must be positive")
    device = model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    state = torch.tensor([prompt_ids + [mask_id] * answer_length], dtype=torch.long, device=device)
    answer_positions = list(range(len(prompt_ids), len(prompt_ids) + answer_length))
    eval_count = 0
    total_steps = int(steps or answer_length)
    reveal_per_step = max(1, math.ceil(answer_length / total_steps))
    trajectory: list[dict[str, Any]] = []
    for step in range(total_steps):
        masked = [position for position in answer_positions if int(state[0, position]) == mask_id]
        if not masked:
            break
        logits = model(input_ids=state).logits[0].float()
        eval_count += 1
        proposals: list[tuple[float, int, int]] = []
        for position in masked:
            probs = F.softmax(logits[position], dim=-1)
            confidence, token_id = probs.max(dim=-1)
            proposals.append((float(confidence), position, int(token_id)))
        proposals.sort(reverse=True)
        committed = proposals[: min(reveal_per_step, len(proposals))]
        for confidence, position, token_id in committed:
            state[0, position] = token_id
        trajectory.append(
            {
                "step": step,
                "masked_before": len(masked),
                "committed": [
                    {"position": position - len(prompt_ids), "token_id": token_id, "confidence": confidence}
                    for confidence, position, token_id in committed
                ],
            }
        )
    output_ids = state[0, answer_positions].detach().cpu().tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {
        "output_text": text,
        "output_token_ids": output_ids,
        "malformed": any(token_id == mask_id for token_id in output_ids),
        "model_eval_count": eval_count,
        "trajectory": trajectory,
    }


@torch.no_grad()
def denoise_answer_spans_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    answer_lengths: Sequence[int],
    *,
    steps: int | None = None,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Batched equivalent of :func:`denoise_answer_span`.

    Rows are grouped by answer length so every batch follows exactly the same
    reveal schedule as the scalar implementation. Left padding is masked out.
    """

    if len(prompts) != len(answer_lengths):
        raise ValueError("prompts and answer_lengths must align")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output: list[dict[str, Any] | None] = [None] * len(prompts)
    grouped: dict[int, list[int]] = {}
    for index, length in enumerate(answer_lengths):
        if int(length) <= 0:
            raise ValueError("answer lengths must be positive")
        grouped.setdefault(int(length), []).append(index)
    device = model_device(model)
    mask_id = infer_mask_id(model)
    pad_id = int(tokenizer.pad_token_id)
    for answer_length, indices in grouped.items():
        total_steps = int(steps or answer_length)
        reveal_per_step = max(1, math.ceil(answer_length / total_steps))
        for start in range(0, len(indices), batch_size):
            member_indices = indices[start : start + batch_size]
            prompt_ids = [
                list(map(int, tokenizer(prompts[index], add_special_tokens=False)["input_ids"]))
                for index in member_indices
            ]
            rows = [
                {"input_ids": ids + [mask_id] * answer_length}
                for ids in prompt_ids
            ]
            batch = pad_batch(rows, pad_id, device)
            state = batch["input_ids"]
            attention = batch["attention_mask"]
            offsets = batch["left_offsets"].tolist()
            answer_positions = [
                list(
                    range(
                        int(offsets[row_index]) + len(prompt_ids[row_index]),
                        int(offsets[row_index]) + len(prompt_ids[row_index]) + answer_length,
                    )
                )
                for row_index in range(len(member_indices))
            ]
            eval_counts = [0] * len(member_indices)
            trajectories: list[list[dict[str, Any]]] = [
                [] for _ in member_indices
            ]
            for step_index in range(total_steps):
                masked_by_row = [
                    [
                        position
                        for position in answer_positions[row_index]
                        if int(state[row_index, position]) == mask_id
                    ]
                    for row_index in range(len(member_indices))
                ]
                if not any(masked_by_row):
                    break
                logits = model(input_ids=state, attention_mask=attention).logits.float()
                for row_index, masked in enumerate(masked_by_row):
                    if not masked:
                        continue
                    eval_counts[row_index] += 1
                    proposals: list[tuple[float, int, int]] = []
                    for position in masked:
                        probs = F.softmax(logits[row_index, position], dim=-1)
                        confidence, token_id = probs.max(dim=-1)
                        proposals.append((float(confidence), position, int(token_id)))
                    proposals.sort(reverse=True)
                    committed = proposals[: min(reveal_per_step, len(proposals))]
                    for confidence, position, token_id in committed:
                        state[row_index, position] = token_id
                    trajectories[row_index].append(
                        {
                            "step": step_index,
                            "masked_before": len(masked),
                            "committed": [
                                {
                                    "position": position - answer_positions[row_index][0],
                                    "token_id": token_id,
                                    "confidence": confidence,
                                }
                                for confidence, position, token_id in committed
                            ],
                        }
                    )
            for row_index, original_index in enumerate(member_indices):
                ids = state[row_index, answer_positions[row_index]].detach().cpu().tolist()
                output[original_index] = {
                    "output_text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                    "output_token_ids": ids,
                    "malformed": any(token_id == mask_id for token_id in ids),
                    "model_eval_count": eval_counts[row_index],
                    "trajectory": trajectories[row_index],
                }
    if any(item is None for item in output):
        raise RuntimeError("Batched denoising did not produce every row")
    return [dict(item) for item in output if item is not None]


def normalized_hit(output: str, target: str) -> bool:
    normalize = lambda text: " ".join(str(text).casefold().split())
    return normalize(target) in normalize(output)


def sparse_support_kl(
    edited_logits: torch.Tensor,
    base_logits: torch.Tensor,
    extra_ids: Iterable[int] = (),
    top_k: int = 32,
) -> torch.Tensor:
    support = set(torch.topk(edited_logits, min(top_k, edited_logits.numel())).indices.tolist())
    support.update(torch.topk(base_logits, min(top_k, base_logits.numel())).indices.tolist())
    support.update(map(int, extra_ids))
    indices = torch.tensor(sorted(support), dtype=torch.long, device=edited_logits.device)
    edited_log = F.log_softmax(edited_logits[indices], dim=-1)
    base_log = F.log_softmax(base_logits[indices], dim=-1)
    return F.kl_div(base_log, edited_log, log_target=True, reduction="sum")


def exact_mask_pattern_bridge(
    costs: Mapping[tuple[int, int], float],
    n: int,
    *,
    beta: float,
    reference: Mapping[tuple[int, int], float] | None = None,
) -> dict[int, dict[int, float]]:
    """Solve the finite reveal-state Doob control by backward DP.

    ``costs[(mask, i)]`` is the cost of revealing position ``i`` from state
    bitmask ``mask``. The returned policy maps each nonterminal mask to a
    normalized next-position distribution.
    """

    terminal = (1 << n) - 1
    h: dict[int, float] = {terminal: 1.0}
    policy: dict[int, dict[int, float]] = {}
    for count in range(n - 1, -1, -1):
        for mask in range(terminal + 1):
            if mask.bit_count() != count:
                continue
            available = [index for index in range(n) if not (mask & (1 << index))]
            weights: dict[int, float] = {}
            for index in available:
                q = (
                    float(reference[(mask, index)])
                    if reference is not None
                    else 1.0 / len(available)
                )
                next_mask = mask | (1 << index)
                weights[index] = q * math.exp(-float(beta) * float(costs[(mask, index)])) * h[next_mask]
            total = sum(weights.values())
            if total <= 0 or not math.isfinite(total):
                raise FloatingPointError(f"Invalid desirability at state {mask}")
            h[mask] = total
            policy[mask] = {index: value / total for index, value in weights.items()}
    return policy
