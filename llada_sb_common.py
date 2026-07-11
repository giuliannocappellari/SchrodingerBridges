
#!/usr/bin/env python3
"""
Common utilities for Colab-scale Schrödinger bridge experiments on LLaDA.

These helpers are intentionally small and explicit. They are not meant to be
the final word on large-scale training; they are meant to let you prototype
Option 2 (bridge teacher + distillation) and Option 3 (local IMF / CSBM style
answer-span bridge) on a single Colab GPU.

Important scope notes
---------------------
1. The code works on the answer span only. That keeps the bridge state tractable.
2. The reverse process follows the official LLaDA generation style:
   predict all masked tokens, then transfer a fixed number of the most confident
   masked positions at each step.
3. Alias tokenizations are assumed to have the same token length as the main
   target string. Mixed-length aliases are filtered out.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from packaging.version import Version
from transformers import AutoModel, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional in some environments
    BitsAndBytesConfig = None  # type: ignore

try:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except Exception:  # pragma: no cover - optional in some environments
    LoraConfig = None  # type: ignore
    PeftModel = None  # type: ignore
    get_peft_model = None  # type: ignore
    prepare_model_for_kbit_training = None  # type: ignore


def disable_incompatible_peft_torchao() -> None:
    """Force PEFT to skip torchao-backed LoRA on incompatible Colab images."""
    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return
    except Exception:
        return

    try:
        if Version(torchao_version) >= Version("0.16.0"):
            return
    except Exception:
        pass

    try:
        import peft.import_utils as peft_import_utils
    except Exception:
        return

    def _torchao_disabled() -> bool:
        return False

    peft_import_utils.is_torchao_available = _torchao_disabled  # type: ignore[attr-defined]
    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = _torchao_disabled  # type: ignore[attr-defined]
    except Exception:
        pass


EVAL_BUCKET_ORDER = (
    "rewrite",
    "declarative_paraphrases",
    "qa_format_generalization",
    "near_locality",
    "far_locality",
)

GENERATION_TARGET_BUCKETS = (
    "rewrite",
    "declarative_paraphrases",
    "qa_format_generalization",
)

MODEL_EVAL_COUNTER: Dict[str, int] = {"count": 0}


@dataclass
class EvalPromptCase:
    """One evaluation prompt with its expected target aliases."""

    prompt: str
    target: str
    aliases: List[str]
    id: Optional[str] = None


@dataclass
class EditExample:
    """One edit request.

    Required JSON/JSONL fields:
        prompt: str
        target: str

    Optional fields:
        aliases: list[str]
        anchor_prompts: list[str]
        bridge_train_prompts: list[str]
        anchor_cases: list[{"prompt": str, "target": str, "aliases": list[str]}]
        eval_anchor_cases: list[{"prompt": str, "target": str, "aliases": list[str]}]
        paraphrase_prompts: list[str]
        declarative_paraphrase_prompts: list[str]
        qa_paraphrase_prompts: list[str]
        locality_cases: list[{"prompt": str, "target": str, "aliases": list[str]}]
        near_locality_cases: list[{"prompt": str, "target": str, "aliases": list[str]}]
        far_locality_cases: list[{"prompt": str, "target": str, "aliases": list[str]}]
        old_target: str
        id: str
    """

    prompt: str
    target: str
    aliases: List[str]
    anchor_prompts: List[str]
    bridge_train_prompts: List[str] = dataclasses.field(default_factory=list)
    anchor_cases: List[EvalPromptCase] = dataclasses.field(default_factory=list)
    eval_anchor_cases: List[EvalPromptCase] = dataclasses.field(default_factory=list)
    old_target: Optional[str] = None
    id: Optional[str] = None
    paraphrase_prompts: List[str] = dataclasses.field(default_factory=list)
    locality_cases: List[EvalPromptCase] = dataclasses.field(default_factory=list)
    declarative_paraphrase_prompts: List[str] = dataclasses.field(default_factory=list)
    qa_paraphrase_prompts: List[str] = dataclasses.field(default_factory=list)
    near_locality_cases: List[EvalPromptCase] = dataclasses.field(default_factory=list)
    far_locality_cases: List[EvalPromptCase] = dataclasses.field(default_factory=list)


@dataclass
class TeacherRecord:
    """Sparse teacher supervision for one masked state.

    teacher_top_ids[j] and teacher_top_probs[j] describe a sparse teacher
    distribution for answer position j. If active_mask[j] is False, those lists
    can be empty and that position is ignored in the loss.
    """

    input_ids: List[int]
    answer_abs_positions: List[int]
    active_mask: List[bool]
    teacher_top_ids: List[List[int]]
    teacher_top_probs: List[List[float]]
    target_ids: List[int]
    kind: str
    prompt: str
    edit_id: Optional[str] = None
    step_index: Optional[int] = None
    source_role: Optional[str] = None


def reset_model_eval_counter() -> None:
    MODEL_EVAL_COUNTER["count"] = 0


def increment_model_eval_counter(amount: int = 1) -> None:
    MODEL_EVAL_COUNTER["count"] += int(amount)


def get_model_eval_counter() -> int:
    return int(MODEL_EVAL_COUNTER["count"])


def default_aliases_for_text(text: str) -> List[str]:
    stripped = text.strip()
    aliases = [f" {stripped}", stripped]
    return list(dict.fromkeys(aliases))


def looks_like_qa_prompt(prompt: str) -> bool:
    lowered = prompt.strip().lower()
    return "?" in lowered or "answer:" in lowered or lowered.endswith(":")


def _parse_eval_prompt_cases(
    raw_cases: Sequence[Dict[str, Any]],
    edit_id: str,
    bucket_name: str,
) -> List[EvalPromptCase]:
    cases: List[EvalPromptCase] = []
    for j, case in enumerate(raw_cases):
        case_prompt = case["prompt"]
        case_target = case["target"]
        case_aliases = case.get("aliases", [case_target])
        case_id = case.get("id", f"{edit_id}_{bucket_name}_{j}")
        cases.append(
            EvalPromptCase(
                prompt=case_prompt,
                target=case_target,
                aliases=list(case_aliases),
                id=case_id,
            )
        )
    return cases


def _content_tokens(text: str) -> List[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "is",
        "was",
        "as",
        "that",
        "which",
        "what",
        "who",
        "answer",
        "city",
        "capital",
        "serves",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z]+", text.lower())
        if len(token) > 2 and token not in stopwords
    ]


def _is_near_locality_prompt(
    case_prompt: str,
    edit_prompt: str,
    target: str,
    old_target: Optional[str],
) -> bool:
    case_tokens = set(_content_tokens(case_prompt))
    reference_tokens = set(_content_tokens(edit_prompt))
    reference_tokens.update(_content_tokens(target))
    if old_target:
        reference_tokens.update(_content_tokens(old_target))
    return bool(case_tokens & reference_tokens)


def _dedupe_eval_prompt_cases(cases: Sequence[EvalPromptCase]) -> List[EvalPromptCase]:
    seen = set()
    unique_cases: List[EvalPromptCase] = []
    for case in cases:
        key = (case.id, case.prompt, case.target, tuple(case.aliases))
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append(case)
    return unique_cases


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_edits(path: str) -> List[EditExample]:
    """Load JSONL or JSON list of edits."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    data: List[Dict[str, Any]]
    if path.endswith(".jsonl"):
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        obj = json.loads(raw)
        if isinstance(obj, list):
            data = obj
        else:
            raise ValueError("JSON file must contain a list of edit objects.")

    edits: List[EditExample] = []
    for i, item in enumerate(data):
        prompt = item["prompt"]
        target = item["target"]
        aliases = item.get("aliases", [target])
        anchor_prompts = item.get("anchor_prompts", [])
        bridge_train_prompts = item.get("bridge_train_prompts", [])
        old_target = item.get("old_target")
        edit_id = item.get("id", f"edit_{i}")
        anchor_cases = _parse_eval_prompt_cases(
            item.get("anchor_cases", []),
            edit_id=edit_id,
            bucket_name="anchor",
        )
        eval_anchor_cases = _parse_eval_prompt_cases(
            item.get("eval_anchor_cases", []),
            edit_id=edit_id,
            bucket_name="eval_anchor",
        )
        legacy_paraphrase_prompts = list(item.get("paraphrase_prompts", []))
        declarative_paraphrase_prompts = list(item.get("declarative_paraphrase_prompts", []))
        qa_paraphrase_prompts = list(item.get("qa_paraphrase_prompts", []))
        for legacy_prompt in legacy_paraphrase_prompts:
            if looks_like_qa_prompt(legacy_prompt):
                qa_paraphrase_prompts.append(legacy_prompt)
            else:
                declarative_paraphrase_prompts.append(legacy_prompt)

        legacy_locality_cases = _parse_eval_prompt_cases(
            item.get("locality_cases", []),
            edit_id=edit_id,
            bucket_name="locality",
        )
        near_locality_cases = _parse_eval_prompt_cases(
            item.get("near_locality_cases", []),
            edit_id=edit_id,
            bucket_name="near_locality",
        )
        far_locality_cases = _parse_eval_prompt_cases(
            item.get("far_locality_cases", []),
            edit_id=edit_id,
            bucket_name="far_locality",
        )
        for legacy_case in legacy_locality_cases:
            if _is_near_locality_prompt(
                case_prompt=legacy_case.prompt,
                edit_prompt=prompt,
                target=target,
                old_target=old_target,
            ):
                near_locality_cases.append(legacy_case)
            else:
                far_locality_cases.append(legacy_case)

        paraphrase_prompts = list(
            dict.fromkeys(declarative_paraphrase_prompts + qa_paraphrase_prompts)
        )
        near_locality_cases = _dedupe_eval_prompt_cases(near_locality_cases)
        far_locality_cases = _dedupe_eval_prompt_cases(far_locality_cases)
        locality_cases = _dedupe_eval_prompt_cases(
            list(near_locality_cases) + list(far_locality_cases)
        )
        edits.append(
            EditExample(
                prompt=prompt,
                target=target,
                aliases=list(aliases),
                anchor_prompts=list(anchor_prompts),
                bridge_train_prompts=list(dict.fromkeys(bridge_train_prompts)),
                anchor_cases=anchor_cases,
                eval_anchor_cases=eval_anchor_cases,
                old_target=old_target,
                id=edit_id,
                paraphrase_prompts=paraphrase_prompts,
                locality_cases=locality_cases,
                declarative_paraphrase_prompts=list(dict.fromkeys(declarative_paraphrase_prompts)),
                qa_paraphrase_prompts=list(dict.fromkeys(qa_paraphrase_prompts)),
                near_locality_cases=list(near_locality_cases),
                far_locality_cases=list(far_locality_cases),
            )
        )
    return edits


def load_llada_model_and_tokenizer(
    model_id: str,
    dtype_name: str = "float16",
    use_4bit: bool = True,
    device_map: str = "auto",
):
    """Load LLaDA using trust_remote_code=True.

    We first try 4-bit loading when requested, because Colab memory is usually
    the bottleneck. If that fails, we fall back to dense loading.
    """
    dtype = choose_dtype(dtype_name)
    load_kwargs: Dict[str, Any] = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )

    if use_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError(
                "BitsAndBytesConfig is unavailable. Install bitsandbytes and a "
                "recent transformers build, or set --use_4bit 0."
            )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    try:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        if not use_4bit:
            raise

        error_text = str(exc)
        if "bitsandbytes" in error_text and "0.46.1" in error_text:
            raise RuntimeError(
                "4-bit loading failed because this runtime needs "
                "bitsandbytes>=0.46.1. In Colab, run "
                "`pip install -U \"bitsandbytes>=0.46.1\"`, restart the "
                "runtime, and rerun with --use_4bit 1."
            ) from exc

        print(f"[WARN] 4-bit load failed: {exc}")
        print("[WARN] Falling back to dense loading.")
        dense_kwargs = dict(load_kwargs)
        dense_kwargs.pop("quantization_config", None)
        if dense_kwargs.get("device_map") == "auto":
            dense_kwargs["device_map"] = None
            print(
                "[WARN] Dense fallback disables device_map='auto' because "
                "the public LLaDA remote class is not compatible with "
                "Transformers auto device mapping in this runtime."
            )
        model = AutoModel.from_pretrained(model_id, **dense_kwargs)
        if dense_kwargs.get("device_map") is None and torch.cuda.is_available():
            try:
                model = model.to("cuda")
            except Exception as move_exc:
                raise RuntimeError(
                    "Dense fallback loaded the model but could not move it to "
                    "CUDA. On Colab, this usually means you should fix 4-bit "
                    "loading instead of relying on dense fallback."
                ) from move_exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    mask_id = getattr(model.config, "mask_token_id", 126336)
    if getattr(tokenizer, "pad_token_id", None) == mask_id:
        # The official LLaDA generation script requires pad_token_id != mask_id.
        # Use eos as padding when possible.
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                "Tokenizer pad token equals mask token and no eos token is available."
            )

    model.eval()
    return model, tokenizer


def get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        return p.device
    raise ValueError("Could not infer model device.")


def infer_mask_id(model) -> int:
    return int(getattr(model.config, "mask_token_id", 126336))


def tokenize_prompt(tokenizer, prompt: str) -> List[int]:
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def tokenize_aliases_same_length(
    tokenizer,
    target: str,
    aliases: Sequence[str],
) -> Tuple[List[int], List[List[int]]]:
    """Tokenize target and keep only aliases with the same token length."""
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    alias_token_lists: List[List[int]] = []
    for alias in aliases:
        alias_ids = tokenizer(alias, add_special_tokens=False)["input_ids"]
        if len(alias_ids) == len(target_ids):
            alias_token_lists.append(alias_ids)
    if not alias_token_lists:
        alias_token_lists = [target_ids]
    return target_ids, alias_token_lists


def build_initial_state(
    prompt_ids: Sequence[int],
    answer_len: int,
    mask_id: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.full((1, len(prompt_ids) + answer_len), mask_id, dtype=torch.long, device=device)
    x[0, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    return x


def build_attention_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x, dtype=torch.long)


def build_transfer_schedule(num_masked: int, steps: int) -> List[int]:
    """Match the official LLaDA logic for distributing transitions across steps."""
    if steps <= 0:
        raise ValueError("steps must be >= 1")
    base = num_masked // steps
    remainder = num_masked % steps
    schedule = [base] * steps
    for i in range(remainder):
        schedule[i] += 1
    return schedule


@torch.no_grad()
def run_logits(model, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = build_attention_mask(x)
    increment_model_eval_counter()
    return model(x, attention_mask=attention_mask).logits


def sanitize_logits_row(logits: torch.Tensor, clamp_value: float = 80.0) -> torch.Tensor:
    """Return a finite float32 logits vector safe for softmax/log-softmax."""
    logits = torch.nan_to_num(
        logits.float(),
        nan=0.0,
        posinf=clamp_value,
        neginf=-clamp_value,
    )
    return logits.clamp(min=-clamp_value, max=clamp_value)


def normalize_probability_vector(probs: torch.Tensor) -> torch.Tensor:
    """Project a 1D tensor onto a valid probability simplex."""
    probs = torch.nan_to_num(
        probs.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    probs = probs.clamp_min(0.0)
    total = probs.sum()
    if probs.numel() == 0:
        raise ValueError("Probability vector must be non-empty.")
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return torch.full_like(probs, 1.0 / probs.numel())
    return probs / total


def sample_token_from_probs(probs: torch.Tensor, temperature: float = 1.0) -> Tuple[int, float]:
    """Sample one token from a 1D probability vector."""
    probs = normalize_probability_vector(probs)
    if temperature <= 0:
        token_id = int(torch.argmax(probs).item())
        confidence = float(probs[token_id].item())
        return token_id, confidence

    logits = torch.log(probs.clamp_min(1e-12)) / temperature
    scaled_probs = normalize_probability_vector(torch.softmax(logits, dim=-1))
    token = int(torch.multinomial(scaled_probs, num_samples=1).item())
    confidence = float(scaled_probs[token].item())
    return token, confidence


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=True)


def exact_alias_match(answer_ids: Sequence[int], alias_token_lists: Sequence[Sequence[int]]) -> bool:
    return any(list(answer_ids) == list(alias) for alias in alias_token_lists)


def soft_overlap_score(answer_ids: Sequence[int], alias_token_lists: Sequence[Sequence[int]]) -> float:
    """A soft endpoint score used when exact bridge support is too sparse.

    Score is in [0, 1]. Exact match gets 1. Otherwise we compute the best
    position-wise token overlap fraction across aliases.
    """
    ans = list(answer_ids)
    best = 0.0
    for alias in alias_token_lists:
        alias_list = list(alias)
        overlap = sum(int(a == b) for a, b in zip(ans, alias_list))
        best = max(best, overlap / max(1, len(alias_list)))
    return best


def endpoint_reward(
    answer_ids: Sequence[int],
    alias_token_lists: Sequence[Sequence[int]],
    reward_mode: str = "soft_overlap",
    reward_beta: float = 6.0,
) -> float:
    """Return a positive endpoint reward.

    hard_exact:
        1.0 on exact match, 1e-6 otherwise.
    soft_overlap:
        exp(beta * overlap_fraction), which is always positive and gives
        smoother guidance on small Colab-scale rollouts.
    """
    if reward_mode == "hard_exact":
        return 1.0 if exact_alias_match(answer_ids, alias_token_lists) else 1e-6
    if reward_mode == "soft_overlap":
        overlap = soft_overlap_score(answer_ids, alias_token_lists)
        return float(math.exp(reward_beta * overlap))
    raise ValueError(f"Unsupported reward_mode: {reward_mode}")


@torch.no_grad()
def local_reference_rollout(
    model,
    x_start: torch.Tensor,
    prompt_len: int,
    answer_len: int,
    steps_remaining: int,
    mask_id: int,
    temperature: float = 1.0,
) -> List[int]:
    """Roll out the frozen answer-span process under the reference Q.

    This follows the same core mechanism as the official generation code:
    at each step, predict all masked tokens, then permanently transfer a
    fixed number of the most confident masked positions.
    """
    if steps_remaining <= 0:
        return x_start[0, prompt_len : prompt_len + answer_len].tolist()

    device = x_start.device
    x = x_start.clone()
    answer_abs_positions = list(range(prompt_len, prompt_len + answer_len))

    for num_transfer in build_transfer_schedule(
        num_masked=int((x[0, answer_abs_positions] == mask_id).sum().item()),
        steps=steps_remaining,
    ):
        if num_transfer <= 0:
            continue

        logits = run_logits(model, x)[:, answer_abs_positions, :][0]
        probs = torch.softmax(logits, dim=-1)

        candidates: List[Tuple[float, int, int]] = []  # (confidence, rel_pos, token_id)
        for rel_pos in range(answer_len):
            abs_pos = answer_abs_positions[rel_pos]
            if int(x[0, abs_pos].item()) != mask_id:
                continue
            token_id, confidence = sample_token_from_probs(
                probs[rel_pos], temperature=temperature
            )
            candidates.append((confidence, rel_pos, token_id))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0], reverse=True)
        for confidence, rel_pos, token_id in candidates[:num_transfer]:
            abs_pos = answer_abs_positions[rel_pos]
            x[0, abs_pos] = token_id

    return x[0, prompt_len : prompt_len + answer_len].tolist()


@torch.no_grad()
def simple_generate_answer(
    model,
    prompt_ids: Sequence[int],
    answer_len: int,
    mask_id: int,
    steps: int,
    temperature: float = 0.0,
) -> List[int]:
    device = get_model_device(model)
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)
    return local_reference_rollout(
        model=model,
        x_start=x,
        prompt_len=len(prompt_ids),
        answer_len=answer_len,
        steps_remaining=steps,
        mask_id=mask_id,
        temperature=temperature,
    )


@torch.no_grad()
def average_target_log_likelihood(
    model,
    tokenizer,
    prompt: str,
    target: str,
) -> float:
    """Length-normalized target score.

    Prefer the model's native likelihood hook when available. Otherwise fall
    back to masked pseudo-log-likelihood over the answer span.
    """
    prompt_ids = tokenize_prompt(tokenizer, prompt)
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return float("-inf")

    native_get_ll = getattr(model, "get_log_likelihood", None)
    if callable(native_get_ll):
        try:
            increment_model_eval_counter()
            native_score = native_get_ll(
                prompt,
                target,
                tokenizer=tokenizer,
            )
            native_score = float(native_score)
            if math.isfinite(native_score):
                return native_score / max(1, len(target_ids))
        except TypeError:
            try:
                increment_model_eval_counter()
                native_score = native_get_ll(prompt, target)
                native_score = float(native_score)
                if math.isfinite(native_score):
                    return native_score / max(1, len(target_ids))
            except Exception:
                pass
        except Exception:
            pass

    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    full_ids = prompt_ids + target_ids
    total_log_prob = 0.0
    prompt_len = len(prompt_ids)

    for rel_pos, target_id in enumerate(target_ids):
        input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        input_ids[0, prompt_len + rel_pos] = mask_id
        logits = run_logits(model, input_ids)[0]
        row = sanitize_logits_row(logits[prompt_len + rel_pos])
        total_log_prob += float(torch.log_softmax(row, dim=-1)[target_id].item())

    return total_log_prob / max(1, len(target_ids))


def best_alias_average_log_likelihood(
    model,
    tokenizer,
    prompt: str,
    aliases: Sequence[str],
) -> float:
    alias_scores = [
        average_target_log_likelihood(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=alias,
        )
        for alias in aliases
    ]
    if not alias_scores:
        return float("-inf")
    return float(max(alias_scores))


def length_normalized_preference_stats(
    model,
    tokenizer,
    prompt: str,
    preferred_aliases: Sequence[str],
    competing_aliases: Sequence[str],
) -> Dict[str, float]:
    preferred_score = best_alias_average_log_likelihood(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        aliases=preferred_aliases,
    )
    competing_score = best_alias_average_log_likelihood(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        aliases=competing_aliases,
    )
    margin = preferred_score - competing_score
    return {
        "preferred_avg_logp": float(preferred_score),
        "competing_avg_logp": float(competing_score),
        "likelihood_margin": float(margin),
        "likelihood_success": float(preferred_score > competing_score),
    }


def first_token_ids_for_aliases(
    tokenizer,
    aliases: Sequence[str],
) -> List[int]:
    token_ids: List[int] = []
    for alias in aliases:
        ids = tokenizer(alias, add_special_tokens=False)["input_ids"]
        if ids:
            token_ids.append(int(ids[0]))
    return list(dict.fromkeys(token_ids))


def distribution_entropy(probs: torch.Tensor) -> float:
    probs = normalize_probability_vector(probs)
    return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())


def preference_stats_from_dense_row(
    row: torch.Tensor,
    preferred_token_ids: Sequence[int],
    competing_token_ids: Sequence[int],
) -> Dict[str, Optional[float]]:
    row = sanitize_logits_row(row)
    probs = normalize_probability_vector(torch.softmax(row, dim=-1))
    log_probs = torch.log(probs.clamp_min(1e-12))

    preferred_rank = None
    if preferred_token_ids:
        preferred_rank = float(
            min(int((row > row[token_id]).sum().item()) + 1 for token_id in preferred_token_ids)
        )

    preferred_score = (
        max(float(log_probs[token_id].item()) for token_id in preferred_token_ids)
        if preferred_token_ids
        else None
    )
    competing_score = (
        max(float(log_probs[token_id].item()) for token_id in competing_token_ids)
        if competing_token_ids
        else None
    )
    first_token_margin = (
        None
        if preferred_score is None or competing_score is None
        else float(preferred_score - competing_score)
    )

    return {
        "preferred_first_token_rank": preferred_rank,
        "first_token_margin": first_token_margin,
        "answer_token_entropy": distribution_entropy(probs),
    }


def preference_stats_from_sparse_distribution(
    candidate_ids: Sequence[int],
    candidate_probs: Sequence[float],
    preferred_token_ids: Sequence[int],
    competing_token_ids: Sequence[int],
) -> Dict[str, Optional[float]]:
    probs = normalize_probability_vector(
        torch.tensor(list(candidate_probs), dtype=torch.float32)
    )
    prob_by_id = {
        int(token_id): float(prob.item())
        for token_id, prob in zip(candidate_ids, probs)
    }
    ranked_ids = [
        token_id
        for _, token_id in sorted(
            zip(probs.detach().cpu().tolist(), candidate_ids),
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    preferred_rank = None
    if preferred_token_ids:
        rank_candidates = [
            idx + 1
            for idx, token_id in enumerate(ranked_ids)
            if int(token_id) in preferred_token_ids
        ]
        preferred_rank = float(min(rank_candidates)) if rank_candidates else float(len(ranked_ids) + 1)

    preferred_score = (
        math.log(max(prob_by_id.get(int(token_id), 1e-12) for token_id in preferred_token_ids))
        if preferred_token_ids
        else None
    )
    competing_score = (
        math.log(max(prob_by_id.get(int(token_id), 1e-12) for token_id in competing_token_ids))
        if competing_token_ids
        else None
    )
    first_token_margin = (
        None
        if preferred_score is None or competing_score is None
        else float(preferred_score - competing_score)
    )

    return {
        "preferred_first_token_rank": preferred_rank,
        "first_token_margin": first_token_margin,
        "answer_token_entropy": distribution_entropy(probs),
    }


@torch.no_grad()
def first_step_model_preference_stats(
    model,
    tokenizer,
    prompt: str,
    answer_len: int,
    preferred_aliases: Sequence[str],
    competing_aliases: Sequence[str],
) -> Dict[str, Optional[float]]:
    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, prompt)
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)
    row = run_logits(model, x)[0, len(prompt_ids)]
    preferred_token_ids = first_token_ids_for_aliases(tokenizer, preferred_aliases)
    competing_token_ids = first_token_ids_for_aliases(tokenizer, competing_aliases)
    return preference_stats_from_dense_row(
        row=row,
        preferred_token_ids=preferred_token_ids,
        competing_token_ids=competing_token_ids,
    )


@torch.no_grad()
def first_step_bridge_preference_stats(
    model,
    tokenizer,
    prompt_text: str,
    guided_target_text: str,
    guided_aliases: Sequence[str],
    answer_len: int,
    steps: int,
    bridge_topk: int,
    mc_rollouts: int,
    guidance_scale: float,
    reward_mode: str,
    reward_beta: float,
    preferred_aliases: Sequence[str],
    competing_aliases: Sequence[str],
) -> Dict[str, Optional[float]]:
    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, prompt_text)
    prompt_len = len(prompt_ids)
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)
    row = sanitize_logits_row(run_logits(model, x)[0, prompt_len])

    guidance_alias_pool = list(
        dict.fromkeys(list(guided_aliases) + default_aliases_for_text(guided_target_text))
    )
    guidance_alias_token_lists = tokenize_alias_lists_for_target_length(
        tokenizer,
        guidance_alias_pool,
        answer_len,
    )
    base_probs = normalize_probability_vector(torch.softmax(row, dim=-1))
    top_ids = torch.topk(base_probs, k=bridge_topk).indices.detach().cpu().tolist()
    candidate_ids = ensure_topk_contains_targets(
        base_top_ids=top_ids,
        alias_token_lists=guidance_alias_token_lists,
        rel_pos=0,
    )
    bridge_scores = estimate_candidate_bridge_scores(
        model=model,
        x=x,
        prompt_len=prompt_len,
        answer_len=answer_len,
        rel_pos=0,
        candidate_ids=candidate_ids,
        alias_token_lists=guidance_alias_token_lists,
        remaining_steps=max(0, int(steps) - 1),
        reward_mode=reward_mode,
        reward_beta=reward_beta,
        mc_rollouts=mc_rollouts,
        mask_id=mask_id,
    )
    sparse_ids, sparse_probs = gather_sparse_teacher_distribution(
        logits_row=row,
        candidate_ids=candidate_ids,
        bridge_scores=bridge_scores,
        guidance_scale=guidance_scale,
    )
    preferred_token_ids = first_token_ids_for_aliases(tokenizer, preferred_aliases)
    competing_token_ids = first_token_ids_for_aliases(tokenizer, competing_aliases)
    return preference_stats_from_sparse_distribution(
        candidate_ids=sparse_ids,
        candidate_probs=sparse_probs,
        preferred_token_ids=preferred_token_ids,
        competing_token_ids=competing_token_ids,
    )


@torch.no_grad()
def estimate_candidate_bridge_scores(
    model,
    x: torch.Tensor,
    prompt_len: int,
    answer_len: int,
    rel_pos: int,
    candidate_ids: Sequence[int],
    alias_token_lists: Sequence[Sequence[int]],
    remaining_steps: int,
    reward_mode: str,
    reward_beta: float,
    mc_rollouts: int,
    mask_id: int,
) -> List[float]:
    """Estimate future-success scores h(v) on a candidate set."""
    scores: List[float] = []
    abs_pos = prompt_len + rel_pos

    for token_id in candidate_ids:
        x_tent = x.clone()
        x_tent[0, abs_pos] = int(token_id)

        if remaining_steps <= 0:
            final_answer = x_tent[0, prompt_len : prompt_len + answer_len].tolist()
            scores.append(
                endpoint_reward(
                    final_answer,
                    alias_token_lists,
                    reward_mode=reward_mode,
                    reward_beta=reward_beta,
                )
            )
            continue

        reward_sum = 0.0
        for _ in range(mc_rollouts):
            out = local_reference_rollout(
                model=model,
                x_start=x_tent,
                prompt_len=prompt_len,
                answer_len=answer_len,
                steps_remaining=remaining_steps,
                mask_id=mask_id,
                temperature=1.0,
            )
            reward_sum += endpoint_reward(
                out,
                alias_token_lists,
                reward_mode=reward_mode,
                reward_beta=reward_beta,
            )
        scores.append(reward_sum / max(1, mc_rollouts))

    return scores


def ensure_topk_contains_targets(
    base_top_ids: List[int],
    alias_token_lists: Sequence[Sequence[int]],
    rel_pos: int,
) -> List[int]:
    needed = {int(alias[rel_pos]) for alias in alias_token_lists}
    merged = list(dict.fromkeys(base_top_ids + list(needed)))
    return merged


def gather_sparse_teacher_distribution(
    logits_row: torch.Tensor,
    candidate_ids: Sequence[int],
    bridge_scores: Sequence[float],
    guidance_scale: float,
) -> Tuple[List[int], List[float]]:
    """Build the approximate bridge teacher on a candidate set.

    Teacher(v) ∝ q(v | x_k) * h(v)^alpha
    which becomes log q + alpha log h in log-space.
    """
    device = logits_row.device
    cand = torch.tensor(list(candidate_ids), dtype=torch.long, device=device)
    base_log_probs = torch.log_softmax(logits_row, dim=-1)[cand]
    bridge_tensor = torch.tensor(list(bridge_scores), dtype=base_log_probs.dtype, device=device)
    bridge_tensor = torch.log(bridge_tensor.clamp_min(1e-12))
    guided = base_log_probs + guidance_scale * bridge_tensor
    guided_probs = torch.softmax(guided, dim=-1)
    return cand.tolist(), guided_probs.detach().cpu().tolist()


def alias_position_probability_mass(
    candidate_ids: Sequence[int],
    guided_probs: Sequence[float],
    alias_token_lists: Sequence[Sequence[int]],
    rel_pos: int,
) -> float:
    if not alias_token_lists:
        return 0.0
    token_ids = {
        int(alias[rel_pos])
        for alias in alias_token_lists
        if rel_pos < len(alias)
    }
    if not token_ids:
        return 0.0
    total = 0.0
    for token_id, prob in zip(candidate_ids, guided_probs):
        if int(token_id) in token_ids:
            total += float(prob)
    return float(total)


def normalize_probs(probs: Sequence[float]) -> List[float]:
    if not probs:
        return []
    total = float(sum(probs))
    if total <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [float(p / total) for p in probs]


def tokenize_alias_lists_for_target_length(
    tokenizer,
    aliases: Sequence[str],
    target_len: int,
) -> List[List[int]]:
    token_lists: List[List[int]] = []
    for alias in aliases:
        alias_ids = tokenizer(alias, add_special_tokens=False)["input_ids"]
        if len(alias_ids) == target_len:
            token_lists.append(alias_ids)
    return token_lists


def make_teacher_record(
    x: torch.Tensor,
    answer_abs_positions: Sequence[int],
    active_mask: Sequence[bool],
    teacher_top_ids: Sequence[Sequence[int]],
    teacher_top_probs: Sequence[Sequence[float]],
    target_ids: Sequence[int],
    kind: str,
    prompt: str,
    edit_id: Optional[str] = None,
    step_index: Optional[int] = None,
    source_role: Optional[str] = None,
) -> TeacherRecord:
    return TeacherRecord(
        input_ids=x[0].detach().cpu().tolist(),
        answer_abs_positions=[int(p) for p in answer_abs_positions],
        active_mask=[bool(v) for v in active_mask],
        teacher_top_ids=[list(map(int, ids)) for ids in teacher_top_ids],
        teacher_top_probs=[normalize_probs(probs) for probs in teacher_top_probs],
        target_ids=[int(t) for t in target_ids],
        kind=kind,
        prompt=prompt,
        edit_id=edit_id,
        step_index=None if step_index is None else int(step_index),
        source_role=source_role,
    )


@torch.no_grad()
def build_bridge_teacher_record_from_state(
    model,
    tokenizer,
    edit: EditExample,
    prompt_text: str,
    x_state: torch.Tensor,
    steps: int,
    step_index: int,
    bridge_topk: int,
    teacher_topk: int,
    mc_rollouts: int,
    guidance_scale: float,
    reward_mode: str,
    reward_beta: float,
    kind: str = "bridge",
    source_role: Optional[str] = None,
) -> Optional[TeacherRecord]:
    """Build one sparse bridge teacher record from an arbitrary masked state."""
    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, prompt_text)
    target_ids, alias_token_lists = tokenize_aliases_same_length(
        tokenizer,
        edit.target,
        edit.aliases,
    )
    answer_len = len(target_ids)
    prompt_len = len(prompt_ids)
    answer_abs_positions = list(range(prompt_len, prompt_len + answer_len))
    x = x_state.clone().to(device)
    logits = run_logits(model, x)[0]

    active_mask: List[bool] = []
    teacher_top_ids: List[List[int]] = []
    teacher_top_probs: List[List[float]] = []
    remaining_steps = max(0, int(steps) - int(step_index) - 1)

    for rel_pos in range(answer_len):
        abs_pos = answer_abs_positions[rel_pos]
        if int(x[0, abs_pos].item()) != mask_id:
            active_mask.append(False)
            teacher_top_ids.append([])
            teacher_top_probs.append([])
            continue

        active_mask.append(True)
        row = sanitize_logits_row(logits[abs_pos])
        base_probs = normalize_probability_vector(torch.softmax(row, dim=-1))
        top_ids = torch.topk(base_probs, k=bridge_topk).indices.detach().cpu().tolist()
        candidate_ids = ensure_topk_contains_targets(
            base_top_ids=top_ids,
            alias_token_lists=alias_token_lists,
            rel_pos=rel_pos,
        )
        bridge_scores = estimate_candidate_bridge_scores(
            model=model,
            x=x,
            prompt_len=prompt_len,
            answer_len=answer_len,
            rel_pos=rel_pos,
            candidate_ids=candidate_ids,
            alias_token_lists=alias_token_lists,
            remaining_steps=remaining_steps,
            reward_mode=reward_mode,
            reward_beta=reward_beta,
            mc_rollouts=mc_rollouts,
            mask_id=mask_id,
        )
        sparse_ids, sparse_probs = gather_sparse_teacher_distribution(
            logits_row=row,
            candidate_ids=candidate_ids,
            bridge_scores=bridge_scores,
            guidance_scale=guidance_scale,
        )
        keep_k = min(teacher_topk, len(sparse_ids))
        prob_tensor = normalize_probability_vector(
            torch.tensor(sparse_probs, dtype=torch.float32, device=row.device)
        )
        top_teacher_idx = torch.topk(prob_tensor, k=keep_k).indices
        kept_ids = [int(sparse_ids[int(i.item())]) for i in top_teacher_idx]
        kept_probs = [float(prob_tensor[int(i.item())].item()) for i in top_teacher_idx]
        teacher_top_ids.append(kept_ids)
        teacher_top_probs.append(normalize_probs(kept_probs))

    if not any(active_mask):
        return None

    return make_teacher_record(
        x=x,
        answer_abs_positions=answer_abs_positions,
        active_mask=active_mask,
        teacher_top_ids=teacher_top_ids,
        teacher_top_probs=teacher_top_probs,
        target_ids=target_ids,
        kind=kind,
        prompt=prompt_text,
        edit_id=edit.id,
        step_index=step_index,
        source_role=source_role,
    )


@torch.no_grad()
def rollout_model_states(
    model,
    prompt_ids: Sequence[int],
    answer_len: int,
    mask_id: int,
    steps: int,
    temperature: float = 1.0,
) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
    """Run the model's native masked rollout and keep each pre-transfer state."""
    device = get_model_device(model)
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)
    prompt_len = len(prompt_ids)
    answer_abs_positions = list(range(prompt_len, prompt_len + answer_len))
    state_trace: List[Tuple[int, torch.Tensor]] = []

    for step_index, num_transfer in enumerate(build_transfer_schedule(answer_len, steps)):
        masked_now = int((x[0, answer_abs_positions] == mask_id).sum().item())
        if masked_now <= 0:
            break
        state_trace.append((step_index, x.clone()))
        if num_transfer <= 0:
            continue

        logits = run_logits(model, x)[:, answer_abs_positions, :][0]
        candidates: List[Tuple[float, int, int]] = []
        for rel_pos in range(answer_len):
            abs_pos = answer_abs_positions[rel_pos]
            if int(x[0, abs_pos].item()) != mask_id:
                continue
            row_probs = normalize_probability_vector(
                torch.softmax(sanitize_logits_row(logits[rel_pos]), dim=-1)
            )
            sampled_idx, confidence = sample_token_from_probs(
                row_probs,
                temperature=temperature,
            )
            candidates.append((confidence, rel_pos, int(sampled_idx)))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, rel_pos, token_id in candidates[:num_transfer]:
            abs_pos = answer_abs_positions[rel_pos]
            x[0, abs_pos] = token_id

    return state_trace, x[0, prompt_len : prompt_len + answer_len].tolist()


@torch.no_grad()
def bridge_guided_rollout(
    model,
    tokenizer,
    prompt_text: str,
    guided_target_text: str,
    guided_aliases: Sequence[str],
    steps: int,
    bridge_topk: int,
    mc_rollouts: int,
    guidance_scale: float,
    reward_mode: str,
    reward_beta: float,
    answer_len: Optional[int] = None,
    competing_aliases: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Run the Option 2 bridge directly at inference time."""
    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, prompt_text)
    guidance_alias_pool = list(
        dict.fromkeys(list(guided_aliases) + default_aliases_for_text(guided_target_text))
    )
    if answer_len is None:
        target_ids, guidance_alias_token_lists = tokenize_aliases_same_length(
            tokenizer,
            guided_target_text,
            guidance_alias_pool,
        )
        answer_len = len(target_ids)
    else:
        guidance_alias_token_lists = tokenize_alias_lists_for_target_length(
            tokenizer,
            guidance_alias_pool,
            int(answer_len),
        )
    competing_token_lists = tokenize_alias_lists_for_target_length(
        tokenizer,
        competing_aliases or [],
        int(answer_len),
    )
    prompt_len = len(prompt_ids)
    answer_abs_positions = list(range(prompt_len, prompt_len + answer_len))
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)

    bridge_new_scores: List[float] = []
    bridge_old_scores: List[float] = []

    for step_index, num_transfer in enumerate(build_transfer_schedule(answer_len, steps)):
        masked_now = int((x[0, answer_abs_positions] == mask_id).sum().item())
        if masked_now <= 0:
            break
        logits = run_logits(model, x)[0]
        candidate_payloads: List[Tuple[float, int, int, int]] = []
        remaining_steps = max(0, steps - step_index - 1)

        for rel_pos in range(answer_len):
            abs_pos = answer_abs_positions[rel_pos]
            if int(x[0, abs_pos].item()) != mask_id:
                continue

            row = sanitize_logits_row(logits[abs_pos])
            base_probs = normalize_probability_vector(torch.softmax(row, dim=-1))
            top_ids = torch.topk(base_probs, k=bridge_topk).indices.detach().cpu().tolist()
            candidate_ids = ensure_topk_contains_targets(
                base_top_ids=top_ids,
                alias_token_lists=guidance_alias_token_lists,
                rel_pos=rel_pos,
            )
            bridge_scores = estimate_candidate_bridge_scores(
                model=model,
                x=x,
                prompt_len=prompt_len,
                answer_len=answer_len,
                rel_pos=rel_pos,
                candidate_ids=candidate_ids,
                alias_token_lists=guidance_alias_token_lists,
                remaining_steps=remaining_steps,
                reward_mode=reward_mode,
                reward_beta=reward_beta,
                mc_rollouts=mc_rollouts,
                mask_id=mask_id,
            )
            sparse_ids, sparse_probs = gather_sparse_teacher_distribution(
                logits_row=row,
                candidate_ids=candidate_ids,
                bridge_scores=bridge_scores,
                guidance_scale=guidance_scale,
            )
            guided_probs = normalize_probability_vector(
                torch.tensor(sparse_probs, dtype=torch.float32, device=row.device)
            )
            bridge_new_scores.append(
                alias_position_probability_mass(
                    sparse_ids,
                    guided_probs.detach().cpu().tolist(),
                    guidance_alias_token_lists,
                    rel_pos,
                )
            )
            bridge_old_scores.append(
                alias_position_probability_mass(
                    sparse_ids,
                    guided_probs.detach().cpu().tolist(),
                    competing_token_lists,
                    rel_pos,
                )
            )
            sampled_local_idx, confidence = sample_token_from_probs(
                guided_probs,
                temperature=temperature,
            )
            sampled_token_id = int(sparse_ids[sampled_local_idx])
            candidate_payloads.append((confidence, rel_pos, sampled_token_id, abs_pos))

        if not candidate_payloads:
            break
        if num_transfer <= 0:
            continue

        candidate_payloads.sort(key=lambda item: item[0], reverse=True)
        for _, rel_pos, sampled_token_id, abs_pos in candidate_payloads[:num_transfer]:
            x[0, abs_pos] = sampled_token_id

    answer_ids = x[0, prompt_len : prompt_len + answer_len].tolist()
    bridge_new_score = mean_or_none(bridge_new_scores)
    bridge_old_score = mean_or_none(bridge_old_scores)
    bridge_margin = (
        None
        if bridge_new_score is None or bridge_old_score is None
        else float(bridge_new_score - bridge_old_score)
    )
    return {
        "answer_ids": answer_ids,
        "bridge_new_score": bridge_new_score,
        "bridge_old_score": bridge_old_score,
        "bridge_margin": bridge_margin,
    }


def convert_teacher_records_to_direct_supervision(
    records: Sequence[TeacherRecord],
) -> List[TeacherRecord]:
    direct_records: List[TeacherRecord] = []
    for record in records:
        direct_top_ids: List[List[int]] = []
        direct_top_probs: List[List[float]] = []
        for rel_pos, is_active in enumerate(record.active_mask):
            if not is_active:
                direct_top_ids.append([])
                direct_top_probs.append([])
                continue
            target_id = int(record.target_ids[rel_pos])
            direct_top_ids.append([target_id])
            direct_top_probs.append([1.0])

        direct_records.append(
            TeacherRecord(
                input_ids=list(record.input_ids),
                answer_abs_positions=list(record.answer_abs_positions),
                active_mask=list(record.active_mask),
                teacher_top_ids=direct_top_ids,
                teacher_top_probs=direct_top_probs,
                target_ids=list(record.target_ids),
                kind="direct_supervision",
                prompt=record.prompt,
                edit_id=record.edit_id,
                step_index=record.step_index,
                source_role=record.source_role,
            )
        )
    return direct_records


@torch.no_grad()
def build_reference_teacher_record(
    model,
    tokenizer,
    prompt_text: str,
    target_text: str,
    aliases: Sequence[str],
    teacher_topk: int,
    kind: str,
    ce_target_text: Optional[str] = None,
    edit_id: Optional[str] = None,
    step_index: Optional[int] = None,
    source_role: Optional[str] = None,
) -> TeacherRecord:
    device = get_model_device(model)
    mask_id = infer_mask_id(model)

    prompt_ids = tokenize_prompt(tokenizer, prompt_text)
    target_ids, _ = tokenize_aliases_same_length(tokenizer, target_text, aliases)
    if ce_target_text is None:
        ce_target_ids = list(target_ids)
    else:
        ce_target_ids, _ = tokenize_aliases_same_length(
            tokenizer,
            ce_target_text,
            [ce_target_text],
        )
        if len(ce_target_ids) != len(target_ids):
            ce_target_ids = list(target_ids)

    answer_len = len(target_ids)
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)
    prompt_len = len(prompt_ids)
    answer_abs_positions = list(range(prompt_len, prompt_len + answer_len))
    logits = run_logits(model, x)[0]

    active_mask = [True] * answer_len
    teacher_top_ids: List[List[int]] = []
    teacher_top_probs: List[List[float]] = []

    for rel_pos in range(answer_len):
        abs_pos = answer_abs_positions[rel_pos]
        probs = normalize_probability_vector(
            torch.softmax(sanitize_logits_row(logits[abs_pos]), dim=-1)
        )
        top = torch.topk(probs, k=teacher_topk)
        ids = top.indices.detach().cpu().tolist()
        p = top.values.detach().cpu().tolist()
        p = [v / max(1e-12, sum(p)) for v in p]
        teacher_top_ids.append(ids)
        teacher_top_probs.append(p)

    return make_teacher_record(
        x=x,
        answer_abs_positions=answer_abs_positions,
        active_mask=active_mask,
        teacher_top_ids=teacher_top_ids,
        teacher_top_probs=teacher_top_probs,
        target_ids=ce_target_ids,
        kind=kind,
        prompt=prompt_text,
        edit_id=edit_id,
        step_index=step_index,
        source_role=source_role,
    )


@torch.no_grad()
def build_eval_anchor_reference_records(
    model,
    tokenizer,
    edit: EditExample,
    teacher_topk: int,
) -> List[TeacherRecord]:
    records: List[TeacherRecord] = []
    for eval_anchor_case in edit.eval_anchor_cases:
        records.append(
            build_reference_teacher_record(
                model=model,
                tokenizer=tokenizer,
                prompt_text=eval_anchor_case.prompt,
                target_text=eval_anchor_case.target,
                aliases=eval_anchor_case.aliases,
                teacher_topk=teacher_topk,
                kind="eval_anchor",
                ce_target_text=eval_anchor_case.target,
                edit_id=edit.id,
                source_role="eval_anchor_ref",
            )
        )
    return records


@torch.no_grad()
def evaluate_anchor_kl_drift_from_records(
    edited_model,
    reference_records: Sequence[TeacherRecord],
) -> Optional[float]:
    if not reference_records:
        return None
    drift_values = [
        float(teacher_record_loss(edited_model, record, ce_weight=0.0).item())
        for record in reference_records
    ]
    return float(sum(drift_values) / max(1, len(drift_values)))


def find_lora_target_modules(model, target_set: str = "broad") -> List[str]:
    """Find common linear module names in the public LLaDA weights."""
    wanted_by_set = {
        "broad": {
            "q_proj",
            "k_proj",
            "v_proj",
            "attn_out",
            "ff_proj",
            "up_proj",
            "ff_out",
            "att_proj",
        },
        "mlp_only": {
            "ff_proj",
            "up_proj",
            "ff_out",
        },
    }
    if target_set not in wanted_by_set:
        raise ValueError(f"Unsupported lora target set: {target_set}")
    wanted = wanted_by_set[target_set]
    found = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in wanted and hasattr(module, "weight"):
            found.add(leaf)
    if not found:
        raise ValueError(
            "Could not find LoRA target modules automatically. "
            "Inspect model.named_modules() and pass a manual list."
        )
    return sorted(found)


def prepare_model_for_peft_training(model, use_4bit: bool):
    if use_4bit and prepare_model_for_kbit_training is not None:
        use_gradient_checkpointing = bool(
            getattr(model, "supports_gradient_checkpointing", False)
        )
        if not use_gradient_checkpointing:
            print(
                f"[WARN] {model.__class__.__name__} does not support gradient "
                "checkpointing; continuing without it."
            )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    return model


def attach_lora(
    model,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
):
    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is not installed. Install peft, or disable distillation."
        )
    disable_incompatible_peft_torchao()
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    return peft_model


def load_existing_lora_adapter(
    model,
    adapter_dir: str,
    trainable: bool = False,
):
    if PeftModel is None:
        raise ImportError(
            "peft is not installed. Install peft to load an existing LoRA adapter."
        )
    try:
        adapted = PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=trainable,
        )
    except TypeError:
        adapted = PeftModel.from_pretrained(model, adapter_dir)
    if trainable:
        mark_existing_lora_params_trainable(adapted)
    return adapted


def mark_existing_lora_params_trainable(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            param.requires_grad = True


def sparse_teacher_kl_loss(
    logits_row: torch.Tensor,
    teacher_ids: Sequence[int],
    teacher_probs: Sequence[float],
) -> torch.Tensor:
    """KL on a sparse teacher support."""
    device = logits_row.device
    ids = torch.tensor(list(teacher_ids), dtype=torch.long, device=device)
    t_probs = normalize_probability_vector(
        torch.tensor(list(teacher_probs), dtype=torch.float32, device=device)
    )
    selected = sanitize_logits_row(logits_row[ids])
    student_log_probs = torch.log_softmax(selected, dim=-1)
    return F.kl_div(student_log_probs, t_probs, reduction="batchmean")


def ce_target_loss(logits_row: torch.Tensor, target_id: int) -> torch.Tensor:
    target = torch.tensor([int(target_id)], dtype=torch.long, device=logits_row.device)
    return F.cross_entropy(sanitize_logits_row(logits_row).unsqueeze(0), target)


def teacher_record_loss(
    model,
    record: TeacherRecord,
    ce_weight: float = 0.25,
) -> torch.Tensor:
    input_ids = torch.tensor(record.input_ids, dtype=torch.long, device=get_model_device(model)).unsqueeze(0)
    attention_mask = build_attention_mask(input_ids)
    logits = model(input_ids, attention_mask=attention_mask).logits[0]
    loss = torch.zeros((), dtype=torch.float32, device=logits.device)
    active_count = 0

    for rel_pos, abs_pos in enumerate(record.answer_abs_positions):
        if not record.active_mask[rel_pos]:
            continue
        teacher_ids = record.teacher_top_ids[rel_pos]
        teacher_probs = record.teacher_top_probs[rel_pos]
        if not teacher_ids or not teacher_probs:
            continue
        row = logits[abs_pos]
        loss = loss + sparse_teacher_kl_loss(row, teacher_ids, teacher_probs)
        if ce_weight > 0:
            loss = loss + ce_weight * ce_target_loss(row, record.target_ids[rel_pos])
        active_count += 1

    if active_count == 0:
        # Return a zero loss that preserves graph/device placement.
        return logits.sum() * 0.0
    return loss / active_count


def train_lora_from_teacher_records(
    model,
    teacher_records: Sequence[TeacherRecord],
    anchor_records: Sequence[TeacherRecord],
    output_dir: str,
    use_4bit: bool,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lr: float = 5e-4,
    epochs: int = 2,
    ce_weight: float = 0.25,
    anchor_weight: float = 0.5,
    grad_clip: float = 1.0,
    seed: int = 0,
    lora_target_set: str = "broad",
) -> Any:
    seed_everything(seed)
    model.train()

    has_existing_adapter = bool(getattr(model, "peft_config", None))
    if has_existing_adapter:
        print("[INFO] Continuing training from an existing LoRA adapter.")
        mark_existing_lora_params_trainable(model)
    else:
        model = prepare_model_for_peft_training(model, use_4bit=use_4bit)
        target_modules = find_lora_target_modules(model, target_set=lora_target_set)
        print(f"[INFO] LoRA target modules: {target_modules}")
        model = attach_lora(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    teacher_records = list(teacher_records)
    anchor_records = list(anchor_records)

    step = 0
    skipped_nonfinite_loss = 0
    skipped_nonfinite_grad = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(epochs):
        random.shuffle(teacher_records)
        running = 0.0
        count = 0
        for record in teacher_records:
            optimizer.zero_grad(set_to_none=True)
            loss = teacher_record_loss(model, record, ce_weight=ce_weight)
            if anchor_records and anchor_weight > 0:
                anchor_record = random.choice(anchor_records)
                anchor_ce_weight = ce_weight if anchor_record.kind == "locality_anchor" else 0.0
                loss = loss + anchor_weight * teacher_record_loss(
                    model, anchor_record, ce_weight=anchor_ce_weight
                )
            if not torch.isfinite(loss):
                skipped_nonfinite_loss += 1
                if skipped_nonfinite_loss <= 5 or skipped_nonfinite_loss % 10 == 0:
                    print(
                        f"[WARN] Skipping non-finite loss for prompt={record.prompt!r} "
                        f"kind={record.kind} skipped_nonfinite_loss={skipped_nonfinite_loss}"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            has_nonfinite_grad = False
            for param in trainable_params:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nonfinite_grad = True
                    break
            if has_nonfinite_grad:
                skipped_nonfinite_grad += 1
                if skipped_nonfinite_grad <= 5 or skipped_nonfinite_grad % 10 == 0:
                    print(
                        f"[WARN] Skipping optimizer step due to non-finite gradients "
                        f"prompt={record.prompt!r} kind={record.kind} "
                        f"skipped_nonfinite_grad={skipped_nonfinite_grad}"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=grad_clip,
            )
            optimizer.step()
            running += float(loss.item())
            count += 1
            step += 1
            if step % 10 == 0:
                print(f"[train] epoch={epoch} step={step} mean_loss={running / max(1, count):.4f}")

        print(f"[train] epoch={epoch} mean_loss={running / max(1, count):.4f}")
        if skipped_nonfinite_loss or skipped_nonfinite_grad:
            print(
                f"[train] epoch={epoch} skipped_nonfinite_loss={skipped_nonfinite_loss} "
                f"skipped_nonfinite_grad={skipped_nonfinite_grad}"
            )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    return model


def save_teacher_records(path: str, records: Sequence[TeacherRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = [dataclasses.asdict(r) for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_teacher_records(path: str) -> List[TeacherRecord]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TeacherRecord(**item) for item in raw]


def estimate_prompt_success_rate(
    model,
    tokenizer,
    prompt: str,
    target: str,
    aliases: Sequence[str],
    steps: int,
    samples: int,
    reward_mode: str = "hard_exact",
    reward_beta: float = 6.0,
) -> Dict[str, float]:
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, prompt)
    target_ids, alias_token_lists = tokenize_aliases_same_length(tokenizer, target, aliases)
    exact = 0
    rewards = []
    for _ in range(samples):
        out = simple_generate_answer(
            model=model,
            prompt_ids=prompt_ids,
            answer_len=len(target_ids),
            mask_id=mask_id,
            steps=steps,
            temperature=1.0,
        )
        if exact_alias_match(out, alias_token_lists):
            exact += 1
        rewards.append(
            endpoint_reward(
                out,
                alias_token_lists,
                reward_mode=reward_mode,
                reward_beta=reward_beta,
            )
        )
    greedy_out = simple_generate_answer(
        model=model,
        prompt_ids=prompt_ids,
        answer_len=len(target_ids),
        mask_id=mask_id,
        steps=steps,
        temperature=0.0,
    )
    return {
        "exact_rate": exact / max(1, samples),
        "greedy_exact_rate": float(exact_alias_match(greedy_out, alias_token_lists)),
        "mean_reward": float(sum(rewards) / max(1, len(rewards))),
    }


@torch.no_grad()
def estimate_prompt_success_rate_with_bridge(
    model,
    tokenizer,
    prompt: str,
    guided_target_text: str,
    guided_aliases: Sequence[str],
    eval_target_text: str,
    eval_aliases: Sequence[str],
    competing_aliases: Sequence[str],
    steps: int,
    samples: int,
    bridge_topk: int,
    mc_rollouts: int,
    guidance_scale: float,
    reward_mode: str = "hard_exact",
    reward_beta: float = 6.0,
) -> Dict[str, Any]:
    eval_target_ids, eval_alias_token_lists = tokenize_aliases_same_length(
        tokenizer,
        eval_target_text,
        eval_aliases,
    )
    first_step_stats = first_step_bridge_preference_stats(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        guided_target_text=guided_target_text,
        guided_aliases=guided_aliases,
        answer_len=len(eval_target_ids),
        steps=steps,
        bridge_topk=bridge_topk,
        mc_rollouts=mc_rollouts,
        guidance_scale=guidance_scale,
        reward_mode=reward_mode,
        reward_beta=reward_beta,
        preferred_aliases=eval_aliases,
        competing_aliases=competing_aliases,
    )
    exact = 0
    rewards: List[float] = []
    bridge_new_scores: List[float] = []
    bridge_old_scores: List[float] = []
    bridge_margins: List[float] = []
    samples_out: List[str] = []

    for _ in range(samples):
        rollout = bridge_guided_rollout(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            guided_target_text=guided_target_text,
            guided_aliases=guided_aliases,
            steps=steps,
            bridge_topk=bridge_topk,
            mc_rollouts=mc_rollouts,
            guidance_scale=guidance_scale,
            reward_mode=reward_mode,
            reward_beta=reward_beta,
            answer_len=len(eval_target_ids),
            competing_aliases=competing_aliases,
            temperature=1.0,
        )
        answer_ids = rollout["answer_ids"]
        if exact_alias_match(answer_ids, eval_alias_token_lists):
            exact += 1
        rewards.append(
            endpoint_reward(
                answer_ids,
                eval_alias_token_lists,
                reward_mode=reward_mode,
                reward_beta=reward_beta,
            )
        )
        if rollout["bridge_new_score"] is not None:
            bridge_new_scores.append(float(rollout["bridge_new_score"]))
        if rollout["bridge_old_score"] is not None:
            bridge_old_scores.append(float(rollout["bridge_old_score"]))
        if rollout["bridge_margin"] is not None:
            bridge_margins.append(float(rollout["bridge_margin"]))
        samples_out.append(decode_ids(tokenizer, answer_ids))

    greedy_rollout = bridge_guided_rollout(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        guided_target_text=guided_target_text,
        guided_aliases=guided_aliases,
        steps=steps,
        bridge_topk=bridge_topk,
        mc_rollouts=mc_rollouts,
        guidance_scale=guidance_scale,
        reward_mode=reward_mode,
        reward_beta=reward_beta,
        answer_len=len(eval_target_ids),
        competing_aliases=competing_aliases,
        temperature=0.0,
    )

    return {
        "exact_rate": exact / max(1, samples),
        "greedy_exact_rate": float(
            exact_alias_match(greedy_rollout["answer_ids"], eval_alias_token_lists)
        ),
        "mean_reward": float(sum(rewards) / max(1, len(rewards))),
        "bridge_new_score": mean_or_none(bridge_new_scores),
        "bridge_old_score": mean_or_none(bridge_old_scores),
        "bridge_margin": mean_or_none(bridge_margins),
        "preferred_avg_logp": None,
        "competing_avg_logp": None,
        "likelihood_margin": None,
        "likelihood_success": None,
        "preferred_first_token_rank": first_step_stats["preferred_first_token_rank"],
        "first_token_margin": first_step_stats["first_token_margin"],
        "answer_token_entropy": first_step_stats["answer_token_entropy"],
        "samples": samples_out,
    }


def preference_alias_sets_for_case(
    edit: EditExample,
    bucket_name: str,
    case: EvalPromptCase,
) -> Tuple[List[str], List[str]]:
    preferred_aliases = list(case.aliases)
    if bucket_name in {"rewrite", "declarative_paraphrases", "qa_format_generalization"}:
        competing_aliases = (
            default_aliases_for_text(edit.old_target)
            if edit.old_target
            else []
        )
    else:
        competing_aliases = list(edit.aliases)
    return preferred_aliases, competing_aliases


def printable_prompt_samples(
    model,
    tokenizer,
    prompt: str,
    target: str,
    aliases: Sequence[str],
    steps: int,
    samples: int,
) -> List[str]:
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, prompt)
    target_ids, _ = tokenize_aliases_same_length(tokenizer, target, aliases)
    outputs = []
    for _ in range(samples):
        out = simple_generate_answer(
            model=model,
            prompt_ids=prompt_ids,
            answer_len=len(target_ids),
            mask_id=mask_id,
            steps=steps,
            temperature=1.0,
        )
        outputs.append(decode_ids(tokenizer, out))
    return outputs


def evaluate_edit_buckets_with_bridge(
    model,
    tokenizer,
    edit: EditExample,
    steps: int,
    samples: int,
    bridge_topk: int,
    mc_rollouts: int,
    guidance_scale: float,
    reward_mode: str = "hard_exact",
    reward_beta: float = 6.0,
    bridge_eval_mode: str = "oracle",
) -> Dict[str, Any]:
    if bridge_eval_mode not in {"oracle", "edit_conditioned"}:
        raise ValueError(f"Unsupported bridge_eval_mode: {bridge_eval_mode}")

    results: Dict[str, Any] = {
        "id": edit.id,
        "prompt": edit.prompt,
        "target": edit.target,
        "old_target": edit.old_target,
        "bridge_eval_mode": bridge_eval_mode,
        "buckets": {},
    }

    bucket_map = build_eval_buckets_for_edit(edit)
    for bucket_name in EVAL_BUCKET_ORDER:
        bucket_results = []
        for case in bucket_map[bucket_name]:
            preferred_aliases, competing_aliases = preference_alias_sets_for_case(
                edit=edit,
                bucket_name=bucket_name,
                case=case,
            )
            if bridge_eval_mode == "oracle":
                guided_target_text = case.target
                guided_aliases = list(case.aliases)
            else:
                guided_target_text = edit.target
                guided_aliases = list(edit.aliases)
            stats = estimate_prompt_success_rate_with_bridge(
                model=model,
                tokenizer=tokenizer,
                prompt=case.prompt,
                guided_target_text=guided_target_text,
                guided_aliases=guided_aliases,
                eval_target_text=case.target,
                eval_aliases=case.aliases,
                competing_aliases=competing_aliases,
                steps=steps,
                samples=samples,
                bridge_topk=bridge_topk,
                mc_rollouts=mc_rollouts,
                guidance_scale=guidance_scale,
                reward_mode=reward_mode,
                reward_beta=reward_beta,
            )
            bucket_results.append(
                {
                    "id": case.id,
                    "prompt": case.prompt,
                    "target": case.target,
                    "aliases": list(case.aliases),
                    "stats": {
                        key: stats[key]
                        for key in (
                            "exact_rate",
                            "greedy_exact_rate",
                            "mean_reward",
                            "bridge_new_score",
                            "bridge_old_score",
                            "bridge_margin",
                            "preferred_avg_logp",
                            "competing_avg_logp",
                            "likelihood_margin",
                            "likelihood_success",
                            "preferred_first_token_rank",
                            "first_token_margin",
                            "answer_token_entropy",
                        )
                    },
                    "samples": list(stats["samples"]),
                }
            )

        num_cases = len(bucket_results)
        mean_exact_rate = None
        mean_greedy_exact_rate = None
        mean_reward = None
        mean_bridge_new_score = None
        mean_bridge_old_score = None
        mean_bridge_margin = None
        mean_preferred_first_token_rank = None
        mean_first_token_margin = None
        mean_answer_token_entropy = None
        if bucket_results:
            mean_exact_rate = float(
                sum(item["stats"]["exact_rate"] for item in bucket_results) / num_cases
            )
            mean_greedy_exact_rate = float(
                sum(item["stats"]["greedy_exact_rate"] for item in bucket_results) / num_cases
            )
            mean_reward = float(
                sum(item["stats"]["mean_reward"] for item in bucket_results) / num_cases
            )
            mean_bridge_new_score = mean_or_none(
                [item["stats"]["bridge_new_score"] for item in bucket_results]
            )
            mean_bridge_old_score = mean_or_none(
                [item["stats"]["bridge_old_score"] for item in bucket_results]
            )
            mean_bridge_margin = mean_or_none(
                [item["stats"]["bridge_margin"] for item in bucket_results]
            )
            mean_preferred_first_token_rank = mean_or_none(
                [item["stats"]["preferred_first_token_rank"] for item in bucket_results]
            )
            mean_first_token_margin = mean_or_none(
                [item["stats"]["first_token_margin"] for item in bucket_results]
            )
            mean_answer_token_entropy = mean_or_none(
                [item["stats"]["answer_token_entropy"] for item in bucket_results]
            )

        results["buckets"][bucket_name] = {
            "num_cases": num_cases,
            "mean_exact_rate": mean_exact_rate,
            "mean_greedy_exact_rate": mean_greedy_exact_rate,
            "mean_reward": mean_reward,
            "mean_likelihood_success": None,
            "mean_likelihood_margin": None,
            "mean_bridge_new_score": mean_bridge_new_score,
            "mean_bridge_old_score": mean_bridge_old_score,
            "mean_bridge_margin": mean_bridge_margin,
            "mean_preferred_first_token_rank": mean_preferred_first_token_rank,
            "mean_first_token_margin": mean_first_token_margin,
            "mean_answer_token_entropy": mean_answer_token_entropy,
            "cases": bucket_results,
        }

    return results


def build_eval_buckets_for_edit(edit: EditExample) -> Dict[str, List[EvalPromptCase]]:
    edit_id = edit.id or "edit"
    rewrite_case = EvalPromptCase(
        prompt=edit.prompt,
        target=edit.target,
        aliases=list(edit.aliases),
        id=f"{edit_id}_rewrite",
    )
    declarative_paraphrase_cases = [
        EvalPromptCase(
            prompt=prompt,
            target=edit.target,
            aliases=list(edit.aliases),
            id=f"{edit_id}_declarative_paraphrase_{idx}",
        )
        for idx, prompt in enumerate(edit.declarative_paraphrase_prompts)
    ]
    qa_paraphrase_cases = [
        EvalPromptCase(
            prompt=prompt,
            target=edit.target,
            aliases=list(edit.aliases),
            id=f"{edit_id}_qa_paraphrase_{idx}",
        )
        for idx, prompt in enumerate(edit.qa_paraphrase_prompts)
    ]
    near_locality_cases = [
        EvalPromptCase(
            prompt=case.prompt,
            target=case.target,
            aliases=list(case.aliases),
            id=case.id,
        )
        for case in edit.near_locality_cases
    ]
    far_locality_cases = [
        EvalPromptCase(
            prompt=case.prompt,
            target=case.target,
            aliases=list(case.aliases),
            id=case.id,
        )
        for case in edit.far_locality_cases
    ]
    return {
        "rewrite": [rewrite_case],
        "declarative_paraphrases": declarative_paraphrase_cases,
        "qa_format_generalization": qa_paraphrase_cases,
        "near_locality": near_locality_cases,
        "far_locality": far_locality_cases,
    }


def build_eval_prompt_text_set(edit: EditExample) -> set[str]:
    bucket_map = build_eval_buckets_for_edit(edit)
    return {
        case.prompt
        for bucket_name in EVAL_BUCKET_ORDER
        for case in bucket_map[bucket_name]
    }


def evaluate_edit_buckets(
    model,
    tokenizer,
    edit: EditExample,
    steps: int,
    samples: int,
    reward_mode: str = "hard_exact",
    reward_beta: float = 6.0,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "id": edit.id,
        "prompt": edit.prompt,
        "target": edit.target,
        "old_target": edit.old_target,
        "buckets": {},
    }

    bucket_map = build_eval_buckets_for_edit(edit)
    for bucket_name in EVAL_BUCKET_ORDER:
        bucket_results = []
        for case in bucket_map[bucket_name]:
            stats = estimate_prompt_success_rate(
                model=model,
                tokenizer=tokenizer,
                prompt=case.prompt,
                target=case.target,
                aliases=case.aliases,
                steps=steps,
                samples=samples,
                reward_mode=reward_mode,
                reward_beta=reward_beta,
            )
            preferred_aliases, competing_aliases = preference_alias_sets_for_case(
                edit=edit,
                bucket_name=bucket_name,
                case=case,
            )
            if competing_aliases:
                stats.update(
                    length_normalized_preference_stats(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=case.prompt,
                        preferred_aliases=preferred_aliases,
                        competing_aliases=competing_aliases,
                    )
                )
            else:
                stats.update(
                    {
                        "preferred_avg_logp": None,
                        "competing_avg_logp": None,
                        "likelihood_margin": None,
                        "likelihood_success": None,
                    }
                )
            target_ids, _ = tokenize_aliases_same_length(
                tokenizer,
                case.target,
                case.aliases,
            )
            stats.update(
                first_step_model_preference_stats(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=case.prompt,
                    answer_len=len(target_ids),
                    preferred_aliases=preferred_aliases,
                    competing_aliases=competing_aliases,
                )
            )
            samples_out = printable_prompt_samples(
                model=model,
                tokenizer=tokenizer,
                prompt=case.prompt,
                target=case.target,
                aliases=case.aliases,
                steps=steps,
                samples=samples,
            )
            bucket_results.append(
                {
                    "id": case.id,
                    "prompt": case.prompt,
                    "target": case.target,
                    "aliases": list(case.aliases),
                    "stats": stats,
                    "samples": samples_out,
                }
            )

        num_cases = len(bucket_results)
        mean_exact_rate = None
        mean_greedy_exact_rate = None
        mean_reward = None
        mean_likelihood_success = None
        mean_likelihood_margin = None
        mean_preferred_first_token_rank = None
        mean_first_token_margin = None
        mean_answer_token_entropy = None
        if bucket_results:
            mean_exact_rate = float(
                sum(item["stats"]["exact_rate"] for item in bucket_results) / num_cases
            )
            mean_greedy_exact_rate = float(
                sum(item["stats"]["greedy_exact_rate"] for item in bucket_results) / num_cases
            )
            mean_reward = float(
                sum(item["stats"]["mean_reward"] for item in bucket_results) / num_cases
            )
            likelihood_successes = [
                item["stats"]["likelihood_success"]
                for item in bucket_results
                if item["stats"]["likelihood_success"] is not None
            ]
            likelihood_margins = [
                item["stats"]["likelihood_margin"]
                for item in bucket_results
                if item["stats"]["likelihood_margin"] is not None
            ]
            if likelihood_successes:
                mean_likelihood_success = float(
                    sum(likelihood_successes) / len(likelihood_successes)
                )
            if likelihood_margins:
                mean_likelihood_margin = float(
                    sum(likelihood_margins) / len(likelihood_margins)
                )
            first_token_ranks = [
                item["stats"]["preferred_first_token_rank"]
                for item in bucket_results
                if item["stats"].get("preferred_first_token_rank") is not None
            ]
            first_token_margins = [
                item["stats"]["first_token_margin"]
                for item in bucket_results
                if item["stats"].get("first_token_margin") is not None
            ]
            answer_entropies = [
                item["stats"]["answer_token_entropy"]
                for item in bucket_results
                if item["stats"].get("answer_token_entropy") is not None
            ]
            if first_token_ranks:
                mean_preferred_first_token_rank = float(
                    sum(first_token_ranks) / len(first_token_ranks)
                )
            if first_token_margins:
                mean_first_token_margin = float(
                    sum(first_token_margins) / len(first_token_margins)
                )
            if answer_entropies:
                mean_answer_token_entropy = float(
                    sum(answer_entropies) / len(answer_entropies)
                )

        results["buckets"][bucket_name] = {
            "num_cases": num_cases,
            "mean_exact_rate": mean_exact_rate,
            "mean_greedy_exact_rate": mean_greedy_exact_rate,
            "mean_reward": mean_reward,
            "mean_likelihood_success": mean_likelihood_success,
            "mean_likelihood_margin": mean_likelihood_margin,
            "mean_preferred_first_token_rank": mean_preferred_first_token_rank,
            "mean_first_token_margin": mean_first_token_margin,
            "mean_answer_token_entropy": mean_answer_token_entropy,
            "cases": bucket_results,
        }

    return results


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


@torch.no_grad()
def evaluate_anchor_kl_drift(
    reference_model,
    edited_model,
    tokenizer,
    edit: EditExample,
    teacher_topk: int,
) -> Optional[float]:
    if not edit.eval_anchor_cases:
        return None

    drift_values = []
    for eval_anchor_case in edit.eval_anchor_cases:
        reference_record = build_reference_teacher_record(
            model=reference_model,
            tokenizer=tokenizer,
            prompt_text=eval_anchor_case.prompt,
            target_text=eval_anchor_case.target,
            aliases=eval_anchor_case.aliases,
            teacher_topk=teacher_topk,
            kind="eval_anchor",
            ce_target_text=eval_anchor_case.target,
        )
        drift = teacher_record_loss(edited_model, reference_record, ce_weight=0.0)
        drift_values.append(float(drift.item()))
    return float(sum(drift_values) / max(1, len(drift_values)))


def bootstrap_mean_confidence_interval(
    values: Sequence[Optional[float]],
    samples: int = 10_000,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    clean_values = [float(v) for v in values if v is not None]
    if not clean_values:
        return None
    if len(clean_values) == 1:
        return {"low": clean_values[0], "high": clean_values[0]}

    rng = random.Random(seed)
    draws = []
    n = len(clean_values)
    for _ in range(samples):
        resample = [clean_values[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(resample) / n)
    draws.sort()
    low_idx = int(0.025 * (len(draws) - 1))
    high_idx = int(0.975 * (len(draws) - 1))
    return {"low": float(draws[low_idx]), "high": float(draws[high_idx])}


def compute_contamination_metrics(
    eval_results: Sequence[Dict[str, Any]],
    edits: Sequence[EditExample],
) -> Dict[str, Any]:
    edit_by_id = {edit.id: edit for edit in edits}
    target_map = {
        edit.id: [alias.strip().lower() for alias in edit.aliases if alias.strip()]
        for edit in edits
    }
    contamination_matrix = {
        edit.id: {other.id: 0 for other in edits if other.id != edit.id}
        for edit in edits
    }
    contaminated_outputs = 0
    total_outputs = 0

    for eval_result in eval_results:
        source_id = eval_result["id"]
        source_aliases = set(target_map.get(source_id, []))
        for bucket_name in GENERATION_TARGET_BUCKETS:
            bucket = eval_result["buckets"].get(bucket_name, {})
            for case in bucket.get("cases", []):
                for sample in case.get("samples", []):
                    sample_norm = " ".join(sample.strip().lower().split())
                    if not sample_norm:
                        continue
                    total_outputs += 1
                    contaminated = False
                    for other_id, aliases in target_map.items():
                        if other_id == source_id:
                            continue
                        if any(alias and alias in sample_norm for alias in aliases):
                            contamination_matrix[source_id][other_id] += 1
                            contaminated = True
                    if contaminated:
                        contaminated_outputs += 1

    contamination_rate = (
        float(contaminated_outputs / total_outputs) if total_outputs else 0.0
    )
    return {
        "contamination_rate": contamination_rate,
        "contamination_matrix": contamination_matrix,
        "total_outputs": int(total_outputs),
        "contaminated_outputs": int(contaminated_outputs),
    }


def directory_size_bytes(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return int(total)


def estimate_reference_success_rate(
    model,
    tokenizer,
    edit: EditExample,
    steps: int,
    samples: int,
    reward_mode: str = "hard_exact",
    reward_beta: float = 6.0,
) -> Dict[str, float]:
    return estimate_prompt_success_rate(
        model=model,
        tokenizer=tokenizer,
        prompt=edit.prompt,
        target=edit.target,
        aliases=edit.aliases,
        steps=steps,
        samples=samples,
        reward_mode=reward_mode,
        reward_beta=reward_beta,
    )


def printable_answer_samples(
    model,
    tokenizer,
    edit: EditExample,
    steps: int,
    samples: int,
) -> List[str]:
    return printable_prompt_samples(
        model=model,
        tokenizer=tokenizer,
        prompt=edit.prompt,
        target=edit.target,
        aliases=edit.aliases,
        steps=steps,
        samples=samples,
    )
