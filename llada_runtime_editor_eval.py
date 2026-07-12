#!/usr/bin/env python3
"""Runtime editor evaluator for ``counterfact_direction1_v1``.

This script intentionally keeps the sprint-1 methods close to the existing
answer-span LLaDA bridge utilities. It is suitable for dev/gpu runs, while the
pure helpers are covered by local unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import string
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from llada_counterfact_protocol import PROTOCOL_VERSION, normalize_counterfact_text
from llada_sb_common import (
    EVAL_BUCKET_ORDER,
    EditExample,
    EvalPromptCase,
    build_eval_buckets_for_edit,
    build_initial_state,
    build_transfer_schedule,
    decode_ids,
    default_aliases_for_text,
    endpoint_reward,
    estimate_candidate_bridge_scores,
    exact_alias_match,
    get_model_device,
    get_model_eval_counter,
    infer_mask_id,
    load_edits,
    load_llada_model_and_tokenizer,
    normalize_probability_vector,
    preference_alias_sets_for_case,
    reset_model_eval_counter,
    run_logits,
    sanitize_logits_row,
    seed_everything,
    soft_overlap_score,
    tokenize_alias_lists_for_target_length,
    tokenize_aliases_same_length,
    tokenize_prompt,
)


SPRINT1_METHODS = (
    "base",
    "target_logit_bias",
    "prompt_memory",
    "target_candidate_insert",
    "myopic_score",
    "no_rollout_bridge",
    "mc_bridge",
    "raw_bridge_gated",
)

INTERVENTION_METHODS = (
    "target_logit_bias",
    "prompt_memory",
    "target_candidate_insert",
    "myopic_score",
    "no_rollout_bridge",
    "mc_bridge",
)

GATED_METHODS = tuple(
    f"{method}_gated_{gate_mode}"
    for method in INTERVENTION_METHODS
    for gate_mode in ("subject", "subject_relation", "hybrid")
)

SUPPORTED_METHODS = tuple(dict.fromkeys(list(SPRINT1_METHODS) + list(GATED_METHODS)))

LOCKED_FINAL_ROLES = {"final_test_500", "final_test_full"}
ANALYSIS_ROLES = {"analysis_500"}


@dataclass(frozen=True)
class RolloutConfig:
    steps: int
    bridge_topk: int
    mc_rollouts: int
    guidance_scale: float
    reward_mode: str
    reward_beta: float
    target_logit_bias: float
    gate_mode: str
    temperature: float
    relation_sim_rewrite_threshold: float = 0.45
    relation_sim_bank_threshold: float = 0.10
    relation_bank_path: str = ""
    relation_bank_source: str = "dev_tune_200_rewrite_templates"


def sparse_support_guidance_kl(
    guided_probs: Sequence[float],
    base_support_probs: Sequence[float],
    eps: float = 1e-12,
) -> float:
    guided = [max(float(p), 0.0) for p in guided_probs]
    base = [max(float(p), 0.0) for p in base_support_probs]
    if len(guided) != len(base):
        raise ValueError("guided_probs and base_support_probs must have the same length")
    g_total = sum(guided)
    b_total = sum(base)
    if g_total <= 0 or b_total <= 0:
        return 0.0
    total = 0.0
    for g, b in zip(guided, base):
        p = g / g_total
        q = b / b_total
        if p > 0:
            total += p * math.log(max(p, eps) / max(q, eps))
    return float(total)


def token_f1(answer_ids: Sequence[int], alias_token_lists: Sequence[Sequence[int]]) -> float:
    answer = list(map(int, answer_ids))
    if not answer or not alias_token_lists:
        return 0.0
    best = 0.0
    for alias in alias_token_lists:
        gold = list(map(int, alias))
        if not gold:
            continue
        remaining = list(gold)
        overlap = 0
        for token_id in answer:
            if token_id in remaining:
                overlap += 1
                remaining.remove(token_id)
        precision = overlap / max(1, len(answer))
        recall = overlap / max(1, len(gold))
        if precision + recall > 0:
            best = max(best, 2 * precision * recall / (precision + recall))
    return float(best)


def is_malformed_answer(tokenizer: Any, answer_ids: Sequence[int], mask_id: int) -> bool:
    if not answer_ids:
        return True
    if any(int(token_id) == int(mask_id) for token_id in answer_ids):
        return True
    try:
        decoded = decode_ids(tokenizer, answer_ids).strip()
    except Exception:
        return False
    return decoded == "" or "[MASK]" in decoded


def relation_keywords(template: str, subject: str) -> List[str]:
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
        "for",
        "by",
        "with",
        "what",
        "who",
        "which",
        "it",
    }
    cleaned = template.replace("{}", " ").replace("{subject}", " ")
    lowered_subject = set(subject.lower().split())
    out: List[str] = []
    for token in re_find_words(cleaned):
        lowered = token.lower()
        if lowered in stopwords or lowered in lowered_subject:
            continue
        if lowered not in out:
            out.append(lowered)
    return out[:4]


RELATION_GATE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "what",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
    "did",
    "does",
    "do",
}

_RELATION_BANK_CACHE: Dict[str, Dict[str, List[str]]] = {}


def normalize_gate_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.replace("[mask]", " ").replace("<mask>", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def remove_phrase_from_gate_text(text: str, phrase: str) -> str:
    text_norm = normalize_gate_text(text)
    phrase_norm = normalize_gate_text(phrase)
    if phrase_norm:
        text_norm = re.sub(r"\b" + re.escape(phrase_norm) + r"\b", " ", text_norm)
    return re.sub(r"\s+", " ", text_norm).strip()


def render_rewrite_template(template: str, subject: str) -> str:
    template = str(template or "")
    if "{}" in template:
        return template.format(subject)
    if "{subject}" in template:
        return template.replace("{subject}", subject)
    return template


def raw_subject(raw_edit: Dict[str, Any]) -> str:
    rewrite = raw_edit.get("requested_rewrite", {})
    if isinstance(rewrite, dict) and rewrite.get("subject"):
        return str(rewrite["subject"])
    return str(raw_edit.get("subject") or "")


def raw_rewrite_template(raw_edit: Dict[str, Any]) -> str:
    rewrite = raw_edit.get("requested_rewrite", {})
    if isinstance(rewrite, dict) and rewrite.get("prompt"):
        return str(rewrite["prompt"])
    return str(raw_edit.get("rewrite_template") or raw_edit.get("prompt") or "")


def raw_relation_id(raw_edit: Dict[str, Any]) -> str:
    rewrite = raw_edit.get("requested_rewrite", {})
    if raw_edit.get("relation_id"):
        return str(raw_edit["relation_id"])
    if isinstance(rewrite, dict) and rewrite.get("relation_id"):
        return str(rewrite["relation_id"])
    return ""


def raw_target_new(raw_edit: Dict[str, Any]) -> str:
    target_new = raw_edit.get("target_new")
    if isinstance(target_new, dict):
        return str(target_new.get("text") or target_new.get("str") or raw_edit.get("target") or "")
    if target_new is not None:
        return str(target_new)
    rewrite = raw_edit.get("requested_rewrite", {})
    if isinstance(rewrite, dict) and isinstance(rewrite.get("target_new"), dict):
        return str(rewrite["target_new"].get("str") or raw_edit.get("target") or "")
    return str(raw_edit.get("target") or "")


def raw_target_true(raw_edit: Dict[str, Any]) -> str:
    target_true = raw_edit.get("target_true")
    if isinstance(target_true, dict):
        return str(target_true.get("text") or target_true.get("str") or raw_edit.get("old_target") or "")
    if target_true is not None:
        return str(target_true)
    rewrite = raw_edit.get("requested_rewrite", {})
    if isinstance(rewrite, dict) and isinstance(rewrite.get("target_true"), dict):
        return str(rewrite["target_true"].get("str") or raw_edit.get("old_target") or "")
    return str(raw_edit.get("old_target") or "")


def relation_content_text(
    prompt: str,
    *,
    subject: str = "",
    target_new: str = "",
    target_true: str = "",
) -> str:
    text = normalize_gate_text(prompt)
    for phrase in (subject, target_new, target_true, "mask"):
        text = remove_phrase_from_gate_text(text, phrase)
    tokens = [tok for tok in text.split() if tok not in RELATION_GATE_STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


def char_ngrams(text: str, n: int = 3) -> Dict[str, int]:
    normalized = f"  {normalize_gate_text(text)}  "
    if len(normalized) < n:
        return {normalized: 1} if normalized.strip() else {}
    counts: Dict[str, int] = {}
    for idx in range(len(normalized) - n + 1):
        gram = normalized[idx : idx + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def cosine_counts(left: Dict[str, int], right: Dict[str, int]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def word_jaccard(left: str, right: str) -> float:
    left_words = set(normalize_gate_text(left).split())
    right_words = set(normalize_gate_text(right).split())
    if not left_words or not right_words:
        return 0.0
    return float(len(left_words & right_words) / len(left_words | right_words))


def relation_similarity(left: str, right: str) -> float:
    return max(cosine_counts(char_ngrams(left), char_ngrams(right)), word_jaccard(left, right))


def relation_text_for_record(raw_edit: Dict[str, Any]) -> str:
    subject = raw_subject(raw_edit)
    rendered = render_rewrite_template(raw_rewrite_template(raw_edit), subject)
    return relation_content_text(
        rendered,
        subject=subject,
        target_new=raw_target_new(raw_edit),
        target_true=raw_target_true(raw_edit),
    )


def load_relation_bank(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    normalized_path = os.path.abspath(os.path.expanduser(path))
    if normalized_path in _RELATION_BANK_CACHE:
        return _RELATION_BANK_CACHE[normalized_path]
    bank: Dict[str, List[str]] = {}
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Missing relation bank source: {normalized_path}")
    for row in load_raw_jsonl(normalized_path):
        relation_id = raw_relation_id(row)
        relation_text = relation_text_for_record(row)
        if relation_id and relation_text:
            bank.setdefault(relation_id, [])
            if relation_text not in bank[relation_id]:
                bank[relation_id].append(relation_text)
    _RELATION_BANK_CACHE[normalized_path] = bank
    return bank


def subject_matches_prompt(subject: str, prompt: str) -> bool:
    subject_norm = normalize_gate_text(subject)
    prompt_norm = normalize_gate_text(prompt)
    if not subject_norm:
        return False
    return re.search(r"\b" + re.escape(subject_norm) + r"\b", prompt_norm) is not None


def hybrid_relation_gate_scores(raw_edit: Dict[str, Any], prompt: str, cfg: RolloutConfig) -> Tuple[bool, float, float]:
    subject = raw_subject(raw_edit)
    if not subject_matches_prompt(subject, prompt):
        return False, 0.0, 0.0
    prompt_relation = relation_content_text(
        prompt,
        subject=subject,
        target_new=raw_target_new(raw_edit),
        target_true=raw_target_true(raw_edit),
    )
    rewrite_relation = relation_text_for_record(raw_edit)
    rewrite_sim = relation_similarity(prompt_relation, rewrite_relation)

    relation_id = raw_relation_id(raw_edit)
    bank = load_relation_bank(cfg.relation_bank_path)
    bank_sim = max((relation_similarity(prompt_relation, proto) for proto in bank.get(relation_id, [])), default=0.0)
    return True, float(rewrite_sim), float(bank_sim)


def hybrid_relation_or_gate_should_activate(raw_edit: Dict[str, Any], prompt: str, cfg: RolloutConfig) -> bool:
    subject_match, rewrite_sim, bank_sim = hybrid_relation_gate_scores(raw_edit, prompt, cfg)
    if not subject_match:
        return False
    return (
        rewrite_sim >= float(cfg.relation_sim_rewrite_threshold)
        or bank_sim >= float(cfg.relation_sim_bank_threshold)
    )


def hybrid_relation_and_gate_should_activate(raw_edit: Dict[str, Any], prompt: str, cfg: RolloutConfig) -> bool:
    subject_match, rewrite_sim, bank_sim = hybrid_relation_gate_scores(raw_edit, prompt, cfg)
    if not subject_match:
        return False
    return (
        rewrite_sim >= float(cfg.relation_sim_rewrite_threshold)
        and bank_sim >= float(cfg.relation_sim_bank_threshold)
    )


def re_find_words(text: str) -> List[str]:
    import re

    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)


def gate_should_activate(
    raw_edit: Dict[str, Any],
    prompt: str,
    gate_mode: str,
    cfg: Optional[RolloutConfig] = None,
) -> bool:
    if gate_mode in {"none", "always"}:
        return True
    subject = raw_subject(raw_edit)
    prompt_lower = prompt.lower()
    subject_match = subject_matches_prompt(subject, prompt)
    if gate_mode == "subject":
        return subject_match
    if gate_mode == "hybrid_relation_or":
        if cfg is None:
            raise ValueError("hybrid_relation_or gate requires RolloutConfig")
        return hybrid_relation_or_gate_should_activate(raw_edit, prompt, cfg)
    if gate_mode == "hybrid_relation_and":
        if cfg is None:
            raise ValueError("hybrid_relation_and gate requires RolloutConfig")
        return hybrid_relation_and_gate_should_activate(raw_edit, prompt, cfg)
    template = str(raw_edit.get("rewrite_template") or raw_edit.get("requested_rewrite", {}).get("prompt", ""))
    keywords = relation_keywords(template, subject)
    relation_match = bool(keywords) and any(keyword in prompt_lower for keyword in keywords)
    if gate_mode in {"subject_relation", "hybrid", "conservative_hybrid"}:
        return subject_match and relation_match
    raise ValueError(f"Unsupported gate_mode: {gate_mode}")


def decompose_method(method: str, cfg: RolloutConfig) -> Tuple[str, Optional[str], str]:
    """Return the intervention method, optional gate, and report label."""

    if method == "raw_bridge_gated":
        return "mc_bridge", cfg.gate_mode, f"raw_bridge_gated_{cfg.gate_mode}"
    hybrid_suffix = "_gated_hybrid"
    if method.endswith(hybrid_suffix):
        base_method = method[: -len(hybrid_suffix)]
        if base_method in INTERVENTION_METHODS:
            return base_method, cfg.gate_mode, method
    for gate_mode in ("subject_relation", "subject"):
        suffix = f"_gated_{gate_mode}"
        if method.endswith(suffix):
            base_method = method[: -len(suffix)]
            if base_method in INTERVENTION_METHODS:
                return base_method, gate_mode, method
    return method, None, method


def prompt_memory_prefix(raw_edit: Dict[str, Any], edit: EditExample) -> str:
    subject = str(raw_edit.get("subject") or raw_edit.get("requested_rewrite", {}).get("subject", "this subject"))
    relation_id = str(raw_edit.get("relation_id") or raw_edit.get("requested_rewrite", {}).get("relation_id", ""))
    return (
        f"For this edit, subject={subject}; relation={relation_id}; "
        f"new_answer={edit.target.strip()}.\n"
    )


def candidate_ids_for_position(
    *,
    base_probs: torch.Tensor,
    bridge_topk: int,
    alias_token_lists: Sequence[Sequence[int]],
    rel_pos: int,
    include_targets: bool,
) -> List[int]:
    top_ids = torch.topk(base_probs, k=min(bridge_topk, int(base_probs.numel()))).indices.detach().cpu().tolist()
    if not include_targets:
        return list(dict.fromkeys(map(int, top_ids)))
    needed = [
        int(alias[rel_pos])
        for alias in alias_token_lists
        if rel_pos < len(alias)
    ]
    return list(dict.fromkeys([int(v) for v in top_ids] + needed))


def target_logit_bias_row(
    row: torch.Tensor,
    target_token_lists: Sequence[Sequence[int]],
    rel_pos: int,
    bias: float,
) -> torch.Tensor:
    biased = row.clone()
    token_ids = {
        int(alias[rel_pos])
        for alias in target_token_lists
        if rel_pos < len(alias)
    }
    for token_id in token_ids:
        if 0 <= token_id < int(biased.numel()):
            biased[token_id] += float(bias)
    return biased


def sparse_guided_distribution(
    *,
    method: str,
    row: torch.Tensor,
    candidate_ids: Sequence[int],
    base_support_probs: torch.Tensor,
    x: torch.Tensor,
    prompt_len: int,
    answer_len: int,
    rel_pos: int,
    target_token_lists: Sequence[Sequence[int]],
    cfg: RolloutConfig,
    mask_id: int,
    remaining_steps: int,
) -> Tuple[List[int], torch.Tensor, float]:
    device = row.device
    cand = torch.tensor(list(candidate_ids), dtype=torch.long, device=device)
    base_log_probs = torch.log(base_support_probs.clamp_min(1e-12))

    if method == "target_candidate_insert":
        guided_probs = base_support_probs
    else:
        if method == "myopic_score":
            target_ids = {
                int(alias[rel_pos])
                for alias in target_token_lists
                if rel_pos < len(alias)
            }
            scores = [math.exp(cfg.reward_beta) if int(token_id) in target_ids else 1.0 for token_id in candidate_ids]
        elif method == "no_rollout_bridge":
            scores = []
            abs_pos = prompt_len + rel_pos
            for token_id in candidate_ids:
                x_tent = x.clone()
                x_tent[0, abs_pos] = int(token_id)
                partial = x_tent[0, prompt_len : prompt_len + answer_len].tolist()
                scores.append(
                    endpoint_reward(
                        partial,
                        target_token_lists,
                        reward_mode=cfg.reward_mode,
                        reward_beta=cfg.reward_beta,
                    )
                )
        elif method in {"mc_bridge", "raw_bridge_gated"}:
            scores = estimate_candidate_bridge_scores(
                model=sparse_guided_distribution.model,  # type: ignore[attr-defined]
                x=x,
                prompt_len=prompt_len,
                answer_len=answer_len,
                rel_pos=rel_pos,
                candidate_ids=candidate_ids,
                alias_token_lists=target_token_lists,
                remaining_steps=remaining_steps,
                reward_mode=cfg.reward_mode,
                reward_beta=cfg.reward_beta,
                mc_rollouts=cfg.mc_rollouts,
                mask_id=mask_id,
            )
        else:
            raise ValueError(f"Unsupported sparse method: {method}")

        score_tensor = torch.tensor(scores, dtype=torch.float32, device=device).clamp_min(1e-12)
        guided_logits = base_log_probs + cfg.guidance_scale * torch.log(score_tensor)
        guided_probs = normalize_probability_vector(torch.softmax(guided_logits, dim=-1))

    kl = sparse_support_guidance_kl(
        guided_probs.detach().cpu().tolist(),
        base_support_probs.detach().cpu().tolist(),
    )
    return cand.detach().cpu().tolist(), guided_probs, kl


@torch.no_grad()
def runtime_rollout(
    *,
    model: Any,
    tokenizer: Any,
    method: str,
    prompt_text: str,
    edit: EditExample,
    raw_edit: Dict[str, Any],
    answer_len: int,
    cfg: RolloutConfig,
) -> Dict[str, Any]:
    intervention_method, gate_mode, method_variant = decompose_method(method, cfg)
    effective_method = intervention_method
    gate_active: Optional[bool] = None
    if gate_mode is not None:
        gate_active = gate_should_activate(raw_edit, prompt_text, gate_mode, cfg)
    if gate_active is False:
        effective_method = "base"

    memory_prefix = ""
    if effective_method == "prompt_memory":
        memory_prefix = prompt_memory_prefix(raw_edit, edit)
        effective_method = "base"

    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, memory_prefix + prompt_text)
    prompt_len = len(prompt_ids)
    answer_abs_positions = list(range(prompt_len, prompt_len + answer_len))
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)

    target_alias_pool = list(dict.fromkeys(list(edit.aliases) + default_aliases_for_text(edit.target)))
    target_token_lists = tokenize_alias_lists_for_target_length(
        tokenizer,
        target_alias_pool,
        answer_len,
    )
    sparse_kl = 0.0

    # Avoid threading model through many pure-ish function signatures.
    sparse_guided_distribution.model = model  # type: ignore[attr-defined]

    for step_index, num_transfer in enumerate(build_transfer_schedule(answer_len, cfg.steps)):
        masked_now = int((x[0, answer_abs_positions] == mask_id).sum().item())
        if masked_now <= 0:
            break
        logits = run_logits(model, x)[0]
        proposals: List[Tuple[float, int, int]] = []
        remaining_steps = max(0, cfg.steps - step_index - 1)

        for rel_pos in range(answer_len):
            abs_pos = answer_abs_positions[rel_pos]
            if int(x[0, abs_pos].item()) != mask_id:
                continue
            row = sanitize_logits_row(logits[abs_pos])
            base_probs = normalize_probability_vector(torch.softmax(row, dim=-1))

            if effective_method == "base":
                token_id, confidence = sample_from_distribution(base_probs, cfg.temperature)
            elif effective_method == "target_logit_bias":
                biased_row = target_logit_bias_row(
                    row,
                    target_token_lists,
                    rel_pos,
                    cfg.target_logit_bias,
                )
                biased_probs = normalize_probability_vector(torch.softmax(biased_row, dim=-1))
                token_id, confidence = sample_from_distribution(biased_probs, cfg.temperature)
                support = candidate_ids_for_position(
                    base_probs=base_probs,
                    bridge_topk=cfg.bridge_topk,
                    alias_token_lists=target_token_lists,
                    rel_pos=rel_pos,
                    include_targets=True,
                )
                support_tensor = torch.tensor(support, dtype=torch.long, device=row.device)
                sparse_kl += sparse_support_guidance_kl(
                    normalize_probability_vector(biased_probs[support_tensor]).detach().cpu().tolist(),
                    normalize_probability_vector(base_probs[support_tensor]).detach().cpu().tolist(),
                )
            else:
                support = candidate_ids_for_position(
                    base_probs=base_probs,
                    bridge_topk=cfg.bridge_topk,
                    alias_token_lists=target_token_lists,
                    rel_pos=rel_pos,
                    include_targets=True,
                )
                support_tensor = torch.tensor(support, dtype=torch.long, device=row.device)
                base_support = normalize_probability_vector(base_probs[support_tensor])
                sparse_ids, guided_probs, kl = sparse_guided_distribution(
                    method=effective_method,
                    row=row,
                    candidate_ids=support,
                    base_support_probs=base_support,
                    x=x,
                    prompt_len=prompt_len,
                    answer_len=answer_len,
                    rel_pos=rel_pos,
                    target_token_lists=target_token_lists,
                    cfg=cfg,
                    mask_id=mask_id,
                    remaining_steps=remaining_steps,
                )
                sparse_kl += kl
                local_idx, confidence = sample_from_distribution(guided_probs, cfg.temperature)
                token_id = int(sparse_ids[local_idx])

            proposals.append((float(confidence), rel_pos, int(token_id)))

        if not proposals or num_transfer <= 0:
            continue
        proposals.sort(key=lambda item: item[0], reverse=True)
        for _, rel_pos, token_id in proposals[:num_transfer]:
            x[0, prompt_len + rel_pos] = int(token_id)

    answer_ids = x[0, prompt_len : prompt_len + answer_len].detach().cpu().tolist()
    return {
        "answer_ids": answer_ids,
        "answer_text": decode_ids(tokenizer, answer_ids),
        "sparse_guidance_kl": float(sparse_kl),
        "effective_method": effective_method,
        "gate_active": gate_active,
        "gate_mode": gate_mode,
        "method_variant": method_variant,
    }


def sample_from_distribution(probs: torch.Tensor, temperature: float) -> Tuple[int, float]:
    probs = normalize_probability_vector(probs)
    if temperature <= 0:
        idx = int(torch.argmax(probs).item())
        return idx, float(probs[idx].item())
    logits = torch.log(probs.clamp_min(1e-12)) / max(1e-6, float(temperature))
    scaled = normalize_probability_vector(torch.softmax(logits, dim=-1))
    idx = int(torch.multinomial(scaled, num_samples=1).item())
    return idx, float(scaled[idx].item())


@torch.no_grad()
def probability_margin(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_new: str,
    target_true: str,
) -> Optional[float]:
    try:
        from llada_sb_common import average_target_log_likelihood

        new_score = average_target_log_likelihood(model, tokenizer, prompt, target_new)
        true_score = average_target_log_likelihood(model, tokenizer, prompt, target_true)
        return float(new_score - true_score)
    except Exception:
        return None


@torch.no_grad()
def candidate_coverage_for_edit(
    *,
    model: Any,
    tokenizer: Any,
    edit: EditExample,
    raw_edit: Dict[str, Any],
    topk: int,
) -> Dict[str, Any]:
    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    prompt_ids = tokenize_prompt(tokenizer, edit.prompt)
    target_new_ids, target_new_aliases = tokenize_aliases_same_length(tokenizer, edit.target, edit.aliases)
    true_aliases = list(raw_edit.get("old_aliases") or default_aliases_for_text(edit.old_target or ""))
    target_true_ids, target_true_aliases = tokenize_aliases_same_length(tokenizer, edit.old_target or "", true_aliases)
    answer_len = len(target_new_ids)
    x = build_initial_state(prompt_ids, answer_len, mask_id, device)
    logits = run_logits(model, x)[0]
    prompt_len = len(prompt_ids)

    new_by_pos = []
    true_by_pos = []
    for rel_pos in range(answer_len):
        row = sanitize_logits_row(logits[prompt_len + rel_pos])
        probs = normalize_probability_vector(torch.softmax(row, dim=-1))
        base_top = set(torch.topk(probs, k=min(topk, int(probs.numel()))).indices.detach().cpu().tolist())
        new_ids = {
            int(alias[rel_pos])
            for alias in target_new_aliases
            if rel_pos < len(alias)
        }
        true_ids = {
            int(alias[rel_pos])
            for alias in target_true_aliases
            if rel_pos < len(alias)
        }
        new_by_pos.append(bool(new_ids & base_top))
        true_by_pos.append(bool(true_ids & base_top))

    return {
        "edit_id": edit.id,
        "case_id": raw_edit.get("case_id", edit.id),
        "relation_id": raw_edit.get("relation_id"),
        "target_length_bin": raw_edit.get("target_length_bin"),
        "target_new_first_token_in_base_topk": bool(new_by_pos[0]) if new_by_pos else False,
        "all_target_new_tokens_in_base_topk": bool(new_by_pos) and all(new_by_pos),
        "target_true_first_token_in_base_topk": bool(true_by_pos[0]) if true_by_pos else False,
        "all_target_true_tokens_in_base_topk": bool(true_by_pos) and all(true_by_pos),
        "all_target_new_tokens_after_candidate_insert": bool(target_new_ids),
        "target_new_token_len": len(target_new_ids),
        "target_true_token_len": len(target_true_ids),
    }


def evaluate_case(
    *,
    model: Any,
    tokenizer: Any,
    method: str,
    edit: EditExample,
    raw_edit: Dict[str, Any],
    bucket_name: str,
    case: EvalPromptCase,
    cfg: RolloutConfig,
    samples: int,
) -> Dict[str, Any]:
    eval_target_ids, eval_alias_token_lists = tokenize_aliases_same_length(
        tokenizer,
        case.target,
        case.aliases,
    )
    answer_len = len(eval_target_ids)
    mask_id = infer_mask_id(model)
    exact_count = 0
    token_f1_values: List[float] = []
    malformed_count = 0
    target_fp_count = 0
    sparse_kls: List[float] = []
    sample_outputs: List[str] = []
    target_new_ids, target_new_aliases = tokenize_aliases_same_length(tokenizer, edit.target, edit.aliases)
    target_new_aliases_for_eval_len = [
        alias for alias in target_new_aliases if len(alias) == answer_len
    ]
    _, configured_gate_mode, configured_method_variant = decompose_method(method, cfg)
    gate_active_values: List[float] = []

    for _ in range(samples):
        rollout = runtime_rollout(
            model=model,
            tokenizer=tokenizer,
            method=method,
            prompt_text=case.prompt,
            edit=edit,
            raw_edit=raw_edit,
            answer_len=answer_len,
            cfg=cfg,
        )
        answer_ids = rollout["answer_ids"]
        exact_count += int(exact_alias_match(answer_ids, eval_alias_token_lists))
        token_f1_values.append(token_f1(answer_ids, eval_alias_token_lists))
        malformed_count += int(is_malformed_answer(tokenizer, answer_ids, mask_id))
        if bucket_name not in {"rewrite", "declarative_paraphrases", "qa_format_generalization"}:
            target_fp_count += int(
                bool(target_new_aliases_for_eval_len)
                and exact_alias_match(answer_ids, target_new_aliases_for_eval_len)
            )
        sparse_kls.append(float(rollout["sparse_guidance_kl"]))
        sample_outputs.append(str(rollout["answer_text"]))
        if rollout.get("gate_active") is not None:
            gate_active_values.append(float(bool(rollout["gate_active"])))

    greedy_cfg = RolloutConfig(**{**cfg.__dict__, "temperature": 0.0})
    greedy = runtime_rollout(
        model=model,
        tokenizer=tokenizer,
        method=method,
        prompt_text=case.prompt,
        edit=edit,
        raw_edit=raw_edit,
        answer_len=answer_len,
        cfg=greedy_cfg,
    )
    greedy_exact = exact_alias_match(greedy["answer_ids"], eval_alias_token_lists)
    base_margin = probability_margin(
        model=model,
        tokenizer=tokenizer,
        prompt=case.prompt,
        target_new=edit.target,
        target_true=edit.old_target or "",
    ) if edit.old_target else None
    prompt_uid = (
        str(case.id)
        if case.id is not None
        else f"{edit.id}::{bucket_name}::{hashlib.sha1(case.prompt.encode('utf-8')).hexdigest()[:16]}"
    )
    gate_activation_rate = (
        sum(gate_active_values) / len(gate_active_values)
        if gate_active_values
        else None
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "split_role": raw_edit.get("split_role"),
        "method": method,
        "method_variant": configured_method_variant,
        "gate_mode": configured_gate_mode,
        "gate_activation_rate": gate_activation_rate,
        "edit_id": edit.id,
        "case_id": raw_edit.get("case_id", edit.id),
        "prompt_uid": prompt_uid,
        "prompt_id": case.id,
        "prompt_text": case.prompt,
        "rendered_prompt": case.prompt,
        "bucket": bucket_name,
        "prompt": case.prompt,
        "target": case.target,
        "relation_id": raw_edit.get("relation_id"),
        "target_length_bin": raw_edit.get("target_length_bin"),
        "exact_rate": exact_count / max(1, samples),
        "greedy_exact": float(greedy_exact),
        "token_f1": sum(token_f1_values) / max(1, len(token_f1_values)),
        "malformed_rate": malformed_count / max(1, samples),
        "target_false_positive_rate": target_fp_count / max(1, samples),
        "sparse_guidance_kl": sum(sparse_kls) / max(1, len(sparse_kls)),
        "base_margin": base_margin,
        "guided_margin": None,
        "sample_outputs": sample_outputs,
        "greedy_output": str(greedy["answer_text"]),
    }


def summarize_case_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["bucket"])), []).append(row)
    by_method_bucket: Dict[str, Dict[str, Any]] = {}
    for (method, bucket), items in sorted(groups.items()):
        key = f"{method}/{bucket}"
        by_method_bucket[key] = {
            "num_cases": len(items),
            "mean_exact_rate": mean_value(items, "exact_rate"),
            "mean_greedy_exact": mean_value(items, "greedy_exact"),
            "mean_token_f1": mean_value(items, "token_f1"),
            "mean_malformed_rate": mean_value(items, "malformed_rate"),
            "mean_target_false_positive_rate": mean_value(items, "target_false_positive_rate"),
            "mean_sparse_guidance_kl": mean_value(items, "sparse_guidance_kl"),
            "mean_base_margin": mean_value(items, "base_margin"),
        }
    return {"by_method_bucket": by_method_bucket}


def mean_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def load_raw_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def enforce_lock_requirements(
    *,
    split_role: str,
    methods: Sequence[str],
    lock_config: Optional[Dict[str, Any]],
) -> None:
    if split_role not in ANALYSIS_ROLES | LOCKED_FINAL_ROLES:
        return
    if lock_config is None:
        raise ValueError(f"{split_role} requires --lock_config")
    required = [
        "thresholds_frozen",
        "span_policy_frozen",
        "gate_policy_frozen",
        "normalization_frozen",
        "metrics_frozen",
        "selected_dev_pareto_point",
    ]
    missing = [key for key in required if not lock_config.get(key)]
    if missing:
        raise ValueError(f"{split_role} lock_config is missing/falsy keys: {missing}")
    if split_role in ANALYSIS_ROLES and not lock_config.get("path_kl_bridge_ready"):
        raise ValueError("analysis_500 requires path_kl_bridge_ready=true")
    if split_role in LOCKED_FINAL_ROLES and not lock_config.get("final_config_locked"):
        raise ValueError(f"{split_role} requires final_config_locked=true")
    if split_role in ANALYSIS_ROLES and "path_kl_bridge" not in methods and not lock_config.get("path_kl_bridge_report_path"):
        raise ValueError("analysis_500 requires path_kl_bridge in methods or path_kl_bridge_report_path")


def infer_split_role(raw_rows: Sequence[Dict[str, Any]], override: str = "") -> str:
    if override:
        return override
    roles = {str(row.get("split_role", "")) for row in raw_rows if row.get("split_role")}
    if len(roles) == 1:
        return next(iter(roles))
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edits_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+", default=["base", "mc_bridge"])
    parser.add_argument("--split_role", type=str, default="")
    parser.add_argument("--lock_config", type=str, default="")
    parser.add_argument("--protocol_version", type=str, default=PROTOCOL_VERSION)
    parser.add_argument("--edit_access", type=str, default="given_at_edit_time")
    parser.add_argument("--training_access", type=str, default="none")
    parser.add_argument("--hyperparameter_access", type=str, default="dev_tune_only")
    parser.add_argument("--stress_eval", type=int, default=0)
    parser.add_argument("--stress_name", type=str, default="")
    parser.add_argument("--target_semantics", type=str, default="")
    parser.add_argument("--analysis_500_used", type=int, default=0)
    parser.add_argument("--final_test_used", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--use_4bit", type=int, default=1)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_edits", type=int, default=0)
    parser.add_argument("--eval_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--bridge_topk", type=int, default=4)
    parser.add_argument("--mc_rollouts", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--reward_mode", type=str, default="soft_overlap", choices=("soft_overlap", "hard_exact"))
    parser.add_argument("--reward_beta", type=float, default=6.0)
    parser.add_argument("--target_logit_bias", type=float, default=5.0)
    parser.add_argument("--gate_mode", type=str, default="subject_relation")
    parser.add_argument("--relation_sim_rewrite_threshold", type=float, default=0.45)
    parser.add_argument("--relation_sim_bank_threshold", type=float, default=0.10)
    parser.add_argument("--relation_bank_path", type=str, default="")
    parser.add_argument("--relation_bank_source", type=str, default="dev_tune_200_rewrite_templates")
    parser.add_argument("--skip_candidate_coverage", type=int, default=0)
    parser.add_argument(
        "--coverage_only",
        type=int,
        default=0,
        help="Write candidate_coverage.jsonl and summary without prompt generation.",
    )
    return parser.parse_args()


def normalize_method_args(method_args: Sequence[str]) -> List[str]:
    methods: List[str] = []
    for item in method_args:
        for method in str(item).split(","):
            method = method.strip()
            if method:
                methods.append(method)
    return list(dict.fromkeys(methods))


def main() -> None:
    args = parse_args()
    if args.protocol_version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol_version: {args.protocol_version}")
    args.methods = normalize_method_args(args.methods)
    unknown = [method for method in args.methods if method not in SUPPORTED_METHODS and method != "path_kl_bridge"]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    raw_rows = load_raw_jsonl(args.edits_path)
    split_role = infer_split_role(raw_rows, args.split_role)
    lock_config = None
    if args.lock_config:
        with open(args.lock_config, "r", encoding="utf-8") as f:
            lock_config = json.load(f)
    enforce_lock_requirements(split_role=split_role, methods=args.methods, lock_config=lock_config)

    seed_everything(args.seed)
    reset_model_eval_counter()
    os.makedirs(args.output_dir, exist_ok=True)
    run_start = time.perf_counter()

    model, tokenizer = load_llada_model_and_tokenizer(
        model_id=args.model_id,
        dtype_name=args.dtype,
        use_4bit=bool(args.use_4bit),
        device_map=args.device_map,
    )
    edits = load_edits(args.edits_path)
    if args.max_edits > 0:
        edits = edits[: args.max_edits]
        raw_rows = raw_rows[: args.max_edits]
    raw_by_id = {str(row.get("id")): row for row in raw_rows}

    cfg = RolloutConfig(
        steps=args.steps,
        bridge_topk=args.bridge_topk,
        mc_rollouts=args.mc_rollouts,
        guidance_scale=args.guidance_scale,
        reward_mode=args.reward_mode,
        reward_beta=args.reward_beta,
        target_logit_bias=args.target_logit_bias,
        gate_mode=args.gate_mode,
        temperature=1.0,
        relation_sim_rewrite_threshold=args.relation_sim_rewrite_threshold,
        relation_sim_bank_threshold=args.relation_sim_bank_threshold,
        relation_bank_path=args.relation_bank_path,
        relation_bank_source=args.relation_bank_source,
    )

    coverage_rows: List[Dict[str, Any]] = []
    if not bool(args.skip_candidate_coverage):
        for edit in edits:
            raw_edit = raw_by_id.get(str(edit.id), {})
            coverage_rows.append(
                candidate_coverage_for_edit(
                    model=model,
                    tokenizer=tokenizer,
                    edit=edit,
                    raw_edit=raw_edit,
                    topk=args.bridge_topk,
                )
            )
        append_jsonl(os.path.join(args.output_dir, "candidate_coverage.jsonl"), coverage_rows)

    case_rows: List[Dict[str, Any]] = []
    if bool(args.coverage_only):
        print("[INFO] coverage_only=1; skipped prompt generation/evaluation.")
    else:
        for method in args.methods:
            if method == "path_kl_bridge":
                print("[WARN] path_kl_bridge is staged for post-sprint implementation; skipping direct eval.")
                continue
            for edit in edits:
                raw_edit = raw_by_id.get(str(edit.id), {})
                bucket_map = build_eval_buckets_for_edit(edit)
                for bucket_name in EVAL_BUCKET_ORDER:
                    for case in bucket_map[bucket_name]:
                        row = evaluate_case(
                            model=model,
                            tokenizer=tokenizer,
                            method=method,
                            edit=edit,
                            raw_edit=raw_edit,
                            bucket_name=bucket_name,
                            case=case,
                            cfg=cfg,
                            samples=args.eval_samples,
                        )
                        case_rows.append(row)
                        print(
                            f"[{method}][{edit.id}][{bucket_name}] "
                            f"exact={row['exact_rate']:.3f} f1={row['token_f1']:.3f} "
                            f"kl={row['sparse_guidance_kl']:.3f}"
                        )

    per_case_path = os.path.join(args.output_dir, "per_case_results.jsonl")
    append_jsonl(per_case_path, case_rows)
    run_config = {
        "protocol_version": PROTOCOL_VERSION,
        "edit_access": args.edit_access,
        "training_access": args.training_access,
        "hyperparameter_access": args.hyperparameter_access,
        "stress_eval": bool(args.stress_eval),
        "stress_name": args.stress_name,
        "target_semantics": args.target_semantics,
        "analysis_500_used": bool(args.analysis_500_used),
        "final_test_used": bool(args.final_test_used),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "edits_path": args.edits_path,
        "split_role": split_role,
        "split": split_role,
        "methods": list(args.methods),
        "coverage_only": bool(args.coverage_only),
        "model_id": args.model_id,
        "seed": args.seed,
        "max_edits": args.max_edits,
        "eval_samples": args.eval_samples,
        "rollout_config": cfg.__dict__,
        "lock_config": lock_config,
    }
    write_json(os.path.join(args.output_dir, "run_config.json"), run_config)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "split_role": split_role,
        "stress_eval": bool(args.stress_eval),
        "stress_name": args.stress_name,
        "target_semantics": args.target_semantics,
        "analysis_500_used": bool(args.analysis_500_used),
        "final_test_used": bool(args.final_test_used),
        "run_config_path": os.path.join(args.output_dir, "run_config.json"),
        "per_case_results_path": per_case_path,
        "candidate_coverage_path": os.path.join(args.output_dir, "candidate_coverage.jsonl"),
        "summary": summarize_case_rows(case_rows),
        "coverage_summary": summarize_coverage_rows(coverage_rows),
        "efficiency": {
            "wall_time_seconds": float(time.perf_counter() - run_start),
            "model_eval_count": get_model_eval_counter(),
            "model_evals_per_edit": get_model_eval_counter() / max(1, len(edits)),
        },
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(f"[INFO] Wrote runtime evaluation to {args.output_dir}")


def summarize_coverage_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "target_new_first_token_in_base_topk",
        "all_target_new_tokens_in_base_topk",
        "target_true_first_token_in_base_topk",
        "all_target_true_tokens_in_base_topk",
        "all_target_new_tokens_after_candidate_insert",
    ]
    summary = {"num_edits": len(rows)}
    for key in keys:
        summary[key] = sum(1 for row in rows if row.get(key)) / max(1, len(rows))
    return summary


if __name__ == "__main__":
    main()
