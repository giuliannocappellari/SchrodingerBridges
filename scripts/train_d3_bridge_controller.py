#!/usr/bin/env python3
"""Train tiny Direction 3 offline controllers from cached teacher rows.

This script never loads LLaDA. ``--fake_model 1`` means the teacher cache was
synthetic; ``--fake_model 0`` means the teacher cache came from a real GPU
teacher-cache run and is being consumed offline. The first real-cache pilot is
intentionally small: a linear top-k value controller plus a linear edit-intent
gate, both trained from cached teacher arrays only.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import (
    D3_PROTOCOL_VERSION,
    D3_ROOT,
    auc_score,
    git_commit,
    mean,
    now_utc,
    read_json,
    read_jsonl,
    repo_path,
    softmax,
    spearman,
    stable_float,
    write_csv,
    write_json,
)


VALUE_FEATURE_NAMES = [
    "bias",
    "base_logit",
    "base_prob",
    "candidate_rank",
    "candidate_is_target_new_token",
    "target_token_position",
    "target_length",
    "step",
    "active_masks",
    "mask_ratio",
    "selected_mask_position",
    "answer_position",
]
GATE_FEATURE_NAMES = [
    "bias",
    "subject_match",
    "relation_token_jaccard_to_rewrite",
    "relation_char3_jaccard_to_rewrite",
    "prompt_token_len",
    "subject_token_len",
    "question_indicator",
    "possessive_indicator",
    "subject_position_frac",
    "relation_id_bucket",
    "step",
    "active_masks",
]

# Backwards-compatible name imported by older tests and replay code.
FEATURE_NAMES = VALUE_FEATURE_NAMES
POSITIVE_PROMPT_TYPES = {"rewrite", "declarative_paraphrase"}
SAME_SUBJECT_PROMPT_TYPES = {"same_subject_different_relation", "same_subject_template"}
LOCALITY_PROMPT_TYPES = {"near_locality", "far_locality"}
FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "does",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "they",
    "to",
    "was",
    "were",
    "which",
    "who",
    "whose",
}


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_phrase(text: str, phrase: Any) -> str:
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return text
    return re.sub(rf"\b{re.escape(phrase_norm)}\b", " ", text)


def relation_text(row: Mapping[str, Any], prompt_text: Any) -> str:
    text = normalize_text(prompt_text)
    for key in ("subject", "target_new", "target_true"):
        text = remove_phrase(text, row.get(key))
    tokens = [token for token in text.split() if token not in FUNCTION_WORDS and len(token) > 1]
    return " ".join(tokens)


def prompt_token_len(prompt_text: Any) -> int:
    return len(normalize_text(prompt_text).split())


def subject_position_fraction(prompt_text: Any, subject: Any) -> float:
    prompt_norm = normalize_text(prompt_text)
    subject_norm = normalize_text(subject)
    if not prompt_norm or not subject_norm:
        return 1.0
    idx = prompt_norm.find(subject_norm)
    if idx < 0:
        return 1.0
    return idx / max(1, len(prompt_norm))


def question_indicator(prompt_text: Any) -> float:
    raw = str(prompt_text or "").strip()
    norm = normalize_text(raw)
    if "?" in raw:
        return 1.0
    return float(norm.split(" ")[:1] and norm.split(" ")[0] in {"who", "what", "where", "when", "which", "whose", "does"})


def possessive_indicator(prompt_text: Any) -> float:
    raw = str(prompt_text or "")
    norm = normalize_text(raw)
    return float("'s" in raw or " of " in f" {norm} ")


def token_set(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token and token not in FUNCTION_WORDS}


def char3_set(text: str) -> set[str]:
    compact = normalize_text(text).replace(" ", "_")
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def annotate_gate_context(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Add deployable text-similarity gate features derived from cached prompts.

    The rewrite prompt for each edit is used as the relation prototype. This is
    allowed for Direction 3 offline work because it comes from the edit request
    and not from analysis/final artifacts or post-decode outcomes.
    """

    rewrite_by_edit: Dict[str, str] = {}
    for row in rows:
        if str(row.get("prompt_type")) != "rewrite":
            continue
        edit_id = str(row.get("edit_id") or row.get("case_id"))
        rewrite_by_edit.setdefault(edit_id, str(row.get("prompt_text") or ""))

    annotated: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        edit_id = str(row.get("edit_id") or row.get("case_id"))
        prompt_text = str(row.get("prompt_text") or "")
        rewrite_text = rewrite_by_edit.get(edit_id, "")
        subject_norm = normalize_text(row.get("subject"))
        prompt_norm = normalize_text(prompt_text)
        subject_match = bool(subject_norm and re.search(rf"\b{re.escape(subject_norm)}\b", prompt_norm))
        row_relation = relation_text(row, prompt_text)
        rewrite_relation = relation_text(row, rewrite_text)
        row["gate_subject_match"] = float(subject_match)
        row["gate_relation_token_jaccard_to_rewrite"] = jaccard(token_set(row_relation), token_set(rewrite_relation))
        row["gate_relation_char3_jaccard_to_rewrite"] = jaccard(char3_set(row_relation), char3_set(rewrite_relation))
        row["gate_prompt_token_len"] = prompt_token_len(prompt_text)
        row["gate_subject_token_len"] = len(subject_norm.split()) if subject_norm else 0
        row["gate_question_indicator"] = question_indicator(prompt_text)
        row["gate_possessive_indicator"] = possessive_indicator(prompt_text)
        row["gate_subject_position_frac"] = subject_position_fraction(prompt_text, row.get("subject"))
        row["gate_relation_id_bucket"] = stable_float(str(row.get("relation_id") or ""), 0.0, 1.0)
        annotated.append(row)
    return annotated


def squash_signed(value: Any, scale: float = 8.0) -> float:
    """Bound heterogeneous teacher scores into a stable SGD feature range."""

    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite feature value: {value}")
    transformed = math.copysign(math.log1p(abs(x)), x) / scale
    return max(-4.0, min(4.0, transformed))


def array_value(row: Mapping[str, Any], keys: Sequence[str], index: int) -> float:
    for key in keys:
        values = row.get(key)
        if isinstance(values, list) and index < len(values):
            return float(values[index])
    raise KeyError(f"Missing candidate score array for keys={keys} index={index}")


def candidate_features(row: Mapping[str, Any], candidate_index: int) -> List[float]:
    candidates = candidate_ids(row)
    positions = set(target_candidate_positions(row))
    target_length = max(1, len(row.get("target_token_ids") or []))
    selected_pos = float(row.get("selected_mask_position", 0))
    answer_pos = selected_pos / max(1.0, target_length - 1.0)
    return [
        1.0,
        squash_signed(array_value(row, ["base_logits_top_k", "base_logits"], candidate_index)),
        float(array_value(row, ["base_probabilities_top_k", "base_probs"], candidate_index)),
        float(candidate_index) / max(1.0, len(candidates) - 1.0),
        float(candidate_index in positions),
        answer_pos,
        min(float(target_length), 8.0) / 8.0,
        float(row["step_index"]) / 3.0,
        float(row["active_mask_count"]) / 4.0,
        float(row.get("mask_ratio", 0.0)),
        selected_pos / 8.0,
        answer_pos,
    ]


def finite_scores(row: Mapping[str, Any], keys: Sequence[str]) -> List[float]:
    values = row.get(keys[0])
    for key in keys:
        if isinstance(row.get(key), list):
            values = row[key]
            break
    if not isinstance(values, list):
        raise KeyError(f"Missing score array for {keys}")
    out = [float(value) for value in values]
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"Non-finite score array for {keys}")
    return out


def margin_for_positions(scores: Sequence[float], positions: Sequence[int]) -> float:
    if not scores:
        return 0.0
    if not positions:
        return min(scores) - max(scores)
    target_score = max(float(scores[idx]) for idx in positions if idx < len(scores))
    non_target = [float(score) for idx, score in enumerate(scores) if idx not in set(positions)]
    return target_score - (max(non_target) if non_target else target_score)


def row_gate_features(row: Mapping[str, Any]) -> List[float]:
    return [
        1.0,
        float(row.get("gate_subject_match", 0.0)),
        float(row.get("gate_relation_token_jaccard_to_rewrite", 0.0)),
        float(row.get("gate_relation_char3_jaccard_to_rewrite", 0.0)),
        min(float(row.get("gate_prompt_token_len", 0)), 80.0) / 80.0,
        min(float(row.get("gate_subject_token_len", 0)), 12.0) / 12.0,
        float(row.get("gate_question_indicator", 0.0)),
        float(row.get("gate_possessive_indicator", 0.0)),
        float(row.get("gate_subject_position_frac", 1.0)),
        float(row.get("gate_relation_id_bucket", 0.0)),
        float(row["step_index"]) / 3.0,
        float(row["active_mask_count"]) / 4.0,
    ]


def row_features(row: Mapping[str, Any]) -> List[float]:
    return row_gate_features(row)


def candidate_ids(row: Mapping[str, Any]) -> List[int]:
    values = row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids")
    if not isinstance(values, list):
        raise KeyError("Missing top-k candidate ids")
    return [int(value) for value in values]


def target_candidate_positions(row: Mapping[str, Any]) -> List[int]:
    targets = {int(value) for value in (row.get("target_token_ids") or [])}
    return [idx for idx, token_id in enumerate(candidate_ids(row)) if token_id in targets]


def candidate_label(row: Mapping[str, Any], candidate_index: int) -> float:
    if int(row.get("label", 0)) != 1:
        return 0.0
    return float(candidate_index in target_candidate_positions(row))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def linear_score(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(w * x for w, x in zip(weights, features))


def predict(weights: Sequence[float], features: Sequence[float]) -> float:
    return sigmoid(linear_score(weights, features))


def softmax_temperature(scores: Sequence[float], temperature: float) -> List[float]:
    temperature = max(float(temperature), 1e-6)
    return softmax([float(score) / temperature for score in scores])


def teacher_distribution(row: Mapping[str, Any], field: str, temperature: float) -> List[float]:
    scores = finite_scores(row, [field])
    return softmax_temperature(scores, temperature)


def value_logits(row: Mapping[str, Any], weights: Sequence[float]) -> List[float]:
    return [linear_score(weights, candidate_features(row, idx)) for idx in range(len(candidate_ids(row)))]


def gate_label(row: Mapping[str, Any]) -> float:
    return float(str(row.get("prompt_type")) in POSITIVE_PROMPT_TYPES or int(row.get("label", 0)) == 1)


def gate_probability(row: Mapping[str, Any], weights: Sequence[float]) -> float:
    return predict(weights, row_gate_features(row))


def value_loss_for_rows(
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    target_teacher: str = "raw_bridge_scores_top_k",
    ranking_teacher: str = "mc_rollout_rewards_top_k",
    teacher_temperature: float = 8.0,
    l2: float = 0.001,
) -> Dict[str, float]:
    eps = 1e-8
    kl_terms: List[float] = []
    ranking_terms: List[float] = []
    spearman_teacher: List[float] = []
    spearman_student: List[float] = []
    target_top3_terms: List[float] = []
    for row in rows:
        candidates = candidate_ids(row)
        logits = value_logits(row, weights)
        student = softmax_temperature(logits, 1.0)
        teacher = teacher_distribution(row, target_teacher, teacher_temperature)
        kl_terms.append(sum(t * math.log((t + eps) / (s + eps)) for t, s in zip(teacher, student)))
        rank_scores = finite_scores(row, [ranking_teacher])
        raw_top_idx = max(range(len(rank_scores)), key=lambda idx: rank_scores[idx])
        pred_top_idx = max(range(len(logits)), key=lambda idx: logits[idx])
        ranking_terms.append(float(raw_top_idx != pred_top_idx))
        spearman_teacher.extend(rank_scores)
        spearman_student.extend(logits)
        positions = set(target_candidate_positions(row))
        if positions:
            order = sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)
            target_top3_terms.append(float(any(idx in positions for idx in order[:3])))
    l2_term = l2 * sum(w * w for w in weights)
    ranking_loss = mean(ranking_terms)
    bridge_loss = mean(kl_terms)
    return {
        "bridge_distillation_loss": bridge_loss,
        "ranking_loss": ranking_loss,
        "bridge_ranking_loss": ranking_loss,
        "locality_kl_proxy_loss": 0.0,
        "gate_loss": 0.0,
        "l2_correction_loss": l2_term,
        "bridge_score_spearman": spearman(spearman_teacher, spearman_student),
        "target_token_ranked_top3_rate": mean(target_top3_terms),
        "total_loss": bridge_loss + 0.25 * ranking_loss + l2_term,
    }


def gate_loss_for_rows(rows: Sequence[Mapping[str, Any]], weights: Sequence[float], l2: float = 0.001) -> Dict[str, float]:
    eps = 1e-8
    bce_terms: List[float] = []
    labels: List[int] = []
    scores: List[float] = []
    neg_scores: List[float] = []
    same_subject_labels: List[int] = []
    same_subject_scores: List[float] = []
    for row in rows:
        pred = gate_probability(row, weights)
        label = gate_label(row)
        bce_terms.append(-(label * math.log(pred + eps) + (1.0 - label) * math.log(1.0 - pred + eps)))
        labels.append(int(label))
        scores.append(pred)
        if not label:
            neg_scores.append(pred)
        if label or str(row.get("prompt_type")) in SAME_SUBJECT_PROMPT_TYPES:
            same_subject_labels.append(int(label))
            same_subject_scores.append(pred)
    l2_term = l2 * sum(w * w for w in weights)
    gate_loss = mean(bce_terms)
    return {
        "bridge_distillation_loss": 0.0,
        "ranking_loss": 0.0,
        "bridge_ranking_loss": 0.0,
        "locality_kl_proxy_loss": mean(neg_scores),
        "gate_loss": gate_loss,
        "same_subject_gate_auc": auc_score(same_subject_labels, same_subject_scores),
        "all_negative_average_guidance": mean(neg_scores),
        "l2_correction_loss": l2_term,
        "total_loss": gate_loss + 0.5 * mean(neg_scores) + l2_term,
    }


def combined_loss_for_rows(
    rows: Sequence[Mapping[str, Any]],
    value_weights: Sequence[float],
    gate_weights: Sequence[float],
    *,
    target_teacher: str = "raw_bridge_scores_top_k",
    ranking_teacher: str = "mc_rollout_rewards_top_k",
    teacher_temperature: float = 8.0,
) -> Dict[str, float]:
    value_metrics = value_loss_for_rows(
        rows,
        value_weights,
        target_teacher=target_teacher,
        ranking_teacher=ranking_teacher,
        teacher_temperature=teacher_temperature,
    )
    gate_metrics = gate_loss_for_rows(rows, gate_weights)
    return {
        **value_metrics,
        "gate_loss": gate_metrics["gate_loss"],
        "same_subject_gate_auc": gate_metrics["same_subject_gate_auc"],
        "locality_kl_proxy_loss": gate_metrics["locality_kl_proxy_loss"],
        "total_loss": value_metrics["total_loss"] + 0.5 * gate_metrics["total_loss"],
    }


def loss_for_rows(rows: Sequence[Mapping[str, Any]], weights: Sequence[float], l2: float = 0.001) -> Dict[str, float]:
    return value_loss_for_rows(rows, weights, l2=l2)


def train_value_controller(
    rows: Sequence[Mapping[str, Any]],
    epochs: int,
    lr: float,
    *,
    target_teacher: str = "raw_bridge_scores_top_k",
    teacher_temperature: float = 8.0,
) -> Tuple[List[float], List[Dict[str, float]]]:
    weights = [0.0 for _ in VALUE_FEATURE_NAMES]
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        for row in rows:
            logits = value_logits(row, weights)
            student = softmax_temperature(logits, 1.0)
            teacher = teacher_distribution(row, target_teacher, teacher_temperature)
            for idx in range(len(candidate_ids(row))):
                features = candidate_features(row, idx)
                grad_scale = student[idx] - teacher[idx]
                for i, feature in enumerate(features):
                    weights[i] -= lr * grad_scale * feature
        metrics = value_loss_for_rows(
            rows,
            weights,
            target_teacher=target_teacher,
            teacher_temperature=teacher_temperature,
        )
        metrics["epoch"] = float(epoch)
        history.append(metrics)
    return weights, history


def train_gate_controller(rows: Sequence[Mapping[str, Any]], epochs: int, lr: float) -> Tuple[List[float], List[Dict[str, float]]]:
    weights = [0.0 for _ in GATE_FEATURE_NAMES]
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        for row in rows:
            features = row_gate_features(row)
            y = gate_label(row)
            pred = predict(weights, features)
            grad_scale = pred - y
            for i, feature in enumerate(features):
                weights[i] -= lr * grad_scale * feature
        metrics = gate_loss_for_rows(rows, weights)
        metrics["epoch"] = float(epoch)
        history.append(metrics)
    return weights, history


def train_controller(rows: Sequence[Mapping[str, Any]], epochs: int, lr: float) -> Tuple[List[float], List[Dict[str, float]]]:
    return train_value_controller(rows, epochs, lr)


def train_fake(rows: Sequence[Mapping[str, Any]], epochs: int, lr: float) -> Tuple[List[float], List[Dict[str, float]]]:
    return train_value_controller(rows, epochs, lr)


def parse_controllers(text: str) -> List[str]:
    controllers = [item.strip() for item in str(text).split(",") if item.strip()]
    allowed = {"value", "gate", "value_gate"}
    unknown = sorted(set(controllers) - allowed)
    if unknown:
        raise ValueError(f"Unknown controller(s): {unknown}")
    return controllers or ["value_gate"]


def train_bundle(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    *,
    controllers: Sequence[str],
    epochs: int,
    lr: float,
    target_teacher: str,
    ranking_teacher: str,
    teacher_temperature: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    models: Dict[str, Any] = {}
    history_rows: List[Dict[str, Any]] = []
    value_weights: List[float] | None = None
    value_history: List[Dict[str, float]] = []
    gate_weights: List[float] | None = None
    gate_history: List[Dict[str, float]] = []

    if any(name in controllers for name in ("value", "value_gate")):
        value_weights, value_history = train_value_controller(
            train_rows,
            epochs,
            lr,
            target_teacher=target_teacher,
            teacher_temperature=teacher_temperature,
        )
    if any(name in controllers for name in ("gate", "value_gate")):
        gate_weights, gate_history = train_gate_controller(train_rows, epochs, lr)

    for name in controllers:
        if name == "value":
            assert value_weights is not None
            train_metrics = value_loss_for_rows(
                train_rows,
                value_weights,
                target_teacher=target_teacher,
                ranking_teacher=ranking_teacher,
                teacher_temperature=teacher_temperature,
            )
            val_metrics = value_loss_for_rows(
                val_rows,
                value_weights,
                target_teacher=target_teacher,
                ranking_teacher=ranking_teacher,
                teacher_temperature=teacher_temperature,
            )
            models[name] = {
                "controller_type": name,
                "value_feature_names": VALUE_FEATURE_NAMES,
                "value_weights": value_weights,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            for row in value_history:
                history_rows.append({"controller": name, **row})
        elif name == "gate":
            assert gate_weights is not None
            train_metrics = gate_loss_for_rows(train_rows, gate_weights)
            val_metrics = gate_loss_for_rows(val_rows, gate_weights)
            models[name] = {
                "controller_type": name,
                "gate_feature_names": GATE_FEATURE_NAMES,
                "gate_weights": gate_weights,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            for row in gate_history:
                history_rows.append({"controller": name, **row})
        elif name == "value_gate":
            assert value_weights is not None and gate_weights is not None
            train_metrics = combined_loss_for_rows(
                train_rows,
                value_weights,
                gate_weights,
                target_teacher=target_teacher,
                ranking_teacher=ranking_teacher,
                teacher_temperature=teacher_temperature,
            )
            val_metrics = combined_loss_for_rows(
                val_rows,
                value_weights,
                gate_weights,
                target_teacher=target_teacher,
                ranking_teacher=ranking_teacher,
                teacher_temperature=teacher_temperature,
            )
            models[name] = {
                "controller_type": name,
                "value_feature_names": VALUE_FEATURE_NAMES,
                "gate_feature_names": GATE_FEATURE_NAMES,
                "value_weights": value_weights,
                "gate_weights": gate_weights,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            for row in value_history:
                history_rows.append({"controller": name, "component": "value", **row})
            for row in gate_history:
                history_rows.append({"controller": name, "component": "gate", **row})
    return models, history_rows


def gate_data_quality_rows(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    quality_rows: List[Dict[str, Any]] = []
    edit_sets: Dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        edit_sets[split] = {str(row.get("edit_id") or row.get("case_id")) for row in rows}
        by_prompt: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            by_prompt.setdefault(str(row.get("prompt_type")), []).append(row)
        for prompt_type, prompt_rows in sorted(by_prompt.items()):
            real_count = sum(1 for row in prompt_rows if str(row.get("prompt_text") or "").strip())
            synthetic_count = sum(1 for row in prompt_rows if bool(row.get("synthetic_from_metadata", False)))
            quality_rows.append(
                {
                    "split": split,
                    "prompt_type": prompt_type,
                    "num_rows": len(prompt_rows),
                    "num_edits": len({str(row.get("edit_id") or row.get("case_id")) for row in prompt_rows}),
                    "real_prompt_text_coverage": real_count / len(prompt_rows) if prompt_rows else 0.0,
                    "synthetic_fallback_rate": synthetic_count / len(prompt_rows) if prompt_rows else 0.0,
                }
            )

    overlaps: Dict[str, List[str]] = {}
    split_names = sorted(edit_sets)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlaps[f"{left}::{right}"] = sorted(edit_sets[left] & edit_sets[right])

    def coverage(split: str, prompt_type: str) -> float:
        matches = [row for row in quality_rows if row["split"] == split and row["prompt_type"] == prompt_type]
        return float(matches[0]["real_prompt_text_coverage"]) if matches else 0.0

    def fallback(split: str, prompt_type: str) -> float:
        matches = [row for row in quality_rows if row["split"] == split and row["prompt_type"] == prompt_type]
        return float(matches[0]["synthetic_fallback_rate"]) if matches else 1.0

    checks = {
        "rewrite_real_text_coverage_ge_0_95": all(coverage(split, "rewrite") >= 0.95 for split in split_names),
        "paraphrase_real_text_coverage_ge_0_95": all(coverage(split, "declarative_paraphrase") >= 0.95 for split in split_names),
        "near_far_real_text_coverage_ge_0_95": all(
            coverage(split, prompt_type) >= 0.95
            for split in split_names
            for prompt_type in ("near_locality", "far_locality")
        ),
        "synthetic_fallback_le_0_20_for_real_source_categories": all(
            fallback(split, prompt_type) <= 0.20
            for split in split_names
            for prompt_type in ("rewrite", "declarative_paraphrase", "near_locality", "far_locality", "generation")
        ),
        "same_subject_negatives_ge_0_80_of_edits": all(
            (
                len({
                    str(row.get("edit_id") or row.get("case_id"))
                    for row in rows_by_split[split]
                    if str(row.get("prompt_type")) == "same_subject_different_relation"
                })
                / max(1, len(edit_sets[split]))
            )
            >= 0.80
            for split in split_names
        ),
        "train_val_edit_ids_disjoint": all(not values for values in overlaps.values()),
    }
    summary = {
        "data_materialization_status": "real_prompt_text_from_teacher_cache",
        "analysis_500_used": False,
        "final_test_used": False,
        "edit_overlaps": overlaps,
        "acceptance_checks": checks,
        "real_prompt_gate_data_pass": all(checks.values()),
    }
    return quality_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_cache_dir", type=Path, default=D3_ROOT / "fake_teacher_cache_v1")
    parser.add_argument("--output_dir", type=Path, default=D3_ROOT / "fake_controller_train_v1")
    parser.add_argument("--fake_model", type=int, default=0)
    parser.add_argument("--controllers", type=str, default="value_gate")
    parser.add_argument("--target_teacher", type=str, default="raw_bridge_scores_top_k")
    parser.add_argument("--ranking_teacher", type=str, default="mc_rollout_rewards_top_k")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--teacher_temperature", type=float, default=8.0)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_train.jsonl")
    val_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_val.jsonl")
    if not train_rows or not val_rows:
        raise AssertionError("Teacher cache train/val rows are required")
    train_rows = annotate_gate_context(train_rows)
    val_rows = annotate_gate_context(val_rows)
    gate_quality_rows, gate_quality_summary = gate_data_quality_rows({"train": train_rows, "val": val_rows})

    cache_summary = read_json(args.teacher_cache_dir / "report_summary.json")
    if bool(cache_summary.get("analysis_500_used", False)) or bool(cache_summary.get("final_test_used", False)):
        raise AssertionError("Locked analysis/final cache is not allowed for D3 offline training")
    controllers = parse_controllers(args.controllers)
    initial_value_metrics = value_loss_for_rows(
        train_rows,
        [0.0 for _ in VALUE_FEATURE_NAMES],
        target_teacher=args.target_teacher,
        ranking_teacher=args.ranking_teacher,
        teacher_temperature=float(args.teacher_temperature),
    )
    initial_gate_metrics = gate_loss_for_rows(train_rows, [0.0 for _ in GATE_FEATURE_NAMES])
    models, history_rows = train_bundle(
        train_rows,
        val_rows,
        controllers=controllers,
        epochs=int(args.epochs),
        lr=float(args.lr),
        target_teacher=args.target_teacher,
        ranking_teacher=args.ranking_teacher,
        teacher_temperature=float(args.teacher_temperature),
    )
    default_controller = "value_gate" if "value_gate" in models else controllers[0]
    default_model = models[default_controller]
    train_metrics = default_model["train_metrics"]
    val_metrics = default_model["val_metrics"]
    fake_model = bool(args.fake_model)
    stage = (
        "Direction 3 fake bridge controller training"
        if fake_model
        else "Direction 3 Stage 1B real-cache offline controller training"
    )
    loss_decreased_by_controller = {}
    for name, model in models.items():
        if name == "gate":
            initial_loss = initial_gate_metrics["total_loss"]
        elif name == "value_gate":
            initial_loss = initial_value_metrics["total_loss"] + 0.5 * initial_gate_metrics["total_loss"]
        else:
            initial_loss = initial_value_metrics["total_loss"]
        loss_decreased_by_controller[name] = model["train_metrics"]["total_loss"] <= initial_loss
    payload = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": stage,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": fake_model,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "teacher_cache_dir": str(args.teacher_cache_dir),
        "controllers": controllers,
        "default_controller": default_controller,
        "target_teacher": args.target_teacher,
        "ranking_teacher": args.ranking_teacher,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "teacher_temperature": float(args.teacher_temperature),
        "feature_contract": "deployable_v1",
        "forbidden_runtime_inputs": [
            "raw_bridge_scores_top_k",
            "mc_rollout_rewards_top_k",
            "myopic_scores_top_k",
            "no_rollout_scores_top_k",
            "chosen_token_id",
            "final_decoded_output",
            "final_edit_success",
            "final_locality_success",
            "malformed",
            "sparse_guidance_kl",
            "prompt_type",
            "negative_type",
            "case_id",
            "edit_id",
            "split_role",
        ],
        "real_prompt_gate_data_pass": bool(gate_quality_summary["real_prompt_gate_data_pass"]),
        "gate_data_quality": gate_quality_summary,
        "value_feature_names": VALUE_FEATURE_NAMES,
        "gate_feature_names": GATE_FEATURE_NAMES,
        "feature_names": VALUE_FEATURE_NAMES,
        "weights": default_model.get("value_weights", default_model.get("gate_weights", [])),
        "initial_train_metrics": initial_value_metrics,
        "initial_gate_metrics": initial_gate_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "models": models,
        "history": history_rows,
        "loss_decreased_by_controller": loss_decreased_by_controller,
        "loss_decreased": all(loss_decreased_by_controller.values()),
        "artifacts": {
            "controller_weights": str(args.output_dir / "controller_weights.json"),
            "train_metrics": str(args.output_dir / "train_metrics.json"),
            "training_history": str(args.output_dir / "training_history.csv"),
            "deployable_gate_data_summary": str(args.output_dir / "deployable_gate_data_summary.json"),
            "deployable_gate_data_quality": str(args.output_dir / "deployable_gate_data_quality.csv"),
        },
    }
    write_json(
        args.output_dir / "controller_weights.json",
        {
            "feature_names": VALUE_FEATURE_NAMES,
            "weights": payload["weights"],
            "value_feature_names": VALUE_FEATURE_NAMES,
            "gate_feature_names": GATE_FEATURE_NAMES,
            "default_controller": default_controller,
            "controllers": models,
        },
    )
    write_csv(args.output_dir / "training_history.csv", history_rows)
    write_csv(args.output_dir / "deployable_gate_data_quality.csv", gate_quality_rows)
    write_json(args.output_dir / "deployable_gate_data_summary.json", gate_quality_summary)
    write_json(args.output_dir / "train_metrics.json", payload)
    write_json(args.output_dir / "report_summary.json", payload)
    print(f"[INFO] Wrote Direction 3 controller training output to {args.output_dir}")


if __name__ == "__main__":
    main()
