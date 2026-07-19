#!/usr/bin/env python3
"""Pure statistical mechanisms used by the bounded N1-N5 pilots."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


FORBIDDEN_RUNTIME_FEATURES = {
    "prompt_type",
    "negative_type",
    "evaluation_bucket",
    "split_label",
    "case_id",
    "final_outcome",
    "teacher_score",
    "future_output",
}


def validate_runtime_feature_schema(features: Sequence[str]) -> None:
    lowered = {str(value).casefold() for value in features}
    forbidden = sorted(lowered & FORBIDDEN_RUNTIME_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden runtime features: {forbidden}")


def _group_means(values: torch.Tensor, labels: Sequence[str]) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("values and labels must align")
    output = torch.empty_like(values.float())
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)
    for indices in groups.values():
        mean = values[indices].float().mean(dim=0)
        output[indices] = mean
    return output


def relation_residualize(
    keys: torch.Tensor,
    relation_ids: Sequence[str],
    *,
    subject_negative_keys: torch.Tensor | None = None,
    mode: str = "full",
    shrinkage: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Remove subject and/or relation main effects from edit keys."""

    if keys.ndim != 2 or keys.shape[0] != len(relation_ids):
        raise ValueError("keys and relation_ids must align")
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if mode not in {"raw", "subject", "relation", "full"}:
        raise ValueError(f"unknown residualization mode: {mode}")
    value = keys.float()
    global_mean = value.mean(dim=0, keepdim=True)
    relation_mean = _group_means(value, relation_ids)
    relation_effect = (1.0 - float(shrinkage)) * (relation_mean - global_mean)
    subject_effect = torch.zeros_like(value)
    if subject_negative_keys is not None:
        if subject_negative_keys.shape != value.shape:
            raise ValueError("subject_negative_keys must match keys")
        negatives = subject_negative_keys.float()
        subject_effect = negatives - negatives.mean(dim=0, keepdim=True)
    if mode == "raw":
        residual = value
    elif mode == "subject":
        residual = value - subject_effect
    elif mode == "relation":
        residual = value - relation_effect
    else:
        if subject_negative_keys is None:
            raise ValueError("full residualization requires subject-negative keys")
        residual = value - subject_effect - relation_effect
    return residual, {
        "raw_norm": float(value.norm(dim=1).mean()),
        "residual_norm": float(residual.norm(dim=1).mean()),
        "relation_effect_norm": float(relation_effect.norm(dim=1).mean()),
        "subject_effect_norm": float(subject_effect.norm(dim=1).mean()),
        "shrinkage": float(shrinkage),
    }

def mean_row_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("cosine inputs must be aligned matrices")
    return float(
        torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=1).mean()
    )


def fisher_discriminant_ratio(
    positives: torch.Tensor, negatives: torch.Tensor
) -> float:
    if positives.ndim != 2 or negatives.ndim != 2:
        raise ValueError("discriminant inputs must be matrices")
    if positives.shape != negatives.shape:
        raise ValueError("paired discriminant inputs must align")
    difference = positives.float().mean(dim=0) - negatives.float().mean(dim=0)
    centered_positive = positives.float() - positives.float().mean(dim=0)
    centered_negative = negatives.float() - negatives.float().mean(dim=0)
    within = centered_positive.square().mean() + centered_negative.square().mean()
    return float(difference.square().mean() / within.clamp_min(1e-12))


def fisher_diagonal(protected_keys: torch.Tensor, damping: float) -> torch.Tensor:
    if protected_keys.ndim != 2 or protected_keys.shape[0] < 2:
        raise ValueError("protected_keys must contain at least two rows")
    diagonal = protected_keys.float().square().mean(dim=0) + float(damping)
    if not torch.isfinite(diagonal).all() or bool((diagonal <= 0).any()):
        raise FloatingPointError("invalid empirical Fisher diagonal")
    return diagonal


def fisher_low_rank(
    protected_keys: torch.Tensor, rank: int, damping: float
) -> dict[str, torch.Tensor | float]:
    values = protected_keys.float()
    centered = values - values.mean(dim=0, keepdim=True)
    _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    effective = min(int(rank), int(vh.shape[0]))
    basis = vh[:effective].T.contiguous()
    eigenvalues = singular[:effective].square() / max(values.shape[0] - 1, 1)
    return {"basis": basis, "eigenvalues": eigenvalues, "damping": float(damping)}


def fisher_quadratic(update: torch.Tensor, fisher: torch.Tensor) -> float:
    value = update.float()
    if fisher.ndim == 1:
        if value.shape[-1] != fisher.shape[0]:
            raise ValueError("Fisher diagonal width mismatch")
        return float((value.square() * fisher.float().unsqueeze(0)).sum())
    if fisher.ndim == 2:
        return float(torch.einsum("oi,ij,oj->", value, fisher.float(), value))
    raise ValueError("Fisher must be diagonal or matrix")


def trust_region_scale(
    update: torch.Tensor, fisher: torch.Tensor, radius: float
) -> tuple[torch.Tensor, dict[str, float]]:
    before = fisher_quadratic(update, fisher)
    scale = min(1.0, math.sqrt(float(radius) / max(before, 1e-12)))
    scaled = update.float() * scale
    return scaled, {
        "fisher_quadratic_before": before,
        "fisher_quadratic_after": fisher_quadratic(scaled, fisher),
        "trust_region_radius": float(radius),
        "scale": float(scale),
    }


def protected_response(update: torch.Tensor, keys: torch.Tensor) -> float:
    if update.ndim != 2 or keys.ndim != 2 or update.shape[1] != keys.shape[1]:
        raise ValueError("update and protected keys are incompatible")
    response = keys.float() @ update.float().T
    return float(response.square().mean())


def primal_dual_project_update(
    update: torch.Tensor,
    protected_by_family: Mapping[str, torch.Tensor],
    limits: Mapping[str, float],
    *,
    multiplier_step: float,
    penalty_growth: float,
    iterations: int = 20,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Minimize protected responses with nonnegative family multipliers."""

    if not protected_by_family:
        raise ValueError("at least one protected family is required")
    value = update.float().clone()
    multipliers = {name: 0.0 for name in protected_by_family}
    trajectory = []
    base_norm = value.norm().clamp_min(1e-12)
    penalty = 1.0
    for step in range(int(iterations)):
        violations = {}
        gradient = torch.zeros_like(value)
        for name, keys in protected_by_family.items():
            current = protected_response(value, keys)
            violation = current - float(limits[name])
            violations[name] = violation
            multipliers[name] = max(
                0.0, multipliers[name] + float(multiplier_step) * violation
            )
            if violation > 0:
                normalized = keys.float() / math.sqrt(max(keys.shape[0], 1))
                family_gradient = 2.0 * (value @ normalized.T) @ normalized
                gradient += (multipliers[name] + penalty * violation) * family_gradient
        gradient_scale = gradient.norm().clamp_min(1e-12)
        if any(value > 0 for value in violations.values()):
            value = value - 0.05 * base_norm * gradient / gradient_scale
        penalty *= float(penalty_growth)
        trajectory.append(
            {
                "step": step,
                "maximum_violation": max(violations.values()),
                "satisfied_fraction": sum(v <= 0 for v in violations.values())
                / len(violations),
                "multiplier_max": max(multipliers.values()),
                "update_norm": float(value.norm()),
            }
        )
        if all(v <= 0 for v in violations.values()):
            break
    final = {
        name: protected_response(value, keys) for name, keys in protected_by_family.items()
    }
    return value, {
        "trajectory": trajectory,
        "multipliers": multipliers,
        "final_responses": final,
        "all_constraints_satisfied": all(
            final[name] <= float(limits[name]) for name in final
        ),
        "finite": bool(torch.isfinite(value).all())
        and all(math.isfinite(value) for value in multipliers.values()),
    }


@dataclass
class LogisticRiskModel:
    mean: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor
    bias: float
    feature_names: tuple[str, ...]

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features.float() - self.mean) / self.scale
        return torch.sigmoid(standardized @ self.weights + float(self.bias))

    def runtime_schema(self) -> list[str]:
        return list(self.feature_names)


def fit_logistic_risk(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    feature_names: Sequence[str],
    steps: int = 500,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
) -> LogisticRiskModel:
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("risk features and labels must align")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names must match feature width")
    validate_runtime_feature_schema(feature_names)
    mean = features.float().mean(dim=0)
    scale = features.float().std(dim=0).clamp_min(1e-6)
    x = (features.float() - mean) / scale
    y = labels.float()
    weights = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad()
        logits = x @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + float(l2) * weights.square().mean()
        loss.backward()
        optimizer.step()
    return LogisticRiskModel(
        mean.detach(),
        scale.detach(),
        weights.detach(),
        float(bias.detach()),
        tuple(map(str, feature_names)),
    )


def binomial_upper_bound(failures: int, total: int, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound via binomial inversion."""

    if total <= 0 or failures < 0 or failures > total:
        raise ValueError("invalid binomial counts")
    if failures == total:
        return 1.0
    alpha = 1.0 - float(confidence)

    def cdf(probability: float) -> float:
        return sum(
            math.comb(total, k)
            * probability**k
            * (1.0 - probability) ** (total - k)
            for k in range(failures + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def calibrate_selective_threshold(
    risks: Sequence[float],
    failures: Sequence[bool],
    *,
    maximum_upper_bound: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if len(risks) != len(failures) or not risks:
        raise ValueError("calibration rows must align and be nonempty")
    candidates = sorted(set(map(float, risks)))
    best = None
    for threshold in candidates:
        accepted = [index for index, value in enumerate(risks) if float(value) <= threshold]
        if not accepted:
            continue
        count = sum(bool(failures[index]) for index in accepted)
        upper = binomial_upper_bound(count, len(accepted), confidence)
        row = {
            "threshold": threshold,
            "accepted": len(accepted),
            "coverage": len(accepted) / len(risks),
            "failures": count,
            "empirical_risk": count / len(accepted),
            "risk_upper_bound": upper,
        }
        if upper <= float(maximum_upper_bound) and (
            best is None or row["coverage"] > best["coverage"]
        ):
            best = row
    return best or {
        "threshold": None,
        "accepted": 0,
        "coverage": 0.0,
        "failures": 0,
        "empirical_risk": 0.0,
        "risk_upper_bound": 1.0,
    }


class LowRankPairwiseCoupler(torch.nn.Module):
    def __init__(self, embedding_width: int, rank: int) -> None:
        super().__init__()
        self.left = torch.nn.Linear(int(embedding_width), int(rank), bias=False)
        self.right = torch.nn.Linear(int(embedding_width), int(rank), bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (self.left(left) * self.right(right)).sum(dim=-1) + self.bias


def fit_pairwise_coupler(
    embeddings: torch.Tensor,
    positive_pairs: Sequence[tuple[int, int]],
    *,
    rank: int,
    steps: int = 400,
    seed: int = 0,
) -> LowRankPairwiseCoupler:
    if not positive_pairs:
        raise ValueError("positive_pairs must not be empty")
    generator = random.Random(int(seed))
    positives = list(positive_pairs)
    vocabulary = sorted({token for pair in positives for token in pair})
    negatives = [
        (left, generator.choice(vocabulary))
        for left, right in positives
        if len(vocabulary) > 1
        for _ in range(1)
    ]
    negatives = [pair if pair not in positives else (pair[0], vocabulary[(vocabulary.index(pair[1]) + 1) % len(vocabulary)]) for pair in negatives]
    pairs = positives + negatives
    labels = torch.tensor([1.0] * len(positives) + [0.0] * len(negatives))
    left_ids = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    right_ids = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    model = LowRankPairwiseCoupler(embeddings.shape[1], int(rank))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    table = embeddings.detach().float().cpu()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        logits = model(table[left_ids], table[right_ids])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    return model.eval()


def exact_coupled_decode(
    candidate_ids: Sequence[Sequence[int]],
    candidate_log_probs: Sequence[Sequence[float]],
    embeddings: torch.Tensor,
    coupler: LowRankPairwiseCoupler,
    *,
    coupling_strength: float,
) -> tuple[list[int], float]:
    if len(candidate_ids) != len(candidate_log_probs) or not candidate_ids:
        raise ValueError("candidate support and log probabilities must align")
    for ids, values in zip(candidate_ids, candidate_log_probs):
        if len(ids) != len(values) or not ids:
            raise ValueError("each candidate position must be nonempty and aligned")
    best_sequence = None
    best_score = -math.inf
    table = embeddings.detach().float().cpu()
    with torch.no_grad():
        for indices in itertools.product(*(range(len(ids)) for ids in candidate_ids)):
            sequence = [int(candidate_ids[position][index]) for position, index in enumerate(indices)]
            score = sum(float(candidate_log_probs[position][index]) for position, index in enumerate(indices))
            for left, right in zip(sequence, sequence[1:]):
                score += float(coupling_strength) * float(
                    coupler(table[left].unsqueeze(0), table[right].unsqueeze(0))[0]
                )
            if score > best_score:
                best_score = score
                best_sequence = sequence
    if best_sequence is None:
        raise RuntimeError("exact coupled inference produced no sequence")
    return best_sequence, best_score


def pairwise_mutual_information(token_sequences: Sequence[Sequence[int]]) -> float:
    pairs = Counter(
        (int(left), int(right))
        for sequence in token_sequences
        for left, right in zip(sequence, sequence[1:])
    )
    if not pairs:
        return 0.0
    left_counts = Counter(left for left, _right in pairs.elements())
    right_counts = Counter(right for _left, right in pairs.elements())
    total = sum(pairs.values())
    value = 0.0
    for (left, right), count in pairs.items():
        joint = count / total
        value += joint * math.log(
            joint / max((left_counts[left] / total) * (right_counts[right] / total), 1e-12)
        )
    return float(value)


def paired_bootstrap_delta(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("paired bootstrap has no aligned cases")
    deltas = [float(left[key]) - float(right[key]) for key in keys]
    generator = random.Random(int(seed))
    samples = []
    for _ in range(int(trials)):
        samples.append(sum(generator.choice(deltas) for _ in deltas) / len(deltas))
    samples.sort()
    return {
        "delta": sum(deltas) / len(deltas),
        "ci_low": samples[int(0.025 * (len(samples) - 1))],
        "ci_high": samples[int(0.975 * (len(samples) - 1))],
        "num_cases": len(keys),
        "trials": int(trials),
    }
