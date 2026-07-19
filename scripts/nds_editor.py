#!/usr/bin/env python3
"""Deployable key-space transforms for the bounded N1-N3 pilots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from scripts.nds_methods import (
    fisher_quadratic,
    primal_dual_project_update,
    protected_response,
    trust_region_scale,
)


@dataclass(frozen=True)
class RelationKeyStatistics:
    global_mean: torch.Tensor
    relation_means: Mapping[str, torch.Tensor]
    fallback_relation_mean: torch.Tensor


def fit_relation_key_statistics(
    keys: torch.Tensor, relation_ids: Sequence[str]
) -> RelationKeyStatistics:
    if keys.ndim != 2 or keys.shape[0] != len(relation_ids):
        raise ValueError("relation statistics inputs must align")
    value = keys.detach().float().cpu()
    global_mean = value.mean(dim=0)
    means = {
        relation: value[
            [index for index, label in enumerate(relation_ids) if str(label) == relation]
        ].mean(dim=0)
        for relation in sorted(set(map(str, relation_ids)))
    }
    return RelationKeyStatistics(global_mean, means, global_mean.clone())


def residualize_runtime_keys(
    keys: torch.Tensor,
    relation_ids: Sequence[str],
    statistics: RelationKeyStatistics,
    *,
    subject_anchor_keys: torch.Tensor | None,
    mode: str,
    shrinkage: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if keys.ndim != 2 or keys.shape[0] != len(relation_ids):
        raise ValueError("runtime keys and relation IDs must align")
    if mode not in {"raw", "subject", "relation", "full"}:
        raise ValueError(f"unsupported residualization mode: {mode}")
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    value = keys.float()
    global_mean = statistics.global_mean.to(value.device)
    relation_effects = []
    for relation in relation_ids:
        relation_mean = statistics.relation_means.get(
            str(relation), statistics.fallback_relation_mean
        ).to(value.device)
        relation_effects.append(
            (1.0 - float(shrinkage)) * (relation_mean - global_mean)
        )
    relation_effect = torch.stack(relation_effects)
    subject_effect = torch.zeros_like(value)
    if subject_anchor_keys is not None:
        if subject_anchor_keys.shape != value.shape:
            raise ValueError("subject anchor keys must match runtime keys")
        subject_effect = subject_anchor_keys.to(value.device).float() - global_mean
    if mode == "raw":
        residual = value
    elif mode == "subject":
        residual = value - subject_effect
    elif mode == "relation":
        residual = value - relation_effect
    else:
        if subject_anchor_keys is None:
            raise ValueError("full residualization requires subject anchor keys")
        residual = value - subject_effect - relation_effect
    return residual, {
        "mode": mode,
        "shrinkage": float(shrinkage),
        "raw_norm": float(value.norm(dim=1).mean()),
        "residual_norm": float(residual.norm(dim=1).mean()),
        "relation_effect_norm": float(relation_effect.norm(dim=1).mean()),
        "subject_effect_norm": float(subject_effect.norm(dim=1).mean()),
        "runtime_features": ["subject", "relation_id", "rewrite_prompt"],
        "evaluation_prompt_used": False,
    }


def linearized_gain(
    update: torch.Tensor, keys: torch.Tensor, residuals: torch.Tensor
) -> float:
    predicted = (update.float() @ keys.float().T).T
    return float((predicted * residuals.float()).sum())


def _match_gain(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    *,
    maximum_scale: float = 10.0,
) -> tuple[torch.Tensor, float]:
    reference_gain = linearized_gain(reference, keys, residuals)
    candidate_gain = linearized_gain(candidate, keys, residuals)
    if abs(candidate_gain) < 1e-12 or reference_gain * candidate_gain <= 0:
        return candidate, 1.0
    scale = min(float(maximum_scale), abs(reference_gain / candidate_gain))
    return candidate * scale, scale


def diagonal_fisher_update(
    update: torch.Tensor,
    fisher_diagonal: torch.Tensor,
    keys: torch.Tensor,
    residuals: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    diagonal = fisher_diagonal.to(update.device).float().clamp_min(1e-8)
    normalized = diagonal / diagonal.mean().clamp_min(1e-8)
    candidate = update.float() / normalized.unsqueeze(0)
    candidate, scale = _match_gain(candidate, update, keys, residuals)
    return candidate, {
        "mode": "diagonal_fisher",
        "matched_gain_scale": scale,
        "linearized_gain_before": linearized_gain(update, keys, residuals),
        "linearized_gain_after": linearized_gain(candidate, keys, residuals),
        "fisher_quadratic_before": fisher_quadratic(update, diagonal),
        "fisher_quadratic_after": fisher_quadratic(candidate, diagonal),
    }


def low_rank_fisher_update(
    update: torch.Tensor,
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    damping: float,
    keys: torch.Tensor,
    residuals: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    value = update.float()
    vectors = basis.to(value.device).float()
    spectrum = eigenvalues.to(value.device).float().clamp_min(0.0)
    if vectors.ndim != 2 or vectors.shape[0] != value.shape[1]:
        raise ValueError("low-rank Fisher basis width mismatch")
    if vectors.shape[1] != spectrum.shape[0]:
        raise ValueError("low-rank Fisher spectrum mismatch")
    damp = max(float(damping), 1e-8)
    coefficients = value @ vectors
    candidate = value / damp
    correction = coefficients * (spectrum / (damp * (spectrum + damp))).unsqueeze(0)
    candidate = candidate - correction @ vectors.T
    candidate, scale = _match_gain(candidate, value, keys, residuals)
    response_before = float((value @ vectors).square().sum())
    response_after = float((candidate @ vectors).square().sum())
    return candidate, {
        "mode": "low_rank_fisher",
        "rank": int(vectors.shape[1]),
        "damping": damp,
        "matched_gain_scale": scale,
        "linearized_gain_before": linearized_gain(value, keys, residuals),
        "linearized_gain_after": linearized_gain(candidate, keys, residuals),
        "protected_subspace_energy_before": response_before,
        "protected_subspace_energy_after": response_after,
    }


def fisher_trust_region_update(
    update: torch.Tensor,
    fisher_diagonal: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    candidate, report = trust_region_scale(update, fisher_diagonal, float(radius))
    return candidate, {"mode": "fisher_trust_region", **report}


def fixed_penalty_update(
    update: torch.Tensor,
    protected_by_family: Mapping[str, torch.Tensor],
    strength: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    value = update.float()
    gradient = torch.zeros_like(value)
    before = {}
    for name, keys in protected_by_family.items():
        key_value = keys.to(value.device).float()
        before[name] = protected_response(value, key_value)
        normalized = key_value / math.sqrt(max(key_value.shape[0], 1))
        gradient += 2.0 * (value @ normalized.T) @ normalized
    candidate = value - float(strength) * gradient / gradient.norm().clamp_min(1e-12) * value.norm()
    after = {
        name: protected_response(candidate, keys.to(value.device))
        for name, keys in protected_by_family.items()
    }
    return candidate, {
        "mode": "fixed_penalty",
        "strength": float(strength),
        "responses_before": before,
        "responses_after": after,
        "finite": bool(torch.isfinite(candidate).all()),
    }


def primal_dual_update(
    update: torch.Tensor,
    protected_by_family: Mapping[str, torch.Tensor],
    limits: Mapping[str, float],
    *,
    multiplier_step: float,
    penalty_growth: float,
    iterations: int = 20,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device_families = {
        name: keys.to(update.device).float() for name, keys in protected_by_family.items()
    }
    candidate, report = primal_dual_project_update(
        update,
        device_families,
        limits,
        multiplier_step=float(multiplier_step),
        penalty_growth=float(penalty_growth),
        iterations=int(iterations),
    )
    return candidate, {"mode": "primal_dual", **report}
