#!/usr/bin/env python3
"""Low-rank residual-memory and state-routing primitives for TRM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


StateBucket = Literal["early", "middle", "late"]


def state_bucket(*, step_index: int, total_steps: int, active_mask_count: int, span_length: int) -> StateBucket:
    if total_steps <= 0 or span_length <= 0:
        raise ValueError("total_steps and span_length must be positive")
    if not 0 <= step_index < total_steps:
        raise ValueError("step_index must be inside the denoising schedule")
    if not 0 <= active_mask_count <= span_length:
        raise ValueError("active_mask_count must be inside the answer span")
    progress = max(step_index / max(total_steps - 1, 1), 1.0 - active_mask_count / span_length)
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


@dataclass(frozen=True)
class ResidualMemory:
    weight: torch.Tensor
    ridge: float
    preservation_strength: float
    state_bucket_name: str

    def predict(self, keys: torch.Tensor, *, alpha: float = 1.0, top_q: int = 0) -> torch.Tensor:
        if keys.shape[-1] != self.weight.shape[0]:
            raise ValueError("key dimension does not match residual memory")
        delta = keys.float() @ self.weight.float()
        if top_q > 0 and top_q < delta.shape[-1]:
            indices = delta.abs().topk(top_q, dim=-1).indices
            sparse = torch.zeros_like(delta)
            sparse.scatter_(-1, indices, delta.gather(-1, indices))
            delta = sparse
        return delta * float(alpha)


def fit_residual_memory(
    edit_keys: torch.Tensor,
    target_deltas: torch.Tensor,
    *,
    ridge: float,
    protect_keys: torch.Tensor | None = None,
    preservation_strength: float = 0.0,
    state_bucket_name: str = "shared",
) -> ResidualMemory:
    if edit_keys.ndim != 2 or target_deltas.ndim != 2:
        raise ValueError("edit_keys and target_deltas must be matrices")
    if edit_keys.shape[0] != target_deltas.shape[0] or edit_keys.shape[0] == 0:
        raise ValueError("edit keys and target deltas must have the same nonzero row count")
    if ridge <= 0 or preservation_strength < 0:
        raise ValueError("ridge must be positive and preservation strength nonnegative")
    x = edit_keys.float()
    y = target_deltas.float()
    if protect_keys is not None and preservation_strength > 0:
        if protect_keys.ndim != 2 or protect_keys.shape[1] != x.shape[1]:
            raise ValueError("protect key dimensions do not match edit keys")
        scale = preservation_strength**0.5
        x = torch.cat((x, protect_keys.float() * scale), dim=0)
        y = torch.cat((y, torch.zeros((len(protect_keys), y.shape[1]), dtype=y.dtype, device=y.device)), dim=0)
    gram = x @ x.T
    system = gram + torch.eye(len(x), dtype=x.dtype, device=x.device) * float(ridge)
    dual_targets = torch.linalg.solve(system, y)
    weight = x.T @ dual_targets
    if not torch.isfinite(weight).all():
        raise FloatingPointError("residual-memory solve produced non-finite parameters")
    return ResidualMemory(
        weight=weight,
        ridge=float(ridge),
        preservation_strength=float(preservation_strength),
        state_bucket_name=str(state_bucket_name),
    )


def fit_state_bucketed_memories(
    banks: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
    *,
    ridge: float,
    preservation_strength: float,
) -> dict[str, ResidualMemory]:
    required = {"early", "middle", "late"}
    if set(banks) != required:
        raise ValueError(f"state banks must be exactly {sorted(required)}")
    return {
        bucket: fit_residual_memory(
            edit_keys,
            deltas,
            ridge=ridge,
            protect_keys=protect_keys,
            preservation_strength=preservation_strength,
            state_bucket_name=bucket,
        )
        for bucket, (edit_keys, deltas, protect_keys) in banks.items()
    }
