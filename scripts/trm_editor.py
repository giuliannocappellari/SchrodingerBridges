#!/usr/bin/env python3
"""Factorized temporal residual memory and evaluation helpers."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch

from scripts.mdm_memit_editor import output_hidden, replace_output_hidden
from scripts.run_dnpe_editor import aggregate


@dataclass
class FactorizedResidualMemory:
    keys: torch.Tensor
    dual: torch.Tensor
    residuals: torch.Tensor
    ridge: float

    @property
    def rank_bound(self) -> int:
        return int(min(self.keys.shape[0], torch.linalg.matrix_rank(self.residuals.float())))

    @property
    def storage_bytes(self) -> int:
        return int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (self.keys, self.dual, self.residuals)
            )
        )

    def predict(self, inputs: torch.Tensor, *, alpha: float = 1.0, top_q: int = 0) -> torch.Tensor:
        coefficients = inputs.float() @ self.dual.T
        delta = coefficients @ self.residuals
        delta = delta * float(alpha)
        if int(top_q) > 0 and int(top_q) < delta.shape[-1]:
            indices = delta.abs().topk(int(top_q), dim=-1).indices
            sparse = torch.zeros_like(delta)
            sparse.scatter_(-1, indices, delta.gather(-1, indices))
            delta = sparse
        return delta

    def cpu_payload(self) -> dict[str, Any]:
        return {
            "keys": self.keys.detach().cpu(),
            "dual": self.dual.detach().cpu(),
            "residuals": self.residuals.detach().cpu(),
            "ridge": float(self.ridge),
        }


def fit_factorized_residual_memory(
    keys: torch.Tensor,
    residuals: torch.Tensor,
    *,
    ridge: float,
) -> FactorizedResidualMemory:
    if keys.ndim != 2 or residuals.ndim != 2:
        raise ValueError("keys and residuals must be matrices")
    if keys.shape[0] != residuals.shape[0] or keys.shape[0] == 0:
        raise ValueError("keys and residuals need the same nonzero row count")
    keys32 = keys.float()
    residuals32 = residuals.float()
    gram = keys32 @ keys32.T
    system = gram + torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge)
    dual = torch.linalg.solve(system, keys32)
    if not all(torch.isfinite(value).all() for value in (keys32, residuals32, dual)):
        raise FloatingPointError("non-finite factorized residual memory")
    return FactorizedResidualMemory(keys32, dual, residuals32, float(ridge))


@contextlib.contextmanager
def install_factorized_residual_memory(
    module: torch.nn.Module,
    memory: FactorizedResidualMemory,
    *,
    alpha: float,
    top_q: int,
) -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {
        "hook_calls": 0,
        "token_vectors_seen": 0,
        "nonzero_delta_coordinates": 0,
        "total_delta_coordinates": 0,
        "delta_norm_sum": 0.0,
    }
    input_box: list[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        input_box.append(inputs[0])

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if not input_box:
            raise RuntimeError("residual-memory pre-hook did not capture module input")
        inputs = input_box.pop(0)
        hidden = output_hidden(output)
        delta = memory.predict(inputs, alpha=alpha, top_q=top_q).to(dtype=hidden.dtype)
        state["hook_calls"] += 1
        state["token_vectors_seen"] += int(delta.numel() // delta.shape[-1])
        state["nonzero_delta_coordinates"] += int((delta != 0).sum())
        state["total_delta_coordinates"] += int(delta.numel())
        state["delta_norm_sum"] += float(delta.float().norm())
        return replace_output_hidden(output, hidden + delta)

    before = module.register_forward_pre_hook(pre_hook)
    after = module.register_forward_hook(hook)
    try:
        yield state
    finally:
        before.remove()
        after.remove()


def harmonic_mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def summarize_editor_rows(
    base_rows: Sequence[Mapping[str, Any]],
    edited_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base = aggregate(base_rows)
    edited = aggregate(edited_rows)
    rewrite = float(edited.get("rewrite", {}).get("expected_exact", 0.0))
    paraphrase = float(
        edited.get("declarative_paraphrase", {}).get("expected_exact", 0.0)
    )
    locality_buckets = ("near_locality", "far_locality")
    base_locality_values = [
        float(base.get(bucket, {}).get("expected_exact", 0.0)) for bucket in locality_buckets
    ]
    edited_locality_values = [
        float(edited.get(bucket, {}).get("expected_exact", 0.0)) for bucket in locality_buckets
    ]
    base_locality = sum(base_locality_values) / len(base_locality_values)
    edited_locality = sum(edited_locality_values) / len(edited_locality_values)
    clipped_self_normalized = min(edited_locality / max(base_locality, 1e-8), 1.0)
    same_subject = float(
        edited.get("same_subject", {}).get("target_new_tfpr_or_exact", 0.0)
    )
    generation = float(edited.get("generation", {}).get("target_new_tfpr_or_exact", 0.0))
    malformed = max(
        (float(values["malformed_rate"]) for values in edited.values()), default=0.0
    )
    selection = harmonic_mean((rewrite, paraphrase, clipped_self_normalized))
    stress_aware = harmonic_mean(
        (rewrite, paraphrase, clipped_self_normalized, max(1.0 - same_subject, 0.0))
    )
    base_agreements = [
        bool(row.get("base_agreement"))
        for row in edited_rows
        if row.get("bucket") in {"near_locality", "far_locality", "generation", "attribute"}
    ]
    return {
        "base_summary": base,
        "edited_summary": edited,
        "base_rewrite_exact": float(base.get("rewrite", {}).get("expected_exact", 0.0)),
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "base_locality_exact": base_locality,
        "locality_exact": edited_locality,
        "clipped_self_normalized_locality": clipped_self_normalized,
        "same_subject_tfpr": same_subject,
        "near_tfpr": float(
            edited.get("near_locality", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "far_tfpr": float(
            edited.get("far_locality", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "generation_tfpr": generation,
        "attribute_tfpr": float(
            edited.get("attribute", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "malformed_rate": malformed,
        "selection_score": selection,
        "stress_aware_aggregate": stress_aware,
        "utility_base_agreement": (
            sum(base_agreements) / len(base_agreements) if base_agreements else 1.0
        ),
        "all_metrics_finite": all(
            math.isfinite(value)
            for value in (
                rewrite,
                paraphrase,
                base_locality,
                edited_locality,
                same_subject,
                generation,
                malformed,
                selection,
                stress_aware,
            )
        ),
    }
