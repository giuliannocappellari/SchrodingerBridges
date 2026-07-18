#!/usr/bin/env python3
"""Factorized temporal residual memory and evaluation helpers."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
import time
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
    input_projection_basis: torch.Tensor | None = None
    edit_row_count: int = 0
    protect_row_count: int = 0

    @property
    def rank_bound(self) -> int:
        return int(min(self.keys.shape[0], torch.linalg.matrix_rank(self.residuals.float())))

    @property
    def storage_bytes(self) -> int:
        tensors = [self.keys, self.dual, self.residuals]
        if self.input_projection_basis is not None:
            tensors.append(self.input_projection_basis)
        return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))

    def predict(self, inputs: torch.Tensor, *, alpha: float = 1.0, top_q: int = 0) -> torch.Tensor:
        projected = inputs.float()
        if self.input_projection_basis is not None:
            basis = self.input_projection_basis.to(projected)
            projected = projected - (projected @ basis) @ basis.T
        coefficients = projected @ self.dual.T
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
            "input_projection_basis": (
                self.input_projection_basis.detach().cpu()
                if self.input_projection_basis is not None
                else torch.empty((self.keys.shape[1], 0))
            ),
            "edit_row_count": int(self.edit_row_count),
            "protect_row_count": int(self.protect_row_count),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: str | torch.device = "cuda",
    ) -> "FactorizedResidualMemory":
        basis = payload.get("input_projection_basis")
        if basis is not None and basis.numel() == 0:
            basis = None
        memory = cls(
            keys=payload["keys"].float().to(device),
            dual=payload["dual"].float().to(device),
            residuals=payload["residuals"].float().to(device),
            ridge=float(payload["ridge"]),
            input_projection_basis=(
                basis.float().to(device) if basis is not None else None
            ),
            edit_row_count=int(payload.get("edit_row_count", payload["keys"].shape[0])),
            protect_row_count=int(payload.get("protect_row_count", 0)),
        )
        if not all(
            torch.isfinite(value).all()
            for value in (memory.keys, memory.dual, memory.residuals)
        ):
            raise FloatingPointError("loaded residual memory contains non-finite values")
        return memory


def fit_factorized_residual_memory(
    keys: torch.Tensor,
    residuals: torch.Tensor,
    *,
    ridge: float,
    protect_keys: torch.Tensor | None = None,
    preservation_strength: float = 0.0,
    input_projection_basis: torch.Tensor | None = None,
) -> FactorizedResidualMemory:
    if keys.ndim != 2 or residuals.ndim != 2:
        raise ValueError("keys and residuals must be matrices")
    if keys.shape[0] != residuals.shape[0] or keys.shape[0] == 0:
        raise ValueError("keys and residuals need the same nonzero row count")
    if preservation_strength < 0:
        raise ValueError("preservation_strength must be nonnegative")
    edit_count = int(keys.shape[0])
    projected_keys = keys.float()
    basis = None
    if input_projection_basis is not None:
        basis = input_projection_basis.float().to(projected_keys)
        if basis.ndim != 2 or basis.shape[0] != projected_keys.shape[1]:
            raise ValueError("input projection basis has the wrong key width")
        projected_keys = projected_keys - (projected_keys @ basis) @ basis.T
    residuals32 = residuals.float()
    protect_count = 0
    if protect_keys is not None and float(preservation_strength) > 0:
        if protect_keys.ndim != 2 or protect_keys.shape[1] != projected_keys.shape[1]:
            raise ValueError("protect key dimensions do not match edit keys")
        protected = protect_keys.float().to(projected_keys)
        if basis is not None:
            protected = protected - (protected @ basis) @ basis.T
        protect_count = int(protected.shape[0])
        scale = math.sqrt(float(preservation_strength))
        projected_keys = torch.cat((projected_keys, protected * scale), dim=0)
        residuals32 = torch.cat(
            (
                residuals32,
                torch.zeros(
                    (protect_count, residuals32.shape[1]),
                    dtype=residuals32.dtype,
                    device=residuals32.device,
                ),
            ),
            dim=0,
        )
    gram = projected_keys @ projected_keys.T
    system = gram + torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge)
    dual = torch.linalg.solve(system, projected_keys)
    if not all(torch.isfinite(value).all() for value in (projected_keys, residuals32, dual)):
        raise FloatingPointError("non-finite factorized residual memory")
    return FactorizedResidualMemory(
        projected_keys,
        dual,
        residuals32,
        float(ridge),
        input_projection_basis=basis,
        edit_row_count=edit_count,
        protect_row_count=protect_count,
    )


def build_input_protection_basis(
    protect_keys: torch.Tensor,
    *,
    explained_variance: float = 0.95,
    maximum_rank: int = 64,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if protect_keys.ndim != 2 or protect_keys.shape[0] < 2:
        raise ValueError("protect_keys must contain at least two rows")
    if not 0 < float(explained_variance) <= 1:
        raise ValueError("explained_variance must be in (0, 1]")
    centered = protect_keys.float() - protect_keys.float().mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
    rank = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(float(explained_variance), device=cumulative.device),
        ).item()
    ) + 1
    rank = min(rank, int(maximum_rank), int(vh.shape[0]))
    basis = vh[:rank].T.contiguous()
    return basis, {
        "protected_rank": rank,
        "key_width": int(protect_keys.shape[1]),
        "captured_variance": float(cumulative[rank - 1]),
        "remaining_dimension": int(protect_keys.shape[1] - rank),
    }


def fit_residual_memory_for_requests(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    ridge: float,
    target_optimization_steps: int,
    learning_rate: float,
    partial_mask_schedule: str,
    reveal_policy: str,
    state_consistency_weight: float,
    old_target_suppression_weight: float,
    seed: int,
    cache_dir: Path,
    protect_keys: torch.Tensor | None = None,
    preservation_strength: float = 0.0,
    input_projection_basis: torch.Tensor | None = None,
) -> tuple[FactorizedResidualMemory, list[dict[str, Any]], float]:
    """Fit one deployable residual memory under a frozen state policy."""

    from scripts.mdm_memit_editor import (
        MemitConfig,
        extract_keys_and_outputs,
        optimize_target_value,
    )

    started = time.monotonic()
    config = MemitConfig(
        layers=(int(layer),),
        target_optimization_steps=int(target_optimization_steps),
        learning_rate=float(learning_rate),
        partial_mask_schedule=str(partial_mask_schedule),
        reveal_policy=str(reveal_policy),
        state_consistency_weight=float(state_consistency_weight),
        old_target_suppression_weight=float(old_target_suppression_weight),
        seed=int(seed),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    diagnostics = []
    for index, request in enumerate(requests, start=1):
        cache_path = cache_dir / f"{request['case_id']}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            target = payload["target_value"].float()
            report = payload["diagnostics"]
        else:
            target, report = optimize_target_value(model, tokenizer, request, config)
            target = target.detach().cpu()
            torch.save({"target_value": target, "diagnostics": report}, cache_path)
        targets.append(target)
        diagnostics.append({"case_id": request["case_id"], **report})
        if index % 10 == 0 or index == len(requests):
            print(
                f"TRM targets schedule={partial_mask_schedule}/{reveal_policy} "
                f"layer={layer} {index}/{len(requests)}",
                flush=True,
            )
    keys, current_outputs = extract_keys_and_outputs(
        model,
        tokenizer,
        requests,
        key_layer=int(layer),
        output_layer=int(layer),
        partial_mask_schedule=str(partial_mask_schedule),
        reveal_policy=str(reveal_policy),
        seed=int(seed),
    )
    residuals = torch.stack(targets) - current_outputs
    memory = fit_factorized_residual_memory(
        keys.to("cuda"),
        residuals.to("cuda"),
        ridge=float(ridge),
        protect_keys=protect_keys,
        preservation_strength=float(preservation_strength),
        input_projection_basis=input_projection_basis,
    )
    return memory, diagnostics, time.monotonic() - started


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


def state_bucket_from_counts(active_mask_count: int, span_length: int) -> str:
    active = int(active_mask_count)
    span = max(int(span_length), 1)
    ratio = active / span
    if ratio >= 2.0 / 3.0:
        return "early"
    if ratio >= 1.0 / 3.0:
        return "middle"
    return "late"


@contextlib.contextmanager
def install_state_bucketed_residual_memories(
    model: torch.nn.Module,
    module: torch.nn.Module,
    memories: Mapping[str, FactorizedResidualMemory],
    *,
    mask_id: int,
    alpha: float,
    top_q: int,
    shuffle_buckets: bool = False,
) -> Iterator[dict[str, Any]]:
    required = {"early", "middle", "late"}
    if set(memories) != required:
        raise ValueError("state-bucketed memory requires early/middle/late")
    state: dict[str, Any] = {
        "hook_calls": 0,
        "bucket_counts": {name: 0 for name in sorted(required)},
        "shuffle_buckets": bool(shuffle_buckets),
    }
    current_buckets: list[str] = []
    previous_max_active: int | None = None
    span_length = 1
    input_box: list[torch.Tensor] = []
    shuffle = {"early": "late", "middle": "early", "late": "middle"}

    def model_pre_hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        nonlocal previous_max_active, span_length, current_buckets
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise RuntimeError("state routing requires runtime input_ids")
        counts = (input_ids == int(mask_id)).sum(dim=1).tolist()
        maximum = max(map(int, counts), default=0)
        if previous_max_active is None or maximum > previous_max_active or (
            maximum == previous_max_active and maximum <= 1
        ):
            span_length = max(maximum, 1)
        previous_max_active = maximum
        current_buckets = [state_bucket_from_counts(int(value), span_length) for value in counts]

    def module_pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        input_box.append(inputs[0])

    def module_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if not input_box or not current_buckets:
            raise RuntimeError("state-routed residual hook lacks runtime state")
        inputs = input_box.pop(0)
        hidden = output_hidden(output)
        delta = torch.zeros_like(hidden, dtype=torch.float32)
        for row_index, bucket in enumerate(current_buckets):
            routed = shuffle[bucket] if shuffle_buckets else bucket
            delta[row_index] = memories[routed].predict(
                inputs[row_index], alpha=alpha, top_q=top_q
            )
            state["bucket_counts"][bucket] += 1
        state["hook_calls"] += 1
        return replace_output_hidden(output, hidden + delta.to(dtype=hidden.dtype))

    model_handle = model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
    before = module.register_forward_pre_hook(module_pre_hook)
    after = module.register_forward_hook(module_hook)
    try:
        yield state
    finally:
        model_handle.remove()
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
