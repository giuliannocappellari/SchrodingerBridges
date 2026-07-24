"""Frozen-base additive update bank with deployable token-only routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _contains_subsequence(values: Sequence[int], pattern: Sequence[int]) -> bool:
    if not pattern or len(pattern) > len(values):
        return False
    width = len(pattern)
    return any(list(values[index : index + width]) == list(pattern) for index in range(len(values) - width + 1))


@dataclass
class DeltaBranch:
    deltas: dict[int, Any]
    subject_token_sequences: tuple[tuple[int, ...], ...]
    relation_token_sets: tuple[frozenset[int], ...]
    block_index: int


class DeltaBranchBank:
    """Keep the base weights frozen and inject additive updates through hooks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layers: Sequence[int],
        *,
        route_mode: str,
        relation_overlap_threshold: float = 0.20,
    ) -> None:
        import torch

        from scripts.mdm_memit_editor import get_module, resolved_key_module_name

        if route_mode not in {"always", "subject", "subject_relation"}:
            raise ValueError(f"Unsupported route mode: {route_mode}")
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.layers = tuple(map(int, layers))
        self.route_mode = route_mode
        self.relation_overlap_threshold = float(relation_overlap_threshold)
        self.branches: list[DeltaBranch] = []
        self.current_gates = None
        self.handles = []
        self.root_handle = model.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True)
        for layer in self.layers:
            module = get_module(model, resolved_key_module_name(model, layer))

            def make_hook(layer_index: int):
                def hook(_module, inputs, output):
                    if not self.branches:
                        return output
                    gates = self.current_gates
                    if gates is None or gates.shape[0] != inputs[0].shape[0]:
                        raise RuntimeError("Delta-bank gates do not align with the forward batch")
                    result = output
                    values = inputs[0]
                    for index, branch in enumerate(self.branches):
                        delta = branch.deltas.get(layer_index)
                        if delta is None:
                            continue
                        residual = torch.nn.functional.linear(values.float(), delta.float())
                        shape = [gates.shape[0]] + [1] * (residual.ndim - 1)
                        residual = residual * gates[:, index].reshape(shape)
                        result = result + residual.to(result.dtype)
                    return result
                return hook

            self.handles.append(module.register_forward_hook(make_hook(layer)))

    def _root_pre_hook(self, _module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise RuntimeError("Delta bank requires input_ids for deployable routing")
        batch = input_ids.detach().cpu().tolist()
        gates = self.torch.zeros((len(batch), len(self.branches)), dtype=self.torch.float32, device=input_ids.device)
        for row_index, tokens in enumerate(batch):
            token_set = set(map(int, tokens))
            for branch_index, branch in enumerate(self.branches):
                if self.route_mode == "always":
                    active = True
                else:
                    subject = any(_contains_subsequence(tokens, sequence) for sequence in branch.subject_token_sequences)
                    if self.route_mode == "subject":
                        active = subject
                    else:
                        relation = any(
                            len(token_set & set(candidate)) / max(len(candidate), 1)
                            >= self.relation_overlap_threshold
                            for candidate in branch.relation_token_sets
                        )
                        active = subject and relation
                gates[row_index, branch_index] = float(active)
        self.current_gates = gates

    def _metadata(self, requests: Sequence[Mapping[str, Any]]):
        subjects = []
        relations = []
        for row in requests:
            subject = tuple(map(int, self.tokenizer(str(row["subject"]), add_special_tokens=False)["input_ids"]))
            relation_text = str(row.get("rewrite_template") or row.get("rewrite_prompt") or "").replace("{}", " ")
            relation = frozenset(map(int, self.tokenizer(relation_text, add_special_tokens=False)["input_ids"]))
            if subject:
                subjects.append(subject)
            if relation:
                relations.append(relation)
        return tuple(dict.fromkeys(subjects)), tuple(dict.fromkeys(relations))

    def add_branch(
        self,
        deltas: Mapping[int, Any],
        requests: Sequence[Mapping[str, Any]],
        *,
        block_index: int,
    ) -> None:
        subjects, relations = self._metadata(requests)
        copied = {int(layer): value.detach().float().clone() for layer, value in deltas.items()}
        self.branches.append(DeltaBranch(copied, subjects, relations, int(block_index)))

    def merge_all(self, *, rank: int, weights: Sequence[float] | None = None) -> dict[str, float]:
        if not self.branches:
            return {"num_merged": 0, "rank": int(rank)}
        torch = self.torch
        raw_weights = list(weights or [1.0] * len(self.branches))
        if len(raw_weights) != len(self.branches):
            raise ValueError("Merge weights must align with branches")
        total = sum(raw_weights)
        normalized = [value / total for value in raw_weights]
        merged = {}
        energy = []
        for layer in self.layers:
            value = sum(
                weight * branch.deltas[layer]
                for weight, branch in zip(normalized, self.branches)
                if layer in branch.deltas
            )
            value = value.float()
            effective = min(int(rank), min(value.shape))
            if min(value.shape) > 1024:
                q = min(effective + 4, min(value.shape))
                u, singular, v = torch.svd_lowrank(value, q=q, niter=2)
                compressed = (u[:, :effective] * singular[:effective]) @ v[:, :effective].T
                denominator = value.square().sum()
            else:
                u, singular, vh = torch.linalg.svd(value, full_matrices=False)
                compressed = (u[:, :effective] * singular[:effective]) @ vh[:effective]
                denominator = singular.square().sum()
            merged[layer] = compressed
            energy.append(float(singular[:effective].square().sum() / denominator.clamp_min(1e-12)))
        subjects = tuple(dict.fromkeys(item for branch in self.branches for item in branch.subject_token_sequences))
        relations = tuple(dict.fromkeys(item for branch in self.branches for item in branch.relation_token_sets))
        count = len(self.branches)
        self.branches = [DeltaBranch(merged, subjects, relations, max(branch.block_index for branch in self.branches))]
        return {"num_merged": count, "rank": int(rank), "mean_explained_energy": sum(energy) / len(energy)}

    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for branch in self.branches
            for tensor in branch.deltas.values()
        )

    def activation_summary(self) -> dict[str, Any]:
        return {
            "route_mode": self.route_mode,
            "num_branches": len(self.branches),
            "runtime_feature_schema": ["input_token_ids", "edit_subject_token_ids", "edit_relation_template_token_ids"],
            "forbidden_runtime_features_used": False,
        }

    def close(self) -> None:
        self.root_handle.remove()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
