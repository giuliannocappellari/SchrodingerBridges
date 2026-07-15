#!/usr/bin/env python3
"""Deployable frozen-text features and model for the T1 edit-intent gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn


WORD_RE = re.compile(r"[a-z0-9]+")
FEATURE_DIM = 768
FEATURE_SCHEMA_VERSION = 1
FORBIDDEN_RUNTIME_FIELDS = {
    "prompt_type",
    "negative_type",
    "split_role",
    "case_id",
    "raw_bridge_scores",
    "mc_rollout_rewards",
    "future_success",
    "final_outcome",
}


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def char_ngrams(text: str, width: int = 3) -> list[str]:
    compact = f" {normalize(text)} "
    return [compact[index : index + width] for index in range(max(0, len(compact) - width + 1))]


def hash_index(namespace: str, value: str, width: int) -> int:
    digest = hashlib.blake2b(f"{namespace}:{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % width


def _add_hashed(vector: torch.Tensor, values: Iterable[str], *, offset: int, width: int, namespace: str) -> None:
    for value in values:
        index = offset + hash_index(namespace, value, width)
        vector[index] += 1.0


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def featurize(prompt: str, subject: str, relation_template: str, relation_id: str) -> torch.Tensor:
    """Create runtime-available features without bucket or teacher information."""

    vector = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    prompt_norm = normalize(prompt)
    subject_norm = normalize(subject)
    relation_text = str(relation_template).replace("{}", " ").replace("{subject}", " ")
    relation_norm = normalize(relation_text)
    prompt_tokens = tokens(prompt_norm)
    relation_tokens = tokens(relation_norm)
    subject_tokens = tokens(subject_norm)

    _add_hashed(vector, prompt_tokens, offset=0, width=192, namespace="prompt_word")
    _add_hashed(vector, relation_tokens, offset=192, width=128, namespace="relation_word")
    _add_hashed(vector, char_ngrams(prompt_norm), offset=320, width=192, namespace="prompt_char3")
    _add_hashed(vector, char_ngrams(relation_norm), offset=512, width=128, namespace="relation_char3")
    interactions = [f"{left}|{right}" for left in prompt_tokens for right in relation_tokens]
    _add_hashed(vector, interactions, offset=640, width=64, namespace="interaction")
    _add_hashed(vector, [str(relation_id)], offset=704, width=32, namespace="relation_id")

    vector[736] = float(bool(subject_norm) and subject_norm in prompt_norm)
    vector[737] = jaccard(prompt_tokens, relation_tokens)
    vector[738] = jaccard(char_ngrams(prompt_norm), char_ngrams(relation_norm))
    vector[739] = min(len(prompt_tokens), 64) / 64.0
    vector[740] = min(len(relation_tokens), 32) / 32.0
    vector[741] = min(len(subject_tokens), 16) / 16.0
    vector[742] = float(prompt_norm.endswith(relation_norm) and bool(relation_norm))
    norm = torch.linalg.vector_norm(vector)
    if norm > 0:
        vector[:736] /= norm
    return vector


class GateMLP(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            self.network = nn.Linear(FEATURE_DIM, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(FEATURE_DIM, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def checkpoint_schema(hidden_dim: int, threshold: float, temperature: float = 1.0) -> dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "runtime_inputs": ["prompt", "subject", "relation_template", "relation_id"],
        "forbidden_runtime_inputs": sorted(FORBIDDEN_RUNTIME_FIELDS),
        "hidden_dim": int(hidden_dim),
        "threshold": float(threshold),
        "temperature": float(temperature),
    }


def save_checkpoint(path: str | Path, model: GateMLP, schema: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "schema": dict(schema)}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[GateMLP, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    schema = dict(payload["schema"])
    if int(schema["feature_schema_version"]) != FEATURE_SCHEMA_VERSION:
        raise RuntimeError("Unsupported T1 gate feature schema")
    if int(schema["feature_dim"]) != FEATURE_DIM:
        raise RuntimeError("T1 gate feature dimension mismatch")
    if set(schema.get("runtime_inputs", [])) & FORBIDDEN_RUNTIME_FIELDS:
        raise RuntimeError("T1 gate checkpoint contains forbidden runtime inputs")
    model = GateMLP(int(schema["hidden_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, schema


@torch.no_grad()
def predict_probability(
    model: GateMLP,
    *,
    prompt: str,
    subject: str,
    relation_template: str,
    relation_id: str,
    temperature: float = 1.0,
) -> float:
    device = next(model.parameters()).device
    feature = featurize(prompt, subject, relation_template, relation_id).to(device)
    logit = model(feature.unsqueeze(0))[0] / max(float(temperature), 1e-6)
    return float(torch.sigmoid(logit).item())
