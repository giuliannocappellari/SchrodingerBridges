#!/usr/bin/env python3
"""Train and replay representation-aware Direction 3 controllers.

This script consumes frozen deployable feature tensors and cached teacher rows.
It never imports or loads LLaDA. It writes Stage 1B.4/1B.5 train, replay,
leakage, and shortcut-audit reports for the bounded representation upgrade.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, auc_score, git_commit, mean, now_utc, read_json, read_jsonl, repo_path, spearman, write_csv, write_json
from scripts.audit_d3_deployable_feature_cache import build_audit as build_feature_cache_audit


DEFAULT_FEATURE_CACHE = D3_ROOT / "deployable_feature_cache_train100_val50_v1"
DEFAULT_TEACHER_CACHE = D3_ROOT / "teacher_cache_train100_val50_v1"
DEFAULT_LOCAL_AUDIT = D3_ROOT / "deployable_feature_cache_train100_val50_v1_local_audit"
DEFAULT_TRAIN_DIR = D3_ROOT / "offline_train_repr_value_gate_train100_val50_v3"
DEFAULT_REPLAY_DIR = D3_ROOT / "offline_replay_repr_train100_val50_v3"
DEFAULT_LEAKAGE_DIR = D3_ROOT / "stage1b_feature_leakage_audit_v3"
DEFAULT_SHORTCUT_DIR = D3_ROOT / "representation_shortcut_audit_v3"

POSITIVE_PROMPT_TYPES = {"rewrite", "declarative_paraphrase"}
NEGATIVE_WEIGHTS = {
    "same_subject_different_relation": 3.0,
    "same_subject_template": 3.0,
    "near_locality": 2.0,
    "far_locality": 2.0,
    "generation": 1.5,
    "attribute": 1.5,
    "unrelated": 1.0,
}
HARD_CRITERIA = {
    "macro_groupwise_spearman": 0.40,
    "ndcg_at_8": 0.70,
    "pairwise_ranking_accuracy": 0.70,
    "teacher_top1_agreement": 0.40,
    "teacher_top3_overlap": 0.65,
    "target_top3_improvement_over_base": 0.15,
    "gate_roc_auc": 0.85,
    "rewrite_activation": 0.90,
    "declarative_paraphrase_activation": 0.85,
    "same_subject_activation_max": 0.05,
    "near_locality_activation_max": 0.02,
    "far_locality_activation_max": 0.02,
    "negative_guidance_ratio": 0.15,
}


def state_key(row: Mapping[str, Any]) -> str:
    import hashlib

    payload = {
        "edit_id": row.get("edit_id") or row.get("case_id"),
        "prompt_id": row.get("prompt_id"),
        "current_state": row.get("current_state") or row.get("fake_state"),
        "step_index": row.get("step_index"),
        "selected_mask_position": row.get("selected_mask_position"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prompt_key(row: Mapping[str, Any]) -> str:
    import hashlib

    payload = {"edit_id": edit_id(row), "prompt_id": row.get("prompt_id")}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def edit_id(row: Mapping[str, Any]) -> str:
    return str(row.get("edit_id") or row.get("case_id"))


def row_split(row: Mapping[str, Any]) -> str:
    value = str(row.get("split_role") or "")
    if "val" in value:
        return "val"
    if "train" in value:
        return "train"
    return value or "unknown"


def prompt_label(row: Mapping[str, Any]) -> int:
    return int(str(row.get("prompt_type")) in POSITIVE_PROMPT_TYPES or int(row.get("label", 0)) == 1)


def target_positions(row: Mapping[str, Any]) -> List[int]:
    target_ids = {int(v) for v in row.get("target_token_ids") or []}
    candidates = [int(v) for v in (row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or [])]
    return [idx for idx, token_id in enumerate(candidates) if token_id in target_ids]


def score_array(row: Mapping[str, Any], keys: Sequence[str]) -> List[float]:
    for key in keys:
        values = row.get(key)
        if isinstance(values, list):
            out = [float(v) for v in values]
            if not all(math.isfinite(v) for v in out):
                raise AssertionError(f"Non-finite teacher values for {key}")
            return out
    raise KeyError(f"Missing score array for {keys}")


def zscore(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / (values.std(dim=1, keepdim=True) + eps)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else 0.0


def pairwise_accuracy(teacher: Sequence[float], pred: Sequence[float]) -> float:
    hits = 0
    total = 0
    for i in range(len(teacher)):
        for j in range(i + 1, len(teacher)):
            dt = teacher[i] - teacher[j]
            if dt == 0:
                continue
            total += 1
            hits += int(dt * (pred[i] - pred[j]) > 0)
    return hits / total if total else 0.0


def ndcg_at_k(teacher: Sequence[float], pred: Sequence[float], k: int = 8) -> float:
    gains = [max(0.0, float(v) - min(teacher)) for v in teacher]
    order = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k]
    ideal = sorted(range(len(teacher)), key=lambda i: teacher[i], reverse=True)[:k]

    def dcg(indices: Sequence[int]) -> float:
        return sum(gains[idx] / math.log2(rank + 2) for rank, idx in enumerate(indices))

    denom = dcg(ideal)
    return dcg(order) / denom if denom > 0 else 0.0


def topk_set(values: Sequence[float], k: int = 3) -> set[int]:
    return set(sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k])


class FeatureBundle:
    def __init__(self, feature_cache_dir: Path, teacher_cache_dir: Path) -> None:
        self.feature_cache_dir = feature_cache_dir
        self.teacher_cache_dir = teacher_cache_dir
        self.train_rows = read_jsonl(teacher_cache_dir / "teacher_states_train.jsonl")
        self.val_rows = read_jsonl(teacher_cache_dir / "teacher_states_val.jsonl")
        self.rows = self.train_rows + self.val_rows
        self.index_rows = read_jsonl(feature_cache_dir / "feature_index.jsonl")
        self.state_features = load_file(str(repo_path(feature_cache_dir / "state_features.safetensors")))
        self.candidate_features = load_file(str(repo_path(feature_cache_dir / "candidate_features.safetensors")))
        self.edit_features = load_file(str(repo_path(feature_cache_dir / "edit_features.safetensors")))
        self.gate_features = load_file(str(repo_path(feature_cache_dir / "gate_features.safetensors")))
        self.state_index = {
            str(row["state_key"]): int(row["row_index"])
            for row in self.index_rows
            if row.get("index_type") == "state"
        }
        self.edit_index = {
            str(row["edit_id"]): int(row["row_index"])
            for row in self.index_rows
            if row.get("index_type") == "edit"
        }
        self.gate_index = {
            str(row["prompt_key"]): int(row["row_index"])
            for row in self.index_rows
            if row.get("index_type") == "gate"
        }
        self.group_indices = [
            int(row["row_index"])
            for row in self.index_rows
            if row.get("index_type") == "candidate_group"
        ]
        if len(self.group_indices) != len(self.rows):
            raise AssertionError(f"Feature group count {len(self.group_indices)} != teacher rows {len(self.rows)}")
        self.row_indices_by_split: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.rows):
            self.row_indices_by_split[row_split(row)].append(idx)

    @property
    def state_dim(self) -> int:
        return int(self.state_features["last_layer_selected"].shape[-1] * 3)

    @property
    def candidate_dim(self) -> int:
        return int(self.candidate_features["candidate_token_embedding"].shape[-1])

    @property
    def edit_dim(self) -> int:
        return int(self.edit_features["target_new_embedding_mean"].shape[-1] * 4)

    @property
    def gate_dim(self) -> int:
        total = 0
        for key, tensor in self.gate_features.items():
            total += int(tensor.shape[-1])
        return total

    def state_tensor(self, row_indices: Sequence[int]) -> torch.Tensor:
        idx = torch.tensor([self.state_index[state_key(self.rows[i])] for i in row_indices], dtype=torch.long)
        pieces = [
            self.state_features["mid_layer_selected"][idx].float(),
            self.state_features["last_layer_selected"][idx].float(),
            self.state_features["answer_span_mean"][idx].float(),
        ]
        return torch.cat(pieces, dim=1)

    def edit_tensor(self, row_indices: Sequence[int]) -> torch.Tensor:
        idx = torch.tensor([self.edit_index[edit_id(self.rows[i])] for i in row_indices], dtype=torch.long)
        pieces = [
            self.edit_features["target_new_embedding_mean"][idx].float(),
            self.edit_features["target_true_embedding_mean"][idx].float(),
            self.edit_features["subject_embedding_mean"][idx].float(),
            self.edit_features["rewrite_relation_embedding_mean"][idx].float(),
        ]
        return torch.cat(pieces, dim=1)

    def gate_tensor(self, row_indices: Sequence[int], relation_shuffle: bool = False) -> torch.Tensor:
        gate_idx = [self.gate_index[prompt_key(self.rows[i])] for i in row_indices]
        idx = torch.tensor(gate_idx, dtype=torch.long)
        keys = sorted(self.gate_features)
        pieces = [self.gate_features[key][idx].float() for key in keys]
        out = torch.cat(pieces, dim=1)
        if relation_shuffle and len(out) > 1:
            out = out[torch.randperm(len(out))]
        return out

    def candidate_tensor(self, row_indices: Sequence[int], candidate_shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.tensor(row_indices, dtype=torch.long)
        emb = self.candidate_features["candidate_token_embedding"][idx].float()
        scalars = torch.stack(
            [
                self.candidate_features["base_logits"][idx].float(),
                self.candidate_features["base_probabilities"][idx].float(),
                self.candidate_features["candidate_rank"][idx].float() / 7.0,
                self.candidate_features["candidate_is_target_new"][idx].float(),
                self.candidate_features["candidate_is_target_true"][idx].float(),
            ],
            dim=-1,
        )
        if candidate_shuffle and emb.shape[1] > 1:
            order = torch.stack([torch.randperm(emb.shape[1]) for _ in range(emb.shape[0])])
            emb = emb[torch.arange(emb.shape[0]).unsqueeze(1), order]
        return emb, scalars

    def teacher_scores(self, row_indices: Sequence[int]) -> torch.Tensor:
        return torch.tensor([score_array(self.rows[i], ["raw_bridge_scores_top_k", "raw_bridge_scores"]) for i in row_indices], dtype=torch.float32)

    def base_scores(self, row_indices: Sequence[int]) -> torch.Tensor:
        return torch.tensor([score_array(self.rows[i], ["base_logits_top_k", "base_logits"]) for i in row_indices], dtype=torch.float32)

    def labels(self, row_indices: Sequence[int]) -> torch.Tensor:
        return torch.tensor([prompt_label(self.rows[i]) for i in row_indices], dtype=torch.float32)

    def sample_weights(self, row_indices: Sequence[int]) -> torch.Tensor:
        weights = []
        for i in row_indices:
            prompt_type = str(self.rows[i].get("prompt_type"))
            weights.append(1.0 if prompt_label(self.rows[i]) else NEGATIVE_WEIGHTS.get(prompt_type, 1.0))
        return torch.tensor(weights, dtype=torch.float32)


class ValueReprModel(nn.Module):
    def __init__(self, state_dim: int, candidate_dim: int, edit_dim: int, proj_dim: int = 128, hidden_dim: int = 256, target_only: bool = False, use_target_indicator: bool = True) -> None:
        super().__init__()
        self.target_only = target_only
        self.use_target_indicator = use_target_indicator
        if target_only:
            self.target_head = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
            return
        self.state_proj = nn.Linear(state_dim, proj_dim)
        self.candidate_proj = nn.Linear(candidate_dim, proj_dim)
        self.edit_proj = nn.Linear(edit_dim, proj_dim)
        scalar_dim = 5 if use_target_indicator else 3
        in_dim = proj_dim * 7 + scalar_dim
        self.head = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, state: torch.Tensor, candidate: torch.Tensor, edit: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if self.target_only:
            return self.target_head(scalars[..., 3:5]).squeeze(-1)
        if not self.use_target_indicator:
            scalars = scalars[..., :3]
        bsz, width, _ = candidate.shape
        zs = self.state_proj(state).unsqueeze(1).expand(-1, width, -1)
        zc = self.candidate_proj(candidate)
        ze = self.edit_proj(edit).unsqueeze(1).expand(-1, width, -1)
        features = torch.cat([zs, zc, ze, zs * zc, zs * ze, zc * ze, torch.abs(zs - zc), scalars], dim=-1)
        return self.head(features).squeeze(-1)


class GateReprModel(nn.Module):
    def __init__(self, gate_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(gate_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(proj_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, gate_features: torch.Tensor) -> torch.Tensor:
        return self.net(gate_features).squeeze(-1)


def batches(indices: Sequence[int], batch_size: int, shuffle: bool = True) -> Iterable[List[int]]:
    items = list(indices)
    if shuffle:
        random.shuffle(items)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def train_value_model(
    bundle: FeatureBundle,
    indices: Sequence[int],
    variant: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    *,
    proj_dim: int,
    hidden_dim: int,
    teacher_temperature: float,
    negative_identity_weight: float,
    target_loss_weight: float,
) -> ValueReprModel:
    model = ValueReprModel(
        bundle.state_dim,
        bundle.candidate_dim,
        bundle.edit_dim,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        target_only=variant == "d3_target_indicator_only",
        use_target_indicator=variant != "d3_value_repr_no_target_indicator",
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for batch in batches(indices, batch_size):
            state = bundle.state_tensor(batch).to(device)
            edit = bundle.edit_tensor(batch).to(device)
            cand, scalars = bundle.candidate_tensor(batch)
            cand = cand.to(device)
            scalars = scalars.to(device)
            teacher = bundle.teacher_scores(batch).to(device)
            labels = bundle.labels(batch).to(device)
            pred = model(state, cand, edit, scalars)
            p_teacher = F.softmax(zscore(teacher) / teacher_temperature, dim=1)
            log_p_student = F.log_softmax(pred, dim=1)
            distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
            diff_t = teacher.unsqueeze(2) - teacher.unsqueeze(1)
            diff_p = pred.unsqueeze(2) - pred.unsqueeze(1)
            rank_mask = (diff_t.abs() > 1e-7).float()
            rank_loss = (F.softplus(-torch.sign(diff_t) * diff_p) * rank_mask).sum() / rank_mask.sum().clamp_min(1.0)
            negative_mask = (1.0 - labels).unsqueeze(1)
            identity = ((pred**2) * negative_mask).sum() / negative_mask.sum().clamp_min(1.0)
            cost = pred.pow(2).mean()
            target_indicator = scalars[..., 3]
            target_loss = -(F.log_softmax(pred, dim=1) * target_indicator).sum(dim=1)
            target_loss = (target_loss * labels).sum() / labels.sum().clamp_min(1.0)
            loss = distill + 0.5 * rank_loss + negative_identity_weight * identity + 0.01 * cost + target_loss_weight * target_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu()


def train_gate_model(bundle: FeatureBundle, indices: Sequence[int], epochs: int, batch_size: int, lr: float, device: torch.device) -> GateReprModel:
    model = GateReprModel(bundle.gate_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for batch in batches(indices, batch_size):
            features = bundle.gate_tensor(batch).to(device)
            labels = bundle.labels(batch).to(device)
            weights = bundle.sample_weights(batch).to(device)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu()


def predict_value(bundle: FeatureBundle, model: ValueReprModel, indices: Sequence[int], batch_size: int, *, state_shuffle: bool = False, candidate_shuffle: bool = False) -> torch.Tensor:
    model.eval()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches(indices, batch_size, shuffle=False):
            state = bundle.state_tensor(batch)
            if state_shuffle and len(state) > 1:
                state = state[torch.randperm(len(state))]
            edit = bundle.edit_tensor(batch)
            cand, scalars = bundle.candidate_tensor(batch, candidate_shuffle=candidate_shuffle)
            preds.append(model(state, cand, edit, scalars))
    return torch.cat(preds, dim=0)


def predict_gate(bundle: FeatureBundle, model: GateReprModel, indices: Sequence[int], batch_size: int, *, relation_shuffle: bool = False) -> torch.Tensor:
    model.eval()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches(indices, batch_size, shuffle=False):
            preds.append(torch.sigmoid(model(bundle.gate_tensor(batch, relation_shuffle=relation_shuffle))))
    return torch.cat(preds, dim=0)


def value_metrics(bundle: FeatureBundle, indices: Sequence[int], preds: torch.Tensor, label: str, split: str) -> Dict[str, Any]:
    rows = [bundle.rows[i] for i in indices]
    teacher = bundle.teacher_scores(indices)
    base = bundle.base_scores(indices)
    spearmans = []
    kendalls = []
    pairwise = []
    ndcgs = []
    top1 = []
    top3_overlap = []
    target_top3 = []
    base_target_top3 = []
    for n, row in enumerate(rows):
        t = teacher[n].tolist()
        p = preds[n].tolist()
        b = base[n].tolist()
        spearmans.append(spearman(t, p))
        kendalls.append(kendall_tau(t, p))
        pairwise.append(pairwise_accuracy(t, p))
        ndcgs.append(ndcg_at_k(t, p, 8))
        top1.append(float(max(range(len(t)), key=lambda i: t[i]) == max(range(len(p)), key=lambda i: p[i])))
        top3_overlap.append(len(topk_set(t, 3) & topk_set(p, 3)) / 3.0)
        positions = set(target_positions(row))
        pred_top3 = topk_set(p, 3)
        base_top3 = topk_set(b, 3)
        target_top3.append(float(bool(positions & pred_top3)))
        base_target_top3.append(float(bool(positions & base_top3)))
    return {
        "controller": label,
        "split": split,
        "num_rows": len(indices),
        "macro_groupwise_spearman": mean(spearmans),
        "median_groupwise_spearman": sorted(spearmans)[len(spearmans) // 2] if spearmans else 0.0,
        "kendall_tau": mean(kendalls),
        "pairwise_ranking_accuracy": mean(pairwise),
        "ndcg_at_8": mean(ndcgs),
        "teacher_top1_agreement": mean(top1),
        "teacher_top3_overlap": mean(top3_overlap),
        "target_top3_rate": mean(target_top3),
        "base_target_top3_rate": mean(base_target_top3),
        "target_top3_improvement_over_base": mean(target_top3) - mean(base_target_top3),
    }


def threshold_sweep(bundle: FeatureBundle, indices: Sequence[int], gate_scores: torch.Tensor, controller: str) -> List[Dict[str, Any]]:
    labels = [prompt_label(bundle.rows[i]) for i in indices]
    scores = [float(v) for v in gate_scores.tolist()]
    rows: List[Dict[str, Any]] = []
    prompt_types = sorted({str(bundle.rows[i].get("prompt_type")) for i in indices})
    for threshold in [i / 100 for i in range(0, 101, 1)]:
        row: Dict[str, Any] = {
            "controller": controller,
            "threshold": threshold,
            "roc_auc": auc_score(labels, scores),
            "pr_auc": pr_auc(labels, scores),
        }
        for prompt_type in prompt_types:
            vals = [float(scores[n] >= threshold) for n, i in enumerate(indices) if str(bundle.rows[i].get("prompt_type")) == prompt_type]
            row[f"{prompt_type}_activation"] = mean(vals)
        rows.append(row)
    return rows


def select_gate_threshold(rows: Sequence[Mapping[str, Any]]) -> Tuple[float, bool]:
    feasible = []
    for row in rows:
        ok = (
            float(row.get("roc_auc", 0.0)) >= HARD_CRITERIA["gate_roc_auc"]
            and float(row.get("rewrite_activation", 0.0)) >= HARD_CRITERIA["rewrite_activation"]
            and float(row.get("declarative_paraphrase_activation", 0.0)) >= HARD_CRITERIA["declarative_paraphrase_activation"]
            and float(row.get("same_subject_different_relation_activation", 0.0)) <= HARD_CRITERIA["same_subject_activation_max"]
            and float(row.get("near_locality_activation", 0.0)) <= HARD_CRITERIA["near_locality_activation_max"]
            and float(row.get("far_locality_activation", 0.0)) <= HARD_CRITERIA["far_locality_activation_max"]
        )
        if ok:
            feasible.append(row)
    if feasible:
        best = max(feasible, key=lambda r: float(r.get("pr_auc", 0.0)))
        return float(best["threshold"]), True
    best = max(rows, key=lambda r: float(r.get("roc_auc", 0.0)) + float(r.get("pr_auc", 0.0)))
    return float(best["threshold"]), False


def negative_guidance_rows(bundle: FeatureBundle, indices: Sequence[int], value_preds: torch.Tensor, gate_scores: torch.Tensor, controller: str) -> List[Dict[str, Any]]:
    by_prompt: Dict[str, List[float]] = defaultdict(list)
    positive_abs: List[float] = []
    negative_abs: List[float] = []
    same_subject_advantage: List[float] = []
    base = bundle.base_scores(indices)
    for n, i in enumerate(indices):
        row = bundle.rows[i]
        positions = target_positions(row)
        target_pred = max([float(value_preds[n, pos]) for pos in positions], default=float(value_preds[n].max()))
        target_base = max([float(base[n, pos]) for pos in positions], default=float(base[n].max()))
        guidance = float(gate_scores[n]) * target_pred
        prompt_type = str(row.get("prompt_type"))
        by_prompt[prompt_type].append(abs(guidance))
        if prompt_label(row):
            positive_abs.append(abs(guidance))
        else:
            negative_abs.append(abs(guidance))
            if prompt_type in {"same_subject_different_relation", "same_subject_template"}:
                same_subject_advantage.append(target_pred - target_base)
    pos_mean = mean(positive_abs)
    rows = []
    for prompt_type, values in sorted(by_prompt.items()):
        rows.append(
            {
                "controller": controller,
                "prompt_type": prompt_type,
                "mean_abs_guidance": mean(values),
                "ratio_to_positive": (mean(values) / pos_mean) if pos_mean else 0.0,
            }
        )
    rows.append(
        {
            "controller": controller,
            "prompt_type": "ALL_NEGATIVE",
            "mean_abs_guidance": mean(negative_abs),
            "ratio_to_positive": (mean(negative_abs) / pos_mean) if pos_mean else 0.0,
            "same_subject_target_advantage_vs_base": mean(same_subject_advantage),
        }
    )
    return rows


def group_rows_by(bundle: FeatureBundle, indices: Sequence[int], key: str, controller: str, preds: torch.Tensor, split: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for local_idx, row_idx in enumerate(indices):
        grouped[str(bundle.rows[row_idx].get(key) or "unknown")].append(local_idx)
    rows = []
    for value, local_positions in sorted(grouped.items()):
        subset_indices = [indices[pos] for pos in local_positions]
        subset_preds = preds[local_positions]
        metric = value_metrics(bundle, subset_indices, subset_preds, controller, split)
        metric[key] = value
        rows.append(metric)
    return rows


def bootstrap_rows(bundle: FeatureBundle, indices: Sequence[int], full_preds: torch.Tensor, target_preds: torch.Tensor, trials: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_edit: Dict[str, List[int]] = defaultdict(list)
    for local_idx, row_idx in enumerate(indices):
        by_edit[edit_id(bundle.rows[row_idx])].append(local_idx)
    edits = sorted(by_edit)
    deltas = []
    for _ in range(trials):
        sampled_positions = []
        for _ in edits:
            sampled_positions.extend(by_edit[rng.choice(edits)])
        sampled_indices = [indices[pos] for pos in sampled_positions]
        full = value_metrics(bundle, sampled_indices, full_preds[sampled_positions], "full", "bootstrap")
        target = value_metrics(bundle, sampled_indices, target_preds[sampled_positions], "target_only", "bootstrap")
        deltas.append(full["macro_groupwise_spearman"] - target["macro_groupwise_spearman"])
    deltas.sort()
    if not deltas:
        return []
    return [
        {
            "comparison": "d3_value_repr_minus_d3_target_indicator_only",
            "metric": "macro_groupwise_spearman",
            "delta_mean": mean(deltas),
            "ci_low": deltas[int(0.025 * (len(deltas) - 1))],
            "ci_high": deltas[int(0.975 * (len(deltas) - 1))],
            "trials": trials,
        }
    ]


def write_leakage_audit(feature_cache_dir: Path, train_dir: Path, replay_dir: Path, output_dir: Path) -> Dict[str, Any]:
    feature_schema = read_json(feature_cache_dir / "feature_schema.json")
    runtime_audit = read_json(feature_cache_dir / "runtime_feature_leakage_audit.json")
    report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1B.5 feature leakage audit v3",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "llada_loaded": False,
        "feature_cache_dir": str(feature_cache_dir),
        "train_dir": str(train_dir),
        "replay_dir": str(replay_dir),
        "num_leaked_runtime_features": int(runtime_audit.get("num_leaked_runtime_features", 0)),
        "feature_leakage_audit_pass": bool(runtime_audit.get("runtime_feature_leakage_audit_pass", False)),
        "forbidden_runtime_fields": feature_schema.get("forbidden_runtime_fields", []),
    }
    repo_path(output_dir).mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report_summary.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_cache_dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--teacher_cache_dir", type=Path, default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--local_audit_dir", type=Path, default=DEFAULT_LOCAL_AUDIT)
    parser.add_argument("--train_dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--replay_dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--leakage_dir", type=Path, default=DEFAULT_LEAKAGE_DIR)
    parser.add_argument("--shortcut_dir", type=Path, default=DEFAULT_SHORTCUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--negative_identity_weight", type=float, default=1.0)
    parser.add_argument("--target_loss_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bootstrap_trials", type=int, default=500)
    parser.add_argument("--allow_overwrite", type=int, default=0)
    parser.add_argument("--expected_candidate_groups", type=int, default=2994)
    parser.add_argument("--expected_candidate_width", type=int, default=8)
    return parser.parse_args()


def ensure_output(path: Path, allow_overwrite: bool) -> None:
    full = repo_path(path)
    if full.exists() and any(full.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"Output directory exists: {path}. Pass --allow_overwrite 1 for intentional reruns.")
    full.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    for out in [args.train_dir, args.replay_dir, args.leakage_dir, args.shortcut_dir]:
        ensure_output(out, bool(args.allow_overwrite))

    feature_audit = build_feature_cache_audit(
        args.feature_cache_dir,
        args.teacher_cache_dir,
        args.local_audit_dir,
        expected_candidate_groups=args.expected_candidate_groups,
        expected_candidate_width=args.expected_candidate_width,
    )
    if not feature_audit["audit_pass"]:
        raise AssertionError(f"Feature-cache local audit failed: {args.local_audit_dir}")

    bundle = FeatureBundle(args.feature_cache_dir, args.teacher_cache_dir)
    train_indices = bundle.row_indices_by_split["train"]
    val_indices = bundle.row_indices_by_split["val"]
    device = torch.device("cpu")

    gate_model = train_gate_model(bundle, train_indices, args.epochs, args.batch_size, args.lr, device)
    value_models = {
        "d3_value_repr": train_value_model(
            bundle,
            train_indices,
            "d3_value_repr",
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            proj_dim=args.proj_dim,
            hidden_dim=args.hidden_dim,
            teacher_temperature=args.teacher_temperature,
            negative_identity_weight=args.negative_identity_weight,
            target_loss_weight=args.target_loss_weight,
        ),
        "d3_value_repr_no_target_indicator": train_value_model(
            bundle,
            train_indices,
            "d3_value_repr_no_target_indicator",
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            proj_dim=args.proj_dim,
            hidden_dim=args.hidden_dim,
            teacher_temperature=args.teacher_temperature,
            negative_identity_weight=args.negative_identity_weight,
            target_loss_weight=args.target_loss_weight,
        ),
        "d3_target_indicator_only": train_value_model(
            bundle,
            train_indices,
            "d3_target_indicator_only",
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            proj_dim=args.proj_dim,
            hidden_dim=args.hidden_dim,
            teacher_temperature=args.teacher_temperature,
            negative_identity_weight=args.negative_identity_weight,
            target_loss_weight=args.target_loss_weight,
        ),
    }
    value_models["d3_value_gate_repr"] = value_models["d3_value_repr"]

    train_value_rows = []
    val_value_rows = []
    predictions: Dict[Tuple[str, str], torch.Tensor] = {}
    for name, model in value_models.items():
        for split, indices, out_rows in [("train", train_indices, train_value_rows), ("val", val_indices, val_value_rows)]:
            preds = predict_value(bundle, model, indices, args.batch_size)
            predictions[(name, split)] = preds
            out_rows.append(value_metrics(bundle, indices, preds, name, split))

    gate_train = predict_gate(bundle, gate_model, train_indices, args.batch_size)
    gate_val = predict_gate(bundle, gate_model, val_indices, args.batch_size)
    sweep = threshold_sweep(bundle, val_indices, gate_val, "d3_gate_repr")
    selected_threshold, threshold_pass = select_gate_threshold(sweep)

    gate_labels_val = [prompt_label(bundle.rows[i]) for i in val_indices]
    gate_scores_val = [float(v) for v in gate_val.tolist()]
    gate_metrics = {
        "controller": "d3_gate_repr",
        "split": "val",
        "gate_roc_auc": auc_score(gate_labels_val, gate_scores_val),
        "gate_pr_auc": pr_auc(gate_labels_val, gate_scores_val),
        "selected_threshold": selected_threshold,
        "threshold_acceptance_pass": threshold_pass,
    }

    groupwise_rows = val_value_rows
    negative_rows = negative_guidance_rows(bundle, val_indices, predictions[("d3_value_repr", "val")], gate_val, "d3_value_gate_repr")
    per_prompt_rows = []
    per_step_rows = []
    per_length_rows = []
    per_relation_rows = []
    for name in ["d3_value_repr", "d3_value_repr_no_target_indicator", "d3_target_indicator_only"]:
        preds = predictions[(name, "val")]
        per_prompt_rows.extend(group_rows_by(bundle, val_indices, "prompt_type", name, preds, "val"))
        per_step_rows.extend(group_rows_by(bundle, val_indices, "step_index", name, preds, "val"))
        per_length_rows.extend(group_rows_by(bundle, val_indices, "target_length_bin", name, preds, "val"))
        per_relation_rows.extend(group_rows_by(bundle, val_indices, "relation_id", name, preds, "val"))

    full_val = next(row for row in val_value_rows if row["controller"] == "d3_value_repr")
    target_val = next(row for row in val_value_rows if row["controller"] == "d3_target_indicator_only")
    no_target_val = next(row for row in val_value_rows if row["controller"] == "d3_value_repr_no_target_indicator")
    state_shuffle_preds = predict_value(bundle, value_models["d3_value_repr"], val_indices, args.batch_size, state_shuffle=True)
    candidate_shuffle_preds = predict_value(bundle, value_models["d3_value_repr"], val_indices, args.batch_size, candidate_shuffle=True)
    relation_shuffle_gate = predict_gate(bundle, gate_model, val_indices, args.batch_size, relation_shuffle=True)
    state_shuffle = value_metrics(bundle, val_indices, state_shuffle_preds, "state_shuffle", "val")
    candidate_shuffle = value_metrics(bundle, val_indices, candidate_shuffle_preds, "candidate_shuffle", "val")
    relation_shuffle_auc = auc_score(gate_labels_val, [float(v) for v in relation_shuffle_gate.tolist()])
    shortcut_rows = [
        {"ablation": "full", **full_val},
        {"ablation": "target_indicator_only", **target_val},
        {"ablation": "no_target_indicator", **no_target_val},
        {"ablation": "state_shuffle", **state_shuffle},
        {"ablation": "candidate_shuffle", **candidate_shuffle},
        {
            "ablation": "relation_shuffle",
            "gate_roc_auc": relation_shuffle_auc,
            "gate_auc_delta_vs_full": gate_metrics["gate_roc_auc"] - relation_shuffle_auc,
        },
    ]
    bootstrap = bootstrap_rows(bundle, val_indices, predictions[("d3_value_repr", "val")], predictions[("d3_target_indicator_only", "val")], args.bootstrap_trials, args.seed)

    neg_all = next(row for row in negative_rows if row["prompt_type"] == "ALL_NEGATIVE")
    hard = {
        "value_spearman_pass": full_val["macro_groupwise_spearman"] >= HARD_CRITERIA["macro_groupwise_spearman"],
        "value_ndcg_pass": full_val["ndcg_at_8"] >= HARD_CRITERIA["ndcg_at_8"],
        "value_pairwise_pass": full_val["pairwise_ranking_accuracy"] >= HARD_CRITERIA["pairwise_ranking_accuracy"],
        "value_top1_pass": full_val["teacher_top1_agreement"] >= HARD_CRITERIA["teacher_top1_agreement"],
        "value_top3_pass": full_val["teacher_top3_overlap"] >= HARD_CRITERIA["teacher_top3_overlap"],
        "target_top3_improvement_pass": full_val["target_top3_improvement_over_base"] >= HARD_CRITERIA["target_top3_improvement_over_base"],
        "gate_auc_pass": gate_metrics["gate_roc_auc"] >= HARD_CRITERIA["gate_roc_auc"],
        "gate_threshold_pass": threshold_pass,
        "negative_guidance_ratio_pass": float(neg_all["ratio_to_positive"]) <= HARD_CRITERIA["negative_guidance_ratio"],
        "same_subject_target_advantage_pass": float(neg_all.get("same_subject_target_advantage_vs_base") or 0.0) <= 0.0,
        "representation_beats_target_indicator_pass": (
            full_val["macro_groupwise_spearman"] >= target_val["macro_groupwise_spearman"] + 0.05
            or full_val["ndcg_at_8"] >= target_val["ndcg_at_8"] + 0.05
        ),
        "state_shuffle_hurts_pass": full_val["macro_groupwise_spearman"] >= state_shuffle["macro_groupwise_spearman"] + 0.05,
        "relation_shuffle_hurts_pass": gate_metrics["gate_roc_auc"] >= relation_shuffle_auc + 0.05,
    }
    all_hard = all(hard.values())
    status = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1B.5 offline replay scientific status v3",
        "analysis_500_used": False,
        "final_test_used": False,
        "llada_loaded": False,
        "actual_decode_allowed": all_hard,
        "scientific_acceptance_pass": all_hard,
        "hard_criteria": hard,
        "decision": "green_pass_ready_for_stage_2a_approval" if all_hard else "do_not_decode_offline_criteria_failed",
    }

    torch.save(
        {
            "gate_model": gate_model.state_dict(),
            "value_models": {name: model.state_dict() for name, model in value_models.items()},
            "selected_gate_threshold": selected_threshold,
            "feature_cache_dir": str(args.feature_cache_dir),
        },
        repo_path(args.train_dir / "repr_controller_weights.pt"),
    )
    train_report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1B.4 representation-aware deployable training v3",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "llada_loaded": False,
        "feature_cache_dir": str(args.feature_cache_dir),
        "teacher_cache_dir": str(args.teacher_cache_dir),
        "model_variants": ["d3_value_repr", "d3_gate_repr", "d3_value_gate_repr", "d3_value_repr_no_target_indicator", "d3_target_indicator_only"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "proj_dim": args.proj_dim,
        "hidden_dim": args.hidden_dim,
        "teacher_temperature": args.teacher_temperature,
        "negative_identity_weight": args.negative_identity_weight,
        "target_loss_weight": args.target_loss_weight,
        "selected_gate_threshold": selected_threshold,
        "threshold_acceptance_pass": threshold_pass,
    }
    replay_report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1B.5 offline replay v3",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "llada_loaded": False,
        "scientific_acceptance_pass": all_hard,
        "selected_gate_threshold": selected_threshold,
        "gate_metrics": gate_metrics,
        "hard_criteria": hard,
    }
    shortcut_report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 representation shortcut audit v3",
        "created_at_utc": now_utc(),
        "analysis_500_used": False,
        "final_test_used": False,
        "llada_loaded": False,
        "representation_beats_target_indicator_pass": hard["representation_beats_target_indicator_pass"],
        "state_shuffle_hurts_pass": hard["state_shuffle_hurts_pass"],
        "relation_shuffle_hurts_pass": hard["relation_shuffle_hurts_pass"],
    }

    write_csv(args.train_dir / "train_metrics.csv", train_value_rows)
    write_csv(args.train_dir / "validation_metrics.csv", val_value_rows + [gate_metrics])
    write_csv(args.train_dir / "gate_threshold_sweep.csv", sweep)
    write_json(args.train_dir / "report_summary.json", train_report)

    write_csv(args.replay_dir / "groupwise_ranking_metrics.csv", groupwise_rows)
    write_csv(args.replay_dir / "negative_guidance_diagnostics.csv", negative_rows)
    write_csv(args.replay_dir / "per_prompt_type_metrics.csv", per_prompt_rows)
    write_csv(args.replay_dir / "per_step_metrics.csv", per_step_rows)
    write_csv(args.replay_dir / "per_target_length_metrics.csv", per_length_rows)
    write_csv(args.replay_dir / "per_relation_metrics.csv", per_relation_rows)
    write_csv(args.replay_dir / "representation_ablation.csv", shortcut_rows)
    write_csv(args.replay_dir / "paired_bootstrap.csv", bootstrap)
    write_csv(args.replay_dir / "gate_threshold_sweep.csv", sweep)
    write_csv(args.replay_dir / "validation_metrics.csv", val_value_rows + [gate_metrics])
    write_json(args.replay_dir / "scientific_status.json", status)
    write_json(args.replay_dir / "report_summary.json", replay_report)

    write_csv(args.shortcut_dir / "representation_ablation.csv", shortcut_rows)
    write_csv(args.shortcut_dir / "paired_bootstrap.csv", bootstrap)
    write_json(args.shortcut_dir / "report_summary.json", shortcut_report)
    write_leakage_audit(args.feature_cache_dir, args.train_dir, args.replay_dir, args.leakage_dir)
    print(f"[INFO] Wrote v3 train outputs to {args.train_dir}")
    print(f"[INFO] Wrote v3 replay outputs to {args.replay_dir}")
    print(f"[INFO] scientific_acceptance_pass={all_hard}")


if __name__ == "__main__":
    main()
