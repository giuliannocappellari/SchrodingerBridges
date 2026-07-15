#!/usr/bin/env python3
"""Train and validate the deployable T1 learned edit-intent gate."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    budget_guard,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage_event,
    repo_path,
    write_csv,
    write_json,
)
from scripts.t1_gate_model import (
    FORBIDDEN_RUNTIME_FIELDS,
    GateMLP,
    checkpoint_schema,
    featurize,
    save_checkpoint,
)


T1_ROOT = Path("runs/counterfact_learned_gate_raw_bridge_v1")
PROMPT_TYPES = (
    "rewrite",
    "declarative_paraphrase",
    "same_subject_different_relation",
    "near_locality",
    "far_locality",
    "generation",
    "attribute",
    "unrelated",
)
THRESHOLDS = [index / 200.0 for index in range(201)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware ROC AUC without a scikit-learn dependency."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in ordered:
        if label:
            true_positives += 1
        else:
            false_positives += 1
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    return torch.stack(
        [
            featurize(
                str(row["prompt"]),
                str(row["subject"]),
                str(row["relation_template"]),
                str(row["relation_id"]),
            )
            for row in rows
        ]
    )


def relation_shuffled_matrix(rows: Sequence[Mapping[str, Any]], seed: int = 1729) -> torch.Tensor:
    relations = [(str(row["relation_template"]), str(row["relation_id"])) for row in rows]
    shuffled = relations[:]
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == relations:
        shuffled = shuffled[1:] + shuffled[:1]
    return torch.stack(
        [
            featurize(str(row["prompt"]), str(row["subject"]), template, relation_id)
            for row, (template, relation_id) in zip(rows, shuffled)
        ]
    )


def loss_for_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if loss_name == "focal":
        probability = torch.sigmoid(logits)
        target_probability = labels * probability + (1.0 - labels) * (1.0 - probability)
        raw = raw * (1.0 - target_probability).pow(2.0)
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def train_model(
    train_features: torch.Tensor,
    train_rows: Sequence[Mapping[str, Any]],
    *,
    hidden_dim: int,
    loss_name: str,
    same_subject_weight: float,
    seed: int,
    epochs: int,
) -> tuple[GateMLP, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GateMLP(hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    labels = torch.tensor([float(row["label"]) for row in train_rows], dtype=torch.float32)
    positive_weight = max(1.0, (len(labels) - float(labels.sum())) / max(float(labels.sum()), 1.0))
    weights = torch.tensor(
        [
            same_subject_weight
            if row.get("negative_type") == "same_subject_different_relation"
            else positive_weight
            if int(row["label"]) == 1
            else 1.0
            for row in train_rows
        ],
        dtype=torch.float32,
    )
    history: list[dict[str, Any]] = []
    batch_size = 512
    for epoch in range(epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            logits = model(train_features[indices])
            loss = loss_for_batch(logits, labels[indices], weights[indices], loss_name)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "train_loss": mean(losses)})
    model.eval()
    return model, history


@torch.no_grad()
def probabilities(model: GateMLP, features: torch.Tensor) -> list[float]:
    output: list[float] = []
    for start in range(0, len(features), 2048):
        output.extend(torch.sigmoid(model(features[start : start + 2048])).tolist())
    return [float(value) for value in output]


def hard_gate_pass(row: Mapping[str, Any]) -> bool:
    return (
        float(row.get("roc_auc", 0.0)) >= 0.90
        and float(row.get("rewrite_activation", 0.0)) >= 0.90
        and float(row.get("declarative_paraphrase_activation", 0.0)) >= 0.85
        and float(row.get("same_subject_different_relation_activation", 1.0)) <= 0.05
        and float(row.get("near_locality_activation", 1.0)) <= 0.02
        and float(row.get("far_locality_activation", 1.0)) <= 0.02
        and float(row.get("relation_shuffle_auc_drop", 0.0)) >= 0.05
    )


def threshold_rows(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float], shuffled_auc: float
) -> list[dict[str, Any]]:
    labels = [int(row["label"]) for row in rows]
    auc = rank_auc(labels, scores)
    precision_area = pr_auc(labels, scores)
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_type[str(row["prompt_type"])].append(index)
    result: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        item: dict[str, Any] = {
            "threshold": threshold,
            "roc_auc": auc,
            "pr_auc": precision_area,
            "relation_shuffle_auc": shuffled_auc,
            "relation_shuffle_auc_drop": auc - shuffled_auc,
        }
        for prompt_type in PROMPT_TYPES:
            indices = by_type.get(prompt_type, [])
            item[f"{prompt_type}_activation"] = mean(
                [float(scores[index] >= threshold) for index in indices]
            )
        item["hard_acceptance_pass"] = hard_gate_pass(item)
        result.append(item)
    return result


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passing = [dict(row) for row in rows if bool(row["hard_acceptance_pass"])]
    if passing:
        return max(
            passing,
            key=lambda row: (
                float(row["declarative_paraphrase_activation"]),
                -float(row["same_subject_different_relation_activation"]),
                -float(row["threshold"]),
            ),
        )
    return max(
        (dict(row) for row in rows),
        key=lambda row: (
            float(row["rewrite_activation"])
            + float(row["declarative_paraphrase_activation"])
            - float(row["same_subject_different_relation_activation"])
            - float(row["near_locality_activation"])
            - float(row["far_locality_activation"]),
            float(row["relation_shuffle_auc_drop"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=T1_ROOT / "gate_data_v1")
    parser.add_argument("--output_dir", type=Path, default=T1_ROOT / "gate_train_v1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)

    guard = budget_guard("T1")
    if not guard["pass"]:
        raise RuntimeError(f"T1 budget guard failed: {guard}")
    train_rows = read_jsonl(args.data_dir / "gate_train.jsonl")
    val_rows = read_jsonl(args.data_dir / "gate_val.jsonl")
    train_features = feature_matrix(train_rows)
    val_features = feature_matrix(val_rows)
    shuffled_features = relation_shuffled_matrix(val_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = [{
        "config_id": "linear_bce_w3_seed0",
        "hidden_dim": 0,
        "loss": "weighted_bce",
        "same_subject_weight": 3,
        "seed": 0,
    }]
    for loss_name in ("weighted_bce", "focal"):
        for hidden_dim in (128, 256):
            for same_subject_weight in (3, 5):
                for seed in (0, 1):
                    configs.append({
                        "config_id": f"mlp{hidden_dim}_{loss_name}_w{same_subject_weight}_seed{seed}",
                        "hidden_dim": hidden_dim,
                        "loss": loss_name,
                        "same_subject_weight": same_subject_weight,
                        "seed": seed,
                    })

    train_metric_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    all_sweeps: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], GateMLP, dict[str, Any]]] = []
    for config in configs:
        model, history = train_model(
            train_features,
            train_rows,
            hidden_dim=int(config["hidden_dim"]),
            loss_name=str(config["loss"]),
            same_subject_weight=float(config["same_subject_weight"]),
            seed=int(config["seed"]),
            epochs=args.epochs,
        )
        train_metric_rows.extend({**config, **row} for row in history)
        scores = probabilities(model, val_features)
        shuffled_scores = probabilities(model, shuffled_features)
        labels = [int(row["label"]) for row in val_rows]
        sweep = threshold_rows(val_rows, scores, rank_auc(labels, shuffled_scores))
        selected = {**select_threshold(sweep), **config}
        validation_rows.append(selected)
        all_sweeps.extend({"config_id": config["config_id"], **row} for row in sweep)
        candidates.append((selected, model, config))

    eligible = [item for item in candidates if bool(item[0]["hard_acceptance_pass"])]
    selection_pool = eligible or candidates
    selected_metrics, selected_model, selected_config = max(
        selection_pool,
        key=lambda item: (
            float(item[0]["pr_auc"]),
            float(item[0]["declarative_paraphrase_activation"]),
            float(item[0]["relation_shuffle_auc_drop"]),
        ),
    )
    schema = checkpoint_schema(int(selected_config["hidden_dim"]), float(selected_metrics["threshold"]))
    schema.update({
        "config_id": selected_config["config_id"],
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_learned_gate_raw_bridge_v1",
    })
    save_checkpoint(output_dir / "checkpoints" / "selected_gate.pt", selected_model, schema)
    write_json(output_dir / "configs" / "selected_config.json", {**selected_config, **schema})
    write_csv(output_dir / "train_metrics.csv", train_metric_rows)
    write_csv(output_dir / "val_metrics.csv", validation_rows)
    write_csv(output_dir / "threshold_sweep.csv", all_sweeps)

    leakage_audit = {
        "runtime_inputs": schema["runtime_inputs"],
        "forbidden_runtime_inputs": sorted(FORBIDDEN_RUNTIME_FIELDS),
        "runtime_forbidden_intersection": sorted(set(schema["runtime_inputs"]) & FORBIDDEN_RUNTIME_FIELDS),
        "teacher_derived_runtime_inputs": False,
        "prompt_type_used_at_runtime": False,
        "negative_type_used_at_runtime": False,
        "evaluation_bucket_used_at_runtime": False,
        "pass": not bool(set(schema["runtime_inputs"]) & FORBIDDEN_RUNTIME_FIELDS),
    }
    write_json(output_dir / "feature_leakage_audit.json", leakage_audit)
    checks = {
        "roc_auc_ge_0_90": float(selected_metrics["roc_auc"]) >= 0.90,
        "rewrite_activation_ge_0_90": float(selected_metrics["rewrite_activation"]) >= 0.90,
        "paraphrase_activation_ge_0_85": float(selected_metrics["declarative_paraphrase_activation"]) >= 0.85,
        "same_subject_activation_le_0_05": float(selected_metrics["same_subject_different_relation_activation"]) <= 0.05,
        "near_activation_le_0_02": float(selected_metrics["near_locality_activation"]) <= 0.02,
        "far_activation_le_0_02": float(selected_metrics["far_locality_activation"]) <= 0.02,
        "relation_shuffle_auc_drop_ge_0_05": float(selected_metrics["relation_shuffle_auc_drop"]) >= 0.05,
        "zero_runtime_feature_leakage": bool(leakage_audit["pass"]),
        "analysis_final_unused": True,
    }
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_learned_gate_raw_bridge_v1",
        "stage": "T1.2 learned gate training",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "budget_guard": guard,
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "num_configs": len(configs),
        "selected_config": selected_config,
        "selected_metrics": selected_metrics,
        "checkpoint": str(args.output_dir / "checkpoints" / "selected_gate.pt"),
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
        "bounded_rescue_available": not all(checks.values()),
        "bounded_rescue_used": False,
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T1",
        stage="T1.2_gate_train",
        event="learned_gate_trained",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=(f"config={selected_config['config_id']} auc={selected_metrics['roc_auc']:.4f} "
               f"relation_shuffle_drop={selected_metrics['relation_shuffle_auc_drop']:.4f}"),
    )
    print(f"acceptance_pass={report['acceptance_pass']}")
    print(f"selected_config={selected_config['config_id']}")
    print(f"selected_threshold={selected_metrics['threshold']}")


if __name__ == "__main__":
    main()
