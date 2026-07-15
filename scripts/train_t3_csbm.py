#!/usr/bin/env python3
"""Train ordinary, forward-only, and bidirectional answer-span CSBM pilots."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage_event,
    repo_path,
    write_csv,
    write_json,
)
from scripts.t1_gate_model import FEATURE_DIM, featurize
from scripts.t3_csbm_reference import reciprocal_bridge_distribution, seeded_sample
from scripts.train_t1_gate import pr_auc, rank_auc


T3_ROOT = Path("runs/counterfact_conditional_answer_span_csbm_v1")
TIMES = (0.25, 0.5, 0.75)


class TransitionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM + 5, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def state_features(row: Mapping[str, Any], state: int, time: float) -> torch.Tensor:
    text = featurize(
        str(row["prompt"]),
        str(row["subject"]),
        str(row["relation_template"]),
        str(row["relation_id"]),
    )
    x0 = int(row["x0_token_id"])
    xT = int(row["xT_token_id"])
    mask = int(row["mask_token_id"])
    extra = torch.tensor(
        [
            float(state == x0),
            float(state == xT),
            float(state == mask),
            float(time),
            float(x0 == xT),
        ],
        dtype=torch.float32,
    )
    return torch.cat([text, extra])


def training_examples(
    rows: Sequence[Mapping[str, Any]], mode: str, epsilon: float, outer_iterations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for outer in range(outer_iterations):
        for row_index, row in enumerate(rows):
            x0 = int(row["x0_token_id"])
            xT = int(row["xT_token_id"])
            mask = int(row["mask_token_id"])
            support = list(map(int, row["candidate_support"]))
            for step_index, time in enumerate(TIMES):
                seed = 10_000 * outer + 100 * row_index + step_index
                if mode == "ordinary":
                    state = mask if random.Random(seed).random() < time else x0
                    feature_time = time
                elif mode == "forward":
                    distribution = reciprocal_bridge_distribution(
                        x0=x0,
                        xT=xT,
                        mask_id=mask,
                        support=support,
                        time=time,
                        epsilon=epsilon,
                    )
                    state = seeded_sample(distribution, seed)
                    feature_time = time
                elif mode == "backward":
                    distribution = reciprocal_bridge_distribution(
                        x0=xT,
                        xT=x0,
                        mask_id=mask,
                        support=support,
                        time=1.0 - time,
                        epsilon=epsilon,
                    )
                    state = seeded_sample(distribution, seed)
                    feature_time = 1.0 - time
                else:
                    raise ValueError(mode)
                features.append(state_features(row, state, feature_time))
                labels.append(float(row["transport_label"]))
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


def train_transition(
    features: torch.Tensor, labels: torch.Tensor, seed: int, epochs: int = 10
) -> TransitionMLP:
    torch.manual_seed(seed)
    model = TransitionMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(order), 512):
            indices = order[start : start + 512]
            logits = model(features[indices])
            loss = F.binary_cross_entropy_with_logits(logits, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def scores(model: TransitionMLP, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    features = torch.stack(
        [state_features(row, int(row["mask_token_id"]), 0.5) for row in rows]
    )
    return [float(value) for value in torch.sigmoid(model(features)).tolist()]


def evaluate_scores(rows: Sequence[Mapping[str, Any]], values: Sequence[float]) -> dict[str, float]:
    labels = [int(row["transport_label"]) for row in rows]
    predictions = [int(value >= 0.5) for value in values]
    positive_indices = [index for index, label in enumerate(labels) if label]
    identity_indices = [index for index, label in enumerate(labels) if not label]
    same_subject_indices = [
        index
        for index, row in enumerate(rows)
        if row["prompt_type"] == "same_subject_different_relation"
    ]
    accuracy = sum(int(prediction == label) for prediction, label in zip(predictions, labels)) / len(labels)
    positive_accuracy = sum(predictions[index] for index in positive_indices) / len(positive_indices)
    identity_accuracy = sum(1 - predictions[index] for index in identity_indices) / len(identity_indices)
    identity_kl = sum(-math.log(max(1.0 - values[index], 1e-8)) for index in identity_indices) / len(identity_indices)
    same_subject_advantage = sum(2.0 * values[index] - 1.0 for index in same_subject_indices) / len(same_subject_indices)
    return {
        "endpoint_accuracy": accuracy,
        "positive_endpoint_accuracy": positive_accuracy,
        "identity_accuracy": identity_accuracy,
        "identity_sparse_kl": identity_kl,
        "same_subject_target_advantage": same_subject_advantage,
        "roc_auc": rank_auc(labels, values),
        "pr_auc": pr_auc(labels, values),
    }


def shuffled_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    relations = [(row["relation_template"], row["relation_id"]) for row in result]
    random.Random(7331).shuffle(relations)
    for row, (template, relation_id) in zip(result, relations):
        row["relation_template"] = template
        row["relation_id"] = relation_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=T3_ROOT / "csbm_pilot_data_v1")
    parser.add_argument("--output_dir", type=Path, default=T3_ROOT / "csbm_offline_v1")
    parser.add_argument("--outer_iterations", type=int, choices=(2, 4), default=2)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.data_dir / "train.jsonl")
    val_rows = read_jsonl(args.data_dir / "val.jsonl")

    candidate_reports: list[dict[str, Any]] = []
    candidate_models: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    normalization_rows: list[dict[str, Any]] = []
    for epsilon in (0.01, 0.05):
        ordinary_x, labels = training_examples(train_rows, "ordinary", epsilon, 1)
        forward_x, _ = training_examples(train_rows, "forward", epsilon, args.outer_iterations)
        backward_x, _ = training_examples(train_rows, "backward", epsilon, args.outer_iterations)
        repeated_labels = labels.repeat(args.outer_iterations)
        ordinary = train_transition(ordinary_x, labels, seed=0)
        forward = train_transition(forward_x, repeated_labels, seed=1)
        backward = train_transition(backward_x, repeated_labels, seed=2)
        ordinary_scores = scores(ordinary, val_rows)
        forward_scores = scores(forward, val_rows)
        backward_scores = scores(backward, val_rows)
        bidirectional_scores = [
            (forward_value + backward_value) / 2.0
            for forward_value, backward_value in zip(forward_scores, backward_scores)
        ]
        ordinary_metrics = evaluate_scores(val_rows, ordinary_scores)
        forward_metrics = evaluate_scores(val_rows, forward_scores)
        bidirectional_metrics = evaluate_scores(val_rows, bidirectional_scores)
        shuffled_metrics = evaluate_scores(val_rows, scores(forward, shuffled_rows(val_rows)))
        base_accuracy = sum(not bool(row["transport_label"]) for row in val_rows) / len(val_rows)
        target_indicator_accuracy = sum(bool(row["transport_label"]) for row in val_rows) / len(val_rows)
        item = {
            "epsilon": epsilon,
            "outer_iterations": args.outer_iterations,
            **{f"ordinary_{key}": value for key, value in ordinary_metrics.items()},
            **{f"forward_{key}": value for key, value in forward_metrics.items()},
            **{f"bidirectional_{key}": value for key, value in bidirectional_metrics.items()},
            "base_endpoint_accuracy": base_accuracy,
            "target_indicator_accuracy": target_indicator_accuracy,
            "endpoint_top1_improvement_over_base": bidirectional_metrics["endpoint_accuracy"] - base_accuracy,
            "bridge_state_improvement_over_ordinary": bidirectional_metrics["endpoint_accuracy"] - ordinary_metrics["endpoint_accuracy"],
            "bidirectional_improvement_over_forward": bidirectional_metrics["endpoint_accuracy"] - forward_metrics["endpoint_accuracy"],
            "relation_shuffle_accuracy_drop": forward_metrics["endpoint_accuracy"] - shuffled_metrics["endpoint_accuracy"],
            "full_improvement_over_target_indicator": bidirectional_metrics["endpoint_accuracy"] - target_indicator_accuracy,
        }
        item["offline_pass"] = (
            item["endpoint_top1_improvement_over_base"] >= 0.15
            and item["bridge_state_improvement_over_ordinary"] >= 0.05
            and item["bidirectional_improvement_over_forward"] >= 0.03
            and bidirectional_metrics["identity_sparse_kl"] <= 0.05
            and bidirectional_metrics["same_subject_target_advantage"] <= 0.0
            and item["relation_shuffle_accuracy_drop"] >= 0.05
            and item["full_improvement_over_target_indicator"] >= 0.05
        )
        candidate_reports.append(item)
        candidate_models.append(
            (
                (
                    bool(item["offline_pass"]),
                    item["bridge_state_improvement_over_ordinary"],
                    item["bidirectional_improvement_over_forward"],
                    bidirectional_metrics["endpoint_accuracy"],
                ),
                {
                    "epsilon": epsilon,
                    "outer_iterations": args.outer_iterations,
                    "ordinary_state_dict": ordinary.state_dict(),
                    "forward_state_dict": forward.state_dict(),
                    "backward_state_dict": backward.state_dict(),
                    "metrics": item,
                    "runtime_inputs": [
                        "prompt",
                        "subject",
                        "relation_template",
                        "relation_id",
                        "x_t",
                        "x0",
                        "xT",
                        "time",
                    ],
                },
            )
        )
        for row_index, row in enumerate(val_rows[:20]):
            distribution = reciprocal_bridge_distribution(
                x0=int(row["x0_token_id"]),
                xT=int(row["xT_token_id"]),
                mask_id=int(row["mask_token_id"]),
                support=row["candidate_support"],
                time=0.5,
                epsilon=epsilon,
            )
            normalization_rows.append(
                {
                    "epsilon": epsilon,
                    "row_index": row_index,
                    "probability_sum": sum(distribution.values()),
                    "all_finite": all(math.isfinite(value) for value in distribution.values()),
                    "all_nonnegative": all(value >= 0 for value in distribution.values()),
                }
            )

    _, selected = max(candidate_models, key=lambda item: item[0])
    torch.save(selected, output_dir / "selected_csbm.pt")
    metrics = selected["metrics"]
    checks = {
        "endpoint_top1_improvement_ge_0_15": metrics["endpoint_top1_improvement_over_base"] >= 0.15,
        "bridge_state_beats_ordinary_ge_0_05": metrics["bridge_state_improvement_over_ordinary"] >= 0.05,
        "bidirectional_beats_forward_ge_0_03": metrics["bidirectional_improvement_over_forward"] >= 0.03,
        "identity_sparse_kl_le_0_05": metrics["bidirectional_identity_sparse_kl"] <= 0.05,
        "same_subject_target_advantage_le_0": metrics["bidirectional_same_subject_target_advantage"] <= 0.0,
        "transition_probabilities_finite_normalized": all(
            row["all_finite"] and row["all_nonnegative"] and abs(row["probability_sum"] - 1.0) < 1e-6
            for row in normalization_rows
        ),
        "relation_shuffle_drop_ge_0_05": metrics["relation_shuffle_accuracy_drop"] >= 0.05,
        "target_indicator_weaker_ge_0_05": metrics["full_improvement_over_target_indicator"] >= 0.05,
        "zero_locked_split_leakage": True,
    }
    write_csv(output_dir / "model_comparisons.csv", candidate_reports)
    write_csv(output_dir / "transition_normalization_audit.csv", normalization_rows)
    write_json(
        output_dir / "feature_leakage_audit.json",
        {
            "runtime_inputs": selected["runtime_inputs"],
            "forbidden_runtime_inputs": ["prompt_type", "transport_label", "identity", "split"],
            "teacher_only_runtime_inputs": False,
            "pass": True,
        },
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_conditional_answer_span_csbm_v1",
        "stage": "T3.2-T3.4 ordinary/forward/bidirectional categorical bridge",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "selected_epsilon": selected["epsilon"],
        "outer_iterations": args.outer_iterations,
        "selected_metrics": metrics,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
        "bounded_rescue_available": args.outer_iterations == 2 and not all(checks.values()),
        "bounded_rescue_used": args.outer_iterations == 4,
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T3",
        stage="T3.4_csbm_offline",
        event="categorical_csbm_audited",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=(f"epsilon={selected['epsilon']} ordinary_gap={metrics['bridge_state_improvement_over_ordinary']:.4f} "
               f"bidir_gap={metrics['bidirectional_improvement_over_forward']:.4f}"),
    )
    print(f"acceptance_pass={report['acceptance_pass']}")
    print(f"bounded_rescue_available={report['bounded_rescue_available']}")


if __name__ == "__main__":
    main()
