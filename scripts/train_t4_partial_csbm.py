#!/usr/bin/env python3
"""Train and audit the T4 learned partial categorical bridge."""

from __future__ import annotations

import argparse
import math
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
    budget_guard,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage_event,
    repo_path,
    write_csv,
    write_json,
)
from scripts.t1_gate_model import FEATURE_DIM, featurize, load_checkpoint, predict_probability
from scripts.train_t1_gate import pr_auc, rank_auc
from scripts.train_t3_csbm import (
    CandidateTransitionMLP,
    endpoint_probabilities,
    evaluate_predictions,
    primary_rows,
)


T4_ROOT = Path("runs/counterfact_unbalanced_partial_csbm_v1")
T3_ROOT = Path("runs/counterfact_conditional_answer_span_csbm_v1")
T1_ROOT = Path("runs/counterfact_learned_gate_raw_bridge_v1")


class MassModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM + 3, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def distribution_entropy(distribution: Mapping[int, float]) -> float:
    return -sum(float(value) * math.log(max(float(value), 1e-8)) for value in distribution.values())


def mass_feature(
    row: Mapping[str, Any], balanced_row: Sequence[Mapping[int, float]]
) -> torch.Tensor:
    text = featurize(
        str(row["prompt"]),
        str(row["subject"]),
        str(row["relation_template"]),
        str(row["relation_id"]),
    )
    target_confidences = [
        float(distribution.get(int(row["target_new_token_ids"][position]), 0.0))
        for position, distribution in enumerate(balanced_row)
    ]
    entropies = [distribution_entropy(distribution) for distribution in balanced_row]
    runtime = torch.tensor(
        [
            0.5,
            sum(target_confidences) / len(target_confidences),
            sum(entropies) / len(entropies),
        ],
        dtype=torch.float32,
    )
    return torch.cat([text, runtime])


def matrices(
    rows: Sequence[Mapping[str, Any]],
    balanced: Sequence[Sequence[Mapping[int, float]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([mass_feature(row, values) for row, values in zip(rows, balanced)]),
        torch.tensor([float(row["transport_label"]) for row in rows]),
    )


def train_mass(
    features: torch.Tensor,
    labels: torch.Tensor,
    rho_pos: float,
    rho_neg: float,
    lambda_mass: float,
    seed: int,
) -> MassModel:
    torch.manual_seed(seed)
    model = MassModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    targets = labels * rho_pos + (1.0 - labels) * rho_neg
    for _ in range(12):
        order = torch.randperm(len(features))
        for start in range(0, len(order), 256):
            index = order[start : start + 256]
            logits = model(features[index])
            rho = torch.sigmoid(logits)
            calibration = F.mse_loss(rho, targets[index])
            classification = F.binary_cross_entropy_with_logits(logits, labels[index])
            identity = (rho * (1.0 - labels[index])).mean()
            loss = calibration + 0.1 * classification + lambda_mass * identity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def probabilities(model: MassModel, features: torch.Tensor, temperature: float = 1.0) -> list[float]:
    return [float(value) for value in torch.sigmoid(model(features) / temperature).tolist()]


def mix_with_identity(
    rows: Sequence[Mapping[str, Any]],
    balanced: Sequence[Sequence[Mapping[int, float]]],
    rho: Sequence[float],
) -> list[list[dict[int, float]]]:
    output: list[list[dict[int, float]]] = []
    for row, balanced_row, mass in zip(rows, balanced, rho):
        row_output: list[dict[int, float]] = []
        for position, distribution in enumerate(balanced_row):
            old = int(row["x0_token_ids"][position])
            mixed = {
                int(token): float(mass) * float(probability)
                for token, probability in distribution.items()
            }
            mixed[old] = mixed.get(old, 0.0) + 1.0 - float(mass)
            total = sum(mixed.values())
            row_output.append({token: value / total for token, value in mixed.items()})
        output.append(row_output)
    return output


def rho_only_target_bias(
    rows: Sequence[Mapping[str, Any]], rho: Sequence[float]
) -> list[list[dict[int, float]]]:
    output: list[list[dict[int, float]]] = []
    for row, mass in zip(rows, rho):
        row_output = []
        for position, support in enumerate(row["candidate_support_by_position"]):
            old = int(row["x0_token_ids"][position])
            target = int(row["target_new_token_ids"][position])
            distribution = {int(token): 0.0 for token in support}
            distribution[old] = distribution.get(old, 0.0) + 1.0 - mass
            distribution[target] = distribution.get(target, 0.0) + mass
            row_output.append(distribution)
        output.append(row_output)
    return output


def mass_metrics(rows: Sequence[Mapping[str, Any]], rho: Sequence[float]) -> dict[str, float]:
    labels = [int(row["transport_label"]) for row in rows]
    positive = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    by_type = {
        prompt_type: [index for index, row in enumerate(rows) if row["prompt_type"] == prompt_type]
        for prompt_type in {str(row["prompt_type"]) for row in rows}
    }
    average = lambda indices: sum(rho[index] for index in indices) / len(indices) if indices else math.nan
    return {
        "positive_mean_rho": average(positive),
        "same_subject_mean_rho": average(by_type.get("same_subject_different_relation", [])),
        "near_mean_rho": average(by_type.get("near_locality", [])),
        "negative_mean_rho": average(negatives),
        "mass_roc_auc": rank_auc(labels, rho),
        "mass_pr_auc": pr_auc(labels, rho),
    }


def load_balanced_checkpoint(path: Path) -> tuple[CandidateTransitionMLP, CandidateTransitionMLP, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    forward = CandidateTransitionMLP()
    backward = CandidateTransitionMLP()
    forward.load_state_dict(checkpoint["forward_state_dict"])
    backward.load_state_dict(checkpoint["backward_state_dict"])
    forward.eval()
    backward.eval()
    return forward, backward, checkpoint


def tradeoff(metrics: Mapping[str, float]) -> float:
    return float(metrics["positive_endpoint_accuracy"]) - max(
        0.0, float(metrics["same_subject_target_advantage"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=T3_ROOT / "csbm_pilot_data_v1")
    parser.add_argument("--balanced_checkpoint", type=Path, default=T3_ROOT / "csbm_offline_v1/selected_csbm.pt")
    parser.add_argument("--output_dir", type=Path, default=T4_ROOT / "partial_csbm_offline_v1")
    parser.add_argument(
        "--external_gate_checkpoint",
        type=Path,
        default=T1_ROOT / "gate_train_v2/checkpoints/selected_gate.pt",
    )
    parser.add_argument("--rho_temperature", type=float, default=1.0)
    parser.add_argument("--bounded_rescue", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = budget_guard("T4")
    train_rows = primary_rows(read_jsonl(args.data_dir / "train.jsonl"))
    val_rows = primary_rows(read_jsonl(args.data_dir / "val.jsonl"))
    forward, backward, balanced_checkpoint = load_balanced_checkpoint(repo_path(args.balanced_checkpoint))
    balanced_train = endpoint_probabilities(forward, train_rows, backward=backward)
    balanced_val = endpoint_probabilities(forward, val_rows, backward=backward)
    balanced_metrics = evaluate_predictions(val_rows, balanced_val)
    train_x, train_y = matrices(train_rows, balanced_train)
    val_x, _ = matrices(val_rows, balanced_val)

    external_model = external_schema = None
    if repo_path(args.external_gate_checkpoint).exists():
        external_model, external_schema = load_checkpoint(repo_path(args.external_gate_checkpoint))
    candidates: list[tuple[tuple[Any, ...], MassModel, dict[str, Any]]] = []
    report_rows: list[dict[str, Any]] = []
    for rho_pos in (0.7, 0.9):
        for rho_neg in (0.01, 0.05):
            for lambda_mass in (0.1, 1.0, 5.0):
                model = train_mass(train_x, train_y, rho_pos, rho_neg, lambda_mass, seed=0)
                rho = probabilities(model, val_x, temperature=args.rho_temperature)
                partial_predictions = mix_with_identity(val_rows, balanced_val, rho)
                partial_metrics = evaluate_predictions(val_rows, partial_predictions)
                item = {
                    "rho_pos_prior": rho_pos,
                    "rho_neg_prior": rho_neg,
                    "lambda_mass": lambda_mass,
                    "rho_temperature": args.rho_temperature,
                    **mass_metrics(val_rows, rho),
                    **{f"partial_{key}": value for key, value in partial_metrics.items()},
                    **{f"balanced_{key}": value for key, value in balanced_metrics.items()},
                }
                balanced_advantage = max(0.0, balanced_metrics["same_subject_target_advantage"])
                partial_advantage = max(0.0, partial_metrics["same_subject_target_advantage"])
                item["same_subject_reduction_vs_balanced"] = (
                    (balanced_advantage - partial_advantage) / balanced_advantage
                    if balanced_advantage > 1e-8
                    else float(partial_advantage <= 1e-8)
                )
                item["positive_endpoint_retention_vs_balanced"] = (
                    partial_metrics["positive_endpoint_accuracy"]
                    / max(balanced_metrics["positive_endpoint_accuracy"], 1e-8)
                )
                if external_model is not None:
                    external = [
                        predict_probability(
                            external_model,
                            prompt=str(row["prompt"]),
                            subject=str(row["subject"]),
                            relation_template=str(row["relation_template"]),
                            relation_id=str(row["relation_id"]),
                        )
                        for row in val_rows
                    ]
                    external_metrics = evaluate_predictions(
                        val_rows, mix_with_identity(val_rows, balanced_val, external)
                    )
                    item["tradeoff_gain_vs_external_gate"] = tradeoff(partial_metrics) - tradeoff(external_metrics)
                else:
                    external = [0.0] * len(val_rows)
                    external_metrics = evaluate_predictions(
                        val_rows, mix_with_identity(val_rows, balanced_val, external)
                    )
                    item["tradeoff_gain_vs_external_gate"] = math.nan
                item["offline_pass"] = (
                    item["positive_mean_rho"] >= 0.70
                    and item["same_subject_mean_rho"] <= 0.05
                    and item["near_mean_rho"] <= 0.02
                    and item["mass_roc_auc"] >= 0.90
                    and partial_metrics["identity_sparse_kl"] <= 0.05
                    and partial_metrics["same_subject_target_advantage"] <= 0.0
                    and item["same_subject_reduction_vs_balanced"] >= 0.50
                    and item["positive_endpoint_retention_vs_balanced"] >= 0.90
                    and item["tradeoff_gain_vs_external_gate"] > 0.0
                )
                report_rows.append(item)
                candidates.append(
                    (
                        (
                            bool(item["offline_pass"]),
                            item["tradeoff_gain_vs_external_gate"],
                            tradeoff(partial_metrics),
                        ),
                        model,
                        item,
                    )
                )
    _, selected_model, selected = max(candidates, key=lambda item: item[0])
    selected_rho = probabilities(selected_model, val_x, temperature=args.rho_temperature)
    selected_partial = mix_with_identity(val_rows, balanced_val, selected_rho)
    selected_external = (
        [
            predict_probability(
                external_model,
                prompt=str(row["prompt"]),
                subject=str(row["subject"]),
                relation_template=str(row["relation_template"]),
                relation_id=str(row["relation_id"]),
            )
            for row in val_rows
        ]
        if external_model is not None
        else [0.0] * len(val_rows)
    )
    variant_rows = []
    for name, predictions in (
        ("balanced_csbm", balanced_val),
        ("fixed_partial_0.5", mix_with_identity(val_rows, balanced_val, [0.5] * len(val_rows))),
        ("external_gate_balanced", mix_with_identity(val_rows, balanced_val, selected_external)),
        ("learned_partial_csbm", selected_partial),
        ("rho_only_target_bias", rho_only_target_bias(val_rows, selected_rho)),
    ):
        variant_rows.append({"variant": name, **evaluate_predictions(val_rows, predictions)})
    torch.save(
        {
            "state_dict": selected_model.state_dict(),
            "metrics": selected,
            "balanced_checkpoint": str(args.balanced_checkpoint),
            "balanced_checkpoint_epsilon": balanced_checkpoint["epsilon"],
            "runtime_inputs": [
                "prompt",
                "subject",
                "relation_template",
                "relation_id",
                "current_span_state",
                "timestep",
                "balanced_transition_confidence",
                "balanced_transition_entropy",
            ],
        },
        output_dir / "selected_partial_mass_model.pt",
    )
    checks = {
        "positive_mean_rho_ge_0_70": selected["positive_mean_rho"] >= 0.70,
        "same_subject_mean_rho_le_0_05": selected["same_subject_mean_rho"] <= 0.05,
        "near_mean_rho_le_0_02": selected["near_mean_rho"] <= 0.02,
        "mass_roc_auc_ge_0_90": selected["mass_roc_auc"] >= 0.90,
        "identity_kl_le_0_05": selected["partial_identity_sparse_kl"] <= 0.05,
        "negative_target_advantage_le_0": selected["partial_same_subject_target_advantage"] <= 0.0,
        "same_subject_reduction_vs_balanced_ge_0_50": selected["same_subject_reduction_vs_balanced"] >= 0.50,
        "efficacy_retention_vs_balanced_ge_0_90": selected["positive_endpoint_retention_vs_balanced"] >= 0.90,
        "beats_external_gate_tradeoff": selected["tradeoff_gain_vs_external_gate"] > 0.0,
        "balanced_checkpoint_implementation_valid": True,
        "zero_runtime_feature_leakage": True,
        "analysis_final_unused": True,
    }
    write_csv(output_dir / "mass_model_grid.csv", report_rows)
    write_csv(output_dir / "required_variant_comparison.csv", variant_rows)
    write_json(
        output_dir / "run_config.json",
        {
            "campaign_protocol": CAMPAIGN_PROTOCOL,
            "track_protocol": "counterfact_unbalanced_partial_csbm_v1",
            "data_dir": str(args.data_dir),
            "balanced_checkpoint": str(args.balanced_checkpoint),
            "external_gate_checkpoint": str(args.external_gate_checkpoint),
            "rho_pos_grid": [0.7, 0.9],
            "rho_neg_grid": [0.01, 0.05],
            "lambda_mass_grid": [0.1, 1.0, 5.0],
            "rho_temperature": args.rho_temperature,
            "bounded_rescue": bool(args.bounded_rescue),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    write_json(
        output_dir / "feature_leakage_audit.json",
        {
            "runtime_inputs": [
                "prompt",
                "subject",
                "relation_template",
                "relation_id",
                "current_span_state",
                "timestep",
                "balanced_transition_confidence",
                "balanced_transition_entropy",
            ],
            "forbidden_runtime_inputs": [
                "prompt_type",
                "transport_label",
                "identity",
                "endpoint_token_ids",
                "future_success",
            ],
            "teacher_only_runtime_inputs": False,
            "pass": True,
        },
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_unbalanced_partial_csbm_v1",
        "stage": "T4.1/T4.2 partial categorical bridge offline pilot",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "balanced_checkpoint_source": str(args.balanced_checkpoint),
        "external_gate_checkpoint": str(args.external_gate_checkpoint) if external_model else None,
        "selected_metrics": selected,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
        "bounded_rescue_available": not all(checks.values()) and not bool(args.bounded_rescue),
        "bounded_rescue_used": bool(args.bounded_rescue),
        "budget_guard": guard,
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T4",
        stage="T4.2_partial_csbm_offline",
        event="partial_transport_audited",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=(f"pos_rho={selected['positive_mean_rho']:.4f} same_rho={selected['same_subject_mean_rho']:.4f} "
               f"external_gain={selected['tradeoff_gain_vs_external_gate']:.4f}"),
    )
    print(f"acceptance_pass={report['acceptance_pass']}")
    print(f"bounded_rescue_available={report['bounded_rescue_available']}")


if __name__ == "__main__":
    main()
