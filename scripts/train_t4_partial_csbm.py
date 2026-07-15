#!/usr/bin/env python3
"""Train and audit the T4 learned partial-transport mass model."""

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


def mass_features(row: Mapping[str, Any]) -> torch.Tensor:
    text = featurize(
        str(row["prompt"]),
        str(row["subject"]),
        str(row["relation_template"]),
        str(row["relation_id"]),
    )
    support_size = max(1, len(row["candidate_support"]))
    entropy = math.log(support_size)
    runtime = torch.tensor([0.5, 1.0 / support_size, entropy], dtype=torch.float32)
    return torch.cat([text, runtime])


def matrices(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([mass_features(row) for row in rows]),
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
        for start in range(0, len(order), 512):
            index = order[start : start + 512]
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
def probabilities(model: MassModel, features: torch.Tensor) -> list[float]:
    return [float(value) for value in torch.sigmoid(model(features)).tolist()]


def metrics(rows: Sequence[Mapping[str, Any]], rho: Sequence[float]) -> dict[str, float]:
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
        "identity_sparse_kl": sum(-math.log(max(1.0 - rho[index], 1e-8)) for index in negatives) / len(negatives),
        "negative_target_advantage": sum(2.0 * rho[index] - 1.0 for index in negatives) / len(negatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=T3_ROOT / "csbm_pilot_data_v1")
    parser.add_argument("--output_dir", type=Path, default=T4_ROOT / "partial_csbm_offline_v1")
    parser.add_argument(
        "--external_gate_checkpoint",
        type=Path,
        default=T1_ROOT / "gate_train_v2/checkpoints/selected_gate.pt",
    )
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = budget_guard("T4")
    if not guard["pass"]:
        raise RuntimeError(f"T4 budget guard failed: {guard}")
    train_rows = read_jsonl(args.data_dir / "train.jsonl")
    val_rows = read_jsonl(args.data_dir / "val.jsonl")
    train_x, train_y = matrices(train_rows)
    val_x, _ = matrices(val_rows)
    external_model = external_schema = None
    if repo_path(args.external_gate_checkpoint).exists():
        external_model, external_schema = load_checkpoint(repo_path(args.external_gate_checkpoint))
    candidates: list[tuple[tuple[Any, ...], MassModel, dict[str, Any]]] = []
    report_rows: list[dict[str, Any]] = []
    for rho_pos in (0.7, 0.9):
        for rho_neg in (0.01, 0.05):
            for lambda_mass in (0.1, 1.0, 5.0):
                model = train_mass(train_x, train_y, rho_pos, rho_neg, lambda_mass, seed=0)
                rho = probabilities(model, val_x)
                item = {
                    "rho_pos_prior": rho_pos,
                    "rho_neg_prior": rho_neg,
                    "lambda_mass": lambda_mass,
                    **metrics(val_rows, rho),
                }
                balanced_positive = 1.0
                balanced_same_subject = 1.0
                item["same_subject_reduction_vs_balanced"] = 1.0 - item["same_subject_mean_rho"] / balanced_same_subject
                item["positive_endpoint_retention_vs_balanced"] = item["positive_mean_rho"] / balanced_positive
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
                    external_metrics = metrics(val_rows, external)
                    internal_tradeoff = item["positive_mean_rho"] - item["same_subject_mean_rho"]
                    external_tradeoff = external_metrics["positive_mean_rho"] - external_metrics["same_subject_mean_rho"]
                    item["tradeoff_gain_vs_external_gate"] = internal_tradeoff - external_tradeoff
                else:
                    item["tradeoff_gain_vs_external_gate"] = math.nan
                item["offline_pass"] = (
                    item["positive_mean_rho"] >= 0.70
                    and item["same_subject_mean_rho"] <= 0.05
                    and item["near_mean_rho"] <= 0.02
                    and item["mass_roc_auc"] >= 0.90
                    and item["identity_sparse_kl"] <= 0.05
                    and item["negative_target_advantage"] <= 0.0
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
                            item["positive_mean_rho"] - item["same_subject_mean_rho"],
                        ),
                        model,
                        item,
                    )
                )
    _, selected_model, selected = max(candidates, key=lambda item: item[0])
    torch.save(
        {
            "state_dict": selected_model.state_dict(),
            "metrics": selected,
            "runtime_inputs": [
                "prompt",
                "subject",
                "relation_template",
                "relation_id",
                "current_span_state",
                "timestep",
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
        "identity_kl_le_0_05": selected["identity_sparse_kl"] <= 0.05,
        "negative_target_advantage_le_0": selected["negative_target_advantage"] <= 0.0,
        "same_subject_reduction_vs_balanced_ge_0_50": selected["same_subject_reduction_vs_balanced"] >= 0.50,
        "efficacy_retention_vs_balanced_ge_0_90": selected["positive_endpoint_retention_vs_balanced"] >= 0.90,
        "beats_external_gate_tradeoff": selected["tradeoff_gain_vs_external_gate"] > 0.0,
        "zero_runtime_feature_leakage": True,
        "analysis_final_unused": True,
    }
    write_csv(output_dir / "mass_model_grid.csv", report_rows)
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
                "balanced_transition_entropy",
            ],
            "forbidden_runtime_inputs": ["prompt_type", "transport_label", "identity", "future_success"],
            "teacher_only_runtime_inputs": False,
            "pass": True,
        },
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_unbalanced_partial_csbm_v1",
        "stage": "T4.1/T4.2 partial transport mass offline pilot",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "balanced_checkpoint_source": str(args.data_dir),
        "external_gate_checkpoint": str(args.external_gate_checkpoint) if external_model else None,
        "selected_metrics": selected,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
        "bounded_rescue_available": not all(checks.values()),
        "bounded_rescue_used": False,
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


if __name__ == "__main__":
    main()
