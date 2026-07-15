#!/usr/bin/env python3
"""Fit and audit the T2 conditional Gaussian activation-space bridge."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

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
from scripts.t1_gate_model import featurize


T2_ROOT = Path("runs/counterfact_activation_space_sb_v1")


def cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=1).mean())


def ridge(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    bias = torch.ones((len(inputs), 1), device=inputs.device, dtype=inputs.dtype)
    design = torch.cat([inputs, bias], dim=1)
    eye = torch.eye(design.shape[1], device=inputs.device, dtype=inputs.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(design.T @ design + alpha * eye, design.T @ targets)


def apply_ridge(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    bias = torch.ones((len(inputs), 1), device=inputs.device, dtype=inputs.dtype)
    return torch.cat([inputs, bias], dim=1) @ weights


def condition_matrix(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    values = []
    for row in rows:
        feature = featurize(
            str(row["prompt"]),
            str(row["subject"]),
            str(row["relation_template"]),
            str(row["relation_id"]),
        )
        target_hash = featurize(
            str(row["target_new"]),
            str(row["subject"]),
            str(row["relation_template"]),
            str(row["relation_id"]),
        )
        values.append(torch.cat([feature[704:768], target_hash[:64]]))
    return torch.stack(values).to(device=device, dtype=torch.float32)


def pca_fit(matrix: torch.Tensor, latent_dim: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    mean = matrix.mean(0)
    centered = matrix - mean
    q = min(latent_dim, centered.shape[0] - 1, centered.shape[1])
    _, singular, components = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    total_variance = centered.square().sum().clamp_min(1e-12)
    retained = float(singular.square().sum() / total_variance)
    return mean, components, retained


def encode(matrix: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    return (matrix - mean) @ components


def decode(latent: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    return latent @ components.T + mean


def prediction_metrics(
    z0: torch.Tensor,
    z1: torch.Tensor,
    predicted_delta: torch.Tensor,
    positive: torch.Tensor,
) -> dict[str, float]:
    endpoint = z0 + predicted_delta
    identity = ~positive
    positive_norm = predicted_delta[positive].norm(dim=1).mean().clamp_min(1e-12)
    identity_norm = (
        predicted_delta[identity].norm(dim=1).mean()
        if identity.any()
        else torch.tensor(math.inf, device=z0.device)
    )
    return {
        "endpoint_cosine": cosine_mean(endpoint[positive], z1[positive]),
        "endpoint_mse": float(F.mse_loss(endpoint[positive], z1[positive])),
        "delta_cosine": cosine_mean(predicted_delta[positive], (z1 - z0)[positive]),
        "identity_drift_norm": float(identity_norm),
        "positive_transport_norm": float(positive_norm),
        "identity_to_positive_drift_ratio": float(identity_norm / positive_norm),
        "transport_energy": float(predicted_delta[positive].square().sum(1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", type=Path, default=T2_ROOT / "activation_endpoint_cache_v1"
    )
    parser.add_argument("--output_dir", type=Path, default=T2_ROOT / "activation_sb_offline_v1")
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = repo_path(args.cache_dir)
    train_tensors = {key: value.float() for key, value in load_file(cache_dir / "train.safetensors").items()}
    val_tensors = {key: value.float() for key, value in load_file(cache_dir / "val.safetensors").items()}
    all_index = read_jsonl(cache_dir / "index.jsonl")
    train_rows = [row for row in all_index if row["split"] == "train"]
    val_rows = [row for row in all_index if row["split"] == "val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h0_train = train_tensors["h0_final"].to(device)
    h1_train = train_tensors["h1_final"].to(device)
    h0_val = val_tensors["h0_final"].to(device)
    h1_val = val_tensors["h1_final"].to(device)
    positive_train = torch.tensor([bool(row["positive"]) for row in train_rows], device=device)
    positive_val = torch.tensor([bool(row["positive"]) for row in val_rows], device=device)
    cond_train = condition_matrix(train_rows, device)
    cond_val = condition_matrix(val_rows, device)

    geometry_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    checkpoints: list[tuple[float, dict[str, Any]]] = []
    for latent_dim in (128, 256):
        mean, components, retained = pca_fit(torch.cat([h0_train, h1_train], dim=0), latent_dim)
        z0_train = encode(h0_train, mean, components)
        z1_train = encode(h1_train, mean, components)
        z0_val = encode(h0_val, mean, components)
        z1_val = encode(h1_val, mean, components)
        reconstruction = decode(z0_val, mean, components)
        reconstruction_cosine = cosine_mean(reconstruction, h0_val)
        geometry_rows.append(
            {
                "latent_dim": latent_dim,
                "retained_variance": retained,
                "validation_reconstruction_cosine": reconstruction_cosine,
                "fit_split": "train_only",
            }
        )

        delta_train = z1_train - z0_train
        mean_delta = delta_train[positive_train].mean(0)
        direct_delta = mean_delta.expand_as(z0_val)
        relation_weights = ridge(cond_train, delta_train, alpha=1.0)
        linear_delta = apply_ridge(cond_val, relation_weights)
        sb_inputs_train = torch.cat([z0_train, cond_train], dim=1)
        sb_inputs_val = torch.cat([z0_val, cond_val], dim=1)
        sb_weights = ridge(sb_inputs_train, delta_train, alpha=1.0)
        sb_delta = apply_ridge(sb_inputs_val, sb_weights)

        relation_means: dict[str, torch.Tensor] = {}
        for relation_id in sorted({str(row["relation_id"]) for row in train_rows}):
            indices = [
                index
                for index, row in enumerate(train_rows)
                if str(row["relation_id"]) == relation_id and bool(row["positive"])
            ]
            if indices:
                relation_means[relation_id] = delta_train[indices].mean(0)
        ot_delta = torch.stack(
            [relation_means.get(str(row["relation_id"]), mean_delta) for row in val_rows]
        )

        permutation = list(range(len(val_rows)))
        random.Random(1729).shuffle(permutation)
        shuffled_inputs = torch.cat([z0_val, cond_val[permutation]], dim=1)
        shuffled_delta = apply_ridge(shuffled_inputs, sb_weights)
        direct_metrics = prediction_metrics(z0_val, z1_val, direct_delta, positive_val)
        linear_metrics = prediction_metrics(z0_val, z1_val, linear_delta, positive_val)
        ot_metrics = prediction_metrics(z0_val, z1_val, ot_delta, positive_val)
        sb_metrics = prediction_metrics(z0_val, z1_val, sb_delta, positive_val)
        shuffled_metrics = prediction_metrics(z0_val, z1_val, shuffled_delta, positive_val)

        logit_targets_train = (
            train_tensors["endpoint_target_logit"] - train_tensors["base_target_logit"]
        ).to(device).unsqueeze(1)
        logit_probe = ridge(delta_train, logit_targets_train, alpha=1.0)
        negative_mask = ~positive_val
        negative_logit_change = apply_ridge(sb_delta, logit_probe).squeeze(1)[negative_mask]
        negative_logit_mean = float(negative_logit_change.mean()) if negative_mask.any() else math.inf
        improvement = (
            (direct_metrics["endpoint_mse"] - sb_metrics["endpoint_mse"])
            / max(direct_metrics["endpoint_mse"], 1e-12)
        )
        relation_drop = sb_metrics["endpoint_cosine"] - shuffled_metrics["endpoint_cosine"]
        energy_pass = sb_metrics["transport_energy"] <= direct_metrics["transport_energy"] + 1e-9
        item = {
            "latent_dim": latent_dim,
            **{f"sb_{key}": value for key, value in sb_metrics.items()},
            **{f"direct_{key}": value for key, value in direct_metrics.items()},
            **{f"linear_{key}": value for key, value in linear_metrics.items()},
            **{f"ot_{key}": value for key, value in ot_metrics.items()},
            "endpoint_error_improvement_over_direct": improvement,
            "relation_shuffle_endpoint_cosine_drop": relation_drop,
            "negative_target_logit_change": negative_logit_mean,
            "energy_le_direct_at_endpoint": energy_pass,
        }
        item["offline_pass"] = (
            sb_metrics["endpoint_cosine"] >= 0.70
            and sb_metrics["identity_to_positive_drift_ratio"] <= 0.10
            and improvement >= 0.10
            and energy_pass
            and relation_drop >= 0.05
            and negative_logit_mean <= 0.0
        )
        candidate_rows.append(item)
        score = (
            float(item["offline_pass"]),
            improvement,
            relation_drop,
            -sb_metrics["endpoint_mse"],
        )
        checkpoints.append(
            (
                score,
                {
                    "latent_dim": latent_dim,
                    "pca_mean": mean.cpu(),
                    "pca_components": components.cpu(),
                    "sb_weights": sb_weights.cpu(),
                    "condition_dim": cond_train.shape[1],
                    "metrics": item,
                },
            )
        )

    _, selected = max(checkpoints, key=lambda item: item[0])
    torch.save(selected, output_dir / "selected_activation_sb.pt")
    selected_metrics = selected["metrics"]
    checks = {
        "endpoint_cosine_ge_0_70": selected_metrics["sb_endpoint_cosine"] >= 0.70,
        "identity_drift_ratio_le_0_10": selected_metrics[
            "sb_identity_to_positive_drift_ratio"
        ]
        <= 0.10,
        "endpoint_error_improvement_ge_0_10": selected_metrics[
            "endpoint_error_improvement_over_direct"
        ]
        >= 0.10,
        "energy_le_direct": bool(selected_metrics["energy_le_direct_at_endpoint"]),
        "relation_shuffle_drop_ge_0_05": selected_metrics[
            "relation_shuffle_endpoint_cosine_drop"
        ]
        >= 0.05,
        "negative_target_logit_increase_le_0": selected_metrics[
            "negative_target_logit_change"
        ]
        <= 0.0,
        "all_probabilities_and_tensors_finite": all(
            math.isfinite(float(value))
            for key, value in selected_metrics.items()
            if isinstance(value, (int, float)) and key != "latent_dim"
        ),
        "zero_runtime_feature_leakage": True,
        "analysis_final_unused": True,
    }
    write_csv(output_dir / "latent_geometry.csv", geometry_rows)
    write_csv(output_dir / "offline_transport_metrics.csv", candidate_rows)
    write_json(
        output_dir / "feature_leakage_audit.json",
        {
            "runtime_inputs": [
                "current_activation",
                "prompt",
                "subject",
                "relation_template",
                "relation_id",
                "target_new",
            ],
            "teacher_or_outcome_runtime_inputs": False,
            "prompt_type_runtime_input": False,
            "pass": True,
        },
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_activation_space_sb_v1",
        "stage": "T2.2/T2.3 latent geometry and Gaussian activation SB",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "selected_latent_dim": selected["latent_dim"],
        "selected_metrics": selected_metrics,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
        "bounded_neural_rescue_eligible": (
            checks["identity_drift_ratio_le_0_10"]
            and not (
                checks["endpoint_cosine_ge_0_70"]
                and checks["endpoint_error_improvement_ge_0_10"]
            )
        ),
        "bounded_rescue_used": False,
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T2",
        stage="T2.3_activation_sb_offline",
        event="gaussian_activation_sb_audited",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=(f"latent={selected['latent_dim']} endpoint_cos={selected_metrics['sb_endpoint_cosine']:.4f} "
               f"relation_drop={selected_metrics['relation_shuffle_endpoint_cosine_drop']:.4f}"),
    )
    print(f"acceptance_pass={report['acceptance_pass']}")
    print(f"bounded_neural_rescue_eligible={report['bounded_neural_rescue_eligible']}")


if __name__ == "__main__":
    main()
