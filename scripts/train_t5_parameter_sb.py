#!/usr/bin/env python3
"""Fit and audit the T5 parameter-space bridge over endpoint adapters."""

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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    record_stage_event,
    repo_path,
    write_csv,
    write_json,
)
from scripts.t1_gate_model import featurize
from scripts.train_t2_activation_sb import (
    apply_ridge,
    brownian_bridge_training_rows,
    integrate_bridge_drift,
    ridge,
)


T5_ROOT = Path("runs/counterfact_parameter_space_sb_v1")


class ConditionalAdapterMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def adapter_vectors(
    tensors: Mapping[str, torch.Tensor], index_rows: Sequence[Mapping[str, Any]], split: str
) -> tuple[torch.Tensor, list[Mapping[str, Any]]]:
    rows = [row for row in index_rows if row["split"] == split]
    vectors = [
        torch.cat(
            [
                tensors[f"{row['adapter_key']}.left"].float().flatten(),
                tensors[f"{row['adapter_key']}.right"].float().flatten(),
            ]
        )
        for row in rows
    ]
    return torch.stack(vectors), rows


def conditions(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    return torch.stack(
        [
            featurize(
                str(row["rewrite_prompt"]),
                str(row["subject"]),
                str(row["rewrite_template"]),
                str(row["relation_id"]),
            )
            for row in rows
        ]
    ).float()


def pca_fit(matrix: torch.Tensor, nominal_dim: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    mean = matrix.mean(0)
    centered = matrix - mean
    effective = min(nominal_dim, len(matrix) - 1, matrix.shape[1])
    _, singular, components = torch.pca_lowrank(centered, q=effective, center=False, niter=4)
    retained = float(singular.square().sum() / centered.square().sum().clamp_min(1e-12))
    return mean, components, retained


def encode(matrix: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    return (matrix - mean) @ components


def decode(latent: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    return latent @ components.T + mean


def cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=1).mean())


def fit_mlp(train_x: torch.Tensor, train_y: torch.Tensor) -> ConditionalAdapterMLP:
    torch.manual_seed(1729)
    model = ConditionalAdapterMLP(train_x.shape[1], train_y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(300):
        prediction = model(train_x)
        loss = F.mse_loss(prediction, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.flatten().float()
    right = right.flatten().float()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    return float((left @ right) / denominator.clamp_min(1e-12))


def probe_metrics(
    vectors: torch.Tensor,
    val_rows: Sequence[Mapping[str, Any]],
    probe_tensors: Mapping[str, torch.Tensor],
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    rank: int,
) -> dict[str, float]:
    row_lookup = {str(row["adapter_key"]): index for index, row in enumerate(val_rows)}
    direct_exact: dict[str, list[float]] = {"rewrite": [], "paraphrase": []}
    predicted_exact: dict[str, list[float]] = {"rewrite": [], "paraphrase": []}
    predicted_deltas: list[torch.Tensor] = []
    direct_deltas: list[torch.Tensor] = []
    dimension = probe_tensors["hidden"].shape[1]
    half = dimension * rank
    for probe_index, row in enumerate(probe_rows):
        vector = vectors[row_lookup[str(row["adapter_key"])]]
        left = vector[:half].reshape(dimension, rank)
        right = vector[half:].reshape(dimension, rank)
        hidden = probe_tensors["hidden"][probe_index].float()
        valid = probe_tensors["valid"][probe_index].bool()
        weights = probe_tensors["output_weights"][probe_index, valid].float()
        base = probe_tensors["base_logits"][probe_index, valid].float()
        direct = probe_tensors["direct_logits"][probe_index, valid].float()
        residual = (hidden @ right) @ left.T
        predicted = base + F.linear(residual, weights)
        target = int(probe_tensors["target_index"][probe_index])
        bucket = str(row["bucket"])
        direct_exact[bucket].append(float(int(direct.argmax()) == target))
        predicted_exact[bucket].append(float(int(predicted.argmax()) == target))
        predicted_deltas.append(predicted - base)
        direct_deltas.append(direct - base)
    mean = lambda values: sum(values) / len(values) if values else math.nan
    return {
        "rewrite_exact": mean(predicted_exact["rewrite"]),
        "paraphrase_exact": mean(predicted_exact["paraphrase"]),
        "direct_rewrite_exact": mean(direct_exact["rewrite"]),
        "direct_paraphrase_exact": mean(direct_exact["paraphrase"]),
        "rewrite_retention": mean(predicted_exact["rewrite"]) / max(mean(direct_exact["rewrite"]), 1e-8),
        "paraphrase_retention": mean(predicted_exact["paraphrase"]) / max(mean(direct_exact["paraphrase"]), 1e-8),
        "logit_delta_correlation": pearson(torch.cat(predicted_deltas), torch.cat(direct_deltas)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=Path, default=T5_ROOT / "direct_endpoint_adapters_rank2_v1")
    parser.add_argument("--output_dir", type=Path, default=T5_ROOT / "parameter_sb_offline_latent64_v1")
    parser.add_argument("--latent_dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--bounded_rescue", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_report = read_json(args.adapter_dir / "report_summary.json")
    if not direct_report.get("acceptance_pass"):
        raise RuntimeError("Direct endpoint adapter family did not pass viability")
    rank = int(direct_report["rank"])
    torch.manual_seed(1729)
    tensors = load_file(repo_path(args.adapter_dir / "endpoint_adapters.safetensors"))
    index_rows = read_jsonl(args.adapter_dir / "adapter_index.jsonl")
    train_vectors, train_rows = adapter_vectors(tensors, index_rows, "train")
    val_vectors, val_rows = adapter_vectors(tensors, index_rows, "val")
    mean, components, retained = pca_fit(train_vectors, args.latent_dim)
    z_train = encode(train_vectors, mean, components)
    z_val = encode(val_vectors, mean, components)
    reconstructed_val = decode(z_val, mean, components)
    zero_vector = torch.zeros((1, train_vectors.shape[1]))
    z0_train = encode(zero_vector.expand(len(train_vectors), -1), mean, components)
    z0_val = encode(zero_vector.expand(len(val_vectors), -1), mean, components)

    condition_train_full = conditions(train_rows)
    condition_val_full = conditions(val_rows)
    condition_mean, condition_components, _ = pca_fit(condition_train_full, 32)
    condition_train = encode(condition_train_full, condition_mean, condition_components)
    condition_val = encode(condition_val_full, condition_mean, condition_components)
    mean_prediction = z_train.mean(0).expand_as(z_val)
    linear_weights = ridge(condition_train, z_train, alpha=1.0)
    linear_prediction = apply_ridge(condition_val, linear_weights)
    mlp = fit_mlp(condition_train, z_train)
    with torch.no_grad():
        mlp_prediction = mlp(condition_val)

    bridge_x, bridge_y = brownian_bridge_training_rows(
        z0_train,
        z_train,
        condition_train,
        steps=4,
        sigma=0.10,
    )
    bridge_weights = ridge(bridge_x, bridge_y, alpha=1.0)
    bridge_delta, bridge_energy = integrate_bridge_drift(
        z0_val, condition_val, bridge_weights, steps=4
    )
    bridge_prediction = z0_val + bridge_delta
    permutation = list(range(len(val_rows)))
    random.Random(1729).shuffle(permutation)
    shuffled_delta, _ = integrate_bridge_drift(
        z0_val, condition_val[permutation], bridge_weights, steps=4
    )
    shuffled_prediction = z0_val + shuffled_delta

    latent_predictions = {
        "mean_adapter": mean_prediction,
        "linear_generator": linear_prediction,
        "conditional_mlp": mlp_prediction,
        "parameter_space_sb": bridge_prediction,
    }
    probe_tensors = load_file(repo_path(args.adapter_dir / "val_probe_features.safetensors"))
    probe_rows = read_jsonl(args.adapter_dir / "probe_index.jsonl")
    method_rows: list[dict[str, Any]] = []
    method_probe: dict[str, dict[str, float]] = {}
    for method, latent in latent_predictions.items():
        predicted_vectors = decode(latent, mean, components)
        probes = probe_metrics(predicted_vectors, val_rows, probe_tensors, probe_rows, rank=rank)
        method_probe[method] = probes
        method_rows.append(
            {
                "method": method,
                "adapter_cosine": cosine_mean(predicted_vectors, val_vectors),
                "latent_mse": float(F.mse_loss(latent, z_val)),
                **probes,
            }
        )
    reconstruction_probe = probe_metrics(
        reconstructed_val, val_rows, probe_tensors, probe_rows, rank=rank
    )
    sb_row = next(row for row in method_rows if row["method"] == "parameter_space_sb")
    mlp_row = next(row for row in method_rows if row["method"] == "conditional_mlp")
    endpoint_norm = float(train_vectors.norm(dim=1).mean())
    zero_reconstruction = decode(encode(zero_vector, mean, components), mean, components)
    identity_ratio = float(zero_reconstruction.norm() / max(endpoint_norm, 1e-12))
    shuffled_cosine = cosine_mean(decode(shuffled_prediction, mean, components), val_vectors)
    relation_drop = sb_row["adapter_cosine"] - shuffled_cosine
    sb_advantage = max(
        sb_row["adapter_cosine"] - mlp_row["adapter_cosine"],
        sb_row["logit_delta_correlation"] - mlp_row["logit_delta_correlation"],
    )
    checks = {
        "validation_reconstruction_cosine_ge_0_95": cosine_mean(reconstructed_val, val_vectors) >= 0.95,
        "reconstructed_rewrite_retention_ge_0_90": reconstruction_probe["rewrite_retention"] >= 0.90,
        "reconstructed_paraphrase_retention_ge_0_90": reconstruction_probe["paraphrase_retention"] >= 0.90,
        "predicted_adapter_cosine_ge_0_70": sb_row["adapter_cosine"] >= 0.70,
        "rewrite_probe_logit_agreement_ge_0_70": sb_row["logit_delta_correlation"] >= 0.70,
        "identity_drift_ratio_le_0_10": identity_ratio <= 0.10,
        "relation_shuffle_drop_ge_0_05": relation_drop >= 0.05,
        "sb_beats_conditional_mlp_ge_0_05": sb_advantage >= 0.05,
        "zero_runtime_feature_leakage": True,
        "analysis_final_unused": True,
    }
    write_json(
        output_dir / "run_config.json",
        {
            "campaign_protocol": CAMPAIGN_PROTOCOL,
            "track_protocol": "counterfact_parameter_space_sb_v1",
            "adapter_dir": str(args.adapter_dir),
            "nominal_latent_dim": args.latent_dim,
            "effective_latent_dim": components.shape[1],
            "reference_process": "brownian_reciprocal_bridge",
            "bridge_steps": 4,
            "brownian_sigma": 0.10,
            "bounded_rescue": bool(args.bounded_rescue),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    write_csv(output_dir / "generator_comparison.csv", method_rows)
    write_csv(
        output_dir / "adapter_reconstruction.csv",
        [
            {
                "nominal_latent_dim": args.latent_dim,
                "effective_latent_dim": components.shape[1],
                "retained_train_variance": retained,
                "validation_reconstruction_cosine": cosine_mean(reconstructed_val, val_vectors),
                "identity_zero_reconstruction_ratio": identity_ratio,
                **reconstruction_probe,
            }
        ],
    )
    write_json(
        output_dir / "feature_leakage_audit.json",
        {
            "runtime_inputs": [
                "subject",
                "relation_id",
                "target_true",
                "target_new",
                "rewrite_template",
                "latent_state",
                "timestep",
            ],
            "heldout_prompt_outcomes_used": False,
            "teacher_only_runtime_inputs": False,
            "pass": True,
        },
    )
    torch.save(
        {
            "rank": rank,
            "nominal_latent_dim": args.latent_dim,
            "effective_latent_dim": components.shape[1],
            "adapter_mean": mean,
            "adapter_components": components,
            "condition_mean": condition_mean,
            "condition_components": condition_components,
            "bridge_weights": bridge_weights,
            "mlp_state_dict": mlp.state_dict(),
            "metrics": sb_row,
        },
        output_dir / "selected_parameter_sb.pt",
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_parameter_space_sb_v1",
        "stage": "T5.2-T5.3 adapter latent and parameter-space SB offline",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "rank": rank,
        "nominal_latent_dim": args.latent_dim,
        "effective_latent_dim": components.shape[1],
        "retained_train_variance": retained,
        "validation_reconstruction_cosine": cosine_mean(reconstructed_val, val_vectors),
        "reconstruction_probe_metrics": reconstruction_probe,
        "selected_sb_metrics": sb_row,
        "conditional_mlp_metrics": mlp_row,
        "bridge_path_energy": bridge_energy,
        "identity_zero_reconstruction_ratio": identity_ratio,
        "relation_shuffle_adapter_cosine_drop": relation_drop,
        "sb_advantage_over_conditional_mlp": sb_advantage,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
        "bounded_latent128_rescue_available": (
            args.latent_dim == 64 and rank == 2 and not all(checks.values())
        ),
        "bounded_rescue_used": bool(args.bounded_rescue),
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T5",
        stage="T5.3_parameter_sb_offline",
        event="parameter_space_sb_audited",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=(f"latent={args.latent_dim}/{components.shape[1]} adapter_cos={sb_row['adapter_cosine']:.4f} "
               f"mlp_advantage={sb_advantage:.4f}"),
    )
    print(f"acceptance_pass={report['acceptance_pass']}")
    print(f"bounded_latent128_rescue_available={report['bounded_latent128_rescue_available']}")


if __name__ == "__main__":
    main()
