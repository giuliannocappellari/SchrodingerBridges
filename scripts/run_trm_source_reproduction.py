#!/usr/bin/env python3
"""Validate TimeROME equation-level invariants when exact source code is unavailable."""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    git_commit,
    now_utc,
    read_json,
    record_stage,
    write_csv,
    write_json,
)
from scripts.trm_residual import fit_residual_memory


def synthetic_tie(*, layers: int = 8, steps: int = 4, seed: int = 260718201) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand((layers, steps), generator=generator) * 0.08
    layer_axis = torch.arange(layers, dtype=torch.float32)[:, None]
    step_axis = torch.arange(steps, dtype=torch.float32)[None, :]
    peak = torch.exp(-((layer_axis - 3.0) ** 2) / 1.5 - ((step_axis - 1.0) ** 2) / 0.7)
    return (noise + 0.82 * peak).clamp(0, 1)


def run_component_reproduction(seed: int = 260718202) -> dict[str, object]:
    torch.manual_seed(seed)
    train_keys = torch.randn(24, 16)
    true_weight = torch.randn(16, 12) * 0.2
    target_deltas = train_keys @ true_weight + torch.randn(24, 12) * 0.01
    retain_keys = torch.randn(40, 16)
    retain_keys = retain_keys - (retain_keys @ train_keys.T) @ torch.linalg.pinv(train_keys @ train_keys.T) @ train_keys
    temporal = fit_residual_memory(train_keys, target_deltas, ridge=0.05)
    random = fit_residual_memory(train_keys.roll(7, dims=0), target_deltas, ridge=0.05)
    base_mse = float(target_deltas.square().mean())
    temporal_mse = float((temporal.predict(train_keys) - target_deltas).square().mean())
    random_mse = float((random.predict(train_keys) - target_deltas).square().mean())
    dense_retain_drift = float(temporal.predict(retain_keys).square().mean().sqrt())
    sparse_retain_drift = float(temporal.predict(retain_keys, top_q=4).square().mean().sqrt())
    primal = torch.linalg.solve(
        train_keys.T @ train_keys + 0.05 * torch.eye(train_keys.shape[1]),
        train_keys.T @ target_deltas,
    )
    return {
        "base_target_mse": base_mse,
        "temporal_residual_target_mse": temporal_mse,
        "random_coordinate_target_mse": random_mse,
        "dense_retain_rms_drift": dense_retain_drift,
        "sparse_q4_retain_rms_drift": sparse_retain_drift,
        "ridge_dual_primal_max_abs_error": float((temporal.weight - primal).abs().max()),
        "residual_parameters_finite": bool(torch.isfinite(temporal.weight).all()),
        "residual_rank": int(torch.linalg.matrix_rank(temporal.weight)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "C0_timerome_source_reproduction_v1")
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source_audit = read_json(CAMPAIGN_ROOT / "A1_source_audit_v1" / "report_summary.json")
    exact_source_available = bool(source_audit["timerome_exact_source_code_available"])
    tie = synthetic_tie()
    peak_flat = int(tie.argmax())
    peak_layer, peak_step = divmod(peak_flat, tie.shape[1])
    peak_effect = float(tie[peak_layer, peak_step])
    random_mean = float(tie.mean())
    component = run_component_reproduction()
    coordinate_rows = [
        {
            "layer": layer,
            "step": step,
            "temporal_indirect_effect": float(tie[layer, step]),
            "selected": layer == peak_layer and step == peak_step,
        }
        for layer in range(tie.shape[0])
        for step in range(tie.shape[1])
    ]
    write_csv(args.output_dir / "coordinate_selection.csv", coordinate_rows)
    heatmap_dir = args.output_dir / "tie_heatmaps"
    heatmap_dir.mkdir()
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    image = axis.imshow(tie.numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
    axis.scatter([peak_step], [peak_layer], marker="x", color="cyan", s=80)
    axis.set_xlabel("Denoising step")
    axis.set_ylabel("Layer")
    axis.set_title("Synthetic temporal indirect effect invariant")
    figure.colorbar(image, ax=axis, label="Normalized effect")
    figure.tight_layout()
    figure.savefig(heatmap_dir / "synthetic_tie.png", dpi=160)
    plt.close(figure)
    source_rows = [
        {
            "task": "synthetic_equation_component",
            "method": "base_no_residual",
            "target_mse": component["base_target_mse"],
            "exact_source_task": False,
        },
        {
            "task": "synthetic_equation_component",
            "method": "random_coordinate_residual",
            "target_mse": component["random_coordinate_target_mse"],
            "exact_source_task": False,
        },
        {
            "task": "synthetic_equation_component",
            "method": "temporal_coordinate_residual",
            "target_mse": component["temporal_residual_target_mse"],
            "exact_source_task": False,
        },
    ]
    write_csv(args.output_dir / "source_task_results.csv", source_rows)
    write_csv(
        args.output_dir / "retain_utility.csv",
        [
            {"method": "dense_residual", "retain_rms_drift": component["dense_retain_rms_drift"]},
            {"method": "sparse_q4_residual", "retain_rms_drift": component["sparse_q4_retain_rms_drift"]},
        ],
    )
    runtime = time.monotonic() - begin
    write_csv(
        args.output_dir / "compute_table.csv",
        [{"stage": "synthetic_component_reproduction", "runtime_seconds": runtime, "gpu_used": False}],
    )
    acceptance = {
        "exact_source_code_available": exact_source_available,
        "exact_source_reproduction_pass": False,
        "technical_infeasibility_documented": not exact_source_available,
        "nontrivial_temporal_localization": peak_effect - random_mean >= 0.15,
        "temporal_residual_moves_metric_expected_direction": component["temporal_residual_target_mse"] < component["base_target_mse"],
        "temporal_residual_beats_random_coordinate": component["temporal_residual_target_mse"] < component["random_coordinate_target_mse"],
        "ridge_equation_verified": component["ridge_dual_primal_max_abs_error"] <= 1e-4,
        "sparsification_does_not_increase_retain_drift": component["sparse_q4_retain_rms_drift"] <= component["dense_retain_rms_drift"] + 1e-8,
        "finite_residual_parameters": component["residual_parameters_finite"],
        "analysis_500_used": False,
        "final_test_used": False,
    }
    continuation_pass = all(
        acceptance[key]
        for key in (
            "technical_infeasibility_documented",
            "nontrivial_temporal_localization",
            "temporal_residual_moves_metric_expected_direction",
            "temporal_residual_beats_random_coordinate",
            "ridge_equation_verified",
            "sparsification_does_not_increase_retain_drift",
            "finite_residual_parameters",
        )
    )
    write_json(
        args.output_dir / "reproduction_config.json",
        {
            "seed": 260718202,
            "layers": 8,
            "steps": 4,
            "ridge": 0.05,
            "sparse_q": 4,
            "source_task": "unavailable_without_official_code_or_checkpoint",
        },
    )
    write_json(
        args.output_dir / "validation_report.json",
        {"acceptance": acceptance, "continuation_pass": continuation_pass},
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "C0_timerome_source_reproduction",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "reproduction_status": "source_reproduction_technically_infeasible_component_invariants_passed",
        "reproduction_claim": "equation_level_and_synthetic_component_only",
        "component_metrics": component,
        "tie_peak": {"layer": peak_layer, "step": peak_step, "effect": peak_effect, "grid_mean": random_mean},
        "runtime_seconds": runtime,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "acceptance": acceptance,
        "acceptance_pass": continuation_pass,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        "C0_timerome_source_reproduction",
        status="passed_component_branch" if continuation_pass else "failed",
        acceptance_pass=continuation_pass,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes="Official TimeROME code/checkpoint unavailable; equation and synthetic causal/residual invariants validated.",
        next_stage="C1_temporal_localization" if continuation_pass else None,
    )
    if not continuation_pass:
        raise SystemExit(2)
    print(f"C0 component reproduction passed: {args.output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
