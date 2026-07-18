#!/usr/bin/env python3
"""Run the complete frozen E1 smoke registry for the TRM campaign."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_trm_state_conditioned_protection import selected_shared_policy
from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_json,
    record_stage,
    record_stage_cost,
    write_csv,
    write_json,
)
from scripts.trm_editor import harmonic_mean


REQUIRED_METHODS = (
    "base",
    "prompt_memory",
    "target_logit_bias",
    "ordinary_mdm_memit",
    "partial_state_mdm_memit",
    "static_nullspace_partial_state_memit",
    "timerome_source_reproduction",
    "timerome_counterfact_fullmask",
    "timerome_counterfact_partial_state",
    "timerome_partial_state_state_bucketed",
    "timerome_partial_state_state_protected",
    "random_site_partial_state_residual",
    "fixed_site_partial_state_residual",
)


def execute(command: Sequence[str]) -> None:
    print("E1 launch:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(command), cwd=ROOT, check=True)


def locality_from_summary(summary: Mapping[str, Any]) -> float:
    return sum(
        float(summary.get(bucket, {}).get("expected_exact", 0.0))
        for bucket in ("near_locality", "far_locality")
    ) / 2.0


def normalized_report_row(
    method: str,
    path: Path,
    *,
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = dict(report or read_json(path / "report_summary.json"))
    base_summary = source.get("base_summary") or {}
    edited_summary = source.get("edited_summary") or {}
    base_locality = float(source.get("base_locality_exact", locality_from_summary(base_summary)))
    locality = float(source.get("locality_exact", locality_from_summary(edited_summary)))
    clipped = float(
        source.get(
            "clipped_self_normalized_locality",
            min(locality / max(base_locality, 1e-8), 1.0),
        )
    )
    rewrite = float(source.get("rewrite_exact", 0.0))
    paraphrase = float(source.get("declarative_paraphrase_exact", 0.0))
    same_subject = float(source.get("same_subject_tfpr", 0.0))
    selection = float(
        source.get("selection_score", harmonic_mean((rewrite, paraphrase, clipped)))
    )
    stress = float(
        source.get(
            "stress_aware_aggregate",
            harmonic_mean((rewrite, paraphrase, clipped, max(1.0 - same_subject, 0.0))),
        )
    )
    try:
        run_path = str(path.relative_to(ROOT))
    except ValueError:
        run_path = str(path)
    row = {
        "method": method,
        "run": run_path,
        "num_edits": int(source.get("num_edits", 20)),
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "target_token_f1": float(
            source.get(
                "target_token_f1",
                edited_summary.get("rewrite", {}).get("target_token_f1", 0.0),
            )
        ),
        "old_target_suppression": float(source.get("old_target_suppression", 0.0)),
        "same_subject_tfpr": same_subject,
        "near_tfpr": float(source.get("near_tfpr", 0.0)),
        "far_tfpr": float(source.get("far_tfpr", 0.0)),
        "generation_tfpr": float(source.get("generation_tfpr", 0.0)),
        "malformed_rate": float(source.get("malformed_rate", 0.0)),
        "base_locality_exact": base_locality,
        "locality_exact": locality,
        "clipped_self_normalized_locality": clipped,
        "selection_score": selection,
        "stress_aware_aggregate": stress,
        "gpu_minutes_per_edit": float(source.get("gpu_minutes_per_edit", 0.0)),
        "memory_storage_bytes": int(source.get("memory_storage_bytes", 0)),
        "runtime_schema_present": bool(source.get("runtime_feature_schema", True)),
        "all_metrics_finite": all(
            math.isfinite(value)
            for value in (
                rewrite,
                paraphrase,
                same_subject,
                float(source.get("near_tfpr", 0.0)),
                float(source.get("far_tfpr", 0.0)),
                float(source.get("malformed_rate", 0.0)),
                selection,
                stress,
            )
        ),
        "analysis_500_used": bool(source.get("analysis_500_used", False)),
        "final_test_used": bool(source.get("final_test_used", False)),
        "comparable_actual_decode": True,
    }
    return row


def base_row(path: Path) -> dict[str, Any]:
    report = read_json(path / "report_summary.json")
    summary = report["base_summary"]
    locality = locality_from_summary(summary)
    rewrite = float(summary.get("rewrite", {}).get("expected_exact", 0.0))
    paraphrase = float(
        summary.get("declarative_paraphrase", {}).get("expected_exact", 0.0)
    )
    same = float(summary.get("same_subject", {}).get("target_new_tfpr_or_exact", 0.0))
    return {
        "method": "base",
        "run": str(path.relative_to(ROOT)),
        "num_edits": int(report.get("num_edits", 20)),
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "target_token_f1": float(
            summary.get("rewrite", {}).get("target_token_f1", 0.0)
        ),
        "old_target_suppression": 0.0,
        "same_subject_tfpr": same,
        "near_tfpr": float(
            summary.get("near_locality", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "far_tfpr": float(
            summary.get("far_locality", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "generation_tfpr": float(
            summary.get("generation", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "malformed_rate": max(
            (float(value.get("malformed_rate", 0.0)) for value in summary.values()),
            default=0.0,
        ),
        "base_locality_exact": locality,
        "locality_exact": locality,
        "clipped_self_normalized_locality": 1.0,
        "selection_score": harmonic_mean((rewrite, paraphrase, 1.0)),
        "stress_aware_aggregate": harmonic_mean(
            (rewrite, paraphrase, 1.0, max(1.0 - same, 0.0))
        ),
        "gpu_minutes_per_edit": 0.0,
        "memory_storage_bytes": 0,
        "runtime_schema_present": True,
        "all_metrics_finite": True,
        "analysis_500_used": False,
        "final_test_used": False,
        "comparable_actual_decode": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "E1_smoke20_v1"
    )
    parser.add_argument(
        "--d1_dir",
        type=Path,
        default=CAMPAIGN_ROOT / "D1_partial_state_target_delta_v1",
    )
    parser.add_argument(
        "--d2_dir",
        type=Path,
        default=CAMPAIGN_ROOT / "D2_state_conditioned_protection_v1",
    )
    parser.add_argument(
        "--c2_dir",
        type=Path,
        default=CAMPAIGN_ROOT / "C2_fullmask_temporal_residual_v1",
    )
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    d1 = read_json(args.d1_dir / "report_summary.json")
    d2 = read_json(args.d2_dir / "report_summary.json")
    if not d1.get("acceptance_pass") or not d2.get("acceptance_pass"):
        raise RuntimeError("D1 and D2 pipeline integrity must pass before E1")
    if d2.get("relation_rescue_triggered") and not (
        CAMPAIGN_ROOT / "D2_relation_conditioned_rescue_v1" / "report_summary.json"
    ).exists():
        raise RuntimeError("The legally triggered relation rescue is incomplete")
    args.output_dir.mkdir(parents=True)
    manifest = PROTOCOL_ROOT / "cf_trm_smoke_20.jsonl"
    anchor = PROTOCOL_ROOT / "cf_trm_anchor_train_500.jsonl"
    d2_config = read_json(args.d2_dir / "run_config.json")
    layer = int(d2_config["layer"])
    schedule, reveal, consistency = selected_shared_policy(d1)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "E1_smoke20",
            "manifest": str(manifest),
            "required_methods": list(REQUIRED_METHODS),
            "selected_D1_partial_method": d1["selected_partial_method"],
            "partial_mask_schedule": schedule,
            "reveal_policy": reveal,
            "state_consistency_weight": consistency,
            "temporal_layer": layer,
            "fixed_source_compatible_layer": 3,
            "random_control_layer": 9,
            "alpha_grid_used": [1.0],
            "ridge_grid_used": [0.01],
            "sparsity_grid_used": [0, 256],
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    python = sys.executable
    covariance = args.output_dir / "train_only_covariance_layer6"
    execute(
        [
            python,
            str(ROOT / "scripts" / "build_trm_covariance.py"),
            "--anchor_manifest",
            str(anchor),
            "--output_dir",
            str(covariance),
            "--layer",
            str(layer),
        ]
    )
    for method in ("prompt_memory", "target_logit_bias"):
        execute(
            [
                python,
                str(ROOT / "scripts" / "run_dnpe_runtime_baseline.py"),
                "--campaign_id",
                CAMPAIGN_ID,
                "--method",
                method,
                "--manifest",
                str(manifest),
                "--output_dir",
                str(args.output_dir / method),
                "--guidance_scale",
                "2.0",
            ]
        )
    basis_cache = args.output_dir / "static_basis_cache"
    basis_cache.mkdir()
    basis_payload = torch.load(
        args.d2_dir / "static_protection_basis.pt",
        map_location="cpu",
        weights_only=True,
    )
    torch.save(
        basis_payload,
        basis_cache / f"layer_{layer}_variance_0.95_basis.pt",
    )
    memit_common = [
        python,
        str(ROOT / "scripts" / "run_dnpe_editor.py"),
        "--campaign_id",
        CAMPAIGN_ID,
        "--manifest",
        str(manifest),
        "--layers",
        str(layer),
        "--covariance_dir",
        str(covariance),
        "--old_target_suppression_weight",
        "0.25",
        "--update_ridge",
        "0.01",
    ]
    execute(
        memit_common
        + [
            "--output_dir",
            str(args.output_dir / "ordinary_mdm_memit"),
            "--method",
            "ordinary_mdm_memit",
            "--partial_mask_schedule",
            "fully_masked",
            "--reveal_policy",
            "random",
            "--state_consistency_weight",
            "0.0",
        ]
    )
    partial_args = [
        "--partial_mask_schedule",
        schedule,
        "--reveal_policy",
        reveal,
        "--state_consistency_weight",
        str(consistency),
    ]
    execute(
        memit_common
        + [
            "--output_dir",
            str(args.output_dir / "partial_state_mdm_memit"),
            "--method",
            "partial_state_mdm_memit",
        ]
        + partial_args
    )
    execute(
        memit_common
        + [
            "--output_dir",
            str(args.output_dir / "static_nullspace_partial_state_memit"),
            "--method",
            "static_nullspace_partial_state_memit",
            "--protected_basis_dir",
            str(basis_cache),
            "--protected_variance",
            "0.95",
        ]
        + partial_args
    )
    residual_specs = [
        {
            "method": "timerome_source_style_fullmask",
            "layer": 3,
            "state_mode": "shared",
            "schedule": "fully_masked",
            "reveal": "random",
            "protection": "none",
            "top_q": 0,
        },
        {
            "method": "timerome_partial_state_state_bucketed",
            "layer": layer,
            "state_mode": "bucketed",
            "schedule": schedule,
            "reveal": reveal,
            "protection": "none",
            "top_q": 0,
        },
        {
            "method": "random_site_partial_state_residual",
            "layer": 9,
            "state_mode": "shared",
            "schedule": schedule,
            "reveal": reveal,
            "protection": "none",
            "top_q": 0,
        },
        {
            "method": "fixed_site_partial_state_residual",
            "layer": 3,
            "state_mode": "shared",
            "schedule": schedule,
            "reveal": reveal,
            "protection": "none",
            "top_q": 0,
        },
    ]
    for spec in residual_specs:
        execute(
            [
                python,
                str(ROOT / "scripts" / "run_trm_editor_experiment.py"),
                "--manifest",
                str(manifest),
                "--anchor_manifest",
                str(anchor),
                "--output_dir",
                str(args.output_dir / spec["method"]),
                "--method",
                spec["method"],
                "--layer",
                str(spec["layer"]),
                "--state_mode",
                spec["state_mode"],
                "--protection_mode",
                spec["protection"],
                "--partial_mask_schedule",
                spec["schedule"],
                "--reveal_policy",
                spec["reveal"],
                "--state_consistency_weight",
                str(consistency),
                "--top_q",
                str(spec["top_q"]),
            ]
        )
    c2_smoke = args.c2_dir / "smoke20_v1" / "stable_temporal_top1"
    rows = [base_row(c2_smoke)]
    for method in ("prompt_memory", "target_logit_bias"):
        rows.append(normalized_report_row(method, args.output_dir / method))
    for method in (
        "ordinary_mdm_memit",
        "partial_state_mdm_memit",
        "static_nullspace_partial_state_memit",
    ):
        rows.append(normalized_report_row(method, args.output_dir / method))
    rows.append(normalized_report_row("timerome_counterfact_fullmask", c2_smoke))
    rows.append(
        normalized_report_row(
            "timerome_counterfact_partial_state",
            args.d2_dir / "unprotected_temporal_residual",
        )
    )
    rows.append(
        normalized_report_row(
            "timerome_partial_state_state_protected",
            args.d2_dir / "state_conditioned_preservation",
        )
    )
    rows.append(
        normalized_report_row(
            "timerome_partial_state_state_protected_sparse",
            args.d2_dir / "state_conditioned_sparsification",
        )
    )
    for spec in residual_specs:
        rows.append(
            normalized_report_row(spec["method"], args.output_dir / spec["method"])
        )
    if d2.get("relation_rescue_triggered"):
        rescue = CAMPAIGN_ROOT / "D2_relation_conditioned_rescue_v1"
        rows.append(
            normalized_report_row(
                "timerome_partial_state_state_relation_protected", rescue
            )
        )
    source_report = read_json(
        CAMPAIGN_ROOT / "C0_timerome_source_reproduction_v1" / "report_summary.json"
    )
    rows.append(
        {
            "method": "timerome_source_reproduction",
            "run": str(
                (CAMPAIGN_ROOT / "C0_timerome_source_reproduction_v1").relative_to(
                    ROOT
                )
            ),
            "num_edits": 0,
            "rewrite_exact": "",
            "declarative_paraphrase_exact": "",
            "target_token_f1": "",
            "old_target_suppression": "",
            "same_subject_tfpr": "",
            "near_tfpr": "",
            "far_tfpr": "",
            "generation_tfpr": "",
            "malformed_rate": "",
            "base_locality_exact": "",
            "locality_exact": "",
            "clipped_self_normalized_locality": "",
            "selection_score": "",
            "stress_aware_aggregate": "",
            "gpu_minutes_per_edit": 0.0,
            "memory_storage_bytes": 0,
            "runtime_schema_present": True,
            "all_metrics_finite": bool(source_report.get("acceptance_pass")),
            "analysis_500_used": False,
            "final_test_used": False,
            "comparable_actual_decode": False,
        }
    )
    by_method = {str(row["method"]): row for row in rows}
    missing = sorted(set(REQUIRED_METHODS) - set(by_method))
    comparable = [
        row for row in rows if row["comparable_actual_decode"] and row["method"] != "base"
    ]
    base = by_method["base"]
    viable = [
        row
        for row in comparable
        if row["all_metrics_finite"]
        and float(row["rewrite_exact"]) > float(base["rewrite_exact"])
        and float(row["malformed_rate"]) <= 0.05
        and float(row["same_subject_tfpr"]) <= 0.30
        and row["runtime_schema_present"]
    ]
    selected = max(
        viable,
        key=lambda row: (
            float(row["stress_aware_aggregate"]),
            float(row["selection_score"]),
            -float(row["gpu_minutes_per_edit"]),
            str(row["method"]),
        ),
    ) if viable else None
    red_failures = {
        "incomplete_required_registry": bool(missing),
        "no_rewrite_gain": not any(
            float(row["rewrite_exact"]) > float(base["rewrite_exact"])
            for row in comparable
        ),
        "no_candidate_with_malformed_at_most_0_05": not any(
            float(row["malformed_rate"]) <= 0.05 for row in comparable
        ),
        "no_candidate_with_same_subject_tfpr_at_most_0_30": not any(
            float(row["same_subject_tfpr"]) <= 0.30 for row in comparable
        ),
        "runtime_schema_mismatch": any(
            not row["runtime_schema_present"] for row in comparable
        ),
        "numerical_instability": any(not row["all_metrics_finite"] for row in comparable),
        "analysis_or_final_used": any(
            row["analysis_500_used"] or row["final_test_used"] for row in rows
        ),
    }
    passed = not any(red_failures.values()) and selected is not None
    write_csv(args.output_dir / "smoke_method_registry.csv", rows)
    write_json(
        args.output_dir / "smoke_selection.json",
        {
            "selected_for_E2": selected,
            "viable_methods": [str(row["method"]) for row in viable],
            "selection_role": "bounded_smoke_calibration_and_integration_only",
        },
    )
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "E1_smoke20",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "required_method_count": len(REQUIRED_METHODS),
        "reported_method_count": len(rows),
        "missing_required_methods": missing,
        "viable_method_count": len(viable),
        "selected_method_for_E2": selected["method"] if selected else None,
        "red_failure_checks": red_failures,
        "runtime_seconds": runtime,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": passed,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {"red_failure_checks": red_failures, "acceptance_pass": passed},
    )
    record_stage_cost(
        "E1_smoke20",
        runtime_seconds=runtime,
        notes="Complete smoke20 method registry and bounded integration calibration",
    )
    record_stage(
        "E1_smoke20",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"selected={selected['method'] if selected else None}; missing={missing}",
        next_stage="E2_pilot100" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(
        json.dumps(
            {
                "acceptance_pass": True,
                "selected_method_for_E2": selected["method"],
                "viable_method_count": len(viable),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
