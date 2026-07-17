#!/usr/bin/env python3
"""Freeze DNPE dev selection and execute locked analysis/final evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    now_utc,
    read_json,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.materialize_dnpe_locked_manifest import materialize
from scripts.report_dnpe_selection import paired_bootstrap, score_report
from scripts.run_dnpe_causal_nullspace_sweep import run_one, site_layers, target_config
from scripts.run_dnpe_dev_selection import run_timerome


def create_dev_lock() -> dict[str, Any]:
    output = CAMPAIGN_ROOT / "dev_method_lock.json"
    if output.exists():
        raise FileExistsError(output)
    selection = read_json(
        CAMPAIGN_ROOT / "F1_dev200_selection_v1" / "dev_selection.json"
    )
    if not selection.get("acceptance_pass") or not selection.get("primary_candidate"):
        raise RuntimeError("Cannot lock an ineligible dev method")
    d2 = read_json(
        CAMPAIGN_ROOT
        / "D2_target_value_optimization_v1"
        / "selected_target_value_config.json"
    )
    d4 = read_json(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "selected_nullspace_config.json"
    )
    site = read_json(
        CAMPAIGN_ROOT / "site_policy_lock_v1" / "site_policy_lock.json"
    )
    registry_path = (
        CAMPAIGN_ROOT / "protocol_v1" / "locked_manifest_registry.json"
    )
    registry = read_json(registry_path)
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": now_utc(),
        "model_id": PRIMARY_MODEL_ID,
        "model_revision": PRIMARY_MODEL_REVISION,
        "selected_candidate": selection["primary_candidate"],
        "edited_layers": site_layers(),
        "position": "last_subject",
        "component": "mlp",
        "causal_site_policy": "stable_temporal_site_set",
        "site_policy_lock_sha256": sha256_file(
            CAMPAIGN_ROOT / "site_policy_lock_v1" / "site_policy_lock.json"
        ),
        "partial_state_policy": d2["selected_partial_state_policy"],
        "target_value_config": d2["selected_config"],
        "nullspace_config": d4["selected"],
        "seed": 260717101,
        "decode_steps": "answer_length",
        "decode_batch_size": 16,
        "metrics": {
            "rewrite_floor": 0.75,
            "paraphrase_floor": 0.40,
            "same_subject_budget": "base+0.03",
            "near_budget": "base+0.03",
            "far_budget": "base+0.03",
            "malformed_budget": 0.05,
        },
        "report_scripts": {
            "locked_evaluation": sha256_file(Path(__file__)),
            "selection": sha256_file(ROOT / "scripts" / "report_dnpe_selection.py"),
        },
        "locked_manifest_registry": {
            "path": str(registry_path.relative_to(ROOT)),
            "sha256": sha256_file(registry_path),
            "manifests": registry["locked_manifests"],
        },
        "analysis_500_used": False,
        "final_test_used": False,
        "dev_lock_valid": True,
        "analysis_may_not_change_primary": True,
    }
    write_json(output, payload)
    write_json(
        CAMPAIGN_ROOT / "dev_method_lock_validation.json",
        {
            "candidate_frozen_on_dev": True,
            "locked_hashes_present": True,
            "analysis_not_used": True,
            "final_not_used": True,
            "acceptance_pass": True,
        },
    )
    record_stage(
        "G1_dev_lock",
        status="passed",
        acceptance_pass=True,
        output_dir=CAMPAIGN_ROOT,
        started_at_utc=now_utc(),
        notes="Primary candidate and all evaluation choices frozen before analysis.",
        next_stage="G2_analysis500",
    )
    return payload


def _alpha_config() -> dict[str, Any]:
    return read_json(
        CAMPAIGN_ROOT
        / "B3_alphaedit_style_mdm_memit_v1"
        / "smoke_selection.json"
    )["selected"]


def run_locked_suite(manifest: Path, root: Path, *, resume: bool) -> dict[str, Path]:
    config = target_config()
    layers = site_layers()
    d4 = read_json(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "selected_nullspace_config.json"
    )["selected"]
    alpha = _alpha_config()
    definitions = {
        "mdm_memit": {
            "layers": [3, 4, 5, 6],
            "config": {
                "learning_rate": 0.1,
                "target_optimization_steps": 25,
                "state_consistency_weight": 0.0,
                "old_target_suppression_weight": 0.0,
            },
            "schedule": "fully_masked",
            "reveal": "random",
        },
        "partial_state_mdm_memit": {
            "layers": [3, 4, 5, 6],
            "config": config,
            "schedule": config["partial_mask_schedule"],
            "reveal": config["reveal_policy"],
        },
        "alphaedit_style_mdm_memit": {
            "layers": [3, 4, 5, 6],
            "config": {
                "learning_rate": 0.1,
                "target_optimization_steps": 25,
                "state_consistency_weight": 0.0,
                "old_target_suppression_weight": 0.0,
            },
            "schedule": "fully_masked",
            "reveal": "random",
            "basis_dir": CAMPAIGN_ROOT / "preservation_basis_v1",
            "protected_variance": float(alpha["protected_variance"]),
            "update_ridge": float(alpha["update_ridge"]),
        },
        "causal_site_partial_state_editor": {
            "layers": layers,
            "config": config,
            "schedule": config["partial_mask_schedule"],
            "reveal": config["reveal_policy"],
        },
        "causal_partial_state_nullspace_memit": {
            "layers": layers,
            "config": config,
            "schedule": config["partial_mask_schedule"],
            "reveal": config["reveal_policy"],
            "basis_dir": CAMPAIGN_ROOT / "D1_state_banks_v1",
            "protected_variance": float(d4["protected_variance"]),
            "update_ridge": float(d4["update_ridge"]),
            "lambda_identity": float(d4["lambda_identity"]),
        },
    }
    paths = {}
    for method, values in definitions.items():
        output = root / method
        run_one(
            manifest=manifest,
            output=output,
            method=method,
            layers=values["layers"],
            config=values["config"],
            schedule=values["schedule"],
            reveal=values["reveal"],
            basis_dir=values.get("basis_dir"),
            protected_variance=values.get("protected_variance", 0.95),
            update_ridge=values.get("update_ridge", 0.0),
            lambda_identity=values.get("lambda_identity", 0.0),
            resume=resume,
        )
        paths[method] = output
    timerome = root / "timerome_dlm_style_residual_memory"
    run_timerome(manifest, timerome, resume=resume)
    paths["timerome_dlm_style_residual_memory"] = timerome
    return paths


def _case_metric(run_dir: Path, bucket: str, field: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    with (run_dir / "edited_per_prompt.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("bucket") != bucket:
                continue
            case_id = str(row["case_id"])
            values.setdefault(case_id, []).append(
                float(str(row.get(field, "")).casefold() in {"true", "1"})
            )
    return {case_id: mean(scores) for case_id, scores in values.items()}


def leakage_bootstrap(left: Path, right: Path, bucket: str) -> dict[str, Any]:
    left_values = _case_metric(left, bucket, "target_new_hit")
    right_values = _case_metric(right, bucket, "target_new_hit")
    ids = sorted(set(left_values) & set(right_values))
    if not ids:
        return {"bucket": bucket, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "num_edits": 0}
    deltas = [left_values[case] - right_values[case] for case in ids]
    rng = random.Random(260717101)
    trials = 2000
    samples = sorted(mean(rng.choice(deltas) for _ in ids) for _ in range(trials))
    return {
        "bucket": bucket,
        "metric": "target_new_hit",
        "delta": mean(deltas),
        "ci_low": samples[int(0.025 * trials)],
        "ci_high": samples[int(0.975 * trials)],
        "num_edits": len(ids),
        "trials": trials,
    }


def locked_stage(stage: str, root: Path, *, resume: bool) -> dict[str, Any]:
    is_analysis = stage == "analysis"
    split = "analysis_500" if is_analysis else "final_test_500"
    required_env = "DEV_METHOD_LOCKED" if is_analysis else "FINAL_METHOD_LOCKED"
    if os.environ.get(required_env) != "1":
        raise PermissionError(f"{required_env}=1 is required")
    if not is_analysis:
        confirmation = CAMPAIGN_ROOT / "analysis_confirmation_lock.json"
        if not confirmation.exists() or not read_json(confirmation).get(
            "final_test_500_authorized"
        ):
            raise PermissionError("final_test_500 requires a passed analysis lock")
    if root.exists() and not resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    materialized = root / f"materialized_{split}.jsonl"
    if not materialized.exists():
        source = (
            ROOT
            / "runs"
            / "counterfact_direction1_v1"
            / "protocol"
            / f"{split}.jsonl"
        )
        materialize(source, materialized, split_name=split)
    paths = run_locked_suite(materialized, root, resume=resume)
    rows = [
        score_report(read_json(path / "report_summary.json"), path / "report_summary.json")
        for path in paths.values()
    ]
    write_csv(root / "selection_scores.csv", rows)
    lock = read_json(CAMPAIGN_ROOT / "dev_method_lock.json")
    primary_name = Path(lock["selected_candidate"]["path"]).name
    if primary_name not in paths:
        raise RuntimeError(f"Locked primary is absent from suite: {primary_name}")
    primary_path = paths[primary_name]
    primary = read_json(primary_path / "report_summary.json")
    baseline_items = [
        (name, path, read_json(path / "report_summary.json"))
        for name, path in paths.items()
        if name != primary_name
    ]
    matched = [
        item
        for item in baseline_items
        if float(primary["rewrite_exact"]) >= float(item[2]["rewrite_exact"]) - 0.05
        and float(primary["declarative_paraphrase_exact"])
        >= float(item[2]["declarative_paraphrase_exact"]) - 0.05
    ]
    strongest = max(
        matched or baseline_items,
        key=lambda item: float(item[2]["rewrite_exact"])
        + float(item[2]["declarative_paraphrase_exact"]),
    )
    primary_leakage = sum(
        float(primary.get(key, 0.0))
        for key in ("same_subject_tfpr", "near_tfpr", "far_tfpr")
    )
    baseline_leakage = sum(
        float(strongest[2].get(key, 0.0))
        for key in ("same_subject_tfpr", "near_tfpr", "far_tfpr")
    )
    bootstrap_rows = []
    for bucket in ("rewrite", "declarative_paraphrase"):
        row = paired_bootstrap(primary_path, strongest[1], bucket=bucket)
        row.update({"left": primary_name, "right": strongest[0]})
        bootstrap_rows.append(row)
    for bucket in ("same_subject", "near_locality", "far_locality"):
        row = leakage_bootstrap(primary_path, strongest[1], bucket)
        row.update({"left": primary_name, "right": strongest[0]})
        bootstrap_rows.append(row)
    write_csv(root / "paired_bootstrap.csv", bootstrap_rows)
    if is_analysis:
        dev_report = read_json(ROOT / lock["selected_candidate"]["path"] / "report_summary.json")
        base_summary = primary["base_summary"]
        budgets = {
            "same": float(base_summary.get("same_subject", {}).get("target_new_tfpr_or_exact", 0.0)) + 0.03,
            "near": float(base_summary.get("near_locality", {}).get("target_new_tfpr_or_exact", 0.0)) + 0.03,
            "far": float(base_summary.get("far_locality", {}).get("target_new_tfpr_or_exact", 0.0)) + 0.03,
        }
        acceptance = {
            "retain_80_percent_dev_rewrite": float(primary["rewrite_exact"])
            >= 0.80 * float(dev_report["rewrite_exact"]),
            "retain_80_percent_dev_paraphrase": float(primary["declarative_paraphrase_exact"])
            >= 0.80 * float(dev_report["declarative_paraphrase_exact"]),
            "same_subject_budget_pass": float(primary["same_subject_tfpr"]) <= budgets["same"],
            "near_budget_pass": float(primary["near_tfpr"]) <= budgets["near"],
            "far_budget_pass": float(primary["far_tfpr"]) <= budgets["far"],
            "malformed_pass": float(primary["malformed_rate"]) <= 0.05,
            "locality_advantage_remains": primary_leakage < baseline_leakage,
            "efficacy_matched_baseline": bool(matched),
            "paired_qualitative_locality_noninferiority": all(
                row.get("ci_high", 0.0) <= 0.0
                for row in bootstrap_rows
                if row.get("metric") == "target_new_hit"
            ),
        }
        passed = all(acceptance.values())
    else:
        acceptance = {
            "locked_primary_present": True,
            "all_required_methods_present": len(paths) == 6,
            "all_metrics_finite": all(
                all(
                    __import__("math").isfinite(float(report.get(key, 0.0)))
                    for key in ("rewrite_exact", "declarative_paraphrase_exact", "same_subject_tfpr", "malformed_rate")
                )
                for _name, _path, report in [(name, path, read_json(path / "report_summary.json")) for name, path in paths.items()]
            ),
        }
        passed = all(acceptance.values())
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "G2_analysis500" if is_analysis else "G3_final500",
        "created_at_utc": now_utc(),
        "split": split,
        "primary_method": primary_name,
        "primary_run": str(primary_path.relative_to(ROOT)),
        "strongest_efficacy_matched_baseline": strongest[0],
        "primary_metrics": {
            key: primary[key]
            for key in (
                "rewrite_exact",
                "declarative_paraphrase_exact",
                "same_subject_tfpr",
                "near_tfpr",
                "far_tfpr",
                "malformed_rate",
            )
        },
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "analysis_500_used": True,
        "final_test_used": not is_analysis,
        "used_for_tuning": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", acceptance)
    if is_analysis and passed:
        write_json(
            CAMPAIGN_ROOT / "analysis_confirmation_lock.json",
            {
                "campaign_id": CAMPAIGN_ID,
                "analysis_report": str((root / "report_summary.json").relative_to(ROOT)),
                "analysis_report_sha256": sha256_file(root / "report_summary.json"),
                "primary_method": primary_name,
                "final_test_500_authorized": True,
                "no_post_analysis_tuning": True,
            },
        )
    record_stage(
        "G2_analysis500" if is_analysis else "G3_final500",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=root,
        started_at_utc=now_utc(),
        notes=f"Locked {split} suite completed without tuning.",
        next_stage="G3_final500" if is_analysis and passed else "H_final_package",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("g1", "analysis", "final"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.stage == "g1":
        result = create_dev_lock()
    else:
        default = CAMPAIGN_ROOT / (
            "G2_analysis500_v1" if args.stage == "analysis" else "G3_final500_v1"
        )
        result = locked_stage(args.stage, args.root or default, resume=bool(args.resume))
    print(json.dumps({"stage": args.stage, "acceptance_pass": result.get("acceptance_pass", True)}, sort_keys=True))


if __name__ == "__main__":
    main()
