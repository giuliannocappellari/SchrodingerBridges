#!/usr/bin/env python3
"""Run the frozen D3 causal-update comparison and staged D4 null-space sweep."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    now_utc,
    read_json,
    read_jsonl,
    record_stage,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.run_dnpe_partial_state_sweep import POLICIES


def target_config() -> dict[str, Any]:
    report = read_json(
        CAMPAIGN_ROOT
        / "D2_target_value_optimization_v1"
        / "selected_target_value_config.json"
    )
    policy = str(report["selected_partial_state_policy"])
    schedule, reveal, _consistency, _old = POLICIES[policy]
    return {
        **report["selected_config"],
        "partial_state_policy": policy,
        "partial_mask_schedule": schedule,
        "reveal_policy": reveal,
    }


def site_layers() -> list[int]:
    lock = read_json(
        CAMPAIGN_ROOT / "site_policy_lock_v1" / "site_policy_lock.json"
    )
    policy = next(
        row
        for row in lock["policies"]
        if row["policy_id"] == "stable_temporal_site_set"
    )
    return list(map(int, policy["layers"]))


def combined_kamel_smoke(path: Path) -> Path:
    rows = []
    for length in (2, 3, 4):
        rows.extend(
            read_jsonl(
                CAMPAIGN_ROOT
                / "protocol_v1"
                / f"dnpe_kamel_smoke_20_n{length}.jsonl"
            )
        )
    if len(rows) != 60 or len({row["case_id"] for row in rows}) != 60:
        raise RuntimeError("Combined KAMEL smoke must contain 60 unique edits")
    if path.exists():
        existing = read_jsonl(path)
        if [row["case_id"] for row in existing] != [row["case_id"] for row in rows]:
            raise RuntimeError("Existing combined KAMEL smoke differs from fresh protocol")
        return path
    write_jsonl(path, rows)
    return path


def run_one(
    *,
    manifest: Path,
    output: Path,
    method: str,
    layers: Sequence[int],
    config: Mapping[str, Any],
    schedule: str,
    reveal: str,
    basis_dir: Path | None = None,
    protected_variance: float = 0.95,
    update_ridge: float = 0.0,
    lambda_identity: float = 0.0,
    resume: bool,
) -> None:
    if output.exists():
        if resume and (output / "report_summary.json").exists():
            return
        raise FileExistsError(output)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_dnpe_editor.py"),
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output),
        "--method",
        method,
        "--layers",
        ",".join(map(str, layers)),
        "--partial_mask_schedule",
        schedule,
        "--reveal_policy",
        reveal,
        "--learning_rate",
        str(config["learning_rate"]),
        "--target_optimization_steps",
        str(config["target_optimization_steps"]),
        "--state_consistency_weight",
        str(config["state_consistency_weight"]),
        "--old_target_suppression_weight",
        str(config["old_target_suppression_weight"]),
        "--update_ridge",
        str(update_ridge),
        "--lambda_identity",
        str(lambda_identity),
        "--include_locality",
        "1",
        "--decode_batch_size",
        "16",
    ]
    if basis_dir is not None:
        command.extend(
            [
                "--protected_basis_dir",
                str(basis_dir),
                "--protected_variance",
                str(protected_variance),
            ]
        )
    subprocess.run(command, cwd=ROOT, check=True)


def update_geometry(run_dir: Path) -> dict[str, float | bool]:
    diagnostics = read_json(run_dir / "target_value_diagnostics.json")
    updates = diagnostics["layer_updates"]
    norms = [float(row["update_norm"]) for row in updates]
    ranks = [
        int((row.get("projection") or {}).get("protected_dimension", 0))
        for row in updates
    ]
    return {
        "mean_update_norm": mean(norms),
        "max_update_norm": max(norms),
        "all_updates_finite": all(math.isfinite(value) for value in norms),
        "mean_protected_dimension": mean(ranks) if ranks else 0.0,
    }


def summary(run_dir: Path) -> dict[str, Any]:
    report = read_json(run_dir / "report_summary.json")
    return {
        "run": run_dir.name,
        "method": report["method"],
        "rewrite_exact": float(report["rewrite_exact"]),
        "declarative_paraphrase_exact": float(
            report["declarative_paraphrase_exact"]
        ),
        "target_token_f1": float(report.get("target_token_f1", 0.0)),
        "same_subject_tfpr": float(report.get("same_subject_tfpr", 0.0)),
        "near_tfpr": float(report.get("near_tfpr", 0.0)),
        "far_tfpr": float(report.get("far_tfpr", 0.0)),
        "malformed_rate": float(report.get("malformed_rate", 0.0)),
        "gpu_minutes_per_edit": float(report.get("gpu_minutes_per_edit", 0.0)),
        **update_geometry(run_dir),
    }


def base_budget(report: Mapping[str, Any], bucket: str) -> float:
    return (
        float(
            report.get("base_summary", {})
            .get(bucket, {})
            .get("target_new_tfpr_or_exact", 0.0)
        )
        + 0.03
    )


def hard_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "rewrite_pass": float(report["rewrite_exact"]) >= 0.75,
        "paraphrase_pass": float(report["declarative_paraphrase_exact"]) >= 0.40,
        "same_subject_pass": float(report.get("same_subject_tfpr", 0.0))
        <= base_budget(report, "same_subject"),
        "near_pass": float(report.get("near_tfpr", 0.0))
        <= base_budget(report, "near_locality"),
        "far_pass": float(report.get("far_tfpr", 0.0))
        <= base_budget(report, "far_locality"),
        "malformed_pass": float(report.get("malformed_rate", 0.0)) <= 0.05,
    }


def locality_agreement(report: Mapping[str, Any]) -> float:
    values = []
    for bucket in ("same_subject", "near_locality", "far_locality"):
        value = report.get("edited_summary", {}).get(bucket, {}).get("base_agreement")
        if value is not None:
            values.append(float(value))
    return mean(values) if values else 0.0


def candidate_score(report: Mapping[str, Any]) -> tuple[Any, ...]:
    checks = hard_checks(report)
    return (
        sum(checks.values()),
        all(checks.values()),
        float(report["rewrite_exact"])
        + float(report["declarative_paraphrase_exact"])
        + locality_agreement(report),
        -float(report.get("same_subject_tfpr", 0.0)),
    )


def run_d3(root: Path, *, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    config = target_config()
    causal = site_layers()
    manifest = combined_kamel_smoke(root / "kamel_smoke_60.jsonl")
    definitions = (
        ("fixed_site_fullmask", [3, 4, 5, 6], "fully_masked", "random"),
        (
            "fixed_site_partial_state",
            [3, 4, 5, 6],
            config["partial_mask_schedule"],
            config["reveal_policy"],
        ),
        ("causal_site_fullmask", causal, "fully_masked", "random"),
        (
            "causal_site_partial_state",
            causal,
            config["partial_mask_schedule"],
            config["reveal_policy"],
        ),
    )
    rows = []
    reports = {}
    for method, layers, schedule, reveal in definitions:
        output = root / method
        run_one(
            manifest=manifest,
            output=output,
            method=method,
            layers=layers,
            config=config,
            schedule=str(schedule),
            reveal=str(reveal),
            resume=resume,
        )
        rows.append(summary(output))
        reports[method] = read_json(output / "report_summary.json")
    write_csv(root / "causal_update_comparison.csv", rows)
    acceptance = {
        "all_four_comparisons_complete": len(rows) == 4,
        "closed_form_updates_finite": all(row["all_updates_finite"] for row in rows),
        "edited_models_decode": all(
            math.isfinite(row["rewrite_exact"])
            and math.isfinite(row["declarative_paraphrase_exact"])
            for row in rows
        ),
        "all_metrics_complete": all(
            all(key in row for key in ("same_subject_tfpr", "near_tfpr", "far_tfpr"))
            for row in rows
        ),
    }
    passed = all(acceptance.values())
    mechanism = {
        "causal_minus_fixed_partial_rewrite": reports["causal_site_partial_state"][
            "rewrite_exact"
        ]
        - reports["fixed_site_partial_state"]["rewrite_exact"],
        "causal_minus_fixed_partial_paraphrase": reports[
            "causal_site_partial_state"
        ]["declarative_paraphrase_exact"]
        - reports["fixed_site_partial_state"]["declarative_paraphrase_exact"],
        "partial_minus_fullmask_causal_rewrite": reports[
            "causal_site_partial_state"
        ]["rewrite_exact"]
        - reports["causal_site_fullmask"]["rewrite_exact"],
        "partial_minus_fullmask_causal_paraphrase": reports[
            "causal_site_partial_state"
        ]["declarative_paraphrase_exact"]
        - reports["causal_site_fullmask"]["declarative_paraphrase_exact"],
    }
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D3_causal_update",
        "created_at_utc": now_utc(),
        "causal_layers": causal,
        "fixed_layers": [3, 4, 5, 6],
        "target_value_config": config,
        "mechanism_deltas": mechanism,
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", acceptance)
    record_stage(
        "D3_causal_update",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=root,
        started_at_utc=started,
        notes="Four frozen site/state update controls completed.",
        next_stage="D4_nullspace_main",
    )
    return report


def d4_id(variance: float, ridge: float, identity: float) -> str:
    return (
        f"variance{variance:.2f}_ridge{ridge:.0e}_identity{identity:.1f}"
    ).replace("+", "").replace(".", "p")


def run_d4(root: Path, *, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    config = target_config()
    layers = site_layers()
    basis = CAMPAIGN_ROOT / "D1_state_banks_v1"
    if not (basis / "report_summary.json").exists():
        raise FileNotFoundError(basis / "report_summary.json")
    smoke_manifest = CAMPAIGN_ROOT / "protocol_v1" / "dnpe_smoke_20.jsonl"
    rows = []
    configs: list[tuple[float, float, float]] = []
    for variance in (0.90, 0.95, 0.99):
        configs.append((variance, 1e-3, 1.0))

    def execute(values: tuple[float, float, float]) -> tuple[Path, dict[str, Any]]:
        variance, ridge, identity = values
        output = root / f"smoke_{d4_id(variance, ridge, identity)}"
        run_one(
            manifest=smoke_manifest,
            output=output,
            method="causal_partial_state_nullspace_memit",
            layers=layers,
            config=config,
            schedule=config["partial_mask_schedule"],
            reveal=config["reveal_policy"],
            basis_dir=basis,
            protected_variance=variance,
            update_ridge=ridge,
            lambda_identity=identity,
            resume=resume,
        )
        report = read_json(output / "report_summary.json")
        rows.append(
            {
                "protected_variance": variance,
                "update_ridge": ridge,
                "lambda_identity": identity,
                **summary(output),
                **hard_checks(report),
                "locality_base_agreement": locality_agreement(report),
            }
        )
        return output, report

    reports: dict[tuple[float, float, float], dict[str, Any]] = {}
    for values in configs:
        _output, reports[values] = execute(values)
    best_variance_config = max(reports, key=lambda values: candidate_score(reports[values]))
    best_variance = best_variance_config[0]
    for ridge in (1e-4, 1e-2):
        values = (best_variance, ridge, 1.0)
        configs.append(values)
        _output, reports[values] = execute(values)
    best_ridge_config = max(reports, key=lambda values: candidate_score(reports[values]))
    best_ridge = best_ridge_config[1]
    for identity in (0.1, 2.0):
        values = (best_variance, best_ridge, identity)
        configs.append(values)
        _output, reports[values] = execute(values)
    if len(set(configs)) > 7:
        raise RuntimeError("D4 staged grid expanded beyond seven configurations")
    selected_values = max(reports, key=lambda values: candidate_score(reports[values]))
    selected = {
        "protected_variance": selected_values[0],
        "update_ridge": selected_values[1],
        "lambda_identity": selected_values[2],
    }
    write_csv(root / "nullspace_staged_grid.csv", rows)
    write_json(
        root / "selected_nullspace_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "selected": selected,
            "selection_source": "smoke20_staged_grid_only",
            "grid_size": len(set(configs)),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    pilot_manifest = CAMPAIGN_ROOT / "protocol_v1" / "dnpe_pilot_100.jsonl"
    no_projection = root / "pilot100_causal_partial_no_projection"
    run_one(
        manifest=pilot_manifest,
        output=no_projection,
        method="causal_site_partial_state_editor",
        layers=layers,
        config=config,
        schedule=config["partial_mask_schedule"],
        reveal=config["reveal_policy"],
        resume=resume,
    )
    pilot = root / "pilot100_selected_nullspace"
    run_one(
        manifest=pilot_manifest,
        output=pilot,
        method="causal_partial_state_nullspace_memit",
        layers=layers,
        config=config,
        schedule=config["partial_mask_schedule"],
        reveal=config["reveal_policy"],
        basis_dir=basis,
        protected_variance=selected["protected_variance"],
        update_ridge=selected["update_ridge"],
        lambda_identity=selected["lambda_identity"],
        resume=resume,
    )
    main_report = read_json(pilot / "report_summary.json")
    baseline_paths = [
        CAMPAIGN_ROOT / "B1_mdm_memit_reproduction_v1" / "pilot100_v1",
        CAMPAIGN_ROOT / "B3_alphaedit_style_mdm_memit_v1" / "pilot100_selected",
        no_projection,
    ]
    baseline_reports = [
        (path, read_json(path / "report_summary.json")) for path in baseline_paths
    ]
    efficacy_matched = [
        (path, report)
        for path, report in baseline_reports
        if float(main_report["rewrite_exact"])
        >= float(report["rewrite_exact"]) - 0.05
        and float(main_report["declarative_paraphrase_exact"])
        >= float(report["declarative_paraphrase_exact"]) - 0.05
    ]
    strongest = (
        max(
            efficacy_matched,
            key=lambda item: float(item[1]["rewrite_exact"])
            + float(item[1]["declarative_paraphrase_exact"]),
        )
        if efficacy_matched
        else None
    )
    same_reduction = 0.0
    if strongest is not None:
        baseline_same = float(strongest[1].get("same_subject_tfpr", 0.0))
        if baseline_same > 0:
            same_reduction = (
                baseline_same - float(main_report.get("same_subject_tfpr", 0.0))
            ) / baseline_same
    hard = hard_checks(main_report)
    acceptance = {
        **hard,
        "efficacy_matched_baseline_exists": strongest is not None,
        "same_subject_tfpr_reduced_at_least_50_percent": same_reduction >= 0.50,
        "bounded_staged_grid": len(set(configs)) <= 7,
        "all_updates_finite": update_geometry(pilot)["all_updates_finite"],
    }
    passed = all(acceptance.values())
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D4_nullspace_main",
        "created_at_utc": now_utc(),
        "selected_config": selected,
        "pilot_run": str(pilot.relative_to(ROOT)),
        "no_projection_pilot_run": str(no_projection.relative_to(ROOT)),
        "strongest_efficacy_matched_baseline": (
            str(strongest[0].relative_to(ROOT)) if strongest else None
        ),
        "same_subject_tfpr_relative_reduction": same_reduction,
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", acceptance)
    record_stage(
        "D4_nullspace_main",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=root,
        started_at_utc=started,
        notes="Staged locality grid selected on smoke20 and validated once on pilot100.",
        next_stage="E1_smoke20",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("d3", "d4"), required=True)
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or (
        CAMPAIGN_ROOT
        / (
            "D3_causal_multi_state_update_v1"
            if args.phase == "d3"
            else "D4_causal_partial_state_nullspace_v1"
        )
    )
    report = (
        run_d3(root, resume=bool(args.resume))
        if args.phase == "d3"
        else run_d4(root, resume=bool(args.resume))
    )
    print(json.dumps({"stage": report["stage"], "acceptance_pass": report["acceptance_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
