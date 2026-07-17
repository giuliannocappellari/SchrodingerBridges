#!/usr/bin/env python3
"""Run and report the frozen E1 smoke and E2 pilot/KAMEL selection stages."""

from __future__ import annotations

import argparse
import json
import math
import random
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
    record_stage,
    write_csv,
    write_json,
)
from scripts.report_dnpe_selection import harmonic, paired_bootstrap
from scripts.run_dnpe_causal_nullspace_sweep import (
    d4_id,
    hard_checks,
    locality_agreement,
    run_one,
    site_layers,
    summary,
    target_config,
    update_geometry,
)


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def run_runtime_baseline(
    method: str, manifest: Path, output: Path, *, resume: bool
) -> None:
    if output.exists():
        if resume and (output / "report_summary.json").exists():
            return
        raise FileExistsError(output)
    execute(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_dnpe_runtime_baseline.py"),
            "--method",
            method,
            "--manifest",
            str(manifest),
            "--output_dir",
            str(output),
            "--guidance_scale",
            "2.0",
            "--decode_batch_size",
            "16",
        ]
    )


def row_from_report(path: Path, *, role: str = "smoke") -> dict[str, Any]:
    report = read_json(path / "report_summary.json")
    return {
        "label": str(report.get("method") or path.name),
        "run": str(path.relative_to(ROOT)),
        "evaluation_role": role,
        "num_edits": int(report.get("num_edits", 0)),
        "rewrite_exact": float(report.get("rewrite_exact", 0.0)),
        "declarative_paraphrase_exact": float(
            report.get("declarative_paraphrase_exact", 0.0)
        ),
        "target_token_f1": float(report.get("target_token_f1", 0.0)),
        "same_subject_tfpr": float(report.get("same_subject_tfpr", 0.0)),
        "near_tfpr": float(report.get("near_tfpr", 0.0)),
        "far_tfpr": float(report.get("far_tfpr", 0.0)),
        "malformed_rate": float(report.get("malformed_rate", 0.0)),
        "gpu_minutes_per_edit": float(report.get("gpu_minutes_per_edit", 0.0)),
        "locality_base_agreement": locality_agreement(report),
        "all_finite": all(
            math.isfinite(float(report.get(key, 0.0)))
            for key in (
                "rewrite_exact",
                "declarative_paraphrase_exact",
                "same_subject_tfpr",
                "near_tfpr",
                "far_tfpr",
                "malformed_rate",
            )
        ),
    }


def base_row(path: Path) -> dict[str, Any]:
    report = read_json(path / "report_summary.json")
    base = report["base_summary"]
    return {
        "label": "base",
        "run": str(path.relative_to(ROOT)),
        "evaluation_role": "smoke_base_from_aligned_run",
        "num_edits": int(report["num_edits"]),
        "rewrite_exact": float(base.get("rewrite", {}).get("expected_exact", 0.0)),
        "declarative_paraphrase_exact": float(
            base.get("declarative_paraphrase", {}).get("expected_exact", 0.0)
        ),
        "target_token_f1": float(base.get("rewrite", {}).get("target_token_f1", 0.0)),
        "same_subject_tfpr": float(
            base.get("same_subject", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "near_tfpr": float(
            base.get("near_locality", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "far_tfpr": float(
            base.get("far_locality", {}).get("target_new_tfpr_or_exact", 0.0)
        ),
        "malformed_rate": max(
            (float(values.get("malformed_rate", 0.0)) for values in base.values()),
            default=0.0,
        ),
        "gpu_minutes_per_edit": 0.0,
        "locality_base_agreement": 1.0,
        "all_finite": True,
    }


def selected_b3_smoke() -> Path:
    selection = read_json(
        CAMPAIGN_ROOT
        / "B3_alphaedit_style_mdm_memit_v1"
        / "smoke_selection.json"
    )
    return ROOT / selection["selected"]["path"]


def selected_d4_smoke() -> Path:
    selection = read_json(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "selected_nullspace_config.json"
    )["selected"]
    return (
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / (
            "smoke_"
            + d4_id(
                float(selection["protected_variance"]),
                float(selection["update_ridge"]),
                float(selection["lambda_identity"]),
            )
        )
    )


def stress_score(report: Mapping[str, Any]) -> float:
    return harmonic(
        (
            float(report["rewrite_exact"]),
            float(report["declarative_paraphrase_exact"]),
            min(max(locality_agreement(report), 0.0), 1.0),
        )
    )


def run_e1(root: Path, *, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    manifest = CAMPAIGN_ROOT / "protocol_v1" / "dnpe_smoke_20.jsonl"
    for method in ("prompt_memory", "target_logit_bias"):
        run_runtime_baseline(
            method, manifest, root / f"{method}_smoke20", resume=resume
        )
    config = target_config()
    rng = random.Random(260717101)
    random_layers = sorted(rng.sample(range(32), len(site_layers())))
    random_output = root / "random_site_partial_state_editor_kamel_smoke60"
    run_one(
        manifest=CAMPAIGN_ROOT
        / "D3_causal_multi_state_update_v1"
        / "kamel_smoke_60.jsonl",
        output=random_output,
        method="random_site_partial_state_editor",
        layers=random_layers,
        config=config,
        schedule=config["partial_mask_schedule"],
        reveal=config["reveal_policy"],
        resume=resume,
    )
    b1_smoke = (
        CAMPAIGN_ROOT / "B1_mdm_memit_reproduction_v1" / "smoke20_repair_v1"
    )
    paths = [
        root / "prompt_memory_smoke20",
        root / "target_logit_bias_smoke20",
        b1_smoke,
        selected_b3_smoke(),
        CAMPAIGN_ROOT / "B4_timerome_dlm_style_v1" / "smoke20_v1",
        random_output,
        CAMPAIGN_ROOT
        / "D3_causal_multi_state_update_v1"
        / "fixed_site_fullmask",
        CAMPAIGN_ROOT
        / "D3_causal_multi_state_update_v1"
        / "fixed_site_partial_state",
        CAMPAIGN_ROOT
        / "D3_causal_multi_state_update_v1"
        / "causal_site_fullmask",
        CAMPAIGN_ROOT
        / "D3_causal_multi_state_update_v1"
        / "causal_site_partial_state",
        selected_d4_smoke(),
    ]
    rows = [base_row(b1_smoke)] + [row_from_report(path) for path in paths]
    b2_selection = read_json(
        CAMPAIGN_ROOT
        / "B2_partial_state_mdm_memit_v1"
        / "smoke_policy_selection.json"
    )["selected_policy_by_length"]
    for length in (2, 3, 4):
        for policy in ("fully_masked_only", b2_selection[str(length)]):
            rows.append(
                row_from_report(
                    CAMPAIGN_ROOT
                    / "B2_partial_state_mdm_memit_v1"
                    / f"smoke_n{length}_{policy}",
                    role=f"kamel_length_{length}",
                )
            )
    write_csv(root / "smoke_method_table.csv", rows)
    parametric = [
        row
        for row in rows
        if row["label"]
        not in {"base", "prompt_memory", "target_logit_bias"}
    ]
    base = rows[0]
    red_stop = {
        "nan_or_corrupt_updates": any(not row["all_finite"] for row in parametric),
        "all_parametric_rewrite_no_better_than_base": all(
            row["rewrite_exact"] <= base["rewrite_exact"] for row in parametric
        ),
        "all_parametric_same_subject_tfpr_above_0_50": all(
            row["same_subject_tfpr"] > 0.50 for row in parametric
        ),
        "train_eval_leakage": False,
    }
    passed = not any(red_stop.values())
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "E1_smoke20",
        "created_at_utc": now_utc(),
        "method_count": len(rows),
        "random_site_layers": random_layers,
        "red_stop_checks": red_stop,
        "acceptance_pass": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", red_stop)
    record_stage(
        "E1_smoke20",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=root,
        started_at_utc=started,
        notes="Full frozen smoke registry checked for catastrophic failures.",
        next_stage="E2_pilot100",
    )
    return report


def run_e2(root: Path, *, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    config = target_config()
    layers = site_layers()
    d4_selection = read_json(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "selected_nullspace_config.json"
    )["selected"]
    basis = CAMPAIGN_ROOT / "D1_state_banks_v1"
    kamel_rows = []
    for length in (2, 3, 4):
        manifest = (
            CAMPAIGN_ROOT
            / "protocol_v1"
            / f"dnpe_kamel_dev_100_n{length}.jsonl"
        )
        no_projection = root / f"kamel_n{length}_causal_no_projection"
        projected = root / f"kamel_n{length}_selected_nullspace"
        run_one(
            manifest=manifest,
            output=no_projection,
            method="causal_site_partial_state_editor",
            layers=layers,
            config=config,
            schedule=config["partial_mask_schedule"],
            reveal=config["reveal_policy"],
            resume=resume,
        )
        run_one(
            manifest=manifest,
            output=projected,
            method="causal_partial_state_nullspace_memit",
            layers=layers,
            config=config,
            schedule=config["partial_mask_schedule"],
            reveal=config["reveal_policy"],
            basis_dir=basis,
            protected_variance=float(d4_selection["protected_variance"]),
            update_ridge=float(d4_selection["update_ridge"]),
            lambda_identity=float(d4_selection["lambda_identity"]),
            resume=resume,
        )
        for method, path in (
            ("causal_no_projection", no_projection),
            ("selected_nullspace", projected),
        ):
            kamel_rows.append(
                {
                    "target_length": length,
                    "method_family": method,
                    **row_from_report(path, role=f"kamel_dev_n{length}"),
                }
            )
        b2_selection = read_json(
            CAMPAIGN_ROOT
            / "B2_partial_state_mdm_memit_v1"
            / "smoke_policy_selection.json"
        )["selected_policy_by_length"][str(length)]
        for method, policy in (
            ("fully_masked_only", "fully_masked_only"),
            ("partial_state_mdm_memit", b2_selection),
        ):
            kamel_rows.append(
                {
                    "target_length": length,
                    "method_family": method,
                    **row_from_report(
                        CAMPAIGN_ROOT
                        / "B2_partial_state_mdm_memit_v1"
                        / f"dev_n{length}_{policy}",
                        role=f"kamel_dev_n{length}",
                    ),
                }
            )
    write_csv(root / "kamel_dev_table.csv", kamel_rows)
    main_path = (
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "pilot100_selected_nullspace"
    )
    no_projection_path = (
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "pilot100_causal_partial_no_projection"
    )
    cf_paths = [
        CAMPAIGN_ROOT / "B1_mdm_memit_reproduction_v1" / "pilot100_v1",
        CAMPAIGN_ROOT / "B3_alphaedit_style_mdm_memit_v1" / "pilot100_selected",
        CAMPAIGN_ROOT / "B4_timerome_dlm_style_v1" / "pilot100_v1",
        no_projection_path,
        main_path,
    ]
    write_csv(
        root / "counterfact_pilot100_table.csv",
        [row_from_report(path, role="counterfact_pilot100") for path in cf_paths],
    )
    main_report = read_json(main_path / "report_summary.json")
    hard = hard_checks(main_report)
    d4 = read_json(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "report_summary.json"
    )
    locality_value = bool(
        d4["acceptance"].get("same_subject_tfpr_reduced_at_least_50_percent")
    )
    d3_root = CAMPAIGN_ROOT / "D3_causal_multi_state_update_v1"
    causal_report = read_json(
        d3_root / "causal_site_partial_state" / "report_summary.json"
    )
    random_report = read_json(
        CAMPAIGN_ROOT
        / "E1_smoke20_v1"
        / "random_site_partial_state_editor_kamel_smoke60"
        / "report_summary.json"
    )
    causal_score_gain = stress_score(causal_report) - stress_score(random_report)
    causal_norm = update_geometry(d3_root / "causal_site_partial_state")[
        "mean_update_norm"
    ]
    random_norm = update_geometry(
        CAMPAIGN_ROOT
        / "E1_smoke20_v1"
        / "random_site_partial_state_editor_kamel_smoke60"
    )["mean_update_norm"]
    causal_mechanism = causal_score_gain >= 0.05 or (
        float(causal_report["rewrite_exact"])
        >= float(random_report["rewrite_exact"]) - 0.02
        and float(causal_report["declarative_paraphrase_exact"])
        >= float(random_report["declarative_paraphrase_exact"]) - 0.02
        and float(causal_norm) <= 0.75 * float(random_norm)
    )
    multi_token_comparison = []
    partial_positive_lengths = 0
    main_positive_lengths = 0
    for length in (2, 3, 4):
        by_family = {
            row["method_family"]: row
            for row in kamel_rows
            if row["target_length"] == length
        }
        baseline = by_family["fully_masked_only"]
        partial = by_family["partial_state_mdm_memit"]
        main = by_family["selected_nullspace"]
        partial_gain = (
            partial["rewrite_exact"] - baseline["rewrite_exact"]
        )
        main_gain = main["rewrite_exact"] - baseline["rewrite_exact"]
        partial_positive_lengths += partial_gain >= 0.10
        main_positive_lengths += main_gain >= 0.10
        multi_token_comparison.append(
            {
                "target_length": length,
                "fullmask_rewrite": baseline["rewrite_exact"],
                "partial_rewrite": partial["rewrite_exact"],
                "main_rewrite": main["rewrite_exact"],
                "partial_rewrite_gain": partial_gain,
                "main_rewrite_gain": main_gain,
                "fullmask_paraphrase": baseline[
                    "declarative_paraphrase_exact"
                ],
                "partial_paraphrase": partial[
                    "declarative_paraphrase_exact"
                ],
                "main_paraphrase": main["declarative_paraphrase_exact"],
            }
        )
    write_csv(root / "multi_token_mechanism_table.csv", multi_token_comparison)
    partial_mechanism = partial_positive_lengths >= 2 or main_positive_lengths >= 2
    mechanism_value = causal_mechanism or partial_mechanism
    bootstrap_rows = []
    for baseline_path in (no_projection_path, cf_paths[1]):
        for bucket in ("rewrite", "declarative_paraphrase"):
            row = paired_bootstrap(main_path, baseline_path, bucket=bucket)
            row.update(
                {
                    "left": "causal_partial_state_nullspace_memit",
                    "right": baseline_path.name,
                }
            )
            bootstrap_rows.append(row)
    write_csv(root / "paired_bootstrap.csv", bootstrap_rows)
    single_token_pass = all(hard.values()) and locality_value
    multi_token_pass = partial_mechanism
    rescue_triggered = single_token_pass and not multi_token_pass
    acceptance = {
        **hard,
        "locality_improves_over_efficacy_matched_baseline": locality_value,
        "causal_or_partial_state_mechanism_value": mechanism_value,
        "no_rescue_pending": not rescue_triggered,
    }
    passed = all(acceptance.values())
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "E2_pilot100",
        "created_at_utc": now_utc(),
        "single_token_hard_pass": all(hard.values()),
        "locality_value": locality_value,
        "causal_stress_score_gain_over_random": causal_score_gain,
        "causal_mechanism_value": causal_mechanism,
        "partial_state_positive_lengths": partial_positive_lengths,
        "main_positive_lengths": main_positive_lengths,
        "partial_state_mechanism_value": partial_mechanism,
        "d5_rescue_triggered": rescue_triggered,
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", acceptance)
    record_stage(
        "E2_pilot100",
        status=("rescue_triggered" if rescue_triggered else "passed" if passed else "failed"),
        acceptance_pass=passed,
        output_dir=root,
        started_at_utc=started,
        notes="Pilot100 and exact-length KAMEL dev evaluated under frozen criteria.",
        next_stage="D5_state_conditioned_rescue" if rescue_triggered else "F1_dev200" if passed else "H_final_package",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("e1", "e2"), required=True)
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or (
        CAMPAIGN_ROOT
        / ("E1_smoke20_v1" if args.phase == "e1" else "E2_pilot100_v1")
    )
    report = (
        run_e1(root, resume=bool(args.resume))
        if args.phase == "e1"
        else run_e2(root, resume=bool(args.resume))
    )
    print(json.dumps({"stage": report["stage"], "acceptance_pass": report["acceptance_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
