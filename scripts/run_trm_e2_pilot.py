#!/usr/bin/env python3
"""Run the fixed E2 CounterFact pilot and select at most three claim candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_trm_e1_smoke import (
    REQUIRED_METHODS,
    base_row,
    normalized_report_row,
)
from scripts.run_trm_state_conditioned_protection import paired_tfpr_bootstrap
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


def execute(command: Sequence[str]) -> None:
    print("E2 launch:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(command), cwd=ROOT, check=True)


def read_prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = dict(source)
            for key in (
                "target_new_hit",
                "target_true_hit",
                "expected_hit",
                "malformed",
                "base_agreement",
            ):
                if key in row and row[key] not in {"", None}:
                    row[key] = str(row[key]).casefold() == "true"
            rows.append(row)
    return rows


def method_is_deployable(row: Mapping[str, Any]) -> bool:
    return bool(
        row["all_metrics_finite"]
        and row["runtime_schema_present"]
        and float(row["malformed_rate"]) <= 0.05
        and float(row["gpu_minutes_per_edit"]) <= 2.0
        and float(row["utility_base_agreement"]) >= 0.25
        and not row["analysis_500_used"]
        and not row["final_test_used"]
    )


def deduplicate_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for source in rows:
        method = str(source["method"])
        if method in seen:
            continue
        seen.add(method)
        output.append(dict(source))
    return output[:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "E2_pilot100_v1"
    )
    parser.add_argument(
        "--e1_dir", type=Path, default=CAMPAIGN_ROOT / "E1_smoke20_v1"
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
    e1 = read_json(args.e1_dir / "report_summary.json")
    if not e1.get("acceptance_pass"):
        raise RuntimeError("E1 must pass before pilot100")
    args.output_dir.mkdir(parents=True)
    e1_config = read_json(args.e1_dir / "run_config.json")
    d1 = read_json(args.d1_dir / "report_summary.json")
    d2_dense = read_json(
        args.d2_dir / "state_conditioned_preservation" / "report_summary.json"
    )
    d2_sparse = read_json(
        args.d2_dir / "state_conditioned_sparsification" / "report_summary.json"
    )
    selected_q = 256 if float(d2_sparse["stress_aware_aggregate"]) >= float(
        d2_dense["stress_aware_aggregate"]
    ) else 0
    layer = int(e1_config["temporal_layer"])
    schedule = str(e1_config["partial_mask_schedule"])
    reveal = str(e1_config["reveal_policy"])
    consistency = float(e1_config["state_consistency_weight"])
    manifest = PROTOCOL_ROOT / "cf_trm_pilot_100.jsonl"
    anchor = PROTOCOL_ROOT / "cf_trm_anchor_train_500.jsonl"
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "E2_pilot100",
            "manifest": str(manifest),
            "required_methods": list(REQUIRED_METHODS),
            "layer": layer,
            "partial_mask_schedule": schedule,
            "reveal_policy": reveal,
            "state_consistency_weight": consistency,
            "selected_state_protected_top_q": selected_q,
            "architectures_frozen_before_pilot": True,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    python = sys.executable
    covariance = args.e1_dir / "train_only_covariance_layer6"
    basis = args.e1_dir / "static_basis_cache"
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
            str(basis),
            "--protected_variance",
            "0.95",
        ]
        + partial_args
    )
    residual_specs = [
        ("timerome_source_style_fullmask", 3, "shared", "none", "fully_masked", "random", 0),
        ("timerome_counterfact_partial_state", layer, "shared", "none", schedule, reveal, 0),
        ("timerome_partial_state_state_bucketed", layer, "bucketed", "none", schedule, reveal, 0),
        ("timerome_shared_soft_preservation", layer, "shared", "shared", schedule, reveal, 0),
        ("timerome_partial_state_state_protected", layer, "bucketed", "state", schedule, reveal, selected_q),
        ("random_site_partial_state_residual", 9, "shared", "none", schedule, reveal, 0),
        ("fixed_site_partial_state_residual", 3, "shared", "none", schedule, reveal, 0),
    ]
    for method, method_layer, state_mode, protection, method_schedule, method_reveal, top_q in residual_specs:
        execute(
            [
                python,
                str(ROOT / "scripts" / "run_trm_editor_experiment.py"),
                "--manifest",
                str(manifest),
                "--anchor_manifest",
                str(anchor),
                "--output_dir",
                str(args.output_dir / method),
                "--method",
                method,
                "--layer",
                str(method_layer),
                "--state_mode",
                state_mode,
                "--protection_mode",
                protection,
                "--partial_mask_schedule",
                method_schedule,
                "--reveal_policy",
                method_reveal,
                "--state_consistency_weight",
                str(consistency),
                "--top_q",
                str(top_q),
            ]
        )
    c2_pilot = args.c2_dir / "pilot100_v1" / "stable_temporal_top1"
    rows = [base_row(c2_pilot)]
    for method in (
        "prompt_memory",
        "target_logit_bias",
        "ordinary_mdm_memit",
        "partial_state_mdm_memit",
        "static_nullspace_partial_state_memit",
    ):
        rows.append(normalized_report_row(method, args.output_dir / method))
    rows.append(normalized_report_row("timerome_counterfact_fullmask", c2_pilot))
    for method, *_rest in residual_specs:
        rows.append(normalized_report_row(method, args.output_dir / method))
    rows.append(
        {
            "method": "timerome_source_reproduction",
            "run": str(
                (CAMPAIGN_ROOT / "C0_timerome_source_reproduction_v1").relative_to(ROOT)
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
            "utility_base_agreement": 1.0,
            "runtime_schema_present": True,
            "all_metrics_finite": True,
            "analysis_500_used": False,
            "final_test_used": False,
            "comparable_actual_decode": False,
        }
    )
    by_method = {str(row["method"]): row for row in rows}
    missing = sorted(set(REQUIRED_METHODS) - set(by_method))
    base = by_method["base"]
    budgets = {
        "same_subject_tfpr": float(base["same_subject_tfpr"]) + 0.03,
        "near_tfpr": float(base["near_tfpr"]) + 0.03,
        "far_tfpr": float(base["far_tfpr"]) + 0.03,
    }
    comparable = [
        row for row in rows if row["comparable_actual_decode"] and row["method"] != "base"
    ]
    full_candidates = [
        row
        for row in comparable
        if method_is_deployable(row)
        and float(row["rewrite_exact"]) >= 0.85
        and float(row["declarative_paraphrase_exact"]) >= 0.40
        and float(row["same_subject_tfpr"]) <= budgets["same_subject_tfpr"]
        and float(row["near_tfpr"]) <= budgets["near_tfpr"]
        and float(row["far_tfpr"]) <= budgets["far_tfpr"]
    ]
    protected = by_method["timerome_partial_state_state_protected"]
    shared = by_method["timerome_shared_soft_preservation"]
    state_bootstrap = paired_tfpr_bootstrap(
        read_prompt_rows(
            args.output_dir
            / "timerome_partial_state_state_protected"
            / "edited_per_prompt.csv"
        ),
        read_prompt_rows(
            args.output_dir
            / "timerome_shared_soft_preservation"
            / "edited_per_prompt.csv"
        ),
        seed=260718901,
    )
    shared_tfpr = float(shared["same_subject_tfpr"])
    state_reduction = (
        (shared_tfpr - float(protected["same_subject_tfpr"]))
        / max(shared_tfpr, 1e-8)
    )
    state_matched = bool(
        float(protected["rewrite_exact"]) >= float(shared["rewrite_exact"]) - 0.02
        and float(protected["declarative_paraphrase_exact"])
        >= float(shared["declarative_paraphrase_exact"]) - 0.02
    )
    state_conditioning_pass = bool(
        method_is_deployable(protected)
        and (
            (state_matched and state_reduction >= 0.20 and state_bootstrap["delta"] < 0)
            or float(protected["stress_aware_aggregate"])
            - float(shared["stress_aware_aggregate"])
            >= 0.05
        )
    )
    baseline_pool = [
        row
        for row in comparable
        if row["method"]
        not in {
            "timerome_partial_state_state_protected",
            "timerome_shared_soft_preservation",
        }
        and method_is_deployable(row)
    ]
    strongest_baseline = max(
        baseline_pool,
        key=lambda row: (float(row["selection_score"]), str(row["method"])),
    )
    baseline_path = ROOT / str(strongest_baseline["run"])
    pareto_bootstrap = paired_tfpr_bootstrap(
        read_prompt_rows(
            args.output_dir
            / "timerome_partial_state_state_protected"
            / "edited_per_prompt.csv"
        ),
        read_prompt_rows(baseline_path / "edited_per_prompt.csv"),
        seed=260718902,
    )
    baseline_tfpr = float(strongest_baseline["same_subject_tfpr"])
    pareto_reduction = (
        (baseline_tfpr - float(protected["same_subject_tfpr"]))
        / max(baseline_tfpr, 1e-8)
    )
    pareto_pass = bool(
        method_is_deployable(protected)
        and float(protected["rewrite_exact"])
        >= float(strongest_baseline["rewrite_exact"]) - 0.02
        and float(protected["declarative_paraphrase_exact"])
        >= float(strongest_baseline["declarative_paraphrase_exact"]) - 0.02
        and pareto_reduction >= 0.25
        and pareto_bootstrap["ci_high"] < 0
        and float(protected["near_tfpr"]) <= float(strongest_baseline["near_tfpr"]) + 0.03
        and float(protected["far_tfpr"]) <= float(strongest_baseline["far_tfpr"]) + 0.03
    )
    diffusion_pass = bool(d1.get("diffusion_specific_pass"))
    positive_classes = {
        "full_editor": bool(full_candidates),
        "pareto_locality": pareto_pass,
        "diffusion_specific_partial_state": diffusion_pass,
        "state_conditioning": state_conditioning_pass,
    }
    nominated = []
    if full_candidates:
        nominated.append(
            max(
                full_candidates,
                key=lambda row: (
                    float(row["selection_score"]),
                    float(row["stress_aware_aggregate"]),
                    str(row["method"]),
                ),
            )
        )
    if pareto_pass or state_conditioning_pass:
        nominated.append(protected)
    if diffusion_pass:
        nominated.append(by_method["timerome_partial_state_state_bucketed"])
    candidates = deduplicate_candidates(nominated)
    integrity = {
        "complete_required_registry": not missing,
        "all_comparable_metrics_finite": all(
            row["all_metrics_finite"] for row in comparable
        ),
        "all_runtime_inputs_deployable": all(
            row["runtime_schema_present"] for row in comparable
        ),
        "no_analysis_or_final_used": not any(
            row["analysis_500_used"] or row["final_test_used"] for row in rows
        ),
        "candidate_count_at_most_three": len(candidates) <= 3,
    }
    passed = all(integrity.values()) and any(positive_classes.values()) and bool(candidates)
    write_csv(args.output_dir / "counterfact_pilot100_method_table.csv", rows)
    write_csv(
        args.output_dir / "paired_bootstrap.csv",
        [
            {
                "comparison": "state_protected_minus_shared_same_subject_tfpr",
                **state_bootstrap,
            },
            {
                "comparison": (
                    "state_protected_minus_"
                    + str(strongest_baseline["method"])
                    + "_same_subject_tfpr"
                ),
                **pareto_bootstrap,
            },
        ],
    )
    write_json(
        args.output_dir / "pilot_candidates.json",
        {
            "positive_classes": positive_classes,
            "candidates": candidates,
            "strongest_efficacy_baseline": strongest_baseline,
            "state_conditioning": {
                "matched_efficacy": state_matched,
                "relative_same_subject_reduction": state_reduction,
                "bootstrap": state_bootstrap,
            },
            "pareto_locality": {
                "relative_same_subject_reduction": pareto_reduction,
                "bootstrap": pareto_bootstrap,
            },
        },
    )
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "E2_pilot100",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "missing_required_methods": missing,
        "budgets": budgets,
        "positive_classes": positive_classes,
        "selected_candidate_methods": [row["method"] for row in candidates],
        "strongest_efficacy_baseline": strongest_baseline["method"],
        "integrity": integrity,
        "runtime_seconds": runtime,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": passed,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "integrity": integrity,
            "positive_classes": positive_classes,
            "acceptance_pass": passed,
        },
    )
    record_stage_cost(
        "E2_pilot100",
        runtime_seconds=runtime,
        notes="Complete fixed pilot100 registry and claim-class selection",
    )
    record_stage(
        "E2_pilot100",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=(
            f"positive_classes={positive_classes}; "
            f"candidates={[row['method'] for row in candidates]}"
        ),
        next_stage="E3_kamel_multi_token" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(
        json.dumps(
            {
                "acceptance_pass": True,
                "positive_classes": positive_classes,
                "candidates": [row["method"] for row in candidates],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
