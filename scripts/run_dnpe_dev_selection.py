#!/usr/bin/env python3
"""Run the frozen F1 DNPE dev200 candidate comparison and selection."""

from __future__ import annotations

import argparse
import json
import subprocess
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
    now_utc,
    read_json,
    record_stage,
    write_csv,
    write_json,
)
from scripts.report_dnpe_selection import harmonic, paired_bootstrap, score_report
from scripts.run_dnpe_causal_nullspace_sweep import run_one, site_layers, target_config
from scripts.run_dnpe_pilot_selection import run_runtime_baseline


def run_timerome(manifest: Path, output: Path, *, resume: bool) -> None:
    if output.exists():
        if resume and (output / "report_summary.json").exists():
            return
        raise FileExistsError(output)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_dnpe_timerome_style.py"),
            "--manifest",
            str(manifest),
            "--output_dir",
            str(output),
            "--layer",
            "6",
            "--ridge",
            "0.01",
            "--similarity_threshold",
            "0.5",
            "--top_k_memory",
            "4",
            "--partial_mask_schedule",
            "cycle",
            "--decode_batch_size",
            "16",
        ],
        cwd=ROOT,
        check=True,
    )


def candidate_row(path: Path) -> dict[str, Any]:
    return score_report(read_json(path / "report_summary.json"), path / "report_summary.json")


def multi_token_family_score(family: str) -> float:
    rows = []
    path = CAMPAIGN_ROOT / "E2_pilot100_v1" / "kamel_dev_table.csv"
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method_family") == family:
                rows.append(
                    mean(
                        (
                            float(row["rewrite_exact"]),
                            float(row["declarative_paraphrase_exact"]),
                        )
                    )
                )
    return mean(rows) if rows else 0.0


def deduplicate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        path = row["path"]
        if path in seen:
            continue
        seen.add(path)
        output.append(row)
    return output[:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=CAMPAIGN_ROOT / "F1_dev200_selection_v1"
    )
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    manifest = CAMPAIGN_ROOT / "protocol_v1" / "dnpe_dev_200.jsonl"
    config = target_config()
    layers = site_layers()
    d4 = read_json(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "selected_nullspace_config.json"
    )["selected"]
    main_path = args.root / "causal_partial_state_nullspace_memit"
    no_projection_path = args.root / "causal_site_partial_state_editor"
    run_one(
        manifest=manifest,
        output=no_projection_path,
        method="causal_site_partial_state_editor",
        layers=layers,
        config=config,
        schedule=config["partial_mask_schedule"],
        reveal=config["reveal_policy"],
        resume=bool(args.resume),
    )
    run_one(
        manifest=manifest,
        output=main_path,
        method="causal_partial_state_nullspace_memit",
        layers=layers,
        config=config,
        schedule=config["partial_mask_schedule"],
        reveal=config["reveal_policy"],
        basis_dir=CAMPAIGN_ROOT / "D1_state_banks_v1",
        protected_variance=float(d4["protected_variance"]),
        update_ridge=float(d4["update_ridge"]),
        lambda_identity=float(d4["lambda_identity"]),
        resume=bool(args.resume),
    )
    alpha_selection = read_json(
        CAMPAIGN_ROOT
        / "B3_alphaedit_style_mdm_memit_v1"
        / "smoke_selection.json"
    )["selected"]
    alpha_path = args.root / "alphaedit_style_mdm_memit"
    run_one(
        manifest=manifest,
        output=alpha_path,
        method="alphaedit_style_mdm_memit",
        layers=[3, 4, 5, 6],
        config={
            "learning_rate": 0.1,
            "target_optimization_steps": 25,
            "state_consistency_weight": 0.0,
            "old_target_suppression_weight": 0.0,
        },
        schedule="fully_masked",
        reveal="random",
        basis_dir=CAMPAIGN_ROOT / "preservation_basis_v1",
        protected_variance=float(alpha_selection["protected_variance"]),
        update_ridge=float(alpha_selection["update_ridge"]),
        resume=bool(args.resume),
    )
    timerome_path = args.root / "timerome_dlm_style_residual_memory"
    run_timerome(manifest, timerome_path, resume=bool(args.resume))
    for method in ("prompt_memory", "target_logit_bias"):
        run_runtime_baseline(
            method,
            manifest,
            args.root / method,
            resume=bool(args.resume),
        )
    report_paths = [
        CAMPAIGN_ROOT / "B1_mdm_memit_reproduction_v1" / "dev200_v1",
        args.root / "prompt_memory",
        args.root / "target_logit_bias",
        alpha_path,
        timerome_path,
        no_projection_path,
        main_path,
    ]
    rows = [candidate_row(path) for path in report_paths]
    rows.sort(
        key=lambda row: (
            bool(row["constraint_pass"]),
            float(row["selection_score"]),
        ),
        reverse=True,
    )
    write_csv(args.root / "dev_selection_scores.csv", rows)
    write_csv(
        args.root / "dev_selection_scores_feasible.csv",
        [row for row in rows if row["constraint_pass"]],
    )
    main_rows = [
        row
        for row in rows
        if row["path"] in {str(main_path.relative_to(ROOT)), str(no_projection_path.relative_to(ROOT))}
        and row["constraint_pass"]
    ]
    main_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    candidates = []
    if main_rows:
        candidates.append({"nomination": "best_stress_aware_aggregate", **main_rows[0]})
        candidates.append(
            {
                "nomination": "best_locality_safety",
                **max(
                    main_rows,
                    key=lambda row: (
                        -float(row["same_subject_tfpr"]),
                        -float(row["near_tfpr"]),
                        -float(row["far_tfpr"]),
                        float(row["selection_score"]),
                    ),
                ),
            }
        )
        family_map = {
            str(main_path.relative_to(ROOT)): "selected_nullspace",
            str(no_projection_path.relative_to(ROOT)): "causal_no_projection",
        }
        candidates.append(
            {
                "nomination": "best_multi_token_robustness",
                **max(
                    main_rows,
                    key=lambda row: multi_token_family_score(family_map[row["path"]]),
                ),
            }
        )
    candidates = deduplicate_candidates(candidates)
    write_csv(args.root / "nominated_candidates.csv", candidates or [{"status": "none"}])
    e2 = read_json(CAMPAIGN_ROOT / "E2_pilot100_v1" / "report_summary.json")
    mechanism_value = bool(
        e2.get("causal_mechanism_value")
        or e2.get("partial_state_mechanism_value")
    )
    passed = bool(candidates) and mechanism_value
    bootstrap_rows = []
    if candidates:
        selected_path = ROOT / candidates[0]["path"]
        for baseline_path in (alpha_path, no_projection_path):
            if selected_path == baseline_path:
                continue
            for bucket in ("rewrite", "declarative_paraphrase"):
                row = paired_bootstrap(selected_path, baseline_path, bucket=bucket)
                row.update(
                    {
                        "left": selected_path.name,
                        "right": baseline_path.name,
                    }
                )
                bootstrap_rows.append(row)
    write_csv(args.root / "paired_bootstrap.csv", bootstrap_rows or [{"status": "no feasible primary"}])
    selection = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "F1_dev200",
        "created_at_utc": now_utc(),
        "primary_candidate": candidates[0] if candidates else None,
        "nominated_candidates": candidates,
        "candidate_count": len(candidates),
        "causal_or_partial_state_mechanism_value": mechanism_value,
        "acceptance_pass": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.root / "dev_selection.json", selection)
    write_json(args.root / "report_summary.json", selection)
    write_json(
        args.root / "validation_report.json",
        {
            "at_most_three_candidates": len(candidates) <= 3,
            "at_least_one_hard_feasible_main_candidate": bool(candidates),
            "mechanism_value": mechanism_value,
            "acceptance_pass": passed,
        },
    )
    record_stage(
        "F1_dev200",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.root,
        started_at_utc=started,
        notes="Dev200 selection applied hard constraints before aggregate ranking.",
        next_stage="F2_scaling" if passed else "H_final_package",
    )
    print(json.dumps({"acceptance_pass": passed, "candidate_count": len(candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
