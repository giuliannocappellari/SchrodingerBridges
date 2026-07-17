#!/usr/bin/env python3
"""Run the predeclared F2 batch/sequential edit scaling comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    read_jsonl,
    record_stage,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.run_dnpe_causal_nullspace_sweep import run_one, site_layers, target_config, update_geometry
from scripts.run_dnpe_dev_selection import run_timerome


COUNTS = (1, 10, 50, 100)


def create_subsets(root: Path) -> dict[int, Path]:
    rows = read_jsonl(CAMPAIGN_ROOT / "protocol_v1" / "dnpe_pilot_100.jsonl")
    if len(rows) != 100:
        raise RuntimeError("Scaling source must contain exactly 100 edits")
    output = {}
    for count in COUNTS:
        path = root / f"scaling_{count}.jsonl"
        selected = rows[:count]
        if path.exists():
            existing = read_jsonl(path)
            if [row["case_id"] for row in existing] != [row["case_id"] for row in selected]:
                raise RuntimeError(f"Existing scaling subset drifted: {path}")
        else:
            write_jsonl(path, selected)
        output[count] = path
    return output


def previous_edit_retention(run_dir: Path, count: int) -> float:
    values = []
    with (run_dir / "edited_per_prompt.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("bucket") == "rewrite"]
    for row in rows[: min(10, count)]:
        values.append(
            float(str(row.get("expected_hit", "")).casefold() in {"true", "1"})
        )
    return mean(values) if values else 0.0


def result_row(method: str, count: int, run_dir: Path) -> dict[str, Any]:
    report = read_json(run_dir / "report_summary.json")
    geometry = (
        update_geometry(run_dir)
        if (run_dir / "target_value_diagnostics.json").exists()
        else {
            "mean_update_norm": 0.0,
            "max_update_norm": 0.0,
            "all_updates_finite": True,
            "mean_protected_dimension": 0.0,
        }
    )
    return {
        "method": method,
        "edit_count": count,
        "run": str(run_dir.relative_to(ROOT)),
        "rewrite_exact": float(report["rewrite_exact"]),
        "declarative_paraphrase_exact": float(
            report["declarative_paraphrase_exact"]
        ),
        "same_subject_tfpr": float(report.get("same_subject_tfpr", 0.0)),
        "near_tfpr": float(report.get("near_tfpr", 0.0)),
        "far_tfpr": float(report.get("far_tfpr", 0.0)),
        "previous_edit_retention": previous_edit_retention(run_dir, count),
        "gpu_minutes_per_edit": float(report.get("gpu_minutes_per_edit", 0.0)),
        "storage_bytes": int(report.get("storage_bytes", 0)),
        **geometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CAMPAIGN_ROOT / "F2_scaling_v1")
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    dev = read_json(CAMPAIGN_ROOT / "F1_dev200_selection_v1" / "report_summary.json")
    if not dev.get("acceptance_pass"):
        raise RuntimeError("F2 is illegal without a passed F1 dev selection")
    manifests = create_subsets(args.root)
    config = target_config()
    layers = site_layers()
    d4 = read_json(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "selected_nullspace_config.json"
    )["selected"]
    alpha = read_json(
        CAMPAIGN_ROOT
        / "B3_alphaedit_style_mdm_memit_v1"
        / "smoke_selection.json"
    )["selected"]
    rows = []
    for count, manifest in manifests.items():
        definitions = (
            (
                "mdm_memit",
                [3, 4, 5, 6],
                {
                    "learning_rate": 0.1,
                    "target_optimization_steps": 25,
                    "state_consistency_weight": 0.0,
                    "old_target_suppression_weight": 0.0,
                },
                "fully_masked",
                "random",
                None,
                0.95,
                0.0,
                0.0,
            ),
            (
                "alphaedit_style_mdm_memit",
                [3, 4, 5, 6],
                {
                    "learning_rate": 0.1,
                    "target_optimization_steps": 25,
                    "state_consistency_weight": 0.0,
                    "old_target_suppression_weight": 0.0,
                },
                "fully_masked",
                "random",
                CAMPAIGN_ROOT / "preservation_basis_v1",
                float(alpha["protected_variance"]),
                float(alpha["update_ridge"]),
                0.0,
            ),
            (
                "causal_partial_state_nullspace_memit",
                layers,
                config,
                config["partial_mask_schedule"],
                config["reveal_policy"],
                CAMPAIGN_ROOT / "D1_state_banks_v1",
                float(d4["protected_variance"]),
                float(d4["update_ridge"]),
                float(d4["lambda_identity"]),
            ),
        )
        for (
            method,
            method_layers,
            method_config,
            schedule,
            reveal,
            basis,
            variance,
            ridge,
            identity,
        ) in definitions:
            output = args.root / f"{method}_n{count}"
            run_one(
                manifest=manifest,
                output=output,
                method=method,
                layers=method_layers,
                config=method_config,
                schedule=str(schedule),
                reveal=str(reveal),
                basis_dir=basis,
                protected_variance=variance,
                update_ridge=ridge,
                lambda_identity=identity,
                resume=bool(args.resume),
            )
            rows.append(result_row(method, count, output))
        timerome = args.root / f"timerome_dlm_style_residual_memory_n{count}"
        run_timerome(manifest, timerome, resume=bool(args.resume))
        rows.append(
            result_row("timerome_dlm_style_residual_memory", count, timerome)
        )
    by_method_count = {
        (row["method"], int(row["edit_count"])): row for row in rows
    }
    main_one = by_method_count[("causal_partial_state_nullspace_memit", 1)]
    main_fifty = by_method_count[("causal_partial_state_nullspace_memit", 50)]
    source_report = read_json(
        ROOT / str(main_fifty["run"]) / "report_summary.json"
    )
    base_same = float(
        source_report.get("base_summary", {})
        .get("same_subject", {})
        .get("target_new_tfpr_or_exact", 0.0)
    )
    acceptance = {
        "fifty_edit_rewrite_retains_80_percent": float(main_fifty["rewrite_exact"])
        >= 0.80 * float(main_one["rewrite_exact"])
        if float(main_one["rewrite_exact"]) > 0
        else False,
        "same_subject_within_base_plus_0_05": float(main_fifty["same_subject_tfpr"])
        <= base_same + 0.05,
        "previous_edit_retention_at_least_0_80": float(
            main_fifty["previous_edit_retention"]
        )
        >= 0.80,
        "update_statistics_finite": all(
            row["all_updates_finite"]
            and math.isfinite(float(row["mean_update_norm"]))
            for row in rows
        ),
    }
    passed = all(acceptance.values())
    write_csv(args.root / "sequential_edit_table.csv", rows)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "F2_scaling",
        "created_at_utc": now_utc(),
        "edit_counts": list(COUNTS),
        "methods": sorted({row["method"] for row in rows}),
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "single_edit_claim_may_continue_if_scaling_fails": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.root / "report_summary.json", report)
    write_json(args.root / "validation_report.json", acceptance)
    record_stage(
        "F2_scaling",
        status="passed" if passed else "failed_secondary_scaling",
        acceptance_pass=passed,
        output_dir=args.root,
        started_at_utc=started,
        notes="Batch/sequential scaling completed at 1,10,50,100 edits.",
        next_stage="F3_second_backbone",
    )
    print(json.dumps({"acceptance_pass": passed, "num_runs": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
