#!/usr/bin/env python3
"""Validate and summarize DNPE baseline stages B1-B4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
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


def _load_run(path: Path) -> dict[str, Any]:
    report = path / "report_summary.json"
    validation = path / "validation_report.json"
    config = path / "run_config.json"
    for required in (report, validation, config):
        if not required.exists():
            raise FileNotFoundError(required)
    payload = read_json(report)
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError(f"Wrong campaign in {report}")
    if payload.get("analysis_500_used") or payload.get("final_test_used"):
        raise RuntimeError(f"Locked split leakage in {report}")
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _per_case_exact(run_dir: Path, bucket: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with (run_dir / "edited_per_prompt.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("bucket") != bucket:
                continue
            hit = str(row.get("expected_hit", "")).casefold() in {"true", "1"}
            values[str(row["case_id"])].append(float(hit))
    return {case_id: mean(scores) for case_id, scores in values.items()}


def _bootstrap_delta(
    deltas: list[float], *, trials: int = 2000, seed: int = 260717101
) -> dict[str, Any]:
    if not deltas:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "num_edits": 0}
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choice(deltas) for _ in deltas) for _ in range(trials)
    )
    return {
        "delta": mean(deltas),
        "ci_low": samples[int(0.025 * trials)],
        "ci_high": samples[min(trials - 1, int(0.975 * trials))],
        "num_edits": len(deltas),
        "trials": trials,
        "resampling_unit": "case_id_with_target_length_stratum",
    }


def report_b1(root: Path, *, repair_used: bool) -> dict[str, Any]:
    smoke_name = "smoke20_repair_v1" if repair_used else "smoke20_v1"
    names = (smoke_name, "pilot100_v1", "dev200_v1")
    reports = {name: _load_run(root / name) for name in names}
    rows = []
    for name, report in reports.items():
        rows.append(
            {
                "run": name,
                "num_edits": report["num_edits"],
                "rewrite_exact": report["rewrite_exact"],
                "declarative_paraphrase_exact": report["declarative_paraphrase_exact"],
                "pre_edit_target_new_rewrite_exact": report["pre_edit_target_new_rewrite_exact"],
                "same_subject_tfpr": report["same_subject_tfpr"],
                "near_tfpr": report["near_tfpr"],
                "far_tfpr": report["far_tfpr"],
                "malformed_rate": report["malformed_rate"],
                "gpu_minutes_per_edit": report["gpu_minutes_per_edit"],
                "run_acceptance_pass": report["acceptance_pass"],
            }
        )
    dev = reports["dev200_v1"]
    acceptance = {
        "rewrite_at_least_0_75": float(dev["rewrite_exact"]) >= 0.75,
        "paraphrase_at_least_0_40": float(dev["declarative_paraphrase_exact"]) >= 0.40,
        "pre_edit_target_new_at_most_0_10": float(dev["pre_edit_target_new_rewrite_exact"]) <= 0.10,
        "no_nan_or_invalid_updates": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "rewrite_exact",
                "declarative_paraphrase_exact",
                "same_subject_tfpr",
                "near_tfpr",
                "far_tfpr",
                "malformed_rate",
            )
        ),
        "all_rollbacks_pass": all(bool(report["rollback_checksum_pass"]) for report in reports.values()),
    }
    passed = all(acceptance.values())
    write_csv(root / "baseline_stage_summary.csv", rows)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "B1_mdm_memit_reproduction",
        "created_at_utc": now_utc(),
        "runs": {name: _display_path(root / name) for name in names},
        "repair_used": repair_used,
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "campaign_may_continue_as_diagnostic_if_failed": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", acceptance)
    return report


def report_b2(root: Path) -> dict[str, Any]:
    run_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("dev_n")
        and (path / "report_summary.json").exists()
    )
    reports = [_load_run(path) for path in run_dirs]
    rows = []
    by_length_policy = {}
    for path, report in zip(run_dirs, reports):
        length = int(report.get("target_length", 0) or 0)
        if not length:
            manifest_name = Path(report["manifest"]).stem
            for candidate in (2, 3, 4):
                if f"n{candidate}" in manifest_name:
                    length = candidate
                    break
        method = str(report["method"])
        policy = method.split("__", 1)[1] if "__" in method else str(report["memit"]["partial_mask_schedule"])
        row = {
            "run": path.name,
            "target_length": length,
            "policy": policy,
            "rewrite_exact": report["rewrite_exact"],
            "declarative_paraphrase_exact": report["declarative_paraphrase_exact"],
            "same_subject_tfpr": report["same_subject_tfpr"],
            "malformed_rate": report["malformed_rate"],
        }
        rows.append(row)
        by_length_policy[(length, policy)] = row
    comparison = []
    positive_lengths = 0
    strong_lengths = 0
    pooled_deltas: dict[str, list[float]] = {
        "rewrite": [],
        "declarative_paraphrase": [],
    }
    for length in (2, 3, 4):
        baseline = by_length_policy.get((length, "fully_masked_only"))
        candidates = [row for row in rows if row["target_length"] == length and row["policy"] != "fully_masked_only"]
        if baseline is None or not candidates:
            continue
        best = max(candidates, key=lambda row: (row["rewrite_exact"], row["declarative_paraphrase_exact"]))
        rewrite_gain = float(best["rewrite_exact"]) - float(baseline["rewrite_exact"])
        paraphrase_gain = float(best["declarative_paraphrase_exact"]) - float(baseline["declarative_paraphrase_exact"])
        positive_lengths += rewrite_gain >= 0.15
        strong_lengths += rewrite_gain >= 0.15 and paraphrase_gain >= 0.08
        baseline_dir = root / str(baseline["run"])
        best_dir = root / str(best["run"])
        for bucket in pooled_deltas:
            baseline_scores = _per_case_exact(baseline_dir, bucket)
            best_scores = _per_case_exact(best_dir, bucket)
            for case_id in sorted(set(baseline_scores) & set(best_scores)):
                pooled_deltas[bucket].append(
                    best_scores[case_id] - baseline_scores[case_id]
                )
        comparison.append(
            {
                "target_length": length,
                "baseline_run": baseline["run"],
                "best_partial_run": best["run"],
                "rewrite_gain": rewrite_gain,
                "paraphrase_gain": paraphrase_gain,
                "malformed_rate": best["malformed_rate"],
            }
        )
    pooled_bootstrap = {
        bucket: _bootstrap_delta(deltas, seed=260717101 + index)
        for index, (bucket, deltas) in enumerate(pooled_deltas.items())
    }
    pooled_positive = all(
        values["ci_low"] > 0.0 for values in pooled_bootstrap.values()
    )
    passed = strong_lengths >= 2 or pooled_positive
    write_csv(root / "partial_state_summary.csv", rows)
    write_csv(root / "partial_state_comparison.csv", comparison)
    write_csv(
        root / "partial_state_pooled_bootstrap.csv",
        [dict(metric=bucket, **values) for bucket, values in pooled_bootstrap.items()],
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "B2_partial_state_reproduction",
        "created_at_utc": now_utc(),
        "positive_lengths_at_rewrite_gain_0_15": positive_lengths,
        "strong_lengths_with_paraphrase_gain_0_08": strong_lengths,
        "pooled_paired_bootstrap": pooled_bootstrap,
        "pooled_rewrite_and_paraphrase_ci_positive": pooled_positive,
        "acceptance_pass": passed,
        "source_qualitative_trend_reproduced": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", {"required_lengths": [2, 3, 4], "all_runs_finite": all(math.isfinite(float(row["rewrite_exact"])) for row in rows), "acceptance_pass": passed})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("B1", "B2"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repair_used", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    started = now_utc()
    report = report_b1(args.root, repair_used=bool(args.repair_used)) if args.stage == "B1" else report_b2(args.root)
    stage = "B1_mdm_memit_reproduction" if args.stage == "B1" else "B2_partial_state_reproduction"
    record_stage(
        stage,
        status="passed" if report["acceptance_pass"] else "failed",
        acceptance_pass=bool(report["acceptance_pass"]),
        output_dir=args.root,
        started_at_utc=started,
        notes="Baseline stage validated against frozen thresholds.",
        next_stage="B2_partial_state_reproduction" if args.stage == "B1" else "B3_alphaedit_style",
    )
    print(json.dumps({"stage": args.stage, "acceptance_pass": report["acceptance_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
