#!/usr/bin/env python3
"""Stress-aware feasibility, selection, and paired bootstrap for DNPE methods."""

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
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import CAMPAIGN_ID, now_utc, read_json, write_csv, write_json


def harmonic(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def base_tfpr(report: Mapping[str, Any], bucket: str) -> float:
    return float(report.get("base_summary", {}).get(bucket, {}).get("target_new_tfpr_or_exact", 0.0))


def locality_score(report: Mapping[str, Any]) -> float:
    values = []
    for bucket in ("same_subject", "near_locality", "far_locality"):
        value = report.get("edited_summary", {}).get(bucket, {}).get("base_agreement")
        if value is not None:
            values.append(float(value))
    return mean(values) if values else 0.0


def score_report(report: Mapping[str, Any], path: Path) -> dict[str, Any]:
    rewrite = float(report["rewrite_exact"])
    paraphrase = float(report["declarative_paraphrase_exact"])
    same = float(report.get("same_subject_tfpr", 0.0))
    near = float(report.get("near_tfpr", 0.0))
    far = float(report.get("far_tfpr", 0.0))
    malformed = float(report.get("malformed_rate", 0.0))
    locality = locality_score(report)
    constraints = {
        "rewrite_pass": rewrite >= 0.75,
        "paraphrase_pass": paraphrase >= 0.40,
        "same_subject_pass": same <= base_tfpr(report, "same_subject") + 0.03,
        "near_pass": near <= base_tfpr(report, "near_locality") + 0.03,
        "far_pass": far <= base_tfpr(report, "far_locality") + 0.03,
        "malformed_pass": malformed <= 0.05,
        "compute_pass": float(report.get("gpu_minutes_per_edit", 0.0)) <= 2.0,
    }
    feasible = all(constraints.values())
    score = harmonic((rewrite, paraphrase, min(max(locality, 0.0), 1.0)))
    return {
        "label": str(report.get("method") or path.parent.name),
        "path": str(path.parent.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path.parent),
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "target_token_f1": float(report.get("target_token_f1", 0.0)),
        "same_subject_tfpr": same,
        "same_subject_budget": base_tfpr(report, "same_subject") + 0.03,
        "near_tfpr": near,
        "near_budget": base_tfpr(report, "near_locality") + 0.03,
        "far_tfpr": far,
        "far_budget": base_tfpr(report, "far_locality") + 0.03,
        "malformed_rate": malformed,
        "locality_base_agreement": locality,
        "gpu_minutes_per_edit": float(report.get("gpu_minutes_per_edit", 0.0)),
        "selection_score": score,
        "constraint_pass": feasible,
        "constraint_violations": ";".join(key for key, value in constraints.items() if not value),
        "feasible_selection_score": score if feasible else "",
        **constraints,
    }


def read_prompt_rows(run_dir: Path, bucket: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with (run_dir / "edited_per_prompt.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("bucket") != bucket:
                continue
            values[str(row["case_id"])].append(1.0 if str(row.get("expected_hit", "")).casefold() in {"true", "1"} else 0.0)
    return {case_id: mean(scores) for case_id, scores in values.items()}


def paired_bootstrap(
    left_dir: Path,
    right_dir: Path,
    *,
    bucket: str,
    trials: int = 2000,
    seed: int = 260717101,
) -> dict[str, Any]:
    left = read_prompt_rows(left_dir, bucket)
    right = read_prompt_rows(right_dir, bucket)
    case_ids = sorted(set(left) & set(right))
    if not case_ids:
        raise RuntimeError(f"No paired rows for {bucket}")
    deltas = [left[case_id] - right[case_id] for case_id in case_ids]
    rng = random.Random(seed)
    samples = []
    for _ in range(trials):
        samples.append(mean(rng.choice(deltas) for _ in case_ids))
    samples.sort()
    return {
        "bucket": bucket,
        "num_edits": len(case_ids),
        "delta": mean(deltas),
        "ci_low": samples[int(0.025 * trials)],
        "ci_high": samples[min(trials - 1, int(0.975 * trials))],
        "trials": trials,
        "resampling_unit": "case_id",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap_pairs", nargs="*", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    scored = []
    reports = {}
    for path in args.reports:
        report = read_json(path)
        if report.get("analysis_500_used") or report.get("final_test_used"):
            raise RuntimeError(f"Locked result supplied to dev selection: {path}")
        row = score_report(report, path)
        scored.append(row)
        reports[row["label"]] = path.parent
    scored.sort(key=lambda row: (bool(row["constraint_pass"]), float(row["selection_score"])), reverse=True)
    write_csv(args.output_dir / "selection_scores.csv", scored)
    write_csv(args.output_dir / "selection_scores_feasible.csv", [row for row in scored if row["constraint_pass"]])
    bootstrap_rows = []
    for pair in args.bootstrap_pairs:
        left, right = pair.split("::", 1)
        for bucket in ("rewrite", "declarative_paraphrase"):
            row = paired_bootstrap(reports[left], reports[right], bucket=bucket)
            row.update({"left": left, "right": right})
            bootstrap_rows.append(row)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    feasible = [row for row in scored if row["constraint_pass"]]
    write_json(
        args.output_dir / "report_summary.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "num_methods": len(scored),
            "num_feasible": len(feasible),
            "feasible_winner": feasible[0]["label"] if feasible else None,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": bool(feasible),
        },
    )
    print(json.dumps({"num_methods": len(scored), "num_feasible": len(feasible), "winner": feasible[0]["label"] if feasible else None}, sort_keys=True))


if __name__ == "__main__":
    main()
