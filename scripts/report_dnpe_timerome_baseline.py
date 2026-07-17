#!/usr/bin/env python3
"""Validate and record the B4 TimeROME-DLM-style residual-memory baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import CAMPAIGN_ID, CAMPAIGN_ROOT, now_utc, read_json, record_stage, write_csv, write_json


def validate(root: Path) -> dict:
    rows = []
    for name in ("smoke20_v1", "pilot100_v1"):
        report = read_json(root / name / "report_summary.json")
        validation = read_json(root / name / "validation_report.json")
        if report.get("analysis_500_used") or report.get("final_test_used"):
            raise RuntimeError("Locked split was used in B4")
        rows.append(
            {
                "run": name,
                "num_edits": report["num_edits"],
                "rewrite_exact": report["rewrite_exact"],
                "declarative_paraphrase_exact": report[
                    "declarative_paraphrase_exact"
                ],
                "same_subject_tfpr": report["same_subject_tfpr"],
                "near_tfpr": report["near_tfpr"],
                "far_tfpr": report["far_tfpr"],
                "malformed_rate": report["malformed_rate"],
                "residual_memory_rank": report["residual_memory_rank"],
                "storage_bytes": report["storage_bytes"],
                "gpu_minutes_per_edit": report["gpu_minutes_per_edit"],
                "residual_memory_finite": report["residual_memory_finite"],
                "metrics_complete": validation["metrics_complete"],
            }
        )
    acceptance = {
        "temporal_localization_runs": all(row["num_edits"] > 0 for row in rows),
        "residual_memory_finite": all(row["residual_memory_finite"] for row in rows),
        "metrics_complete": all(row["metrics_complete"] for row in rows),
        "runtime_and_storage_reported": all(
            float(row["gpu_minutes_per_edit"]) >= 0
            and int(row["storage_bytes"]) > 0
            for row in rows
        ),
        "all_numeric_finite": all(
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
    }
    passed = all(acceptance.values())
    write_csv(root / "baseline_stage_summary.csv", rows)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "B4_timerome_style",
        "created_at_utc": now_utc(),
        "reproduction_label": "timerome_dlm_style_not_exact_reproduction",
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", acceptance)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=CAMPAIGN_ROOT / "B4_timerome_dlm_style_v1"
    )
    args = parser.parse_args()
    started = now_utc()
    report = validate(args.root)
    record_stage(
        "B4_timerome_style",
        status="passed" if report["acceptance_pass"] else "failed",
        acceptance_pass=bool(report["acceptance_pass"]),
        output_dir=args.root,
        started_at_utc=started,
        notes="TimeROME-DLM-style baseline validated as an inspired comparator.",
        next_stage="C1_standard_causal_tracing",
    )
    print(json.dumps({"acceptance_pass": report["acceptance_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
