#!/usr/bin/env python3
"""Select and validate the bounded AlphaEdit-style MDM-MEMIT baseline."""

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


def _locality_score(report: dict) -> float:
    summary = report["edited_summary"]
    values = [
        summary.get(bucket, {}).get("base_agreement")
        for bucket in ("same_subject", "near_locality", "far_locality")
    ]
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / max(len(numeric), 1)


def select_smoke(root: Path, baseline_path: Path) -> dict:
    baseline = read_json(baseline_path)
    rows = []
    for path in sorted(root.glob("smoke_variance_*_ridge_*")):
        report_path = path / "report_summary.json"
        if not report_path.exists():
            continue
        report = read_json(report_path)
        row = {
            "run": path.name,
            "path": str(path.relative_to(ROOT)),
            "protected_variance": float(report["protected_variance"]),
            "update_ridge": float(report["memit"].get("update_ridge", 0.0)),
            "rewrite_exact": float(report["rewrite_exact"]),
            "paraphrase_exact": float(report["declarative_paraphrase_exact"]),
            "same_subject_tfpr": float(report["same_subject_tfpr"]),
            "near_tfpr": float(report["near_tfpr"]),
            "far_tfpr": float(report["far_tfpr"]),
            "malformed_rate": float(report["malformed_rate"]),
            "locality_score": _locality_score(report),
        }
        row["efficacy_valid"] = row["rewrite_exact"] >= float(baseline["rewrite_exact"]) - 0.10
        row["locality_improved"] = (
            row["same_subject_tfpr"] < float(baseline["same_subject_tfpr"])
            or row["near_tfpr"] < float(baseline["near_tfpr"])
            or row["far_tfpr"] < float(baseline["far_tfpr"])
            or row["locality_score"] > _locality_score(baseline)
        )
        row["finite"] = all(math.isfinite(row[key]) for key in ("rewrite_exact", "paraphrase_exact", "same_subject_tfpr", "locality_score"))
        rows.append(row)
    if not rows:
        raise RuntimeError("No AlphaEdit-style smoke runs")
    eligible = [row for row in rows if row["efficacy_valid"] and row["finite"]]
    selected = max(
        eligible or rows,
        key=lambda row: (
            row["efficacy_valid"],
            row["locality_improved"],
            row["locality_score"],
            row["rewrite_exact"] + row["paraphrase_exact"],
            -row["protected_variance"],
        ),
    )
    write_csv(root / "smoke_grid_summary.csv", rows)
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "selection_source": "dnpe_smoke_20_only",
        "selected": selected,
        "bounded_grid": {
            "protected_variance": [0.90, 0.95, 0.99],
            "update_ridge": [1e-4, 1e-3, 1e-2],
            "staged_not_full_cartesian": True,
        },
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "smoke_selection.json", payload)
    return payload


def validate_pilot(root: Path, baseline_path: Path) -> dict:
    selected = read_json(root / "smoke_selection.json")["selected"]
    pilot = read_json(root / "pilot100_selected" / "report_summary.json")
    baseline = read_json(baseline_path)
    acceptance = {
        "rewrite_within_0_10_of_mdm_memit": float(pilot["rewrite_exact"]) >= float(baseline["rewrite_exact"]) - 0.10,
        "no_numerical_collapse": bool(pilot["rollback_checksum_pass"]) and float(pilot["malformed_rate"]) <= 0.05,
        "projector_dimension_reported": bool(pilot.get("protected_basis_dir")),
        "same_subject_or_locality_improves": (
            float(pilot["same_subject_tfpr"]) < float(baseline["same_subject_tfpr"])
            or float(pilot["near_tfpr"]) < float(baseline["near_tfpr"])
            or float(pilot["far_tfpr"]) < float(baseline["far_tfpr"])
            or _locality_score(pilot) > _locality_score(baseline)
        ),
    }
    passed = all(acceptance.values())
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "B3_alphaedit_style",
        "created_at_utc": now_utc(),
        "selected_smoke_config": selected,
        "pilot100": str((root / "pilot100_selected").relative_to(ROOT)),
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "reproduction_label": "alphaedit_style_mdm_memit",
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    write_json(root / "validation_report.json", acceptance)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select_smoke", "validate_pilot"), required=True)
    parser.add_argument("--root", type=Path, default=CAMPAIGN_ROOT / "B3_alphaedit_style_mdm_memit_v1")
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    started = now_utc()
    report = select_smoke(args.root, args.baseline) if args.phase == "select_smoke" else validate_pilot(args.root, args.baseline)
    if args.phase == "validate_pilot":
        record_stage(
            "B3_alphaedit_style",
            status="passed" if report["acceptance_pass"] else "failed",
            acceptance_pass=bool(report["acceptance_pass"]),
            output_dir=args.root,
            started_at_utc=started,
            notes="AlphaEdit-style baseline validated on fresh pilot100.",
            next_stage="B4_timerome_style",
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
