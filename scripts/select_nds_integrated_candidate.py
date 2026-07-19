#!/usr/bin/env python3
"""Select one N6 composition from calibration-only actual-decode reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import CAMPAIGN_ID, CAMPAIGN_ROOT, git_commit, now_utc, read_json, sha256_file, update_track, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "N6_calibration_selection_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    baseline = read_json(args.baseline_dir / "report_summary.json")
    rows = []
    for path in args.candidate_dirs:
        report = read_json(path / "report_summary.json")
        if "calibration" not in Path(report["manifest"]).name.casefold():
            raise RuntimeError("N6 selection may use calibration reports only")
        feasible = (
            float(report["same_subject_tfpr"]) <= float(baseline["same_subject_tfpr"]) + 0.03
            and float(report["near_tfpr"]) <= float(baseline["near_tfpr"]) + 0.03
            and float(report["far_tfpr"]) <= float(baseline["far_tfpr"]) + 0.03
            and float(report["malformed_rate"]) <= 0.05
        )
        rows.append(
            {
                "candidate_dir": str(path),
                "candidate_id": report["method"],
                "rewrite_exact": report["rewrite_exact"],
                "declarative_paraphrase_exact": report["declarative_paraphrase_exact"],
                "same_subject_tfpr": report["same_subject_tfpr"],
                "near_tfpr": report["near_tfpr"],
                "far_tfpr": report["far_tfpr"],
                "protected_distributional_kl": report["protected_distributional_kl"],
                "selection_score": report["selection_score"],
                "feasible": feasible,
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    selected = max(feasible_rows, key=lambda row: float(row["selection_score"])) if feasible_rows else None
    write_csv(args.output_dir / "calibration_pareto.csv", rows)
    write_json(
        args.output_dir / "candidate_lock.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "track_id": "N6",
            "candidate_id": selected["candidate_id"] if selected else None,
            "candidate_dir": selected["candidate_dir"] if selected else None,
            "candidate_run_config_sha256": sha256_file(Path(selected["candidate_dir"]) / "run_config.json") if selected else None,
            "selected_on_calibration_only": True,
            "frozen_before_pilot": bool(selected),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N6",
        "stage": "calibration_selection",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "num_candidates": len(rows),
        "num_feasible_candidates": len(feasible_rows),
        "selected_candidate": selected,
        "triggered": True,
        "mechanism_pass": bool(selected),
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": bool(selected),
    }
    write_json(args.output_dir / "report_summary.json", report)
    update_track("N6", status="running" if selected else "pilot_failed", candidate_id=selected["candidate_id"] if selected else None, mechanism_pass=bool(selected), output_dir=args.output_dir)
    print(json.dumps({"selected_candidate": selected["candidate_id"] if selected else None}))


if __name__ == "__main__":
    main()
