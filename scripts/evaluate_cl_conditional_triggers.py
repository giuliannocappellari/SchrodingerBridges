#!/usr/bin/env python3
"""Evaluate the frozen C10-C14 triggers without inventing new experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    autonomous_enabled,
    now_utc,
    read_csv,
    read_json,
    record_stage,
    update_track,
    write_csv,
    write_json,
)


PILOT_ROOT = CAMPAIGN_ROOT / "D_breadth_first_pilots_v1"
CONFIRMATION_ROOT = CAMPAIGN_ROOT / "F_fresh_confirmation_v1"
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "G_conditional_tracks_v1"


def confirmed_equivalence_classes(
    confirmation_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    classes = set()
    for row in confirmation_rows:
        if str(row.get("confirmation_pass", "")).casefold() != "true":
            continue
        report = read_json(str(row["report_path"]))
        # report_path is the track report; resolve the actual selected run through
        # its candidate table so runtime implementation provenance is preserved.
        candidate_rows = read_csv(Path(str(row["report_path"])).parent / "candidate_results.csv")
        if len(candidate_rows) != 1:
            raise RuntimeError(f"Expected one confirmation candidate for {row['track_id']}")
        equivalence = candidate_rows[0].get("implementation_equivalence_class")
        if equivalence:
            classes.add(str(equivalence))
        if report.get("analysis_500_used") or report.get("final_test_used"):
            raise RuntimeError("Locked split use detected in confirmation report")
    return classes


def trigger_decisions(
    confirmation_rows: Sequence[Mapping[str, Any]],
    pilot_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    confirmed = {
        str(row["track_id"])
        for row in confirmation_rows
        if str(row.get("confirmation_pass", "")).casefold() == "true"
    }
    equivalence = confirmed_equivalence_classes(confirmation_rows)
    c10 = bool({"C1", "C4"} & confirmed)
    c11 = any(
        str(row.get("track_id")) == "C5"
        and str(row.get("mechanism_signal_pass", "")).casefold() == "true"
        and str(row.get("fisher_signal_present", "")).casefold() == "true"
        for row in pilot_rows
    ) and "C5" not in confirmed
    c12 = any(
        str(row.get("track_id")) in {"C1", "C4", "C5", "C6"}
        and float(row.get("current_rewrite_exact") or 0.0) >= 0.80
        and float(row.get("average_forgetting") or 0.0) > 0.10
        for row in pilot_rows
    )
    c13 = False  # No validated pre-edit risk AUC artifact is produced by the frozen pilots.
    c14 = len(equivalence) >= 2
    values = {
        "C10": (c10, "C1 or C4 confirmed with a finite rank-bounded branch"),
        "C11": (c11, "C5 Fisher/orthogonal mechanism signal without confirmation"),
        "C12": (c12, "strong parametric acquisition with identifiable forgetting"),
        "C13": (c13, "validated pre-edit risk AUC >= 0.80"),
        "C14": (c14, "at least two confirmed non-equivalent compatible components"),
    }
    return [
        {
            "track_id": track,
            "triggered": triggered,
            "status": "triggered_pending_pilot" if triggered else "not_triggered",
            "trigger_rule": rule,
            "confirmed_equivalence_class_count": len(equivalence),
        }
        for track, (triggered, rule) in values.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not autonomous_enabled():
        raise PermissionError("CL_DLLM_AUTONOMOUS_MODE=1 is required")
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (ROOT / args.output_dir).resolve()
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    confirmation = read_json(CONFIRMATION_ROOT / "report_summary.json")
    if not confirmation.get("all_eligible_tracks_terminal"):
        raise RuntimeError("Fresh confirmation must be terminal before conditional triggers")
    confirmation_rows = read_csv(CONFIRMATION_ROOT / "confirmation_results.csv")
    pilot_rows = []
    for track in (f"C{index}" for index in range(1, 10)):
        path = PILOT_ROOT / "track_reports" / f"{track}_pilot_v1" / "candidate_results.csv"
        pilot_rows.extend(read_csv(path))
    decisions = trigger_decisions(confirmation_rows, pilot_rows)
    pending = [row for row in decisions if row["triggered"]]
    write_csv(args.output_dir / "trigger_registry.csv", decisions)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "trigger_rules_frozen": True,
            "implementation_equivalence_deduplicated": True,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "G_conditional_trigger_evaluation",
        "created_at_utc": now_utc(),
        "num_triggered_tracks": len(pending),
        "triggered_tracks": [row["track_id"] for row in pending],
        "all_conditional_tracks_terminal": not pending,
        "analysis_500_used": False,
        "final_test_used": False,
        "trigger_evaluation_pass": True,
        "acceptance_pass": not pending,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "confirmation_terminal": True,
            "equivalence_classes_deduplicated": True,
            "unvalidated_risk_auc_used": False,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": True,
        },
    )
    for row in decisions:
        update_track(
            row["track_id"],
            status=row["status"],
            trigger_rule=row["trigger_rule"],
            triggered=bool(row["triggered"]),
        )
    if not pending:
        record_stage(
            "G_conditional_tracks",
            status="passed",
            acceptance_pass=True,
            output_dir=args.output_dir,
            started_at_utc=report["created_at_utc"],
            notes="no conditional track triggered",
            next_stage="H_final_selection",
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
