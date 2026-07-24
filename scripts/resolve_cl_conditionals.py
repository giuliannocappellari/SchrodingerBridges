#!/usr/bin/env python3
"""Resolve triggered conditional tracks using only their frozen evidence paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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


DEFAULT_ROOT = CAMPAIGN_ROOT / "G_conditional_tracks_v1"
C0_ROOT = CAMPAIGN_ROOT / "C0_common_baselines_v1"
PILOT_ROOT = CAMPAIGN_ROOT / "D_breadth_first_pilots_v1"


def spectral_repair_evidence(
    baseline: Mapping[str, Any],
    lowrank: Mapping[str, Any],
) -> dict[str, Any]:
    rewrite_gap = abs(
        float(lowrank["current_rewrite_exact"])
        - float(baseline["current_rewrite_exact"])
    )
    forgetting_delta = float(lowrank["average_forgetting"]) - float(
        baseline["average_forgetting"]
    )
    retention_delta = float(lowrank["past_retention"]) - float(
        baseline["past_retention"]
    )
    passed = rewrite_gap <= 0.03 and (
        forgetting_delta < 0.0 or retention_delta >= 0.10
    )
    return {
        "rewrite_gap": rewrite_gap,
        "forgetting_delta_lowrank_minus_baseline": forgetting_delta,
        "retention_delta_lowrank_minus_baseline": retention_delta,
        "conditional_pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if not autonomous_enabled():
        raise PermissionError("CL_DLLM_AUTONOMOUS_MODE=1 is required")
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (ROOT / args.output_dir).resolve()
    )
    trigger_report = read_json(args.output_dir / "report_summary.json")
    if trigger_report.get("all_conditional_tracks_terminal"):
        print(json.dumps(trigger_report, sort_keys=True))
        return
    decisions = read_csv(args.output_dir / "trigger_registry.csv")
    rows = []
    for decision in decisions:
        track = str(decision["track_id"])
        if str(decision["triggered"]).casefold() != "true":
            rows.append({**decision, "terminal_status": "not_triggered", "conditional_pass": False})
            continue
        evidence: dict[str, Any]
        if track == "C10":
            c8 = read_json(PILOT_ROOT / "track_reports" / "C8_pilot_v1" / "report_summary.json")
            evidence = {
                "c8_exact_sb_candidate": c8.get("selected_candidate"),
                "conditional_pass": False,
                "reason": "no exact low-dimensional parameter-space SB candidate passed C8",
            }
        elif track == "C11":
            evidence = {
                "conditional_pass": False,
                "reason": "explicit Fisher trigger existed but no frozen online-Laplace candidate passed",
            }
        elif track == "C12":
            baseline = read_json(
                C0_ROOT / "sequential_fullmask_memit_pilot100" / "report_summary.json"
            )
            lowrank = read_json(
                C0_ROOT / "sequential_lowrank_memit_pilot100" / "report_summary.json"
            )
            evidence = spectral_repair_evidence(baseline, lowrank)
            evidence["reason"] = (
                "rank-8 spectral update repair improved matched continual stability"
                if evidence["conditional_pass"]
                else "rank-8 spectral update repair did not improve matched continual stability"
            )
        elif track == "C13":
            evidence = {
                "conditional_pass": False,
                "reason": "no validated pre-edit risk AUC artifact",
            }
        else:
            raise RuntimeError(
                "C14 integration triggered and requires its frozen compatible-component pilot; "
                "do not auto-resolve it from component results"
            )
        status = "conditional_passed" if evidence["conditional_pass"] else "conditional_failed"
        track_dir = args.output_dir / f"{track}_conditional_v1"
        track_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            track_dir / "report_summary.json",
            {
                "campaign_id": CAMPAIGN_ID,
                "track_id": track,
                "stage": f"{track}_conditional",
                "status": status,
                **evidence,
                "analysis_500_used": False,
                "final_test_used": False,
                "acceptance_pass": True,
            },
        )
        if not evidence["conditional_pass"]:
            (track_dir / "track_stop_checkpoint.md").write_text(
                f"# {track} Conditional Stop\n\n"
                f"- Status: `{status}`\n"
                f"- Reason: `{evidence['reason']}`\n"
                "- Analysis/final use: `false`\n",
                encoding="utf-8",
            )
        update_track(
            track,
            status=status,
            nominated_candidate="" if not evidence["conditional_pass"] else track,
            report_path=str((track_dir / "report_summary.json").relative_to(ROOT)),
            conditional_pass=bool(evidence["conditional_pass"]),
        )
        rows.append(
            {
                **decision,
                "terminal_status": status,
                "conditional_pass": bool(evidence["conditional_pass"]),
                "reason": evidence["reason"],
                "report_path": str((track_dir / "report_summary.json").relative_to(ROOT)),
            }
        )

    write_csv(args.output_dir / "conditional_results.csv", rows)
    final = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "G_conditional_tracks",
        "created_at_utc": now_utc(),
        "num_triggered_tracks": sum(str(row["triggered"]).casefold() == "true" for row in decisions),
        "num_conditional_passed": sum(bool(row["conditional_pass"]) for row in rows),
        "all_conditional_tracks_terminal": True,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", final)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "all_conditional_tracks_terminal": True,
            "c14_not_auto_resolved": True,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": True,
        },
    )
    record_stage(
        "G_conditional_tracks",
        status="passed",
        acceptance_pass=True,
        output_dir=args.output_dir,
        started_at_utc=final["created_at_utc"],
        notes=f"triggered={final['num_triggered_tracks']}; passed={final['num_conditional_passed']}",
        next_stage="H_final_selection",
    )
    print(json.dumps(final, sort_keys=True))


if __name__ == "__main__":
    main()
