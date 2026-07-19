#!/usr/bin/env python3
"""Write the required formal stop package for a bounded NDS track failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import CAMPAIGN_ID, CAMPAIGN_ROOT, git_commit, now_utc, read_json, update_track, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("N1", "N2", "N3", "N4", "N5", "N6"), required=True)
    parser.add_argument("--classification", choices=("implementation_failure", "protocol_infeasibility", "offline_scientific_failure", "actual_decode_failure", "generalization_failure", "budget_not_run", "infrastructure_blocked"), required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or CAMPAIGN_ROOT / f"{args.track}_track_stop_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    evidence_report = args.evidence_dir / "report_summary.json"
    evidence = read_json(evidence_report) if evidence_report.is_file() else {}
    artifacts = [
        {
            "path": str(path.relative_to(ROOT)),
            "exists": path.is_file(),
            "source": "bounded_track_evidence",
        }
        for path in sorted(args.evidence_dir.rglob("*"))
        if path.is_file()
    ]
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": args.track,
        "status": "pilot_failed" if args.classification != "budget_not_run" else "budget_not_run",
        "failure_classification": args.classification,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "evidence_dir": str(args.evidence_dir),
        "evidence_acceptance_pass": evidence.get("acceptance_pass"),
        "analysis_500_used": False,
        "final_test_used": False,
        "formal_stop_package_complete": True,
    }
    write_json(output / "report_summary.json", report)
    write_csv(
        output / "track_evidence_table.csv",
        [
            {"field": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
            for key, value in sorted(evidence.items())
        ],
    )
    write_json(output / "artifact_availability_manifest.json", {"track_id": args.track, "artifacts": artifacts})
    (output / "track_stop_checkpoint.md").write_text(
        f"# {args.track} Track Stop Checkpoint\n\n"
        f"Failure classification: `{args.classification}`. The bounded track may not be silently resumed or reclassified as a passed method.\n",
        encoding="utf-8",
    )
    (output / "negative_result_report.md").write_text(
        f"# {args.track} Bounded Negative Result\n\n"
        f"The track stopped at `{args.classification}` under its frozen criteria. This does not establish universal impossibility.\n",
        encoding="utf-8",
    )
    (output / "next_recommendation.md").write_text(
        "# Next Recommendation\n\nContinue the breadth-first campaign and retain this result as immutable evidence.\n",
        encoding="utf-8",
    )
    update_track(
        args.track,
        status=report["status"],
        output_dir=output,
        failure_classification=args.classification,
        notes="Formal bounded track stop package written.",
    )
    print(json.dumps({"track": args.track, "status": report["status"], "classification": args.classification}))


if __name__ == "__main__":
    main()
