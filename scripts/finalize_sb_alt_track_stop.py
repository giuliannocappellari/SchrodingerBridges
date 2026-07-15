#!/usr/bin/env python3
"""Write a formal scientific stop package for one alternatives track."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    TRACKS,
    git_commit,
    now_utc,
    record_stage_event,
    repo_path,
    set_track_status,
    sha256_file,
    write_csv,
    write_json,
)


TRACK_ROOTS = {
    "T1": Path("runs/counterfact_learned_gate_raw_bridge_v1"),
    "T2": Path("runs/counterfact_activation_space_sb_v1"),
    "T3": Path("runs/counterfact_conditional_answer_span_csbm_v1"),
    "T4": Path("runs/counterfact_unbalanced_partial_csbm_v1"),
    "T5": Path("runs/counterfact_parameter_space_sb_v1"),
}

TRACK_NAMES = {
    "T1": "learned edit-intent gate + raw bridge",
    "T2": "activation-space Schrodinger bridge",
    "T3": "conditional answer-span categorical Schrodinger bridge",
    "T4": "unbalanced / partial categorical Schrodinger bridge",
    "T5": "parameter-space Schrodinger bridge",
}

REQUIRED_FILES = (
    "report_summary.json",
    "track_stop_checkpoint.md",
    "negative_result_report.md",
    "track_evidence_table.csv",
    "artifact_availability_manifest.json",
    "next_recommendation.md",
)


def write_text(path: Path, value: str) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(value.rstrip() + "\n", encoding="utf-8")


def artifact_row(path: str | Path, role: str) -> dict[str, Any]:
    relative = Path(path)
    full = repo_path(relative)
    return {
        "artifact_role": role,
        "path": str(relative),
        "exists": full.exists(),
        "size_bytes": full.stat().st_size if full.is_file() else "",
        "sha256": sha256_file(full) if full.is_file() else "",
    }


def parse_evidence(values: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role.strip() or not path.strip():
            raise ValueError(f"Evidence must use role=path, got {value!r}")
        parsed.append((role.strip(), path.strip()))
    return parsed


def build_stop_package(
    *,
    track_id: str,
    failure_category: str,
    result_summary: str,
    evidence: Sequence[tuple[str, str]],
    next_recommendation: str,
    bounded_rescue_used: bool,
    pilot_actual_decode_completed: bool,
    output_dir: str | Path | None = None,
) -> Path:
    if track_id not in TRACK_ROOTS:
        raise ValueError(f"Unknown track: {track_id}")
    track = next(item for item in TRACKS if item["id"] == track_id)
    output = Path(output_dir) if output_dir else TRACK_ROOTS[track_id] / "pilot_stop_package_v1"
    output_full = repo_path(output)
    if output_full.exists():
        raise FileExistsError(output_full)

    evidence_rows = [artifact_row(path, role) for role, path in evidence]
    missing = [row["path"] for row in evidence_rows if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Required stop evidence is missing: {missing}")

    report: dict[str, Any] = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_id": track_id,
        "track_protocol": track["protocol"],
        "track_name": TRACK_NAMES[track_id],
        "status": "formal_negative",
        "failure_category": failure_category,
        "scientific_hypothesis_tested": True,
        "pilot_actual_decode_completed": pilot_actual_decode_completed,
        "bounded_scientific_rescue_used": bounded_rescue_used,
        "analysis_500_used": False,
        "final_test_used": False,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "result_summary": result_summary,
        "required_files": list(REQUIRED_FILES),
    }
    write_json(output / "report_summary.json", report)
    write_csv(output / "track_evidence_table.csv", evidence_rows)
    write_json(
        output / "artifact_availability_manifest.json",
        {
            "track_id": track_id,
            "created_at_utc": now_utc(),
            "all_declared_required_evidence_present": True,
            "artifacts": evidence_rows,
        },
    )
    write_text(
        output / "track_stop_checkpoint.md",
        f"""# {track_id} Track Stop Checkpoint

Status: `formal_negative`

Failure category: `{failure_category}`

{result_summary}

The bounded pilot was evaluated under its predeclared criteria. No locked
analysis or final split was used, and the failed criteria were not lowered.
The campaign must continue to the next mandatory breadth-first pilot.
""",
    )
    write_text(
        output / "negative_result_report.md",
        f"""# {track_id} Negative Result

The {TRACK_NAMES[track_id]} pilot reached a bounded scientific stop.

{result_summary}

This result applies to the declared pilot and its allowed rescue policy. It
does not claim that every possible implementation of the hypothesis fails.
""",
    )
    write_text(
        output / "next_recommendation.md",
        f"""# Next Recommendation

{next_recommendation}
""",
    )
    missing_output = [name for name in REQUIRED_FILES if not (output_full / name).exists()]
    if missing_output:
        raise RuntimeError(f"Incomplete stop package: {missing_output}")

    set_track_status(
        track_id,
        "formal_negative",
        evidence_path=str(output),
        rescue_used=bounded_rescue_used,
    )
    record_stage_event(
        track=track_id,
        stage=f"{track_id}_pilot_stop",
        event="pilot_formal_negative",
        status="fail",
        notes=f"{failure_category}: {result_summary}",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_id", choices=sorted(TRACK_ROOTS), required=True)
    parser.add_argument("--failure_category", required=True)
    parser.add_argument("--result_summary", required=True)
    parser.add_argument("--next_recommendation", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--bounded_rescue_used", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pilot_actual_decode_completed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output = build_stop_package(
        track_id=args.track_id,
        failure_category=args.failure_category,
        result_summary=args.result_summary,
        evidence=parse_evidence(args.evidence),
        next_recommendation=args.next_recommendation,
        bounded_rescue_used=bool(args.bounded_rescue_used),
        pilot_actual_decode_completed=bool(args.pilot_actual_decode_completed),
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": "formal_negative", "output_dir": str(output)}))


if __name__ == "__main__":
    main()
