#!/usr/bin/env python3
"""Reconcile M4 after correcting its synthetic analytical-test fixture.

This script never loads a model or reruns decoding. It preserves the original
reports, reruns only the finite-state analytical checks, and updates M4's
terminal status when the already-written scientific outputs were otherwise
valid and SB-positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import STATE_ROOT, git_commit, now_utc, record_stage, write_json
from scripts.run_mask_pattern_sb_track import M4_ROOT, _analytical_tests


FAILED_CHECK = "higher_beta_favors_lower_cost"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _backup(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copy2(source, destination)


def _record_reconciliation_if_missing(output_dir: Path, started_at_utc: str) -> None:
    history_path = STATE_ROOT / "stage_history.csv"
    rows: list[dict[str, str]] = []
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    already_recorded = any(
        row.get("stage") == "M4_complete"
        and str(row.get("acceptance_pass", "")).casefold() == "true"
        and "analytical_fixture_reconciled" in str(row.get("notes", ""))
        for row in rows
    )
    if not already_recorded:
        record_stage(
            stage="M4_complete",
            track="M4",
            status="passed",
            output_dir=output_dir,
            acceptance_pass=True,
            started_at_utc=started_at_utc,
            notes="analytical_fixture_reconciled; no model load or decoding rerun",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=M4_ROOT)
    args = parser.parse_args()
    started_at_utc = now_utc()
    output_dir = args.output_dir
    report_path = output_dir / "report_summary.json"
    analytical_path = output_dir / "analytical_test_report.json"
    final_report_path = output_dir / "final_track_report.md"

    if not report_path.exists() or not analytical_path.exists():
        raise FileNotFoundError("M4 terminal and analytical reports are required")

    report = _read(report_path)
    if report.get("analytical_test_correction", {}).get("status") == "validated":
        if not report.get("acceptance_pass") or not _read(analytical_path).get("acceptance_pass"):
            raise RuntimeError("Existing M4 reconciliation is internally inconsistent")
        _record_reconciliation_if_missing(output_dir, started_at_utc)
        print(json.dumps({"acceptance_pass": True, "already_reconciled": True}))
        return

    original_analytical = _read(analytical_path)
    failed = sorted(key for key, value in original_analytical.get("checks", {}).items() if not value)
    if failed != [FAILED_CHECK]:
        raise RuntimeError(f"Unexpected original analytical failures: {failed}")
    if report.get("analytical_tests_pass") is not False:
        raise RuntimeError("M4 report does not record the expected analytical-only failure")
    if not report.get("integration_smoke_pass") or not report.get("sb_specific_positive_result"):
        raise RuntimeError("M4 scientific outputs are not eligible for analytical-only reconciliation")
    if report.get("old_analysis_500_used") or report.get("old_final_test_used"):
        raise RuntimeError("Locked historical split usage detected")

    _backup(analytical_path, output_dir / "analytical_test_report_pre_fix.json")
    _backup(report_path, output_dir / "report_summary_pre_analytical_fix.json")
    if final_report_path.exists():
        _backup(final_report_path, output_dir / "final_track_report_pre_analytical_fix.md")

    analytical_pass = _analytical_tests(output_dir)
    corrected_analytical = _read(analytical_path)
    if not analytical_pass or not corrected_analytical.get("acceptance_pass"):
        raise RuntimeError("Corrected analytical validation still failed")

    correction = {
        "status": "validated",
        "reconciled_at_utc": now_utc(),
        "reconciliation_git_commit": git_commit(),
        "model_loaded": False,
        "decoding_rerun": False,
        "scientific_outputs_changed": False,
        "original_failed_check": FAILED_CHECK,
        "root_cause": (
            "The original synthetic fixture assigned equal total cost to every complete "
            "reveal permutation, so a lower-cost first-action preference was not implied."
        ),
        "correction": (
            "Use an order-dependent trajectory-cost fixture while retaining normalization, "
            "beta-zero, terminal, no-forcing, and brute-force-DP checks."
        ),
        "preserved_original_reports": [
            "analytical_test_report_pre_fix.json",
            "report_summary_pre_analytical_fix.json",
            "final_track_report_pre_analytical_fix.md",
        ],
    }
    report["analytical_tests_pass"] = True
    report["acceptance_pass"] = bool(
        report["integration_smoke_pass"]
        and report["sb_specific_positive_result"]
        and analytical_pass
    )
    report["analytical_test_correction"] = correction
    write_json(report_path, report)
    write_json(
        output_dir / "analytical_test_correction_report.json",
        {
            **correction,
            "acceptance_pass": report["acceptance_pass"],
            "original_analytical_report": original_analytical,
            "corrected_analytical_report": corrected_analytical,
        },
    )

    final_report_path.write_text(
        "# M4 Exact Mask-Pattern Schrodinger Bridge\n\n"
        "Status: **passed after analytical-fixture correction**\n\n"
        "- Integration smoke: passed\n"
        "- Corrected analytical finite-state checks: passed\n"
        "- SB-specific positive criterion: passed\n"
        "- Model loaded during reconciliation: no\n"
        "- Decoding rerun during reconciliation: no\n"
        "- Original reports preserved with `pre_fix` suffixes\n",
        encoding="utf-8",
    )
    _record_reconciliation_if_missing(output_dir, started_at_utc)
    print(json.dumps({"acceptance_pass": report["acceptance_pass"], "already_reconciled": False}))


if __name__ == "__main__":
    main()
