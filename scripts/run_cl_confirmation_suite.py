#!/usr/bin/env python3
"""Confirm every breadth-first pilot winner on the frozen fresh stream."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    autonomous_enabled,
    now_utc,
    read_csv,
    read_json,
    record_stage,
    update_track,
    write_csv,
    write_json,
)
from scripts.run_cl_track_suite import (
    KL_BASELINE,
    MATCHED_NON_SB,
    MECHANISM_BASELINE,
    method_run_dir,
)


PILOT_ROOT = CAMPAIGN_ROOT / "D_breadth_first_pilots_v1"
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "F_fresh_confirmation_v1"


def frozen_editor_command(
    *,
    pilot_dir: Path,
    manifest: Path,
    output_dir: Path,
) -> list[str]:
    config = read_json(pilot_dir / "run_config.json")
    memit = config["memit"]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_cl_sequential_editor.py"),
        "--manifest",
        str(manifest),
        "--retention_manifest",
        str(PROTOCOL_ROOT / "base_denoising_retention_500.jsonl"),
        "--output_dir",
        str(output_dir),
        "--method",
        str(config["method"]),
        "--model_id",
        str(config["model_id"]),
        "--model_revision",
        str(config["model_revision"]),
        "--layers",
        ",".join(map(str, config["layers"])),
        "--covariance_dir",
        str(CAMPAIGN_ROOT / "B1_covariance_cache_v1"),
        "--covariance_representation",
        str(config["covariance_representation"]),
        "--target_optimization_steps",
        str(memit["target_optimization_steps"]),
        "--learning_rate",
        str(memit["learning_rate"]),
        "--covariance_weight",
        str(memit["covariance_weight"]),
        "--lowrank_rank",
        str(config["lowrank_rank"]),
        "--lora_rank",
        str(config["lora_rank"]),
        "--lora_steps",
        str(config["lora_steps"]),
        "--lora_learning_rate",
        str(config.get("lora_learning_rate", 1e-3)),
        "--replay_items_per_block",
        str(config["replay_items_per_block"]),
        "--relation_overlap_threshold",
        str(config.get("relation_overlap_threshold", 0.20)),
        "--decode_batch_size",
        "16",
        "--allow_confirmation",
        "1",
    ]
    return command


def run_process(name: str, command: list[str], log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{name}.log"
    exit_path = log_root / f"{name}.exitcode"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    exit_path.write_text(f"{process.returncode}\n", encoding="utf-8")
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(f"{name} failed with exit {process.returncode}:\n" + "\n".join(tail))


def ensure_confirmation_run(
    *,
    name: str,
    pilot_dir: Path,
    output_dir: Path,
    log_root: Path,
) -> dict[str, Any]:
    report_path = output_dir / "report_summary.json"
    if not report_path.is_file():
        if output_dir.exists():
            raise RuntimeError(f"Partial confirmation output requires integrity review: {output_dir}")
        print(f"START {name}", flush=True)
        run_process(
            name,
            frozen_editor_command(
                pilot_dir=pilot_dir,
                manifest=PROTOCOL_ROOT / "cf_cl_confirmation_200.jsonl",
                output_dir=output_dir,
            ),
            log_root,
        )
    report = read_json(report_path)
    if not report.get("acceptance_pass") or not report.get("all_metrics_finite"):
        raise RuntimeError(f"Invalid confirmation report: {report_path}")
    if report.get("analysis_500_used") or report.get("final_test_used"):
        raise RuntimeError(f"Locked split contamination in {report_path}")
    return report


def confirmation_acceptance(
    pilot_classes: list[str],
    confirmation_row: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    def metric(name: str, default: float = 1.0) -> float:
        value = confirmation_row.get(name)
        return default if value in {None, ""} else float(value)

    confirmation_classes = [
        value for value in str(confirmation_row.get("success_classes") or "").split(",") if value
    ]
    reasons = []
    if not set(pilot_classes) & set(confirmation_classes):
        reasons.append("pilot_success_class_not_reproduced")
    if metric("same_subject_tfpr") > 0.03:
        reasons.append("same_subject_tfpr_above_0.03")
    if metric("near_tfpr") > 0.03:
        reasons.append("near_tfpr_above_0.03")
    if metric("far_tfpr") > 0.03:
        reasons.append("far_tfpr_above_0.03")
    if metric("base_retention_loss_fraction") > 0.05:
        reasons.append("base_retention_loss_above_0.05")
    if metric("malformed_rate") > 0.05:
        reasons.append("malformed_above_0.05")
    return not reasons, reasons


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_report = args.output_dir / "report_summary.json"
    if final_report.is_file():
        print(f"Confirmation suite already complete: {final_report}")
        return
    pilot_summary = read_json(PILOT_ROOT / "report_summary.json")
    if not pilot_summary.get("all_mandatory_tracks_terminal"):
        raise RuntimeError("All mandatory pilots must be terminal before confirmation")
    track_rows = read_csv(PILOT_ROOT / "track_results.csv")
    eligible = [row for row in track_rows if row.get("selected_candidate")]
    started = time.monotonic()
    started_at = now_utc()
    log_root = ROOT / "logs" / CAMPAIGN_ID / "F_fresh_confirmation_v1"
    result_rows = []
    if eligible:
        baseline_output = args.output_dir / "comparison_baseline_cf_confirmation200"
        ensure_confirmation_run(
            name="confirmation_comparison_baseline",
            pilot_dir=KL_BASELINE,
            output_dir=baseline_output,
            log_root=log_root,
        )
    else:
        baseline_output = args.output_dir / "comparison_baseline_cf_confirmation200"

    for pilot_row in eligible:
        track = str(pilot_row["track_id"])
        method = str(pilot_row["selected_candidate"])
        pilot_dir = method_run_dir(PILOT_ROOT, method, "pilot100")
        candidate_output = args.output_dir / "method_runs" / f"{track}_{method}_cf_confirmation200"
        ensure_confirmation_run(
            name=f"{track.lower()}_{method}_cf_confirmation200",
            pilot_dir=pilot_dir,
            output_dir=candidate_output,
            log_root=log_root,
        )
        mechanism_method = MECHANISM_BASELINE.get(track)
        mechanism_output = None
        if mechanism_method:
            mechanism_output = args.output_dir / "comparators" / f"{mechanism_method}_cf_confirmation200"
            ensure_confirmation_run(
                name=f"confirmation_{mechanism_method}",
                pilot_dir=method_run_dir(PILOT_ROOT, mechanism_method, "pilot100"),
                output_dir=mechanism_output,
                log_root=log_root,
            )
        matched_method = MATCHED_NON_SB.get(track)
        matched_output = None
        if matched_method:
            matched_output = args.output_dir / "comparators" / f"{matched_method}_cf_confirmation200"
            ensure_confirmation_run(
                name=f"confirmation_{matched_method}",
                pilot_dir=method_run_dir(PILOT_ROOT, matched_method, "pilot100"),
                output_dir=matched_output,
                log_root=log_root,
            )
        report_dir = args.output_dir / "track_reports" / f"{track}_confirmation_v1"
        if not (report_dir / "report_summary.json").is_file():
            command = [
                sys.executable,
                str(ROOT / "scripts" / "report_cl_track.py"),
                "--track",
                track,
                "--baseline_dir",
                str(baseline_output),
                "--candidate_dir",
                str(candidate_output),
                "--output_dir",
                str(report_dir),
                "--update_state",
                "0",
            ]
            if mechanism_output:
                command.extend(("--mechanism_baseline_dir", str(mechanism_output)))
            if matched_output:
                command.extend(("--matched_non_sb_dir", str(matched_output)))
            run_process(f"{track.lower()}_confirmation_report", command, log_root)
        confirmation_candidates = read_csv(report_dir / "candidate_results.csv")
        if len(confirmation_candidates) != 1:
            raise RuntimeError(f"Expected one confirmation candidate for {track}")
        pilot_report = read_json(PILOT_ROOT / "track_reports" / f"{track}_pilot_v1" / "report_summary.json")
        passed, reasons = confirmation_acceptance(
            list(pilot_report.get("selected_success_classes") or []),
            confirmation_candidates[0],
        )
        status = "confirmed" if passed else "confirmation_failed"
        update_track(
            track,
            status=status,
            nominated_candidate=method if passed else "",
            report_path=str((report_dir / "report_summary.json").relative_to(ROOT)),
            confirmation_pass=passed,
            confirmation_failed_candidate=method if not passed else None,
        )
        if not passed:
            (report_dir / "confirmation_stop_checkpoint.md").write_text(
                f"# {track} Confirmation Stop\n\n"
                f"- Method: `{method}`\n"
                f"- Failed checks: `{','.join(reasons)}`\n"
                "- Decision: bounded negative; do not tune on confirmation.\n",
                encoding="utf-8",
            )
        result_rows.append(
            {
                "track_id": track,
                "method": method,
                "status": status,
                "confirmation_pass": passed,
                "failed_checks": ",".join(reasons),
                "pilot_success_classes": ",".join(pilot_report.get("selected_success_classes") or []),
                "confirmation_success_classes": confirmation_candidates[0]["success_classes"],
                "report_path": str((report_dir / "report_summary.json").relative_to(ROOT)),
                "kamel_confirmation_status": "not_run_no_multi_token_claim_preselected",
            }
        )

    all_terminal = len(result_rows) == len(eligible)
    confirmed = [row for row in result_rows if row["confirmation_pass"]]
    write_csv(args.output_dir / "confirmation_results.csv", result_rows)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "manifest": str((PROTOCOL_ROOT / "cf_cl_confirmation_200.jsonl").relative_to(ROOT)),
            "fresh_confirmation_no_tuning": True,
            "eligible_tracks": [row["track_id"] for row in eligible],
            "kamel_multi_token_claim_made": False,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "F_fresh_confirmation",
        "created_at_utc": now_utc(),
        "num_eligible_tracks": len(eligible),
        "num_confirmed_tracks": len(confirmed),
        "confirmed_tracks": [row["track_id"] for row in confirmed],
        "all_eligible_tracks_terminal": all_terminal,
        "analysis_500_used": False,
        "final_test_used": False,
        "runtime_seconds": time.monotonic() - started,
        "acceptance_pass": all_terminal,
    }
    write_json(final_report, report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "fresh_stream_only": True,
            "no_confirmation_tuning": True,
            "all_eligible_tracks_terminal": all_terminal,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": all_terminal,
        },
    )
    record_stage(
        "F_fresh_confirmation",
        status="passed",
        acceptance_pass=all_terminal,
        output_dir=args.output_dir,
        started_at_utc=started_at,
        notes=f"confirmed_tracks={','.join(row['track_id'] for row in confirmed) or 'none'}",
        next_stage="G_conditional_tracks",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
