#!/usr/bin/env python3
"""Run the frozen C1-C9 breadth-first continual pilot matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    autonomous_enabled,
    now_utc,
    read_json,
    record_stage,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT = CAMPAIGN_ROOT / "D_breadth_first_pilots_v1"
C0_SUITE = CAMPAIGN_ROOT / "C0_common_baselines_v1"
KL_BASELINE = CAMPAIGN_ROOT / "C0_kl_comparison_baseline_v1"
TRACK_ORDER = tuple(f"C{index}" for index in range(1, 10))

# These are the bounded repository-local pilot implementations. Source/exactness
# status is serialized by run_cl_sequential_editor.py and enforced by the track
# reporter; in particular, C7/C8 proxies cannot enter confirmation as SB methods.
TRACK_METHODS: dict[str, tuple[str, ...]] = {
    "C1": ("growth_shared", "growth_block", "growth_block_gate"),
    "C2": ("replay_partial",),
    "C3": ("sparse_routed_memory",),
    "C4": ("gated_adapter_expansion",),
    "C5": ("oedit_partial_memit",),
    "C6": ("lwf_partial_memit", "der_partial", "agem_partial"),
    "C7": ("bridge_replay",),
    "C8": ("sb_function_barycenter",),
    "C9": ("dual_memory_10", "dual_memory_25", "dual_memory_50"),
}

TRACK_COMPARATORS: dict[str, tuple[str, ...]] = {
    "C1": (),
    "C2": ("replay_clean",),
    "C3": ("growth_block",),
    "C4": ("growth_block",),
    "C5": (),
    "C6": ("replay_clean",),
    "C7": ("replay_partial",),
    "C8": ("growth_shared",),
    "C9": ("growth_block",),
}

MECHANISM_BASELINE = {
    "C2": "replay_clean",
    "C6": "replay_clean",
}

MATCHED_NON_SB = {
    "C7": "replay_partial",
    "C8": "growth_shared",
}

TRACK_PLAN_COVERAGE = {
    "C1": "shared branch; block branches; relation-gated block branches; partial-mask training",
    "C2": "clean replay comparator; partial-state label replay",
    "C3": "dense branch comparator; sparse subject+relation routed memory",
    "C4": "ungated branch comparator; relation-gated adapter expansion",
    "C5": "O-Edit protected-basis partial-state updates",
    "C6": "LwF adaptation; DER proxy; A-GEM proxy; clean replay comparator",
    "C7": "ordinary partial replay comparator; endpoint-biased bridge proxy",
    "C8": "linear/shared merge comparator; norm-weighted parameter barycenter proxy",
    "C9": "fast branches with consolidation intervals 10, 25, and 50",
}


def unique_methods() -> tuple[str, ...]:
    ordered: list[str] = []
    for track in TRACK_ORDER:
        for method in (*TRACK_METHODS[track], *TRACK_COMPARATORS[track]):
            if method not in ordered:
                ordered.append(method)
    return tuple(ordered)


def method_run_dir(output_dir: Path, method: str, scale: str) -> Path:
    return output_dir / "method_runs" / f"{method}_{scale}"


def editor_command(method: str, manifest: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_cl_sequential_editor.py"),
        "--manifest",
        str(manifest),
        "--retention_manifest",
        str(PROTOCOL_ROOT / "base_denoising_retention_500.jsonl"),
        "--output_dir",
        str(output_dir),
        "--method",
        method,
        "--covariance_dir",
        str(CAMPAIGN_ROOT / "B1_covariance_cache_v1"),
        "--covariance_representation",
        "diagonal",
        "--decode_batch_size",
        "16",
    ]


def run_process(name: str, command: list[str], log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{name}.log"
    exit_path = log_root / f"{name}.exitcode"
    print(f"START {name}", flush=True)
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


def require_valid_report(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "report_summary.json"
    if not report_path.is_file():
        raise RuntimeError(f"Missing report: {report_path}")
    report = read_json(report_path)
    if not report.get("acceptance_pass") or not report.get("all_metrics_finite"):
        raise RuntimeError(f"Invalid sequential-editor report: {report_path}")
    if report.get("analysis_500_used") or report.get("final_test_used"):
        raise RuntimeError(f"Locked split contamination in {report_path}")
    if report.get("protected_kl") is None:
        raise RuntimeError(f"Protected KL missing from {report_path}")
    return report


def ensure_run(
    *,
    name: str,
    method: str,
    manifest: Path,
    output_dir: Path,
    log_root: Path,
) -> dict[str, Any]:
    report_path = output_dir / "report_summary.json"
    if report_path.is_file():
        return require_valid_report(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"Partial output requires integrity review: {output_dir}")
    run_process(name, editor_command(method, manifest, output_dir), log_root)
    report = require_valid_report(output_dir)
    print(
        json.dumps(
            {
                "completed": name,
                "rewrite": report["current_rewrite_exact"],
                "paraphrase": report["current_paraphrase_exact"],
                "retention": report["past_retention"],
                "forgetting": report["average_forgetting"],
                "same_subject_tfpr": report["same_subject_tfpr"],
                "protected_kl": report["protected_kl"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def ensure_kl_baseline(log_root: Path) -> tuple[str, dict[str, Any]]:
    c0 = read_json(C0_SUITE / "report_summary.json")
    method = str(c0.get("selected_acquisition_baseline") or "")
    if not method:
        raise RuntimeError("C0 has no selected acquisition baseline")
    report = ensure_run(
        name="c0_kl_comparison_baseline_pilot100",
        method=method,
        manifest=PROTOCOL_ROOT / "cf_cl_pilot_100.jsonl",
        output_dir=KL_BASELINE,
        log_root=log_root,
    )
    return method, report


def report_command(track: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "report_cl_track.py"),
        "--track",
        track,
        "--baseline_dir",
        str(KL_BASELINE),
        "--output_dir",
        str(output_dir / "track_reports" / f"{track}_pilot_v1"),
    ]
    for method in TRACK_METHODS[track]:
        command.extend(("--candidate_dir", str(method_run_dir(output_dir, method, "pilot100"))))
    mechanism = MECHANISM_BASELINE.get(track)
    if mechanism:
        command.extend(
            ("--mechanism_baseline_dir", str(method_run_dir(output_dir, mechanism, "pilot100")))
        )
    matched = MATCHED_NON_SB.get(track)
    if matched:
        command.extend(
            ("--matched_non_sb_dir", str(method_run_dir(output_dir, matched, "pilot100")))
        )
    return command


def reports_for(methods: Iterable[str], output_dir: Path, scale: str) -> list[dict[str, Any]]:
    return [require_valid_report(method_run_dir(output_dir, method, scale)) for method in methods]


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
        print(f"Breadth-first suite already complete: {final_report}")
        return
    started = time.monotonic()
    started_at = now_utc()
    log_root = ROOT / "logs" / CAMPAIGN_ID / "D_breadth_first_pilots_v1"
    baseline_method, baseline_report = ensure_kl_baseline(log_root)
    track_rows = []
    method_rows = []
    for track_index, track in enumerate(TRACK_ORDER):
        required = tuple(dict.fromkeys((*TRACK_METHODS[track], *TRACK_COMPARATORS[track])))
        for scale, manifest in (
            ("smoke20", PROTOCOL_ROOT / "cf_cl_smoke_20.jsonl"),
            ("pilot100", PROTOCOL_ROOT / "cf_cl_pilot_100.jsonl"),
        ):
            for method in required:
                report = ensure_run(
                    name=f"{track.lower()}_{method}_{scale}",
                    method=method,
                    manifest=manifest,
                    output_dir=method_run_dir(args.output_dir, method, scale),
                    log_root=log_root,
                )
                method_rows.append(
                    {
                        "track_id": track,
                        "method": method,
                        "role": "candidate" if method in TRACK_METHODS[track] else "comparator",
                        "scale": scale,
                        "report_path": str(
                            (method_run_dir(args.output_dir, method, scale) / "report_summary.json")
                            .relative_to(ROOT)
                        ),
                        "current_rewrite_exact": report["current_rewrite_exact"],
                        "current_paraphrase_exact": report["current_paraphrase_exact"],
                        "past_retention": report["past_retention"],
                        "average_forgetting": report["average_forgetting"],
                        "same_subject_tfpr": report["same_subject_tfpr"],
                        "protected_kl": report["protected_kl"],
                    }
                )
        track_report_dir = args.output_dir / "track_reports" / f"{track}_pilot_v1"
        if not (track_report_dir / "report_summary.json").is_file():
            if track_report_dir.exists():
                raise RuntimeError(f"Partial track report requires integrity review: {track_report_dir}")
            run_process(f"{track.lower()}_pilot_report", report_command(track, args.output_dir), log_root)
        report = read_json(track_report_dir / "report_summary.json")
        if report.get("analysis_500_used") or report.get("final_test_used"):
            raise RuntimeError(f"Locked split contamination in {track_report_dir}")
        track_rows.append(
            {
                "track_id": track,
                "status": report["status"],
                "selected_candidate": report.get("selected_candidate"),
                "num_confirmation_eligible": report["num_confirmation_eligible"],
                "num_mechanism_signals": report["num_mechanism_signals"],
                "plan_coverage": TRACK_PLAN_COVERAGE[track],
                "report_path": str((track_report_dir / "report_summary.json").relative_to(ROOT)),
            }
        )
        next_stage = f"C{track_index + 2}_pilot" if track_index + 1 < len(TRACK_ORDER) else "E_pilot_eligibility"
        record_stage(
            f"{track}_pilot",
            status="passed" if report["acceptance_pass"] else "failed",
            acceptance_pass=bool(report["acceptance_pass"]),
            output_dir=track_report_dir,
            started_at_utc=started_at,
            notes=(
                f"selected={report.get('selected_candidate')}; "
                f"mechanism_signals={report['num_mechanism_signals']}; continue breadth-first"
            ),
            next_stage=next_stage,
        )

    terminal = len(track_rows) == len(TRACK_ORDER)
    eligible = [row for row in track_rows if row["num_confirmation_eligible"]]
    write_csv(args.output_dir / "method_results.csv", method_rows)
    write_csv(args.output_dir / "track_results.csv", track_rows)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "track_order": list(TRACK_ORDER),
            "track_methods": {key: list(value) for key, value in TRACK_METHODS.items()},
            "track_comparators": {key: list(value) for key, value in TRACK_COMPARATORS.items()},
            "comparison_baseline_method": baseline_method,
            "comparison_baseline_report": str((KL_BASELINE / "report_summary.json").relative_to(ROOT)),
            "comparison_baseline_protected_kl": baseline_report["protected_kl"],
            "all_tracks_before_confirmation": True,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D_breadth_first_pilots",
        "created_at_utc": now_utc(),
        "num_terminal_tracks": len(track_rows),
        "num_confirmation_eligible_tracks": len(eligible),
        "eligible_tracks": [row["track_id"] for row in eligible],
        "all_mandatory_tracks_terminal": terminal,
        "analysis_500_used": False,
        "final_test_used": False,
        "runtime_seconds": time.monotonic() - started,
        "acceptance_pass": terminal,
    }
    write_json(final_report, report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "all_mandatory_tracks_terminal": terminal,
            "all_track_reports_present": all(
                (args.output_dir / "track_reports" / f"{track}_pilot_v1" / "report_summary.json").is_file()
                for track in TRACK_ORDER
            ),
            "all_runs_have_protected_kl": all(row["protected_kl"] is not None for row in method_rows),
            "sb_proxies_confirmation_eligible": any(
                row["track_id"] in {"C7", "C8"} and row["num_confirmation_eligible"]
                for row in track_rows
            ),
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": terminal,
        },
    )
    record_stage(
        "E_pilot_eligibility",
        status="passed",
        acceptance_pass=terminal,
        output_dir=args.output_dir,
        started_at_utc=started_at,
        notes=f"eligible_tracks={','.join(row['track_id'] for row in eligible) or 'none'}",
        next_stage="F_fresh_confirmation" if eligible else "G_conditional_tracks",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
