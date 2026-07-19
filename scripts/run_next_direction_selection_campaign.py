#!/usr/bin/env python3
"""Execute the full bounded next-direction selection campaign on RunPod."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    STATE_ROOT,
    autonomous_enabled,
    initialize_state,
    now_utc,
    read_json,
    record_stage,
    update_track,
    write_json,
)


PYTHON = sys.executable
LOG_ROOT = ROOT / "logs" / CAMPAIGN_ID


def run_command(
    name: str,
    command: Sequence[str],
    *,
    output_dir: Path | None = None,
    expected_report: str = "report_summary.json",
) -> dict[str, Any]:
    if output_dir is not None:
        report = output_dir / expected_report
        if report.is_file():
            return read_json(report)
        if output_dir.exists():
            raise RuntimeError(f"partial output directory requires integrity review: {output_dir}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    exit_path = LOG_ROOT / f"{name}.exitcode"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    exit_path.write_text(f"{process.returncode}\n", encoding="utf-8")
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"{name} failed with exit {process.returncode}:\n" + "\n".join(tail)
        )
    if output_dir is None:
        return {"runtime_seconds": time.monotonic() - started, "exit_code": 0}
    report = output_dir / expected_report
    if not report.is_file():
        raise RuntimeError(f"{name} completed without {report}")
    return read_json(report)


def python_script(script: str, *args: str) -> list[str]:
    return [PYTHON, str(ROOT / "scripts" / script), *map(str, args)]


def editor_command(
    method: str,
    manifest: Path,
    output: Path,
    *,
    mechanism_dir: Path | None = None,
    allow_confirmation: bool = False,
) -> list[str]:
    command = python_script(
        "run_nds_counterfact_editor.py",
        "--manifest",
        manifest,
        "--output_dir",
        output,
        "--method",
        method,
        "--measurement_dir",
        CAMPAIGN_ROOT / "S1_shared_measurements_v1",
        "--layers",
        "4,5,6,7",
        "--allow_confirmation",
        "1" if allow_confirmation else "0",
    )
    if mechanism_dir and (mechanism_dir / "report_summary.json").is_file():
        report = read_json(mechanism_dir / "report_summary.json")
        if method in {"fisher_lowrank", "relation_fisher_integrated"}:
            selected = report.get("selected") or {}
            command += [
                "--fisher_damping",
                str(selected.get("damping", 1e-3)),
                "--fisher_rank",
                str(selected.get("rank", 64)),
            ]
        if method in {"primal_dual", "relation_primal_dual_integrated"}:
            selected = report.get("selected") or {}
            command += [
                "--multiplier_step",
                str(selected.get("multiplier_step", 0.05)),
                "--penalty_growth",
                str(selected.get("penalty_growth", 1.5)),
            ]
    return command


def formal_stop(track: str, evidence_dir: Path, classification: str) -> None:
    output = CAMPAIGN_ROOT / f"{track}_track_stop_v1"
    run_command(
        f"{track.lower()}_formal_stop",
        python_script(
            "finalize_nds_track_stop.py",
            "--track",
            track,
            "--classification",
            classification,
            "--evidence_dir",
            evidence_dir,
            "--output_dir",
            output,
        ),
        output_dir=output,
    )


def baseline_floor(report: dict[str, Any]) -> bool:
    base_rewrite = report.get("base_summary", {}).get("rewrite", {})
    return bool(
        float(report.get("rewrite_exact", 0.0)) >= 0.75
        and float(report.get("declarative_paraphrase_exact", 0.0)) >= 0.40
        and float(base_rewrite.get("target_new_rate", 1.0)) <= 0.10
        and float(report.get("malformed_rate", 1.0)) <= 0.05
    )


def run_counterfact_track(
    track: str,
    baseline_dir: Path,
    *,
    fixed_baseline_dir: Path | None = None,
) -> dict[str, Any]:
    started = now_utc()
    mechanism_dir = CAMPAIGN_ROOT / f"{track}_offline_mechanism_v1"
    mechanism = run_command(
        f"{track.lower()}_offline_mechanism",
        python_script(
            "run_nds_offline_mechanism.py",
            "--track",
            track,
            "--measurement_dir",
            CAMPAIGN_ROOT / "S1_shared_measurements_v1",
            "--calibration_manifest",
            PROTOCOL_ROOT / "cf_nds_calibration_200.jsonl",
            "--layer",
            "6",
            "--output_dir",
            mechanism_dir,
        ),
        output_dir=mechanism_dir,
    )
    if not mechanism["mechanism_pass"]:
        formal_stop(track, mechanism_dir, "offline_scientific_failure")
        record_stage(
            f"{track}_pilot",
            status="failed",
            acceptance_pass=False,
            output_dir=mechanism_dir,
            started_at_utc=started,
            notes="Offline mechanism gate failed after bounded rescue.",
            next_stage={"N1": "N2_pilot", "N2": "N3_pilot", "N3": "N4_pilot"}[track],
        )
        return {"pilot_pass": False, "mechanism_pass": False}
    method = str(mechanism["selected_candidate"])
    smoke_dir = CAMPAIGN_ROOT / f"{track}_{method}_smoke20_v1"
    pilot_decode_dir = CAMPAIGN_ROOT / f"{track}_{method}_pilot100_decode_v1"
    run_command(
        f"{track.lower()}_{method}_smoke20",
        editor_command(
            method,
            PROTOCOL_ROOT / "cf_nds_smoke_20.jsonl",
            smoke_dir,
            mechanism_dir=mechanism_dir,
        ),
        output_dir=smoke_dir,
    )
    run_command(
        f"{track.lower()}_{method}_pilot100",
        editor_command(
            method,
            PROTOCOL_ROOT / "cf_nds_pilot_100.jsonl",
            pilot_decode_dir,
            mechanism_dir=mechanism_dir,
        ),
        output_dir=pilot_decode_dir,
    )
    report_dir = CAMPAIGN_ROOT / f"{track}_pilot_v1"
    report = run_command(
        f"{track.lower()}_pilot_report",
        python_script(
            "report_nds_counterfact_track.py",
            "--track",
            track,
            "--baseline_dir",
            fixed_baseline_dir or baseline_dir,
            "--candidate_dir",
            pilot_decode_dir,
            "--smoke_dir",
            smoke_dir,
            "--mechanism_dir",
            mechanism_dir,
            "--output_dir",
            report_dir,
        ),
        output_dir=report_dir,
    )
    record_stage(
        f"{track}_pilot",
        status="passed" if report["pilot_pass"] else "failed",
        acceptance_pass=bool(report["pilot_pass"]),
        output_dir=report_dir,
        started_at_utc=started,
        notes=f"candidate={method}; class={report.get('success_class')}",
        next_stage={"N1": "N2_pilot", "N2": "N3_pilot", "N3": "N4_pilot"}[track],
    )
    return {
        **report,
        "method": method,
        "mechanism_dir": mechanism_dir,
        "smoke_dir": smoke_dir,
        "pilot_decode_dir": pilot_decode_dir,
        "pilot_report_dir": report_dir,
    }


def terminal_protocol_infeasible(baseline_dir: Path) -> None:
    for track in ("N1", "N2", "N3", "N4", "N5"):
        formal_stop(track, baseline_dir, "protocol_infeasibility")
    update_track("N6", status="not_triggered", notes="Baseline floor failed; integration was not legal.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_preflight_tests", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if not autonomous_enabled():
        raise PermissionError("NEXT_DIRECTION_AUTONOMOUS_MODE=1 is required")
    initialize_state()
    if not args.skip_preflight_tests:
        run_command("preflight_pytest", [PYTHON, "-m", "pytest", "tests", "-q"])
    bootstrap = CAMPAIGN_ROOT / "S0_bootstrap_v1"
    run_command(
        "s0_bootstrap",
        python_script("bootstrap_next_direction_campaign.py"),
        output_dir=bootstrap,
    )
    protocol = PROTOCOL_ROOT
    run_command(
        "s0_fresh_protocol",
        python_script("build_next_direction_protocol.py", "--output_dir", protocol),
        output_dir=protocol,
    )
    measurements = CAMPAIGN_ROOT / "S1_shared_measurements_v1"
    run_command(
        "s1_shared_measurements",
        python_script(
            "build_nds_shared_measurements.py",
            "--statistics_manifest",
            protocol / "cf_nds_statistics_train_500.jsonl",
            "--calibration_manifest",
            protocol / "cf_nds_calibration_200.jsonl",
            "--output_dir",
            measurements,
            "--layers",
            "4,5,6,7",
        ),
        output_dir=measurements,
    )
    baseline_started = now_utc()
    baseline_root = CAMPAIGN_ROOT / "S1_common_baselines_v1"
    baseline_root.mkdir(exist_ok=True)
    base_dir = baseline_root / "base_pilot100"
    ordinary_dir = baseline_root / "ordinary_memit_pilot100"
    partial_dir = baseline_root / "partial_state_memit_pilot100"
    static_dir = baseline_root / "static_nullspace_pilot100"
    temporal_dir = baseline_root / "historical_style_temporal_residual_pilot100"
    for name, method, output in (
        ("s1_base_pilot100", "base", base_dir),
        ("s1_ordinary_pilot100", "ordinary_memit", ordinary_dir),
        ("s1_partial_pilot100", "partial_state_memit", partial_dir),
        ("s1_static_nullspace_pilot100", "static_nullspace_partial_state_memit", static_dir),
    ):
        run_command(
            name,
            editor_command(method, protocol / "cf_nds_pilot_100.jsonl", output),
            output_dir=output,
        )
    run_command(
        "s1_temporal_residual_diagnostic_pilot100",
        python_script(
            "run_trm_editor_experiment.py",
            "--manifest",
            protocol / "cf_nds_pilot_100.jsonl",
            "--output_dir",
            temporal_dir,
            "--method",
            "best_historical_style_temporal_residual",
            "--layer",
            "6",
            "--state_mode",
            "shared",
            "--protection_mode",
            "none",
            "--anchor_manifest",
            protocol / "cf_nds_statistics_train_500.jsonl",
            "--partial_mask_schedule",
            "cycle",
            "--reveal_policy",
            "base_confidence",
            "--limit",
            "100",
        ),
        output_dir=temporal_dir,
    )
    partial_report = read_json(partial_dir / "report_summary.json")
    repair_used = False
    if not baseline_floor(partial_report):
        repair_dir = baseline_root / "partial_state_memit_source_repair_pilot100"
        command = editor_command(
            "partial_state_memit", protocol / "cf_nds_pilot_100.jsonl", repair_dir
        ) + ["--covariance_weight", "7500"]
        partial_report = run_command(
            "s1_partial_source_repair_pilot100", command, output_dir=repair_dir
        )
        partial_dir = repair_dir
        repair_used = True
    suite_report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "S1_common_baselines",
        "created_at_utc": now_utc(),
        "partial_state_baseline_dir": str(partial_dir),
        "ordinary_memit_dir": str(ordinary_dir),
        "static_nullspace_dir": str(static_dir),
        "historical_style_temporal_residual_dir": str(temporal_dir),
        "partial_state_baseline_floor_pass": baseline_floor(partial_report),
        "source_compatible_repair_used": repair_used,
        "baseline_floor": {
            "rewrite_exact": partial_report["rewrite_exact"],
            "declarative_paraphrase_exact": partial_report["declarative_paraphrase_exact"],
            "pre_edit_target_new_rewrite": partial_report["base_summary"]["rewrite"]["target_new_rate"],
            "malformed_rate": partial_report["malformed_rate"],
        },
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": baseline_floor(partial_report),
    }
    write_json(baseline_root / "report_summary.json", suite_report)
    record_stage(
        "S1_common_baselines",
        status="passed" if suite_report["acceptance_pass"] else "failed",
        acceptance_pass=bool(suite_report["acceptance_pass"]),
        output_dir=baseline_root,
        started_at_utc=baseline_started,
        notes=f"partial-state floor pass={suite_report['acceptance_pass']}; repair={repair_used}",
        next_stage="S1_shared_measurements" if suite_report["acceptance_pass"] else "S6_final_package",
    )
    if not suite_report["acceptance_pass"]:
        terminal_protocol_infeasible(baseline_root)
        run_command(
            "terminal_protocol_infeasible_package",
            python_script("finalize_next_direction_selection.py"),
            output_dir=CAMPAIGN_ROOT / "final_direction_selection_package_v1",
        )
        return

    n1 = run_counterfact_track("N1", partial_dir)
    n2 = run_counterfact_track("N2", partial_dir)
    fixed_penalty_dir = CAMPAIGN_ROOT / "N3_fixed_penalty_pilot100_decode_v1"
    run_command(
        "n3_fixed_penalty_pilot100",
        editor_command(
            "fixed_penalty",
            protocol / "cf_nds_pilot_100.jsonl",
            fixed_penalty_dir,
        ),
        output_dir=fixed_penalty_dir,
    )
    n3 = run_counterfact_track(
        "N3", partial_dir, fixed_baseline_dir=fixed_penalty_dir
    )

    n4_started = now_utc()
    passed_counterfact = [row for row in (n1, n2, n3) if row.get("pilot_pass")]
    if passed_counterfact:
        underlying = max(
            passed_counterfact,
            key=lambda row: float(
                read_json(row["pilot_decode_dir"] / "report_summary.json")["selection_score"]
            ),
        )
        underlying_method = underlying["method"]
        underlying_mechanism = underlying["mechanism_dir"]
        pilot_underlying_dir = underlying["pilot_decode_dir"]
    else:
        underlying_method = "partial_state_memit"
        underlying_mechanism = None
        pilot_underlying_dir = partial_dir
    n4_stats = CAMPAIGN_ROOT / f"N4_{underlying_method}_statistics500_v1"
    n4_cal = CAMPAIGN_ROOT / f"N4_{underlying_method}_calibration200_v1"
    for name, manifest, output in (
        ("n4_underlying_statistics500", protocol / "cf_nds_statistics_train_500.jsonl", n4_stats),
        ("n4_underlying_calibration200", protocol / "cf_nds_calibration_200.jsonl", n4_cal),
    ):
        run_command(
            name,
            editor_command(
                underlying_method,
                manifest,
                output,
                mechanism_dir=underlying_mechanism,
            ),
            output_dir=output,
        )
    n4_dir = CAMPAIGN_ROOT / "N4_selective_conformal_pilot_v1"
    n4 = run_command(
        "n4_selective_pilot",
        python_script(
            "run_nds_selective_conformal.py",
            "--statistics_dir",
            n4_stats,
            "--calibration_dir",
            n4_cal,
            "--pilot_dir",
            pilot_underlying_dir,
            "--output_dir",
            n4_dir,
        ),
        output_dir=n4_dir,
    )
    record_stage(
        "N4_pilot",
        status="passed" if n4["pilot_pass"] else "failed",
        acceptance_pass=bool(n4["pilot_pass"]),
        output_dir=n4_dir,
        started_at_utc=n4_started,
        notes=f"underlying={underlying_method}; coverage={n4['coverage']}",
        next_stage="N5_pilot",
    )

    n5_started = now_utc()
    n5_dir = CAMPAIGN_ROOT / "N5_joint_span_rank32_pilot_v1"
    n5 = run_command(
        "n5_joint_span_rank32",
        python_script(
            "run_nds_joint_span.py",
            "--protocol_dir",
            protocol,
            "--measurement_dir",
            measurements,
            "--output_dir",
            n5_dir,
            "--rank",
            "32",
        ),
        output_dir=n5_dir,
    )
    if not n5["pilot_pass"]:
        rescue_dir = CAMPAIGN_ROOT / "N5_joint_span_rank64_rescue_pilot_v1"
        n5 = run_command(
            "n5_joint_span_rank64_rescue",
            python_script(
                "run_nds_joint_span.py",
                "--protocol_dir",
                protocol,
                "--measurement_dir",
                measurements,
                "--output_dir",
                rescue_dir,
                "--rank",
                "64",
                "--rescue_used",
                "1",
            ),
            output_dir=rescue_dir,
        )
        n5_dir = rescue_dir
    record_stage(
        "N5_pilot",
        status="passed" if n5["pilot_pass"] else "failed",
        acceptance_pass=bool(n5["pilot_pass"]),
        output_dir=n5_dir,
        started_at_utc=n5_started,
        notes=f"rank={n5['rank']}; lengths_with_gain={n5['lengths_with_10pp_gain']}",
        next_stage="N6_integrated_pilot",
    )

    n6_started = now_utc()
    mechanism_passes = {
        "N1": bool(n1.get("mechanism_pass")),
        "N2": bool(n2.get("mechanism_pass")),
        "N3": bool(n3.get("mechanism_pass")),
        "N4": n4.get("calibration_threshold", {}).get("threshold") is not None,
    }
    compositions = []
    if mechanism_passes["N1"] and mechanism_passes["N2"]:
        compositions.append(("relation_fisher_integrated", n2.get("mechanism_dir")))
    if mechanism_passes["N1"] and mechanism_passes["N3"]:
        compositions.append(("relation_primal_dual_integrated", n3.get("mechanism_dir")))
    n6 = {"pilot_pass": False, "triggered": bool(any(mechanism_passes.values()))}
    n6_report_dir = CAMPAIGN_ROOT / "N6_not_triggered_v1"
    if compositions:
        n6_baseline_calibration = n4_cal
        if underlying_method != "partial_state_memit":
            n6_baseline_calibration = CAMPAIGN_ROOT / "N6_partial_state_calibration200_v1"
            run_command(
                "n6_partial_state_calibration200",
                editor_command(
                    "partial_state_memit",
                    protocol / "cf_nds_calibration_200.jsonl",
                    n6_baseline_calibration,
                ),
                output_dir=n6_baseline_calibration,
            )
        n6_calibration_dirs = []
        for method, mechanism_dir in compositions:
            output = CAMPAIGN_ROOT / f"N6_{method}_calibration200_v1"
            run_command(
                f"n6_{method}_calibration200",
                editor_command(
                    method,
                    protocol / "cf_nds_calibration_200.jsonl",
                    output,
                    mechanism_dir=mechanism_dir,
                ),
                output_dir=output,
            )
            n6_calibration_dirs.append(output)
        selector_dir = CAMPAIGN_ROOT / "N6_calibration_selection_v1"
        selector = run_command(
            "n6_calibration_selection",
            python_script(
                "select_nds_integrated_candidate.py",
                "--baseline_dir",
                n6_baseline_calibration,
                "--candidate_dirs",
                *n6_calibration_dirs,
                "--output_dir",
                selector_dir,
            ),
            output_dir=selector_dir,
        )
        if selector.get("selected_candidate"):
            selected_method = selector["selected_candidate"]["candidate_id"]
            selected_mechanism = next(value for method, value in compositions if method == selected_method)
            n6_smoke = CAMPAIGN_ROOT / f"N6_{selected_method}_smoke20_v1"
            n6_decode = CAMPAIGN_ROOT / f"N6_{selected_method}_pilot100_decode_v1"
            run_command("n6_smoke20", editor_command(selected_method, protocol / "cf_nds_smoke_20.jsonl", n6_smoke, mechanism_dir=selected_mechanism), output_dir=n6_smoke)
            run_command("n6_pilot100", editor_command(selected_method, protocol / "cf_nds_pilot_100.jsonl", n6_decode, mechanism_dir=selected_mechanism), output_dir=n6_decode)
            n6_report_dir = CAMPAIGN_ROOT / "N6_integrated_pilot_v1"
            n6 = run_command(
                "n6_pilot_report",
                python_script(
                    "report_nds_counterfact_track.py",
                    "--track",
                    "N6",
                    "--baseline_dir",
                    partial_dir,
                    "--candidate_dir",
                    n6_decode,
                    "--smoke_dir",
                    n6_smoke,
                    "--mechanism_dir",
                    selector_dir,
                    "--output_dir",
                    n6_report_dir,
                ),
                output_dir=n6_report_dir,
            )
            n6.update({"method": selected_method, "mechanism_dir": selected_mechanism, "pilot_decode_dir": n6_decode, "pilot_report_dir": n6_report_dir})
    if not compositions:
        n6_report_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            n6_report_dir / "report_summary.json",
            {
                "campaign_id": CAMPAIGN_ID,
                "track_id": "N6",
                "triggered": bool(any(mechanism_passes.values())),
                "composable_components_available": False,
                "pilot_pass": False,
                "status": "not_triggered",
                "analysis_500_used": False,
                "final_test_used": False,
                "acceptance_pass": True,
            },
        )
        update_track("N6", status="not_triggered", output_dir=n6_report_dir, notes="No predeclared composition had independently passed components.")
    record_stage(
        "N6_integrated_pilot",
        status="passed" if n6.get("pilot_pass") else "not_triggered" if not compositions else "failed",
        acceptance_pass=bool(n6.get("pilot_pass") or not compositions),
        output_dir=n6_report_dir,
        started_at_utc=n6_started,
        notes=f"mechanism_trigger={any(mechanism_passes.values())}; compositions={len(compositions)}",
        next_stage="S4_fresh_confirmation",
    )

    confirmation_started = now_utc()
    counterfact_nominees = [
        ("N1", n1),
        ("N2", n2),
        ("N3", n3),
        ("N6", n6),
    ]
    needs_counterfact_confirmation = any(row.get("pilot_pass") for _track, row in counterfact_nominees) or bool(n4.get("pilot_pass"))
    confirmation_baseline = CAMPAIGN_ROOT / "S4_partial_state_confirmation200_v1"
    if needs_counterfact_confirmation:
        run_command(
            "s4_partial_state_confirmation200",
            editor_command(
                "partial_state_memit",
                protocol / "cf_nds_confirmation_200.jsonl",
                confirmation_baseline,
                allow_confirmation=True,
            ),
            output_dir=confirmation_baseline,
        )
    for track, row in counterfact_nominees:
        if not row.get("pilot_pass"):
            continue
        method = row["method"]
        decode_dir = CAMPAIGN_ROOT / f"{track}_{method}_confirmation200_decode_v1"
        run_command(
            f"{track.lower()}_confirmation_decode",
            editor_command(
                method,
                protocol / "cf_nds_confirmation_200.jsonl",
                decode_dir,
                mechanism_dir=row.get("mechanism_dir"),
                allow_confirmation=True,
            ),
            output_dir=decode_dir,
        )
        confirm_dir = CAMPAIGN_ROOT / f"{track}_confirmation_v1"
        run_command(
            f"{track.lower()}_confirmation_report",
            python_script(
                "report_nds_confirmation.py",
                "--track",
                track,
                "--pilot_dir",
                row["pilot_report_dir"],
                "--baseline_dir",
                confirmation_baseline,
                "--candidate_dir",
                decode_dir,
                "--output_dir",
                confirm_dir,
            ),
            output_dir=confirm_dir,
        )
    if n4.get("pilot_pass"):
        n4_confirmation_editor = CAMPAIGN_ROOT / f"N4_{underlying_method}_confirmation200_v1"
        run_command(
            "n4_underlying_confirmation200",
            editor_command(
                underlying_method,
                protocol / "cf_nds_confirmation_200.jsonl",
                n4_confirmation_editor,
                mechanism_dir=underlying_mechanism,
                allow_confirmation=True,
            ),
            output_dir=n4_confirmation_editor,
        )
        n4_confirmation = CAMPAIGN_ROOT / "N4_confirmation_v1"
        run_command(
            "n4_selective_confirmation",
            python_script(
                "run_nds_selective_confirmation.py",
                "--pilot_dir",
                n4_dir,
                "--confirmation_editor_dir",
                n4_confirmation_editor,
                "--output_dir",
                n4_confirmation,
            ),
            output_dir=n4_confirmation,
        )
    if n5.get("pilot_pass"):
        n5_confirmation = CAMPAIGN_ROOT / "N5_confirmation_v1"
        run_command(
            "n5_joint_span_confirmation",
            python_script(
                "run_nds_joint_span_confirmation.py",
                "--pilot_dir",
                n5_dir,
                "--protocol_dir",
                protocol,
                "--measurement_dir",
                measurements,
                "--output_dir",
                n5_confirmation,
            ),
            output_dir=n5_confirmation,
        )
    record_stage(
        "S4_fresh_confirmation",
        status="passed",
        acceptance_pass=True,
        output_dir=STATE_ROOT,
        started_at_utc=confirmation_started,
        notes="Every pilot-passed candidate was evaluated once on its fresh frozen confirmation split.",
        next_stage="S5_final_selection",
    )
    final_dir = CAMPAIGN_ROOT / "final_direction_selection_package_v1"
    run_command(
        "s5_s6_final_selection_package",
        python_script("finalize_next_direction_selection.py", "--output_dir", final_dir),
        output_dir=final_dir,
    )


if __name__ == "__main__":
    main()
