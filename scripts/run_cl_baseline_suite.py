#!/usr/bin/env python3
"""Run and report the frozen C0 continual factual-edit baseline suite."""

from __future__ import annotations

import argparse
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

from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    SOURCE_AUDIT_ROOT,
    autonomous_enabled,
    git_commit,
    now_utc,
    read_json,
    record_stage,
    update_track,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT = CAMPAIGN_ROOT / "C0_common_baselines_v1"
BASELINE_METHODS = (
    "base",
    "sequential_partial_memit",
    "sequential_fullmask_memit",
    "sequential_lowrank_memit",
    "sequential_lora",
    "oedit_partial_memit",
    "ordinary_replay_memit",
    "lwf_partial_memit",
)
REQUIRED_PLAN_METHODS = {
    "sequential_partial_memit",
    "sequential_lora",
    "sequential_fullmask_memit",
    "oedit_partial_memit",
    "ordinary_replay_memit",
    "lwf_partial_memit",
}


def run_specs(output_dir: Path) -> list[dict[str, Any]]:
    specs = []
    for method in BASELINE_METHODS:
        for scale, manifest in (
            ("smoke20", PROTOCOL_ROOT / "cf_cl_smoke_20.jsonl"),
            ("pilot100", PROTOCOL_ROOT / "cf_cl_pilot_100.jsonl"),
        ):
            if method == "sequential_partial_memit":
                external = CAMPAIGN_ROOT / f"C0_partial_memit_{scale}_v1"
                destination = external if (external / "report_summary.json").is_file() else output_dir / f"{method}_{scale}"
            else:
                destination = output_dir / f"{method}_{scale}"
            specs.append(
                {
                    "method": method,
                    "scale": scale,
                    "manifest": manifest,
                    "output_dir": destination,
                }
            )
    return specs


def command_for(spec: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_cl_sequential_editor.py"),
        "--manifest",
        str(spec["manifest"]),
        "--retention_manifest",
        str(PROTOCOL_ROOT / "base_denoising_retention_500.jsonl"),
        "--output_dir",
        str(spec["output_dir"]),
        "--method",
        str(spec["method"]),
        "--covariance_dir",
        str(CAMPAIGN_ROOT / "B1_covariance_cache_v1"),
        "--covariance_representation",
        "diagonal",
        "--decode_batch_size",
        "16",
    ]


def run_one(spec: dict[str, Any], log_root: Path) -> dict[str, Any]:
    report_path = spec["output_dir"] / "report_summary.json"
    if report_path.is_file():
        return read_json(report_path)
    if spec["output_dir"].exists():
        raise RuntimeError(f"Partial C0 output requires integrity review: {spec['output_dir']}")
    name = f"c0_{spec['method']}_{spec['scale']}"
    log_path = log_root / f"{name}.log"
    exit_path = log_root / f"{name}.exitcode"
    command = command_for(spec)
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
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        raise RuntimeError(f"{name} failed with exit {process.returncode}:\n" + "\n".join(tail))
    if not report_path.is_file():
        raise RuntimeError(f"{name} completed without {report_path}")
    report = read_json(report_path)
    if not report.get("acceptance_pass") or not report.get("all_metrics_finite"):
        raise RuntimeError(f"{name} produced an invalid report")
    print(
        json.dumps(
            {
                "completed": name,
                "rewrite": report["current_rewrite_exact"],
                "paraphrase": report["current_paraphrase_exact"],
                "retention": report["past_retention"],
                "forgetting": report["average_forgetting"],
                "baseline_floor_pass": report["baseline_floor_pass"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def result_row(spec: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "current_rewrite_exact",
        "current_paraphrase_exact",
        "past_retention",
        "average_retention",
        "average_forgetting",
        "backward_transfer",
        "same_subject_tfpr",
        "near_tfpr",
        "far_tfpr",
        "malformed_rate",
        "base_retention_loss_fraction",
        "pre_edit_target_new_rewrite",
        "storage_mb_per_edit",
        "gpu_minutes_per_edit",
        "baseline_floor_pass",
        "implementation_status",
        "exact_method_claim_eligible",
    )
    return {
        "method": spec["method"],
        "scale": spec["scale"],
        "report_path": str((spec["output_dir"] / "report_summary.json").relative_to(ROOT)),
        **{key: report.get(key) for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not autonomous_enabled():
        raise PermissionError("CL_DLLM_AUTONOMOUS_MODE=1 is required")
    args.output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_report = args.output_dir / "report_summary.json"
    if final_report.is_file():
        print(f"C0 suite already complete: {final_report}")
        return
    started_at = now_utc()
    started = time.monotonic()
    log_root = ROOT / "logs" / CAMPAIGN_ID / "C0_common_baselines_v1"
    log_root.mkdir(parents=True, exist_ok=True)
    specs = run_specs(args.output_dir)
    reports = [run_one(spec, log_root) for spec in specs]
    rows = [result_row(spec, report) for spec, report in zip(specs, reports)]
    pilot_rows = [row for row in rows if row["scale"] == "pilot100"]
    missing_required = sorted(REQUIRED_PLAN_METHODS - {row["method"] for row in pilot_rows})
    acquisition = [row for row in pilot_rows if bool(row["baseline_floor_pass"])]
    best = max(
        acquisition,
        key=lambda row: (
            float(row["current_rewrite_exact"]),
            float(row["current_paraphrase_exact"]),
            float(row["past_retention"]),
        ),
    ) if acquisition else None
    source_registry = read_json(SOURCE_AUDIT_ROOT / "source_registry.json")
    diffusiongrow = source_registry["diffusiongrow"]
    source_status = {
        "source_id": "diffusiongrow",
        "implementation_status": diffusiongrow["implementation_status"],
        "exact_source_reproduction_available": False,
        "source_compatible_domain_adaptation_run": False,
        "reason": "No verifiable source repository or checkpoint was available; factual growth adaptations are evaluated under C1 and cannot be called exact reproduction.",
    }
    acceptance = not missing_required and bool(acquisition)
    write_csv(args.output_dir / "baseline_results.csv", rows)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "methods": list(BASELINE_METHODS),
            "scales": ["smoke20", "pilot100"],
            "covariance_representation": "diagonal",
            "source_style_status": source_status,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "C0_common_baselines",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "num_runs": len(rows),
        "missing_required_methods": missing_required,
        "num_acquisition_baselines": len(acquisition),
        "selected_acquisition_baseline": best["method"] if best else None,
        "selected_acquisition_report": best["report_path"] if best else None,
        "source_style_status": source_status,
        "implementation_repair_used": False,
        "baseline_infeasible": not bool(acquisition),
        "analysis_500_used": False,
        "final_test_used": False,
        "runtime_seconds": time.monotonic() - started,
        "acceptance_pass": acceptance,
    }
    write_json(final_report, report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "all_required_methods_run": not missing_required,
            "all_reports_valid": all(row.get("current_rewrite_exact") is not None for row in rows),
            "factual_acquisition_baseline_available": bool(acquisition),
            "diffusiongrow_exact_reproduction_misclaimed": False,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": acceptance,
        },
    )
    update_track(
        "C0",
        status="pilot_passed" if acceptance else "baseline_infeasible",
        nominated_candidate=best["method"] if best else None,
        report_path=str(final_report.relative_to(ROOT)),
        pilot_pass=acceptance,
    )
    record_stage(
        "C0_common_baselines",
        status="passed" if acceptance else "failed",
        acceptance_pass=acceptance,
        output_dir=args.output_dir,
        started_at_utc=started_at,
        notes=f"selected_acquisition={best['method'] if best else None}; exact DiffusionGrow source reproduction unavailable",
        next_stage="C1_pilot" if acceptance else "I_final_package",
    )
    print(json.dumps(report, sort_keys=True))
    if not acceptance:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
