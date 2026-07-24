#!/usr/bin/env python3
"""Run only the frozen, single-use C1/C3/C4/C5 pilot rescues."""

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
from scripts.run_cl_sequential_editor import (
    OEDIT_RESCUE_BASIS_RANKS,
    RELATION_GATE_RESCUE_THRESHOLDS,
)
from scripts.run_cl_track_suite import KL_BASELINE


# Keep this explicit rather than importing a mutable runtime location alias.
INITIAL_ROOT = CAMPAIGN_ROOT / "D_breadth_first_pilots_v1"
DEFAULT_RESCUE_ROOT = CAMPAIGN_ROOT / "D_bounded_rescues_v1"


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value in {None, ""} else float(value)


def frozen_rescue_specs(track: str, initial: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(initial.get("confirmation_eligible", "")).casefold() == "true":
        return []
    acquisition = (
        _number(initial, "current_rewrite_exact") >= 0.75
        and _number(initial, "current_paraphrase_exact") >= 0.40
    )
    if not acquisition:
        return []
    safety_failure = any(
        _number(initial, key, 1.0) > threshold
        for key, threshold in (
            ("same_subject_tfpr", 0.03),
            ("near_tfpr", 0.03),
            ("far_tfpr", 0.03),
            ("base_retention_loss_fraction", 0.05),
        )
    )
    if track in {"C1", "C3"} and safety_failure:
        method = "growth_block_gate" if track == "C1" else "sparse_routed_memory"
        return [
            {
                "track_id": track,
                "label": f"{method}_relation_threshold_{threshold:.2f}",
                "method": method,
                "extra_args": ["--relation_overlap_threshold", str(threshold)],
                "rescue_rule": "predeclared_relation_routing_sparsity",
            }
            for threshold in RELATION_GATE_RESCUE_THRESHOLDS
        ]
    if track == "C4" and (
        _number(initial, "past_retention") < 0.75
        or _number(initial, "average_forgetting", 1.0) > 0.10
    ):
        return [
            {
                "track_id": track,
                "label": "gated_adapter_shared_basis",
                "method": "gated_adapter_shared_basis",
                "extra_args": [],
                "rescue_rule": "predeclared_shared_lowrank_basis",
            }
        ]
    if track == "C5":
        return [
            {
                "track_id": track,
                "label": f"oedit_partial_memit_basis_rank_{rank}",
                "method": "oedit_partial_memit",
                "extra_args": ["--protected_basis_rank", str(rank)],
                "rescue_rule": "predeclared_oedit_basis_rank",
            }
            for rank in OEDIT_RESCUE_BASIS_RANKS
        ]
    return []


def initial_candidate(track: str, method: str) -> dict[str, Any]:
    rows = read_csv(INITIAL_ROOT / "track_reports" / f"{track}_pilot_v1" / "candidate_results.csv")
    selected = [row for row in rows if row["method"] == method]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {track}/{method} initial candidate")
    return selected[0]


def editor_command(spec: Mapping[str, Any], manifest: Path, output_dir: Path) -> list[str]:
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
        str(spec["method"]),
        "--covariance_dir",
        str(CAMPAIGN_ROOT / "B1_covariance_cache_v1"),
        "--covariance_representation",
        "diagonal",
        "--decode_batch_size",
        "16",
        *map(str, spec["extra_args"]),
    ]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RESCUE_ROOT)
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
    initial_report = read_json(INITIAL_ROOT / "report_summary.json")
    if not initial_report.get("all_mandatory_tracks_terminal"):
        raise RuntimeError("Every initial C1-C9 pilot must be terminal before rescue")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    log_root = ROOT / "logs" / CAMPAIGN_ID / "D_bounded_rescues_v1"
    initial_by_track = {
        "C1": initial_candidate("C1", "growth_block_gate"),
        "C3": initial_candidate("C3", "sparse_routed_memory"),
        "C4": initial_candidate("C4", "gated_adapter_expansion"),
        "C5": initial_candidate("C5", "oedit_partial_memit"),
    }
    specs = [
        spec
        for track, initial in initial_by_track.items()
        for spec in frozen_rescue_specs(track, initial)
    ]
    candidate_registry = []
    report_registry = []
    for track in ("C1", "C3", "C4", "C5"):
        track_specs = [spec for spec in specs if spec["track_id"] == track]
        if not track_specs:
            continue
        candidate_dirs = []
        for spec in track_specs:
            for scale, manifest in (
                ("smoke20", PROTOCOL_ROOT / "cf_cl_smoke_20.jsonl"),
                ("pilot100", PROTOCOL_ROOT / "cf_cl_pilot_100.jsonl"),
            ):
                output = args.output_dir / "method_runs" / f"{spec['label']}_{scale}"
                print(f"START rescue_{track.lower()}_{spec['label']}_{scale}", flush=True)
                run_process(
                    f"rescue_{track.lower()}_{spec['label']}_{scale}",
                    editor_command(spec, manifest, output),
                    log_root,
                )
                report = read_json(output / "report_summary.json")
                if not report.get("acceptance_pass") or report.get("protected_kl") is None:
                    raise RuntimeError(f"Invalid rescue run: {output}")
                if scale == "pilot100":
                    candidate_dirs.append(output)
        report_dir = args.output_dir / "track_reports" / f"{track}_rescue_v1"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "report_cl_track.py"),
            "--track",
            track,
            "--baseline_dir",
            str(KL_BASELINE),
            "--output_dir",
            str(report_dir),
        ]
        for candidate_dir in candidate_dirs:
            command.extend(("--candidate_dir", str(candidate_dir)))
        run_process(f"{track.lower()}_rescue_report", command, log_root)
        report = read_json(report_dir / "report_summary.json")
        result_rows = read_csv(report_dir / "candidate_results.csv")
        selected_method = report.get("selected_candidate")
        selected_report_path = report.get("selected_report_path")
        selected_row = next(
            (row for row in result_rows if row["report_path"] == selected_report_path), None
        )
        selected_dir = None
        if selected_row is not None:
            selected_dir = str((ROOT / selected_row["report_path"]).parent.relative_to(ROOT))
        candidate_registry.append(
            {
                "track_id": track,
                "method": selected_method or "",
                "pilot_dir": selected_dir or "",
                "pilot_report_dir": str(report_dir.relative_to(ROOT)),
                "success_classes": ",".join(report.get("selected_success_classes") or []),
                "rescue_pass": bool(selected_method),
            }
        )
        report_registry.append(
            {
                "track_id": track,
                "status": report["status"],
                "rescue_rule": track_specs[0]["rescue_rule"],
                "num_configs": len(track_specs),
                "selected_candidate": selected_method,
            }
        )
        update_track(
            track,
            status=report["status"],
            rescue_used=True,
            nominated_candidate=selected_method or "",
            report_path=str((report_dir / "report_summary.json").relative_to(ROOT)),
            pilot_pass=bool(selected_method),
        )
        record_stage(
            f"{track}_pilot",
            status="passed" if selected_method else "failed",
            acceptance_pass=bool(selected_method),
            output_dir=report_dir,
            started_at_utc=now_utc(),
            notes=f"single bounded rescue used; selected={selected_method}",
            next_stage="E_pilot_eligibility",
        )

    write_csv(
        args.output_dir / "rescued_candidates.csv",
        candidate_registry,
        ("track_id", "method", "pilot_dir", "pilot_report_dir", "success_classes", "rescue_pass"),
    )
    write_csv(
        args.output_dir / "rescue_registry.csv",
        report_registry,
        ("track_id", "status", "rescue_rule", "num_configs", "selected_candidate"),
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D_bounded_rescues",
        "created_at_utc": now_utc(),
        "num_tracks_rescued": len(report_registry),
        "num_rescues_passed": sum(bool(row["selected_candidate"]) for row in report_registry),
        "all_rescues_terminal": True,
        "analysis_500_used": False,
        "final_test_used": False,
        "runtime_seconds": time.monotonic() - started,
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "run_config.json",
        {
            "relation_thresholds": list(RELATION_GATE_RESCUE_THRESHOLDS),
            "oedit_basis_ranks": list(OEDIT_RESCUE_BASIS_RANKS),
            "c4_rescue": "shared_lowrank_basis",
            "one_rescue_per_track": True,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
