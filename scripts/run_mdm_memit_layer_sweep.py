#!/usr/bin/env python3
"""Run the predeclared coarse-to-fine contiguous MEMIT layer search."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)


M1_ROOT = CAMPAIGN_ROOT / "M1_mdm_memit_reproduction_v1"
DEFAULT_OUTPUT = M1_ROOT / "layer_selection_v1"


def _rank_desc(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row[key]), int(row["start_layer"])),
    )
    return {str(row["window"]): index + 1 for index, row in enumerate(ordered)}


def _run_or_reuse(
    *,
    output_dir: Path,
    manifest: Path,
    layers: tuple[int, int, int, int],
    covariance_dir: Path,
    limit: int,
) -> dict[str, Any]:
    report_path = output_dir / "report_summary.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if output_dir.exists():
        raise RuntimeError(f"Partial layer-window output requires audit: {output_dir}")
    command = [
        sys.executable,
        str(ROOT / "scripts/run_mdm_memit_stage.py"),
        "--stage",
        "batch",
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output_dir),
        "--covariance_dir",
        str(covariance_dir),
        "--layers",
        ",".join(map(str, layers)),
        "--include_locality",
        "0",
        "--limit",
        str(limit),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _result_row(
    report: dict[str, Any],
    layers: tuple[int, int, int, int],
    phase: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "window": "-".join(map(str, layers)),
        "start_layer": layers[0],
        "end_layer": layers[-1],
        "num_edits": report["num_edits"],
        "rewrite_exact": report["rewrite_exact"],
        "paraphrase_exact": report["paraphrase_exact"],
        "malformed_rate": report["malformed_rate"],
        "gpu_minutes_per_edit": report["gpu_minutes_per_edit"],
        "acceptance_pass": report["acceptance_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROTOCOL_ROOT / "cf_layer_select_500.jsonl")
    parser.add_argument("--covariance_dir", type=Path, default=CAMPAIGN_ROOT / "covariance_cache_v1")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coarse_limit", type=int, default=34)
    parser.add_argument("--full_candidates", type=int, default=3)
    args = parser.parse_args()
    started = now_utc()
    if args.coarse_limit <= 0 or args.full_candidates <= 0:
        raise ValueError("coarse_limit and full_candidates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = [tuple(range(start, start + 4)) for start in range(29)]
    missing_covariance = [
        layer
        for layer in range(32)
        if not (args.covariance_dir / f"layer_{layer}_covariance.pt").exists()
    ]
    if missing_covariance:
        raise FileNotFoundError(f"Missing covariance layers: {missing_covariance}")

    coarse_rows: list[dict[str, Any]] = []
    for layers in windows:
        run_dir = args.output_dir / f"coarse_l{layers[0]:02d}_{layers[-1]:02d}"
        report = _run_or_reuse(
            output_dir=run_dir,
            manifest=args.manifest,
            layers=layers,
            covariance_dir=args.covariance_dir,
            limit=args.coarse_limit,
        )
        coarse_rows.append(_result_row(report, layers, "coarse"))

    efficacy_ranks = _rank_desc(coarse_rows, "rewrite_exact")
    generalization_ranks = _rank_desc(coarse_rows, "paraphrase_exact")
    for row in coarse_rows:
        window = str(row["window"])
        row["efficacy_rank"] = efficacy_ranks[window]
        row["generalization_rank"] = generalization_ranks[window]
        row["rank_sum"] = efficacy_ranks[window] + generalization_ranks[window]
    ordered = sorted(
        coarse_rows,
        key=lambda row: (
            int(row["rank_sum"]),
            -float(row["rewrite_exact"]),
            int(row["start_layer"]),
        ),
    )
    candidate_windows = [str(row["window"]) for row in ordered[: args.full_candidates]]
    paper_window = "4-5-6-7"
    if paper_window not in candidate_windows:
        candidate_windows.append(paper_window)

    full_rows: list[dict[str, Any]] = []
    for window in candidate_windows:
        layers = tuple(map(int, window.split("-")))
        run_dir = args.output_dir / f"full_l{layers[0]:02d}_{layers[-1]:02d}"
        report = _run_or_reuse(
            output_dir=run_dir,
            manifest=args.manifest,
            layers=layers,  # type: ignore[arg-type]
            covariance_dir=args.covariance_dir,
            limit=500,
        )
        full_rows.append(_result_row(report, layers, "full"))  # type: ignore[arg-type]

    efficacy_ranks = _rank_desc(full_rows, "rewrite_exact")
    generalization_ranks = _rank_desc(full_rows, "paraphrase_exact")
    for row in full_rows:
        window = str(row["window"])
        row["efficacy_rank"] = efficacy_ranks[window]
        row["generalization_rank"] = generalization_ranks[window]
        row["rank_sum"] = efficacy_ranks[window] + generalization_ranks[window]
    selected = sorted(
        full_rows,
        key=lambda row: (
            int(row["rank_sum"]),
            -float(row["rewrite_exact"]),
            int(row["start_layer"]),
        ),
    )[0]
    all_rows = coarse_rows + full_rows
    write_csv(args.output_dir / "layer_sweep.csv", all_rows)
    write_csv(M1_ROOT / "layer_sweep.csv", all_rows)
    selection = {
        "window": selected["window"],
        "layers": list(range(int(selected["start_layer"]), int(selected["end_layer"]) + 1)),
        "selection_rule": "rank efficacy + rank generalization; tie-break efficacy",
        "coarse_to_fine": {
            "all_contiguous_width_4_windows_screened": True,
            "coarse_edits_per_window": args.coarse_limit,
            "coarse_manifest": str(args.manifest),
            "full_candidate_windows": candidate_windows,
            "full_edits_per_candidate": 500,
            "paper_window_forced_into_full_comparison": True,
        },
        "selected_metrics": selected,
        "manifest_sha256": sha256_file(args.manifest),
    }
    write_json(args.output_dir / "selected_layer_window.json", selection)
    acceptance = (
        len(coarse_rows) == 29
        and all(int(row["num_edits"]) == args.coarse_limit for row in coarse_rows)
        and all(int(row["num_edits"]) == 500 for row in full_rows)
        and paper_window in candidate_windows
    )
    report = {
        "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
        "track": "M1",
        "stage": "M1_layer_window_selection",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "acceptance_pass": acceptance,
        "all_windows_screened": len(coarse_rows) == 29,
        "num_coarse_windows": len(coarse_rows),
        "num_full_candidates": len(full_rows),
        "selected_window": selection,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="M1_layer_window_selection",
        track="M1",
        status="passed" if acceptance else "failed",
        output_dir=args.output_dir,
        acceptance_pass=acceptance,
        started_at_utc=started,
        notes=f"Selected contiguous window {selected['window']} after all-window coarse screen.",
    )
    print(json.dumps({"acceptance_pass": acceptance, "selected_window": selected["window"]}))


if __name__ == "__main__":
    main()
