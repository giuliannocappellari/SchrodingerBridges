#!/usr/bin/env python3
"""Run the frozen staged D2 multi-state target-value sweep."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    now_utc,
    read_json,
    record_stage,
    write_csv,
    write_json,
)
from scripts.run_dnpe_partial_state_sweep import POLICIES


def staged_configs() -> list[dict[str, float | int]]:
    return [
        {
            "learning_rate": lr,
            "target_optimization_steps": steps,
            "state_consistency_weight": 0.1,
            "old_target_suppression_weight": 0.25,
        }
        for lr, steps in ((0.05, 25), (0.10, 25), (0.05, 50), (0.10, 50))
    ]


def objective_configs(best: Mapping[str, Any]) -> list[dict[str, float | int]]:
    output = []
    for consistency, old_suppress in ((0.1, 0.25), (0.0, 0.0), (0.1, 0.0), (0.0, 0.25)):
        output.append(
            {
                "learning_rate": float(best["learning_rate"]),
                "target_optimization_steps": int(best["target_optimization_steps"]),
                "state_consistency_weight": consistency,
                "old_target_suppression_weight": old_suppress,
            }
        )
    return output


def config_id(config: Mapping[str, Any]) -> str:
    return (
        f"lr{float(config['learning_rate']):.2f}_"
        f"steps{int(config['target_optimization_steps'])}_"
        f"cons{float(config['state_consistency_weight']):.1f}_"
        f"old{float(config['old_target_suppression_weight']):.2f}"
    ).replace(".", "p")


def diagnostic_summary(run_dir: Path) -> dict[str, Any]:
    report = read_json(run_dir / "report_summary.json")
    diagnostics = read_json(run_dir / "target_value_diagnostics.json")
    rows = diagnostics["target_optimization"]
    starts = [float(row["history"][0]["total_loss"]) for row in rows]
    ends = [float(row["history"][-1]["total_loss"]) for row in rows]
    base_probs = [float(row["heldout_base_target_probability"]) for row in rows]
    edited_probs = [float(row["heldout_edited_target_probability"]) for row in rows]
    norm_ratios = [
        float(row["target_value_norm"]) / max(float(row["initial_value_norm"]), 1e-8)
        for row in rows
    ]
    finite_values = starts + ends + base_probs + edited_probs + norm_ratios
    return {
        "run": run_dir.name,
        "rewrite_exact": float(report["rewrite_exact"]),
        "declarative_paraphrase_exact": float(report["declarative_paraphrase_exact"]),
        "target_token_f1": float(report.get("target_token_f1", 0.0)),
        "mean_initial_loss": mean(starts),
        "mean_final_loss": mean(ends),
        "loss_decreased_fraction": mean(
            float(end < start) for start, end in zip(starts, ends)
        ),
        "mean_heldout_base_target_probability": mean(base_probs),
        "mean_heldout_edited_target_probability": mean(edited_probs),
        "heldout_probability_improved_fraction": mean(
            float(edited > base) for base, edited in zip(base_probs, edited_probs)
        ),
        "max_target_value_norm_ratio": max(norm_ratios),
        "all_finite": all(math.isfinite(value) for value in finite_values),
    }


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["all_finite"]
        and row["mean_final_loss"] < row["mean_initial_loss"]
        and row["max_target_value_norm_ratio"] <= 1.7501
    ]
    if not eligible:
        raise RuntimeError("No finite target-value configuration remained editable")
    return max(
        eligible,
        key=lambda row: (
            row["mean_heldout_edited_target_probability"]
            - row["mean_heldout_base_target_probability"],
            row["rewrite_exact"] + row["declarative_paraphrase_exact"],
            -row["mean_final_loss"],
            row["run"],
        ),
    )


def run_one(
    *,
    manifest: Path,
    output: Path,
    layers: list[int],
    schedule: str,
    reveal: str,
    config: Mapping[str, Any],
    resume: bool,
) -> None:
    if output.exists():
        if resume and (output / "report_summary.json").exists():
            return
        raise FileExistsError(output)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_dnpe_editor.py"),
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output),
        "--method",
        "multi_state_target_value_editor",
        "--layers",
        ",".join(map(str, layers)),
        "--partial_mask_schedule",
        schedule,
        "--reveal_policy",
        reveal,
        "--learning_rate",
        str(config["learning_rate"]),
        "--target_optimization_steps",
        str(config["target_optimization_steps"]),
        "--state_consistency_weight",
        str(config["state_consistency_weight"]),
        "--old_target_suppression_weight",
        str(config["old_target_suppression_weight"]),
        "--include_locality",
        "0",
        "--decode_batch_size",
        "16",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=CAMPAIGN_ROOT / "D2_target_value_optimization_v1",
    )
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    b2 = read_json(
        CAMPAIGN_ROOT
        / "B2_partial_state_mdm_memit_v1"
        / "smoke_policy_selection.json"
    )
    policy = str(b2["selected_policy_by_length"]["2"])
    schedule, reveal, _consistency, _old = POLICIES[policy]
    site_lock = read_json(
        CAMPAIGN_ROOT / "site_policy_lock_v1" / "site_policy_lock.json"
    )
    site = next(
        row
        for row in site_lock["policies"]
        if row["policy_id"] == "stable_temporal_site_set"
    )
    layers = list(map(int, site["layers"]))
    protocol = CAMPAIGN_ROOT / "protocol_v1"
    stage1_rows = []
    stage1_by_id = {}
    for config in staged_configs():
        identifier = config_id(config)
        output = args.root / f"n2_stage1_{identifier}"
        run_one(
            manifest=protocol / "dnpe_kamel_smoke_20_n2.jsonl",
            output=output,
            layers=layers,
            schedule=schedule,
            reveal=reveal,
            config=config,
            resume=bool(args.resume),
        )
        row = {**config, **diagnostic_summary(output)}
        stage1_rows.append(row)
        stage1_by_id[identifier] = row
    stage1_best = choose(stage1_rows)
    all_rows = list(stage1_rows)
    seen = {config_id(row) for row in stage1_rows}
    for config in objective_configs(stage1_best):
        identifier = config_id(config)
        if identifier in seen:
            continue
        seen.add(identifier)
        output = args.root / f"n2_stage2_{identifier}"
        run_one(
            manifest=protocol / "dnpe_kamel_smoke_20_n2.jsonl",
            output=output,
            layers=layers,
            schedule=schedule,
            reveal=reveal,
            config=config,
            resume=bool(args.resume),
        )
        all_rows.append({**config, **diagnostic_summary(output)})
    selected = choose(all_rows)
    selected_config = {
        key: selected[key]
        for key in (
            "learning_rate",
            "target_optimization_steps",
            "state_consistency_weight",
            "old_target_suppression_weight",
        )
    }
    validation_rows = []
    for length in (3, 4):
        output = args.root / f"heldout_n{length}_{config_id(selected_config)}"
        run_one(
            manifest=protocol / f"dnpe_kamel_smoke_20_n{length}.jsonl",
            output=output,
            layers=layers,
            schedule=schedule,
            reveal=reveal,
            config=selected_config,
            resume=bool(args.resume),
        )
        validation_rows.append(
            {"target_length": length, **diagnostic_summary(output)}
        )
    write_csv(args.root / "target_value_grid.csv", all_rows)
    write_csv(args.root / "heldout_partial_state_validation.csv", validation_rows)
    acceptance = {
        "loss_finite_and_decreases": all(
            row["all_finite"] and row["mean_final_loss"] < row["mean_initial_loss"]
            for row in validation_rows
        ),
        "heldout_target_probability_improves": all(
            row["mean_heldout_edited_target_probability"]
            > row["mean_heldout_base_target_probability"]
            for row in validation_rows
        ),
        "target_value_norm_within_cap": all(
            row["max_target_value_norm_ratio"] <= 1.7501
            for row in validation_rows
        ),
        "bounded_grid_respected": len(all_rows) <= 7,
    }
    passed = all(acceptance.values())
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D2_target_value_optimization",
        "created_at_utc": now_utc(),
        "selected_partial_state_policy": policy,
        "selected_layers": layers,
        "selected_config": selected_config,
        "stage1_grid_size": len(stage1_rows),
        "total_grid_size": len(all_rows),
        "validation_target_lengths": [3, 4],
        "acceptance": acceptance,
        "acceptance_pass": passed,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.root / "selected_target_value_config.json", report)
    write_json(args.root / "report_summary.json", report)
    write_json(args.root / "validation_report.json", acceptance)
    record_stage(
        "D2_target_value_optimization",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.root,
        started_at_utc=started,
        notes="Frozen staged target-value grid validated on held-out partial states.",
        next_stage="D3_causal_update",
    )
    print(json.dumps({"acceptance_pass": passed, "selected_config": selected_config}, sort_keys=True))


if __name__ == "__main__":
    main()
