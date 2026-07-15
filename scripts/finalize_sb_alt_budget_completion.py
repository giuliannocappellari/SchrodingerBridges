#!/usr/bin/env python3
"""Finalize the SB alternatives campaign at a validated budget stop."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    CAMPAIGN_ROOT,
    STATE_ROOT,
    TRACKS,
    append_log,
    git_commit,
    now_utc,
    read_csv,
    read_json,
    read_jsonl,
    record_stage_event,
    refresh_budget,
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

STOP_FILES = (
    "report_summary.json",
    "track_stop_checkpoint.md",
    "negative_result_report.md",
    "track_evidence_table.csv",
    "artifact_availability_manifest.json",
    "next_recommendation.md",
)

FINAL_FILES = (
    "report_summary.json",
    "cross_track_status_table.csv",
    "cross_track_main_results.csv",
    "cross_track_pilot_results.csv",
    "same_subject_stress_table.csv",
    "target_length_table.csv",
    "relation_table.csv",
    "compute_storage_table.csv",
    "paired_bootstrap.csv",
    "rewrite_locality_pareto.png",
    "aggregate_compute_pareto.png",
    "same_subject_plot.png",
    "failure_cases.csv",
    "track_failure_taxonomy.csv",
    "artifact_availability_manifest.json",
    "reproducibility_manifest.json",
    "final_research_report.md",
    "paper_claim_matrix.md",
    "next_research_recommendation.md",
)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def budget_completion_required(
    budget: Mapping[str, Any], untested_tracks: Sequence[str]
) -> dict[str, Any]:
    estimates = {key: float(value) for key, value in budget["pilot_estimates"].items()}
    pilot_reserve = sum(estimates[track_id] for track_id in untested_tracks)
    terminal_reserve = float(budget["reserve_usd"])
    required = pilot_reserve + terminal_reserve
    remaining = float(budget["remaining_budget_usd"])
    return {
        "untested_tracks": list(untested_tracks),
        "mandatory_pilot_reserve_usd": round(pilot_reserve, 6),
        "terminal_reporting_reserve_usd": round(terminal_reserve, 6),
        "required_available_usd": round(required, 6),
        "remaining_budget_usd": round(remaining, 6),
        "shortfall_usd": round(max(0.0, required - remaining), 6),
        "budget_completion_required": remaining + 1e-9 < required,
    }


def artifact_row(name: str, path: str | Path, notes: str = "") -> dict[str, Any]:
    full = repo_path(path)
    row: dict[str, Any] = {
        "artifact_name": name,
        "path": str(Path(path)),
        "exists": full.exists(),
        "size_bytes": full.stat().st_size if full.is_file() else "",
        "sha256": sha256_file(full) if full.is_file() else "",
        "notes": notes,
    }
    return row


def write_text(path: str | Path, text: str) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_stop_package(
    *,
    track_id: str,
    status: str,
    failure_category: str,
    evidence: Sequence[Mapping[str, Any]],
    notes: str,
    budget_audit: Mapping[str, Any],
) -> Path:
    output = TRACK_ROOTS[track_id] / "pilot_stop_package_v1"
    output_full = repo_path(output)
    if (output_full / "report_summary.json").exists():
        raise FileExistsError(output_full)
    evidence_rows = [dict(row) for row in evidence]
    availability = [artifact_row(str(row["artifact_name"]), str(row["path"]), str(row.get("notes", ""))) for row in evidence_rows]
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_id": track_id,
        "track_protocol": next(track["protocol"] for track in TRACKS if track["id"] == track_id),
        "track_name": TRACK_NAMES[track_id],
        "status": status,
        "failure_category": failure_category,
        "scientific_hypothesis_tested": track_id == "T1",
        "pilot_actual_decode_completed": track_id == "T1",
        "bounded_scientific_rescue_used": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "budget_audit": dict(budget_audit),
        "notes": notes,
    }
    write_json(output / "report_summary.json", report)
    write_csv(output / "track_evidence_table.csv", evidence_rows)
    write_json(
        output / "artifact_availability_manifest.json",
        {
            "track_id": track_id,
            "created_at_utc": now_utc(),
            "artifacts": availability,
            "all_declared_required_evidence_present": all(row["exists"] for row in availability),
        },
    )
    if status == "formal_negative":
        checkpoint = f"""# {track_id} Track Stop Checkpoint

Status: `formal_negative`

The learned gate passed its offline acceptance and runtime parity checks, but the
predeclared smoke20 actual decode produced no declarative-paraphrase gain and no
candidate reached the green or yellow efficacy criteria. Confirmation30 was not
legally required after a red smoke result. Historical campaigns were not modified,
and no locked analysis or final split was read.

Failure category: `{failure_category}`.
"""
        negative = f"""# {track_id} Negative Result

This is an actual-decode negative at the tested pilot setting, not an
implementation failure. The learned edit-intent gate localized activation well,
but useful controller efficacy did not survive the smoke evaluation. The result
does not establish that all learned-gate configurations fail; it establishes that
the bounded tested pilot did not pass its predeclared criteria.

{notes}
"""
        recommendation = """# Next Recommendation

Do not scale T1 or send it to `dev_tune_200`. A future separately funded protocol
could precommit a stronger guidance calibration, but this campaign may not spend
the protected breadth-first reserve on that rescue.
"""
    else:
        checkpoint = f"""# {track_id} Track Stop Checkpoint

Status: `budget_not_run`

This track did not receive its mandatory pilot because the remaining authorized
budget could not cover all untested pilots plus the fixed terminal reserve.
It is untested, not scientifically failed.

Failure category: `{failure_category}`.
"""
        negative = f"""# {track_id} Budget-Not-Run Result

No scientific result is claimed for {TRACK_NAMES[track_id]}. Implementation or
plan artifacts may exist, but no valid mandatory pilot was executed. The campaign
stopped under its breadth-first budget guard before consuming more compute.

{notes}
"""
        recommendation = f"""# Next Recommendation

Retest {track_id} only under a new, explicitly funded campaign that preserves the
same locked-split and breadth-first rules. Do not describe this unrun hypothesis
as rejected.
"""
    write_text(output / "track_stop_checkpoint.md", checkpoint)
    write_text(output / "negative_result_report.md", negative)
    write_text(output / "next_recommendation.md", recommendation)
    missing = [name for name in STOP_FILES if not (output_full / name).exists()]
    if missing:
        raise RuntimeError(f"Incomplete {track_id} stop package: {missing}")
    return output


def paired_bootstrap(rows: Sequence[Mapping[str, Any]], trials: int = 2000) -> list[dict[str, Any]]:
    methods = ("learned_gate_myopic", "learned_gate_no_rollout", "learned_gate_mc_bridge")
    metrics = (
        ("rewrite_exact", "rewrite", False),
        ("declarative_paraphrase_exact", "declarative_paraphrases", False),
        ("same_subject_tfpr", "rewrite", True),
    )
    output: list[dict[str, Any]] = []
    for method in methods:
        for metric, bucket, stress_only in metrics:
            by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for row in rows:
                split_role = str(row.get("split_role") or "")
                is_stress = "same_subject_stress" in split_role
                if is_stress != stress_only or str(row.get("bucket")) != bucket:
                    continue
                row_method = str(row.get("method"))
                if row_method not in {"base", method}:
                    continue
                edit_id = str(row.get("edit_id") or row.get("case_id")).replace("__same_subject_stress", "")
                by_method[row_method][edit_id].append(to_float(row.get("exact_rate")))
            common = sorted(set(by_method["base"]) & set(by_method[method]))
            if not common:
                continue
            deltas = [
                mean(by_method[method][edit_id]) - mean(by_method["base"][edit_id])
                for edit_id in common
            ]
            rng = random.Random(f"{method}:{metric}:20260715")
            samples = sorted(
                mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
                for _ in range(trials)
            )
            output.append(
                {
                    "method": method,
                    "baseline": "base",
                    "metric": metric,
                    "num_edits": len(common),
                    "delta": mean(deltas),
                    "ci95_low": samples[int(0.025 * (trials - 1))],
                    "ci95_high": samples[int(0.975 * (trials - 1))],
                    "bootstrap_unit": "edit_id",
                    "trials": trials,
                }
            )
    return output


def grouped_metric_rows(
    rows: Sequence[Mapping[str, Any]], group_key: str
) -> list[dict[str, Any]]:
    allowed_methods = {"base", "learned_gate_myopic", "learned_gate_no_rollout", "learned_gate_mc_bridge"}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if "same_subject_stress" in str(row.get("split_role") or ""):
            continue
        method = str(row.get("method"))
        bucket = str(row.get("bucket"))
        if method not in allowed_methods or bucket not in {"rewrite", "declarative_paraphrases"}:
            continue
        group = str(row.get(group_key) or "unknown")
        grouped[(method, group, bucket)].append(to_float(row.get("exact_rate")))
    return [
        {
            "track_id": "T1",
            "method": method,
            "group": group,
            "bucket": bucket,
            "exact_rate": mean(values),
            "num_prompt_rows": len(values),
        }
        for (method, group, bucket), values in sorted(grouped.items())
    ]


def write_scatter_plot(path: Path, rows: Sequence[Mapping[str, Any]], *, x_key: str, y_key: str, title: str) -> None:
    image = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), title, fill="black", font=font)
    left, top, right, bottom = 90, 60, 860, 500
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, bottom, left, top), fill="black", width=2)
    values = [(to_float(row.get(x_key)), to_float(row.get(y_key)), str(row.get("method", row.get("track_id", "")))) for row in rows]
    xmax = max((x for x, _, _ in values), default=1.0) or 1.0
    ymax = max((y for _, y, _ in values), default=1.0) or 1.0
    colors = ((35, 87, 137), (201, 66, 56), (45, 132, 83), (182, 124, 38), (117, 82, 150))
    for index, (x, y, label) in enumerate(values):
        px = left + int((x / xmax) * (right - left - 20))
        py = bottom - int((y / ymax) * (bottom - top - 20))
        color = colors[index % len(colors)]
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline="black")
        draw.text((px + 9, py - 7), label, fill=color, font=font)
    draw.text((left, bottom + 22), x_key, fill="black", font=font)
    draw.text((left, top - 18), y_key, fill="black", font=font)
    repo_path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(repo_path(path), format="PNG")


def write_bar_plot(path: Path, rows: Sequence[Mapping[str, Any]], *, value_key: str, title: str) -> None:
    image = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 18), title, fill="black", font=font)
    left, top, right, bottom = 80, 60, 870, 500
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    values = [to_float(row.get(value_key)) for row in rows]
    vmax = max(values, default=1.0) or 1.0
    width = max(18, int((right - left) / max(1, len(rows)) * 0.65))
    slot = (right - left) / max(1, len(rows))
    colors = ((35, 87, 137), (201, 66, 56), (45, 132, 83), (182, 124, 38), (117, 82, 150))
    for index, (row, value) in enumerate(zip(rows, values)):
        x = int(left + slot * index + (slot - width) / 2)
        height = int((value / vmax) * (bottom - top - 30))
        color = colors[index % len(colors)]
        draw.rectangle((x, bottom - height, x + width, bottom), fill=color, outline="black")
        label = str(row.get("method", row.get("track_id", "")))
        draw.text((x, bottom + 8), label[:18], fill="black", font=font)
        draw.text((x, bottom - height - 14), f"{value:.3f}", fill="black", font=font)
    repo_path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(repo_path(path), format="PNG")


def read_t1_failure_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("method")) not in {"learned_gate_myopic", "learned_gate_mc_bridge"}:
            continue
        if "same_subject_stress" in str(row.get("split_role") or ""):
            continue
        if str(row.get("bucket")) not in {"rewrite", "declarative_paraphrases"}:
            continue
        if to_float(row.get("exact_rate")) > 0.0:
            continue
        output.append(
            {
                "track_id": "T1",
                "method": row.get("method"),
                "edit_id": row.get("edit_id"),
                "relation_id": row.get("relation_id"),
                "bucket": row.get("bucket"),
                "prompt": row.get("prompt_text") or row.get("prompt"),
                "target": row.get("target"),
                "sample_outputs": json.dumps(row.get("sample_outputs") or []),
                "failure_type": "target_not_produced",
            }
        )
        if len(output) >= 40:
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t1_report_dir",
        type=Path,
        default=Path("runs/counterfact_learned_gate_raw_bridge_v1/smoke20_report_v3"),
    )
    parser.add_argument(
        "--t1_decode_dir",
        type=Path,
        default=Path("runs/counterfact_learned_gate_raw_bridge_v1/smoke20_decode_v2"),
    )
    args = parser.parse_args()

    t1_report = read_json(args.t1_report_dir / "report_summary.json")
    if t1_report.get("acceptance_pass") or t1_report.get("selected_candidate"):
        raise RuntimeError("T1 is not eligible for a formal negative stop")
    t1_results = read_csv(args.t1_report_dir / "pilot_results.csv")
    t1_rows = read_jsonl(args.t1_decode_dir / "per_case_results.jsonl")

    budget = refresh_budget("campaign_budget_terminal_audit", "Final budget-stop audit.")
    set_track_status(
        "T1",
        "formal_negative",
        evidence_path=str(TRACK_ROOTS["T1"] / "pilot_stop_package_v1"),
        rescue_used=False,
    )
    record_stage_event(
        track="T1",
        stage="T1_pilot_stop",
        event="pilot_formal_negative",
        status="fail",
        notes="Offline gate passed; smoke20 had no viable efficacy candidate and confirmation was not required.",
    )

    budget = read_json(STATE_ROOT / "budget_state.json")
    untested = [track["id"] for track in TRACKS if track["id"] != "T1"]
    budget_audit = budget_completion_required(budget, untested)
    if not budget_audit["budget_completion_required"]:
        raise RuntimeError(f"Budget stop is not justified: {budget_audit}")

    t1_evidence = [
        {"artifact_name": "gate training", "path": "runs/counterfact_learned_gate_raw_bridge_v1/gate_train_v2/report_summary.json", "role": "offline gate acceptance", "notes": "required"},
        {"artifact_name": "runtime gate audit", "path": "runs/counterfact_learned_gate_raw_bridge_v1/runtime_gate_audit_v2/report_summary.json", "role": "runtime parity and leakage", "notes": "required"},
        {"artifact_name": "smoke decode summary", "path": str(args.t1_decode_dir / "summary.json"), "role": "actual decode", "notes": "required"},
        {"artifact_name": "smoke decision", "path": str(args.t1_report_dir / "report_summary.json"), "role": "pilot decision", "notes": "required"},
        {"artifact_name": "smoke metrics", "path": str(args.t1_report_dir / "pilot_results.csv"), "role": "method metrics", "notes": "required"},
    ]
    t1_stop = write_stop_package(
        track_id="T1",
        status="formal_negative",
        failure_category="actual_decode_failed",
        evidence=t1_evidence,
        notes="The best learned-gated rewrite exact was 0.05 and declarative paraphrase exact was 0.0 at the tested pilot point.",
        budget_audit=budget_audit,
    )

    code_evidence = {
        "T2": [
            {"artifact_name": "endpoint collector implementation", "path": "scripts/build_t2_activation_endpoints.py", "role": "implementation only", "notes": "pilot not run"},
            {"artifact_name": "offline activation SB implementation", "path": "scripts/train_t2_activation_sb.py", "role": "implementation only", "notes": "pilot not run"},
        ],
        "T3": [
            {"artifact_name": "CSBM data builder", "path": "scripts/build_t3_csbm_data.py", "role": "implementation only", "notes": "pilot not run"},
            {"artifact_name": "CSBM trainer", "path": "scripts/train_t3_csbm.py", "role": "implementation only", "notes": "pilot not run"},
        ],
        "T4": [
            {"artifact_name": "partial CSBM trainer", "path": "scripts/train_t4_partial_csbm.py", "role": "implementation only", "notes": "pilot not run"},
            {"artifact_name": "track plan", "path": "UNBALANCED_PARTIAL_CSBM_PLAN.md", "role": "protocol", "notes": "pilot not run"},
        ],
        "T5": [
            {"artifact_name": "track plan", "path": "PARAMETER_SPACE_SB_PLAN.md", "role": "protocol only", "notes": "implementation and pilot not run"},
        ],
    }
    stop_paths = {"T1": t1_stop}
    for track_id in untested:
        stop_paths[track_id] = write_stop_package(
            track_id=track_id,
            status="budget_not_run",
            failure_category="budget_not_run",
            evidence=code_evidence[track_id],
            notes=(
                f"Remaining ${budget_audit['remaining_budget_usd']:.2f}; required "
                f"${budget_audit['required_available_usd']:.2f} for all untested pilots plus reserve."
            ),
            budget_audit=budget_audit,
        )
        set_track_status(
            track_id,
            "budget_not_run",
            evidence_path=str(stop_paths[track_id]),
            rescue_used=False,
        )
        record_stage_event(
            track=track_id,
            stage=f"{track_id}_budget_stop",
            event="mandatory_pilot_not_run",
            status="budget_not_run",
            notes="Hypothesis untested; breadth-first budget guard prevented launch.",
        )

    final_root = CAMPAIGN_ROOT / "final_research_package_v1"
    final_full = repo_path(final_root)
    if (final_full / "report_summary.json").exists():
        raise FileExistsError(final_full)

    track_rows = [
        {
            "track_id": "T1",
            "track_name": TRACK_NAMES["T1"],
            "status": "formal_negative",
            "pilot_executed": True,
            "failure_category": "actual_decode_failed",
            "scientific_hypothesis_tested": True,
            "scale_up_allowed": False,
            "evidence_path": str(stop_paths["T1"]),
        }
    ] + [
        {
            "track_id": track_id,
            "track_name": TRACK_NAMES[track_id],
            "status": "budget_not_run",
            "pilot_executed": False,
            "failure_category": "budget_not_run",
            "scientific_hypothesis_tested": False,
            "scale_up_allowed": False,
            "evidence_path": str(stop_paths[track_id]),
        }
        for track_id in untested
    ]
    write_csv(final_root / "cross_track_status_table.csv", track_rows)

    main_rows: list[dict[str, Any]] = []
    for row in t1_results:
        main_rows.append({"track_id": "T1", "result_available": True, **row})
    for track_id in untested:
        main_rows.append(
            {
                "track_id": track_id,
                "method": "not_run",
                "result_available": False,
                "status": "budget_not_run",
                "notes": "No pilot metrics; hypothesis untested.",
            }
        )
    write_csv(final_root / "cross_track_main_results.csv", main_rows)
    write_csv(final_root / "cross_track_pilot_results.csv", main_rows)

    stress_rows = [
        {
            "track_id": "T1",
            "method": row.get("method"),
            "same_subject_tfpr": row.get("same_subject_tfpr"),
            "base_same_subject_tfpr": next((item.get("same_subject_tfpr") for item in t1_results if item.get("method") == "base"), ""),
            "budget": row.get("same_subject_budget", ""),
            "available": True,
        }
        for row in t1_results
    ]
    stress_rows.extend(
        {"track_id": track_id, "method": "not_run", "available": False, "notes": "budget_not_run"}
        for track_id in untested
    )
    write_csv(final_root / "same_subject_stress_table.csv", stress_rows)

    target_rows = grouped_metric_rows(t1_rows, "target_length_bin")
    for row in target_rows:
        row["target_length_bin"] = row.pop("group")
    target_rows.extend(
        {"track_id": track_id, "method": "not_run", "target_length_bin": "", "available": False, "notes": "budget_not_run"}
        for track_id in untested
    )
    write_csv(final_root / "target_length_table.csv", target_rows)

    relation_rows = grouped_metric_rows(t1_rows, "relation_id")
    for row in relation_rows:
        row["relation_id"] = row.pop("group")
    relation_rows.extend(
        {"track_id": track_id, "method": "not_run", "relation_id": "", "available": False, "notes": "budget_not_run"}
        for track_id in untested
    )
    write_csv(final_root / "relation_table.csv", relation_rows)

    compute_rows = [
        {
            "track_id": "T1",
            "method": row.get("method"),
            "gpu_minutes_per_edit": row.get("gpu_minutes_per_edit_method_share"),
            "model_eval_count_total": t1_report.get("model_eval_count"),
            "runtime_seconds_total": t1_report.get("runtime_seconds"),
            "storage_bytes_per_edit": "",
            "status": "measured_smoke20",
        }
        for row in t1_results
    ]
    compute_rows.extend(
        {
            "track_id": track["id"],
            "method": "not_run",
            "gpu_minutes_per_edit": "",
            "model_eval_count_total": 0,
            "runtime_seconds_total": 0,
            "storage_bytes_per_edit": "",
            "status": "budget_not_run",
            "planned_pilot_estimate_usd": track["pilot_estimate_usd"],
        }
        for track in TRACKS
        if track["id"] in untested
    )
    write_csv(final_root / "compute_storage_table.csv", compute_rows)

    bootstrap_rows = paired_bootstrap(t1_rows)
    write_csv(final_root / "paired_bootstrap.csv", bootstrap_rows)
    failure_rows = read_t1_failure_cases(t1_rows)
    write_csv(final_root / "failure_cases.csv", failure_rows)
    taxonomy_rows = [
        {
            "track_id": row["track_id"],
            "primary_failure_category": row["failure_category"],
            "scientific_hypothesis_tested": row["scientific_hypothesis_tested"],
            "interpretation": (
                "actual decode did not meet efficacy criteria"
                if row["track_id"] == "T1"
                else "untested because authorized budget was insufficient"
            ),
        }
        for row in track_rows
    ]
    write_csv(final_root / "track_failure_taxonomy.csv", taxonomy_rows)

    learned_plot_rows = [row for row in t1_results if str(row.get("method", "")).startswith("learned_gate_")]
    for row in learned_plot_rows:
        row["efficacy_mean"] = mean(
            [to_float(row.get("rewrite_exact")), to_float(row.get("declarative_paraphrase_exact"))]
        )
    write_scatter_plot(
        final_root / "rewrite_locality_pareto.png",
        learned_plot_rows,
        x_key="same_subject_tfpr",
        y_key="efficacy_mean",
        title="T1 smoke20 efficacy vs same-subject TFPR",
    )
    write_scatter_plot(
        final_root / "aggregate_compute_pareto.png",
        learned_plot_rows,
        x_key="gpu_minutes_per_edit_method_share",
        y_key="efficacy_mean",
        title="T1 smoke20 efficacy vs GPU minutes per edit",
    )
    write_bar_plot(
        final_root / "same_subject_plot.png",
        learned_plot_rows,
        value_key="same_subject_tfpr",
        title="T1 learned-gate same-subject TFPR",
    )

    pilot_lock = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "created_at_utc": now_utc(),
        "analysis_500_used": False,
        "final_test_used": False,
        "breadth_first_pilots_complete": False,
        "terminal_reason": "budget_completion",
        "budget_audit": budget_audit,
        "tracks": track_rows,
    }
    write_json(CAMPAIGN_ROOT / "pilot_registry_lock.json", pilot_lock)

    strongest_claim = (
        "The T1 learned edit-intent gate passed strong offline localization and runtime-parity audits, "
        "but the bounded initial actual-decode point did not produce useful paraphrase efficacy. "
        "No Schrodinger-bridge alternative earned a positive scientific claim in this budget-limited campaign."
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "campaign_status": "budget_completion",
        "completion_kind": "budget",
        "package_validation_pass": True,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "tracks_terminal": True,
        "all_mandatory_pilots_executed": False,
        "pilot_passed_tracks": [],
        "formal_negative_tracks": ["T1"],
        "budget_not_run_tracks": untested,
        "budget_audit": budget_audit,
        "estimated_spend_usd": budget.get("estimated_spend_usd"),
        "remaining_budget_usd": budget.get("remaining_budget_usd"),
        "strongest_defensible_claim": strongest_claim,
        "limitations": [
            "T1 used only the initial predeclared smoke20 guidance point; no bounded calibration was funded.",
            "T2-T5 mandatory pilots were not run and must not be interpreted as scientific failures.",
            "No common dev, analysis_500, or final_test_500 evaluation was justified.",
        ],
    }
    write_json(final_root / "report_summary.json", report)

    write_json(
        final_root / "reproducibility_manifest.json",
        {
            "campaign_protocol": CAMPAIGN_PROTOCOL,
            "git_commit": git_commit(),
            "python_version": platform.python_version(),
            "analysis_500_used": False,
            "final_test_used": False,
            "t1_decode_summary_sha256": sha256_file(args.t1_decode_dir / "summary.json"),
            "t1_results_sha256": sha256_file(args.t1_report_dir / "pilot_results.csv"),
            "common_split_summary_sha256": sha256_file(CAMPAIGN_ROOT / "common_protocol_v1/split_summary.json"),
            "terminal_budget_audit": budget_audit,
        },
    )

    write_text(
        final_root / "final_research_report.md",
        f"""# Schrodinger-Bridge Alternatives Campaign: Budget Completion

## Outcome

The campaign reached its predeclared budget terminal condition before all five
mandatory breadth-first pilots could run. T1 completed a real smoke20 actual
decode and is a formal negative at the tested pilot point. T2-T5 are
`budget_not_run`, not scientifically rejected.

## T1

The learned gate passed offline discrimination, runtime parity, leakage, and
same-subject activation checks. Actual decoding remained localized, but the best
learned-gated rewrite exact was 0.05 and declarative paraphrase exact was 0.0.
No green or yellow candidate existed, so confirmation30 and scale-up were not
authorized.

## Budget guard

After T1 reporting, `${budget_audit['remaining_budget_usd']:.2f}` remained while
`${budget_audit['required_available_usd']:.2f}` was required to preserve all
untested pilots and the fixed terminal reserve. The shortfall was
`${budget_audit['shortfall_usd']:.2f}`. Starting T2 would have violated the
breadth-first policy.

## Strongest defensible claim

{strongest_claim}

## Locked evaluation

`analysis_500` and `final_test_500` were not used. No primary dev candidate was
locked, and no final-test run was justified.
""",
    )
    write_text(
        final_root / "paper_claim_matrix.md",
        """# Paper Claim Matrix

| Claim | Status | Evidence |
|---|---|---|
| Edit-intent localization | Diagnostic support | T1 gate passed offline and runtime parity audits. |
| Strong SB editing method | Not supported | T1 actual decode failed efficacy; T2-T5 untested. |
| Activation-space transport | Untested | T2 budget-not-run. |
| Categorical CSBM | Untested | T3 budget-not-run. |
| Partial/unbalanced transport | Untested | T4 budget-not-run. |
| Parameter-space editing | Untested | T5 budget-not-run. |
| Diagnostic/negative result | Supported | Learned localization alone did not yield useful T1 pilot efficacy. |
""",
    )
    write_text(
        final_root / "next_research_recommendation.md",
        """# Next Research Recommendation

Do not inspect locked analysis or final splits. A successor campaign should fund
all five minimum pilots plus reporting reserve before launch, then resume from
T2 while treating T1's bounded actual-decode result as historical evidence. The
first scientific priority is activation-space SB, followed by the already
implemented categorical T3/T4 pilots. T5 should remain last because its endpoint
adapter prerequisite is the most expensive and highest risk.
""",
    )

    availability_paths: list[Path] = [final_root / name for name in FINAL_FILES if name != "artifact_availability_manifest.json"]
    for track_id, stop_path in stop_paths.items():
        availability_paths.extend(stop_path / name for name in STOP_FILES)
    availability = [artifact_row(path.name, path) for path in availability_paths]
    missing = [row["path"] for row in availability if not row["exists"]]
    if missing:
        raise RuntimeError(f"Final package artifacts missing: {missing}")
    write_json(
        final_root / "artifact_availability_manifest.json",
        {
            "campaign_protocol": CAMPAIGN_PROTOCOL,
            "created_at_utc": now_utc(),
            "all_required_artifacts_present": True,
            "artifacts": availability,
        },
    )

    terminal_status = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "campaign_status": "budget_completion",
        "terminal": True,
        "created_at_utc": now_utc(),
        "analysis_500_used": False,
        "final_test_used": False,
        "final_package": str(final_root),
        "budget_audit": budget_audit,
    }
    write_json(CAMPAIGN_ROOT / "campaign_terminal_status.json", terminal_status)

    record_stage_event(
        track="campaign",
        stage="phase_f_final_reporting",
        event="budget_completion_package_validated",
        status="budget_completion",
        notes=f"T1 formal negative; T2-T5 budget-not-run; shortfall=${budget_audit['shortfall_usd']:.2f}.",
    )
    state = read_json(STATE_ROOT / "campaign_state.json")
    state.update(
        {
            "campaign_status": "budget_completion",
            "current_track": None,
            "current_stage": "terminal",
            "terminal_reason": "authorized_budget_insufficient_for_remaining_mandatory_pilots_plus_reserve",
            "terminal_at_utc": now_utc(),
            "analysis_500_used": False,
            "final_test_used": False,
            "last_git_commit": git_commit(),
        }
    )
    write_json(STATE_ROOT / "campaign_state.json", state)
    budget = read_json(STATE_ROOT / "budget_state.json")
    budget.update(
        {
            "budget_guard_pass": False,
            "campaign_terminal_reason": "budget_completion",
            "terminal_budget_audit": budget_audit,
            "updated_at_utc": now_utc(),
        }
    )
    write_json(STATE_ROOT / "budget_state.json", budget)
    append_log("Campaign terminal package validated. Pod may now be stopped under the budget-completion rule.")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
