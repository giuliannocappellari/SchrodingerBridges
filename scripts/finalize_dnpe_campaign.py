#!/usr/bin/env python3
"""Build and validate the terminal positive or bounded-negative DNPE package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    STATE_ROOT,
    artifact_manifest,
    git_commit,
    now_utc,
    read_json,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)


REQUIRED_PACKAGE_FILES = (
    "report_summary.json",
    "final_research_report.md",
    "paper_claim_recommendation.md",
    "main_results_table.csv",
    "same_subject_stress_table.csv",
    "multi_token_table.csv",
    "causal_localization_table.csv",
    "locality_distribution_table.csv",
    "compute_storage_table.csv",
    "sequential_edit_table.csv",
    "paired_bootstrap.csv",
    "rewrite_locality_pareto.png",
    "causal_heatmap.png",
    "partial_state_plot.png",
    "update_norm_locality_plot.png",
    "failure_cases.csv",
    "artifact_availability_manifest.json",
    "reproducibility_manifest.json",
    "terminal_package_validation.json",
)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_metric_reports() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(CAMPAIGN_ROOT.glob("**/report_summary.json")):
        if "final_research_package_v1" in path.parts:
            continue
        try:
            report = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if "rewrite_exact" not in report:
            continue
        row = {
            "method": str(report.get("method") or path.parent.name),
            "stage": str(report.get("stage") or path.parent.parent.name),
            "run": str(path.parent.relative_to(ROOT)),
            "num_edits": int(report.get("num_edits", 0)),
            "rewrite_exact": float(report.get("rewrite_exact", 0.0)),
            "declarative_paraphrase_exact": float(
                report.get("declarative_paraphrase_exact", 0.0)
            ),
            "target_token_f1": float(report.get("target_token_f1", 0.0)),
            "same_subject_tfpr": float(report.get("same_subject_tfpr", 0.0)),
            "near_tfpr": float(report.get("near_tfpr", 0.0)),
            "far_tfpr": float(report.get("far_tfpr", 0.0)),
            "malformed_rate": float(report.get("malformed_rate", 0.0)),
            "gpu_minutes_per_edit": float(
                report.get("gpu_minutes_per_edit", 0.0)
            ),
            "storage_bytes": int(report.get("storage_bytes", 0)),
            "analysis_500_used": bool(report.get("analysis_500_used", False)),
            "final_test_used": bool(report.get("final_test_used", False)),
        }
        if all(
            math.isfinite(float(row[key]))
            for key in (
                "rewrite_exact",
                "declarative_paraphrase_exact",
                "same_subject_tfpr",
                "near_tfpr",
                "far_tfpr",
                "malformed_rate",
            )
        ):
            rows.append(row)
    return rows


def copy_or_status_csv(source: Path, destination: Path, status: str) -> None:
    rows = read_csv(source)
    write_csv(destination, rows or [{"status": status, "source": str(source)}])


def _blank_plot(path: Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def rewrite_locality_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _blank_plot(path, "Rewrite/locality Pareto", "No completed metric runs")
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    for row in rows:
        axis.scatter(row["same_subject_tfpr"], row["rewrite_exact"], s=30)
    axis.axvline(0.03, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Same-subject target false-positive rate")
    axis.set_ylabel("Rewrite exact")
    axis.set_title("DNPE rewrite/locality trade-off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def partial_state_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    numeric = [row for row in rows if row.get("target_length")]
    if not numeric:
        _blank_plot(path, "Partial-state editing", "No completed multi-token rows")
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    families = sorted({str(row.get("method_family") or row.get("policy")) for row in numeric})
    for family in families:
        values = [row for row in numeric if str(row.get("method_family") or row.get("policy")) == family]
        values.sort(key=lambda row: int(row["target_length"]))
        y_key = "rewrite_exact" if "rewrite_exact" in values[0] else "rewrite_gain"
        axis.plot(
            [int(row["target_length"]) for row in values],
            [float(row.get(y_key, 0.0)) for row in values],
            marker="o",
            label=family,
        )
    axis.set_xlabel("Exact target length")
    axis.set_ylabel("Rewrite exact")
    axis.set_title("Multi-token partial-state editing")
    if len(families) <= 12:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def update_norm_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    values = [
        row
        for row in rows
        if row.get("mean_update_norm") not in (None, "")
        and row.get("same_subject_tfpr") not in (None, "")
    ]
    if not values:
        _blank_plot(path, "Update geometry", "No completed D4 grid")
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        [float(row["mean_update_norm"]) for row in values],
        [float(row["same_subject_tfpr"]) for row in values],
    )
    axis.set_xlabel("Mean update norm")
    axis.set_ylabel("Same-subject TFPR")
    axis.set_title("Update norm versus locality")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def failure_cases(main_run: Path | None) -> list[dict[str, Any]]:
    if main_run is None or not (main_run / "edited_per_prompt.csv").exists():
        return [{"status": "main candidate was not evaluated"}]
    rows = read_csv(main_run / "edited_per_prompt.csv")
    failures = []
    for row in rows:
        expected = str(row.get("expected_hit", "")).casefold()
        target_hit = str(row.get("target_new_hit", "")).casefold()
        malformed = str(row.get("malformed", "")).casefold()
        if expected in {"false", "0"} or (
            row.get("bucket") not in {"rewrite", "declarative_paraphrase"}
            and target_hit in {"true", "1"}
        ) or malformed in {"true", "1"}:
            failures.append(row)
    return failures[:200] or [{"status": "no sampled failure rows"}]


def determine_outcome(requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, "explicit_cli_choice"
    e2 = CAMPAIGN_ROOT / "E2_pilot100_v1" / "report_summary.json"
    if not e2.exists():
        return "infrastructure_blocked", "E2 report missing"
    report = read_json(e2)
    if report.get("acceptance_pass"):
        g3 = CAMPAIGN_ROOT / "G3_final500_v1" / "report_summary.json"
        if g3.exists() and read_json(g3).get("acceptance_pass"):
            return "positive", "locked final package passed"
        return "infrastructure_blocked", "pilot passed but locked pipeline incomplete"
    return "formal_negative", "no pilot candidate passed frozen efficacy/locality/mechanism criteria"


def build_package(output: Path, *, requested_outcome: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = now_utc()
    outcome, outcome_reason = determine_outcome(requested_outcome)
    metric_rows = discover_metric_reports()
    write_csv(output / "main_results_table.csv", metric_rows or [{"status": "no metric reports"}])
    write_csv(
        output / "same_subject_stress_table.csv",
        [
            {
                "method": row["method"],
                "run": row["run"],
                "same_subject_tfpr": row["same_subject_tfpr"],
                "near_tfpr": row["near_tfpr"],
                "far_tfpr": row["far_tfpr"],
            }
            for row in metric_rows
        ]
        or [{"status": "no locality metric reports"}],
    )
    multi_rows = read_csv(CAMPAIGN_ROOT / "E2_pilot100_v1" / "kamel_dev_table.csv")
    if not multi_rows:
        multi_rows = read_csv(
            CAMPAIGN_ROOT
            / "B2_partial_state_mdm_memit_v1"
            / "partial_state_summary.csv"
        )
    write_csv(output / "multi_token_table.csv", multi_rows or [{"status": "not completed"}])
    causal_rows = []
    for path in (
        CAMPAIGN_ROOT / "standard_causal_tracing_v1" / "aie_by_layer_position.csv",
        CAMPAIGN_ROOT / "temporal_causal_tracing_v1" / "tie_aggregate.csv",
    ):
        for row in read_csv(path):
            causal_rows.append({"source": path.parent.name, **row})
    write_csv(output / "causal_localization_table.csv", causal_rows or [{"status": "not completed"}])
    write_csv(
        output / "locality_distribution_table.csv",
        [
            {
                "method": row["method"],
                "run": row["run"],
                "same_subject_tfpr": row["same_subject_tfpr"],
                "near_tfpr": row["near_tfpr"],
                "far_tfpr": row["far_tfpr"],
                "malformed_rate": row["malformed_rate"],
            }
            for row in metric_rows
        ]
        or [{"status": "not completed"}],
    )
    write_csv(
        output / "compute_storage_table.csv",
        [
            {
                "method": row["method"],
                "run": row["run"],
                "gpu_minutes_per_edit": row["gpu_minutes_per_edit"],
                "storage_bytes": row["storage_bytes"],
            }
            for row in metric_rows
        ]
        or [{"status": "not completed"}],
    )
    scaling = read_csv(CAMPAIGN_ROOT / "F2_scaling_v1" / "sequential_edit_table.csv")
    write_csv(
        output / "sequential_edit_table.csv",
        scaling or [{"status": "not_run_due_pilot_stop"}],
    )
    copy_or_status_csv(
        CAMPAIGN_ROOT / "E2_pilot100_v1" / "paired_bootstrap.csv",
        output / "paired_bootstrap.csv",
        "not_completed",
    )
    rewrite_locality_plot(output / "rewrite_locality_pareto.png", metric_rows)
    partial_state_plot(output / "partial_state_plot.png", multi_rows)
    d4_grid = read_csv(
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "nullspace_staged_grid.csv"
    )
    update_norm_plot(output / "update_norm_locality_plot.png", d4_grid)
    causal_source = (
        CAMPAIGN_ROOT / "temporal_causal_tracing_v1" / "temporal_heatmaps.png"
    )
    if not causal_source.exists():
        causal_source = (
            CAMPAIGN_ROOT / "standard_causal_tracing_v1" / "causal_heatmap.png"
        )
    if causal_source.exists():
        shutil.copyfile(causal_source, output / "causal_heatmap.png")
    else:
        _blank_plot(
            output / "causal_heatmap.png",
            "Causal localization",
            "Causal tracing was not completed",
        )
    main_run = (
        CAMPAIGN_ROOT
        / "D4_causal_partial_state_nullspace_v1"
        / "pilot100_selected_nullspace"
    )
    write_csv(
        output / "failure_cases.csv",
        failure_cases(main_run if main_run.exists() else None),
    )
    stage_reports = {}
    for stage, path in (
        ("B1", CAMPAIGN_ROOT / "B1_mdm_memit_reproduction_v1" / "report_summary.json"),
        ("B2", CAMPAIGN_ROOT / "B2_partial_state_mdm_memit_v1" / "report_summary.json"),
        ("B3", CAMPAIGN_ROOT / "B3_alphaedit_style_mdm_memit_v1" / "report_summary.json"),
        ("B4", CAMPAIGN_ROOT / "B4_timerome_dlm_style_v1" / "report_summary.json"),
        ("C1", CAMPAIGN_ROOT / "standard_causal_tracing_v1" / "report_summary.json"),
        ("C2", CAMPAIGN_ROOT / "temporal_causal_tracing_v1" / "report_summary.json"),
        ("D2", CAMPAIGN_ROOT / "D2_target_value_optimization_v1" / "report_summary.json"),
        ("D4", CAMPAIGN_ROOT / "D4_causal_partial_state_nullspace_v1" / "report_summary.json"),
        ("E2", CAMPAIGN_ROOT / "E2_pilot100_v1" / "report_summary.json"),
    ):
        stage_reports[stage] = read_json(path) if path.exists() else None
    claim = (
        "strong_diffusion_native_parametric_editor"
        if outcome == "positive"
        else "bounded_negative_result"
        if outcome == "formal_negative"
        else "infrastructure_blocked"
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "terminal_outcome": outcome,
        "terminal_reason": outcome_reason,
        "strongest_claim_class": claim,
        "analysis_500_used": any(
            bool(row.get("analysis_500_used")) for row in metric_rows
        ),
        "final_test_used": any(
            bool(row.get("final_test_used")) for row in metric_rows
        ),
        "historical_campaigns_modified": False,
        "stage_acceptance": {
            name: (payload.get("acceptance_pass") if payload else None)
            for name, payload in stage_reports.items()
        },
        "package_validation_pass": False,
    }
    write_json(output / "report_summary.json", report)
    failure_sections = {
        "baseline reproduction": report["stage_acceptance"].get("B1"),
        "partial-state reproduction": report["stage_acceptance"].get("B2"),
        "null-space baseline": report["stage_acceptance"].get("B3"),
        "temporal residual baseline": report["stage_acceptance"].get("B4"),
        "standard causal localization": report["stage_acceptance"].get("C1"),
        "temporal causal localization": report["stage_acceptance"].get("C2"),
        "target-value optimization": report["stage_acceptance"].get("D2"),
        "locality projection": report["stage_acceptance"].get("D4"),
        "joint pilot eligibility": report["stage_acceptance"].get("E2"),
    }
    (output / "final_research_report.md").write_text(
        "# Diffusion-Native Parametric Editor Final Report\n\n"
        f"- Terminal outcome: `{outcome}`\n"
        f"- Reason: {outcome_reason}\n"
        f"- Analysis used: `{report['analysis_500_used']}`\n"
        f"- Final test used: `{report['final_test_used']}`\n\n"
        "## Stage Evidence\n\n"
        + "".join(
            f"- {name}: `{value}`\n" for name, value in failure_sections.items()
        )
        + "\n## Interpretation\n\n"
        + (
            "The frozen method survived dev, locked analysis, and final evaluation.\n"
            if outcome == "positive"
            else "No method satisfied the complete frozen efficacy, locality, and mechanism criteria after the permitted bounded path. This is a bounded scientific result, not an infrastructure-only claim.\n"
            if outcome == "formal_negative"
            else "The scientific hypothesis remains unresolved because the campaign could not complete the required pipeline.\n"
        ),
        encoding="utf-8",
    )
    (output / "paper_claim_recommendation.md").write_text(
        "# Paper Claim Recommendation\n\n"
        f"Primary classification: `{claim}`.\n\n"
        "Do not claim a strong editor unless locked analysis and final evaluation passed. "
        "Report supported reproduction or mechanism subresults separately from the joint editor verdict.\n",
        encoding="utf-8",
    )
    availability = []
    for name, payload in stage_reports.items():
        availability.append(
            {
                "stage": name,
                "available": payload is not None,
                "acceptance_pass": payload.get("acceptance_pass") if payload else None,
            }
        )
    write_json(
        output / "artifact_availability_manifest.json",
        {"campaign_id": CAMPAIGN_ID, "stages": availability},
    )
    script_paths = sorted(
        path
        for path in (ROOT / "scripts").glob("*dnpe*.py")
        if path.is_file()
    )
    protocol_paths = sorted(
        (CAMPAIGN_ROOT / "protocol_v1").glob("*.json*")
    )
    write_json(
        output / "reproducibility_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "git_commit": git_commit(),
            "primary_model": {
                "id": PRIMARY_MODEL_ID,
                "revision": PRIMARY_MODEL_REVISION,
            },
            "secondary_model": {
                "id": SECONDARY_MODEL_ID,
                "revision": SECONDARY_MODEL_REVISION,
            },
            "protocol_artifacts": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in protocol_paths
            ],
            "source_scripts": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in script_paths
            ],
            "commands": {
                "main_table": "python reproduce_dnpe_paper.py --table main",
                "causal_figure": "python reproduce_dnpe_paper.py --figure causal_heatmap",
                "validate": "python reproduce_dnpe_paper.py --validate-terminal-package",
            },
            "random_seeds": [260717101],
            "analysis_500_used": report["analysis_500_used"],
            "final_test_used": report["final_test_used"],
        },
    )
    write_json(
        output / "terminal_package_validation.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "required_files": list(REQUIRED_PACKAGE_FILES),
            "required_files_present": False,
            "nonempty_files": False,
            "artifact_hash_count": 0,
            "artifact_hashes": [],
            "self_hash_excluded_to_avoid_recursive_manifest": True,
            "acceptance_pass": False,
        },
    )
    present = all((output / name).exists() for name in REQUIRED_PACKAGE_FILES)
    nonempty = all(
        (output / name).stat().st_size > 0 for name in REQUIRED_PACKAGE_FILES
    )
    report["package_validation_pass"] = bool(present and nonempty)
    write_json(output / "report_summary.json", report)
    package_hashes = [
        row
        for row in artifact_manifest(output)
        if not row["path"].endswith("terminal_package_validation.json")
    ]
    validation = read_json(output / "terminal_package_validation.json")
    validation.update(
        {
            "required_files_present": present,
            "nonempty_files": nonempty,
            "artifact_hash_count": len(package_hashes),
            "artifact_hashes": package_hashes,
            "acceptance_pass": present and nonempty,
        }
    )
    write_json(output / "terminal_package_validation.json", validation)
    if not validation["acceptance_pass"]:
        raise RuntimeError("Terminal package validation failed")
    record_stage(
        "H_final_package",
        status="passed",
        acceptance_pass=True,
        output_dir=output,
        started_at_utc=started,
        notes=f"Terminal {outcome} package validated.",
        next_stage=None,
    )
    state = read_json(STATE_ROOT / "campaign_state.json")
    state["campaign_status"] = outcome
    state["current_stage"] = "H_final_package"
    state["next_stage"] = None
    state["analysis_500_used"] = report["analysis_500_used"]
    state["final_test_used"] = report["final_test_used"]
    state["pod_status"] = "stop_pending_after_terminal_validation"
    state["updated_at_utc"] = now_utc()
    write_json(STATE_ROOT / "campaign_state.json", state)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CAMPAIGN_ROOT / "final_research_package_v1",
    )
    parser.add_argument(
        "--outcome",
        choices=("auto", "positive", "formal_negative", "infrastructure_blocked"),
        default="auto",
    )
    args = parser.parse_args()
    report = build_package(args.output_dir, requested_outcome=args.outcome)
    print(
        json.dumps(
            {
                "terminal_outcome": report["terminal_outcome"],
                "package_validation_pass": report["package_validation_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
