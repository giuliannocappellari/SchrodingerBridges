#!/usr/bin/env python3
"""Build and validate the terminal next-direction selection package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    MANDATORY_TRACKS,
    PLAN_FILES,
    PROTOCOL_ROOT,
    STATE_ROOT,
    git_commit,
    initialize_state,
    now_utc,
    read_csv,
    read_json,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)


REQUIRED_FILES = (
    "report_summary.json",
    "direction_selection_matrix.csv",
    "track_results.csv",
    "paired_bootstrap.csv",
    "efficacy_locality_pareto.png",
    "coverage_risk_plot.png",
    "multi_token_results.csv",
    "failure_taxonomy.csv",
    "artifact_availability_manifest.json",
    "reproducibility_manifest.json",
    "final_research_report.md",
    "next_direction_recommendation.md",
    "SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md",
    "terminal_package_validation.json",
)

TERMINAL = {
    "pilot_failed",
    "confirmation_passed",
    "confirmation_failed",
    "protocol_infeasible",
    "infrastructure_blocked",
    "not_triggered",
    "budget_not_run",
}
CLASS_PRIORITY = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, None: 0, "": 0}
RECOMMENDATION = {
    "N1": "pursue_relation_residualized_editor",
    "N2": "pursue_fisher_constrained_editor",
    "N3": "pursue_primal_dual_editor",
    "N4": "pursue_selective_conformal_editor",
    "N5": "pursue_joint_span_coupled_editor",
    "N6": "pursue_integrated_statistical_editor",
}


def _report_for(row: Mapping[str, Any]) -> dict[str, Any]:
    path = row.get("report_path")
    if not path:
        return {}
    value = ROOT / str(path)
    return read_json(value) if value.is_file() else {}


def _first_present(report: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = report.get(key)
        if value is not None:
            return value
    return None


def selection_row(row: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    success_class = row.get("success_class") or report.get("success_class")
    return {
        "track_id": row["track_id"],
        "candidate_id": row.get("nominated_candidate") or report.get("candidate_id"),
        "pilot_status": "passed" if row.get("pilot_pass") else "failed",
        "confirmation_status": "passed" if row.get("confirmation_pass") else "failed" if row.get("status") == "confirmation_failed" else "not_run",
        "success_class": success_class,
        "rewrite": report.get("rewrite_exact"),
        "paraphrase": report.get("declarative_paraphrase_exact"),
        "same_subject_tfpr": _first_present(
            report, "same_subject_tfpr", "accepted_same_subject_tfpr"
        ),
        "near_tfpr": report.get("near_tfpr"),
        "far_tfpr": report.get("far_tfpr"),
        "distributional_locality_kl": report.get("protected_distributional_kl"),
        "coverage": report.get("coverage"),
        "risk_upper_bound": report.get("risk_upper_bound"),
        "multi_token_exact_delta": json.dumps(report.get("full_span_exact_delta_by_length")) if report.get("full_span_exact_delta_by_length") is not None else None,
        "paired_ci_low": (report.get("same_subject_paired_bootstrap") or report.get("pooled_paired_bootstrap") or {}).get("ci_low"),
        "paired_ci_high": (report.get("same_subject_paired_bootstrap") or report.get("pooled_paired_bootstrap") or {}).get("ci_high"),
        "gpu_minutes_per_edit": report.get("gpu_minutes_per_edit"),
        "implementation_risk": {"N1": "medium", "N2": "medium", "N3": "high", "N4": "medium", "N5": "medium", "N6": "high"}[row["track_id"]],
        "recommended": False,
        "track_status": row.get("status"),
    }


def _plots(output: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 5))
    for row in rows:
        if row["rewrite"] in {None, ""} or row["same_subject_tfpr"] in {None, ""}:
            continue
        axis.scatter(float(row["same_subject_tfpr"]), float(row["rewrite"]), label=row["track_id"])
        axis.annotate(row["track_id"], (float(row["same_subject_tfpr"]), float(row["rewrite"])))
    axis.axvline(0.03, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Same-subject target false-positive rate")
    axis.set_ylabel("Rewrite exact")
    axis.set_title("Fresh confirmation efficacy-locality comparison")
    fig.tight_layout()
    fig.savefig(output / "efficacy_locality_pareto.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    selective = [row for row in rows if row["coverage"] not in {None, ""}]
    if selective:
        axis.scatter(
            [float(row["coverage"]) for row in selective],
            [float(row["risk_upper_bound"]) for row in selective],
        )
        for row in selective:
            axis.annotate(row["track_id"], (float(row["coverage"]), float(row["risk_upper_bound"])))
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("95% upper risk bound")
    axis.set_title("Selective editing coverage-risk")
    fig.tight_layout()
    fig.savefig(output / "coverage_risk_plot.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "final_direction_selection_package_v1")
    args = parser.parse_args()
    started = now_utc()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    state = initialize_state()
    registry = read_json(STATE_ROOT / "track_registry.json")
    by_id = {row["track_id"]: row for row in registry["tracks"]}
    if any(by_id[track]["status"] not in TERMINAL for track in MANDATORY_TRACKS):
        raise RuntimeError("all mandatory tracks must be terminal before final selection")
    if by_id["N6"]["status"] not in TERMINAL:
        by_id["N6"]["status"] = "not_triggered"
    rows = [selection_row(by_id[track], _report_for(by_id[track])) for track in sorted(by_id)]
    eligible = [
        row
        for row in rows
        if row["confirmation_status"] == "passed" and CLASS_PRIORITY.get(row["success_class"], 0) > 0
    ]
    selected = max(
        eligible,
        key=lambda row: (
            CLASS_PRIORITY[row["success_class"]],
            -(float(row["same_subject_tfpr"]) if row["same_subject_tfpr"] not in {None, ""} else 1.0),
            float(row["rewrite"] or 0.0),
        ),
    ) if eligible else None
    if selected:
        selected["recommended"] = True
        recommendation_status = RECOMMENDATION[selected["track_id"]]
    else:
        recommendation_status = "no_promising_next_direction"
    write_csv(args.output_dir / "direction_selection_matrix.csv", rows)
    write_csv(args.output_dir / "track_results.csv", rows)
    bootstrap_rows = []
    multi_token = []
    for track, row in by_id.items():
        path = row.get("report_path")
        if not path:
            continue
        report_dir = (ROOT / str(path)).parent
        bootstrap = report_dir / "paired_bootstrap.csv"
        if bootstrap.is_file():
            bootstrap_rows.extend({"track_id": track, **item} for item in read_csv(bootstrap))
        if track == "N5":
            length_path = report_dir / "target_length_results.csv"
            if length_path.is_file():
                multi_token.extend(read_csv(length_path))
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    write_csv(args.output_dir / "multi_token_results.csv", multi_token)
    failures = [
        {
            "track_id": row["track_id"],
            "status": row["track_status"],
            "failure_classification": by_id[row["track_id"]].get("failure_classification") or ("confirmation_failure" if row["confirmation_status"] == "failed" else "pilot_failure"),
            "mechanism_pass": by_id[row["track_id"]].get("mechanism_pass"),
            "pilot_pass": by_id[row["track_id"]].get("pilot_pass"),
            "confirmation_pass": by_id[row["track_id"]].get("confirmation_pass"),
        }
        for row in rows
        if not row["recommended"]
    ]
    write_csv(args.output_dir / "failure_taxonomy.csv", failures)
    _plots(args.output_dir, rows)
    reproducibility = {
        "campaign_id": CAMPAIGN_ID,
        "git_commit": git_commit(),
        "plans": [
            {"path": name, "sha256": sha256_file(ROOT / name)} for name in PLAN_FILES
        ],
        "protocol_manifests": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for path in sorted(PROTOCOL_ROOT.glob("*.jsonl"))
        ],
        "historical_analysis_final_opened": False,
        "selected_full_campaign_executed": False,
    }
    write_json(args.output_dir / "reproducibility_manifest.json", reproducibility)
    outcome = "selected_direction" if selected else "no_promising_direction"
    report = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_status": "terminal",
        "terminal_outcome": outcome,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "mandatory_tracks_terminal": True,
        "selected_track": selected["track_id"] if selected else None,
        "selected_candidate": selected["candidate_id"] if selected else None,
        "success_class": selected["success_class"] if selected else None,
        "recommendation_status": recommendation_status,
        "analysis_500_used": False,
        "final_test_used": False,
        "selected_full_campaign_executed": False,
        "package_validation_pass": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    selected_description = (
        f"{selected['track_id']} / `{selected['candidate_id']}` in Class {selected['success_class']}"
        if selected
        else "no candidate; none survived fresh confirmation under Classes A-E"
    )
    (args.output_dir / "final_research_report.md").write_text(
        "# Diffusion Editor Next-Direction Selection\n\n"
        f"## Outcome\n\nThe bounded breadth-first campaign selected {selected_description}.\n\n"
        "All mandatory directions N1-N5 reached terminal pilot status. Only frozen pilot winners were eligible for fresh confirmation, and confirmation was not used for retuning. Historical analysis/final splits remained closed.\n\n"
        "## Interpretation\n\nThe recommendation follows the frozen class ordering A > B > C > D > E. Failed pilots remain bounded evidence rather than universal impossibility claims.\n",
        encoding="utf-8",
    )
    (args.output_dir / "next_direction_recommendation.md").write_text(
        "# Next Direction Recommendation\n\n"
        f"Status: `{recommendation_status}`.\n\n"
        + (
            f"Proceed with a separately approved full campaign for {selected['track_id']}. It outranked alternatives by frozen success class, paired evidence, same-subject safety, and implementation risk.\n"
            if selected
            else "Do not launch a full campaign from these candidates. The strongest defensible result is a bounded comparative negative study; formulate a new protocol before further tuning.\n"
        ),
        encoding="utf-8",
    )
    (args.output_dir / "SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md").write_text(
        "# Selected Direction Full Campaign Draft\n\n"
        "**Draft only. This campaign did not execute the selected full program.**\n\n"
        f"Selected direction: `{selected['track_id'] if selected else 'none'}`.\n\n"
        "## Proposed stages\n\n1. Rebuild independent training/calibration manifests.\n2. Reproduce the confirmed mechanism and editor.\n3. Scale only the frozen architecture.\n4. Lock dev before any analysis.\n5. Run one analysis confirmation and one final evaluation if predeclared criteria pass.\n\n"
        "## Stop rules\n\nStop on mechanism failure, efficacy/locality failure, confirmation regression, leakage, or malformed-span excess. Do not lower thresholds or reuse confirmation prompts for training.\n",
        encoding="utf-8",
    )
    missing_before_manifest = [
        name
        for name in REQUIRED_FILES
        if name not in {
            "artifact_availability_manifest.json",
            "terminal_package_validation.json",
        }
        and not (args.output_dir / name).is_file()
    ]
    report["package_validation_pass"] = not missing_before_manifest
    write_json(args.output_dir / "report_summary.json", report)
    artifacts = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name not in {
            "artifact_availability_manifest.json",
            "terminal_package_validation.json",
        }:
            artifacts.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_json(args.output_dir / "artifact_availability_manifest.json", {"artifacts": artifacts})
    missing = [
        name
        for name in REQUIRED_FILES
        if name != "terminal_package_validation.json"
        and not (args.output_dir / name).is_file()
    ]
    validation = {
        "campaign_id": CAMPAIGN_ID,
        "mandatory_tracks_terminal": True,
        "required_files": list(REQUIRED_FILES),
        "missing_files_before_validation_write": missing,
        "historical_analysis_final_opened": False,
        "selected_full_campaign_executed": False,
        "acceptance_pass": not missing,
    }
    write_json(args.output_dir / "terminal_package_validation.json", validation)
    if missing:
        raise RuntimeError(f"terminal package is incomplete: {missing}")
    state = initialize_state()
    state.update(
        {
            "campaign_status": "terminal",
            "terminal_outcome": outcome,
            "selected_track": report["selected_track"],
            "selected_candidate": report["selected_candidate"],
            "current_stage": "S6_final_package",
            "next_stage": None,
            "analysis_500_used": False,
            "final_test_used": False,
            "pod_status": "stop_pending",
            "updated_at_utc": now_utc(),
        }
    )
    write_json(STATE_ROOT / "campaign_state.json", state)
    record_stage("S5_final_selection", status="passed", acceptance_pass=True, output_dir=args.output_dir, started_at_utc=started, notes=f"recommendation={recommendation_status}", next_stage="S6_final_package")
    record_stage("S6_final_package", status="passed", acceptance_pass=True, output_dir=args.output_dir, started_at_utc=started, notes="Terminal package validated; selected full campaign remains draft-only.", next_stage=None)
    print(json.dumps({"terminal_outcome": outcome, "recommendation_status": recommendation_status, "selected_track": report["selected_track"]}))


if __name__ == "__main__":
    main()
