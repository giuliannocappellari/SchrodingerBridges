#!/usr/bin/env python3
"""Build and validate the terminal continual-direction selection package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    STATE_ROOT,
    autonomous_enabled,
    now_utc,
    read_csv,
    read_json,
    record_stage,
    register_artifacts,
    sha256_file,
    write_csv,
    write_json,
)


PILOT_ROOT = CAMPAIGN_ROOT / "D_breadth_first_pilots_v1"
CONFIRMATION_ROOT = CAMPAIGN_ROOT / "F_fresh_confirmation_v1"
CONDITIONAL_ROOT = CAMPAIGN_ROOT / "G_conditional_tracks_v1"
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "final_direction_selection_package_v1"

REQUIRED_FILES = (
    "report_summary.json",
    "final_research_report.md",
    "direction_selection_matrix.csv",
    "sequential_retention_table.csv",
    "forgetting_curves.csv",
    "same_subject_table.csv",
    "base_retention_table.csv",
    "multi_token_table.csv",
    "compute_storage_table.csv",
    "paired_bootstrap.csv",
    "track_status_registry.json",
    "artifact_availability_manifest.json",
    "reproducibility_manifest.json",
    "next_direction_recommendation.md",
    "SELECTED_CONTINUAL_DIRECTION_FULL_CAMPAIGN_DRAFT.md",
    "terminal_package_validation.json",
)

TRACK_RECOMMENDATIONS = {
    "C1": "pursue_diffusiongrow_continual_editor",
    "C2": "pursue_partial_state_replay_editor",
    "C3": "pursue_sparse_memory_editor",
    "C4": "pursue_gated_adapter_editor",
    "C5": "pursue_orthogonal_fisher_editor",
    "C6": "pursue_functional_replay_editor",
    "C7": "pursue_bridge_replay_editor",
    "C8": "pursue_sb_consolidation_editor",
    "C9": "pursue_dual_memory_editor",
    "C14": "pursue_integrated_continual_editor",
}
CLASS_PRIORITY = {"A": 4, "C": 3, "B": 2, "D": 1}


def _truthy(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    classes = [item for item in str(row.get("success_classes") or "").split(",") if item]
    priority = max((CLASS_PRIORITY.get(item, 0) for item in classes), default=0)

    def number(key: str, default: float) -> float:
        value = row.get(key)
        return default if value in {None, ""} else float(value)

    return (
        priority,
        number("past_retention", 0.0),
        -number("average_forgetting", 1.0),
        -number("same_subject_tfpr", 1.0),
        number("current_rewrite_exact", 0.0),
    )


def select_recommendation(
    confirmed_rows: Sequence[Mapping[str, Any]],
    *,
    mechanism_signal_count: int,
) -> tuple[str, Mapping[str, Any] | None, str]:
    # Equivalent executable mechanisms count once. This prevents the same
    # subject+relation rank-8 router from appearing as three independent wins.
    best_by_equivalence: dict[str, Mapping[str, Any]] = {}
    for row in confirmed_rows:
        equivalence = str(row.get("implementation_equivalence_class") or row.get("method"))
        previous = best_by_equivalence.get(equivalence)
        if previous is None or selection_key(row) > selection_key(previous):
            best_by_equivalence[equivalence] = row
    if best_by_equivalence:
        selected = max(best_by_equivalence.values(), key=selection_key)
        track = str(selected["track_id"])
        recommendation = TRACK_RECOMMENDATIONS.get(track, "no_promising_continual_direction")
        classes = str(selected.get("success_classes") or "").split(",")
        claim = (
            "full_continual_editor"
            if "A" in classes
            else "sb_specific_continual_result"
            if "C" in classes
            else "retention_locality_pareto_result"
            if "B" in classes
            else "efficiency_scaling_result"
            if "D" in classes
            else "mechanism_only_result"
        )
        return recommendation, selected, claim
    if mechanism_signal_count:
        return "mechanism_only_result", None, "mechanism_only_result"
    return "no_promising_continual_direction", None, "no_promising_continual_direction"


def candidate_rows() -> list[dict[str, Any]]:
    output = []
    for index in range(1, 10):
        track = f"C{index}"
        path = PILOT_ROOT / "track_reports" / f"{track}_pilot_v1" / "candidate_results.csv"
        output.extend(read_csv(path))
    return output


def confirmed_rows() -> list[dict[str, Any]]:
    output = []
    for confirmation in read_csv(CONFIRMATION_ROOT / "confirmation_results.csv"):
        if not _truthy(confirmation.get("confirmation_pass")):
            continue
        report_dir = ROOT / str(confirmation["report_path"])
        rows = read_csv(report_dir.parent / "candidate_results.csv")
        if len(rows) != 1:
            raise RuntimeError(f"Expected one confirmed candidate for {confirmation['track_id']}")
        output.append({**rows[0], "confirmation_status": "confirmed"})
    return output


def collect_table_rows(candidates: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    retention = []
    curves = []
    for candidate in candidates:
        report_path = ROOT / str(candidate["report_path"])
        run_dir = report_path.parent
        for row in read_csv(run_dir / "block_metrics.csv"):
            retention.append(
                {
                    "track_id": candidate["track_id"],
                    "method": candidate["method"],
                    **row,
                }
            )
        for row in read_csv(run_dir / "retention_matrix.csv"):
            curves.append(
                {
                    "track_id": candidate["track_id"],
                    "method": candidate["method"],
                    **row,
                }
            )
    return {"retention": retention, "curves": curves}


def write_markdown(
    output_dir: Path,
    recommendation: str,
    selected: Mapping[str, Any] | None,
    claim: str,
    track_matrix: Sequence[Mapping[str, Any]],
) -> None:
    selected_text = (
        f"`{selected['method']}` from `{selected['track_id']}`"
        if selected is not None
        else "none"
    )
    terminal_counts = {}
    for row in track_matrix:
        terminal_counts[str(row["status"])] = terminal_counts.get(str(row["status"]), 0) + 1
    (output_dir / "final_research_report.md").write_text(
        "\n".join(
            (
                "# Continual Diffusion Editing Selection Report",
                "",
                f"- Recommendation: `{recommendation}`",
                f"- Claim class: `{claim}`",
                f"- Selected confirmed method: {selected_text}",
                "- Historical analysis/final splits used: `false`",
                f"- Terminal track-status counts: `{json.dumps(terminal_counts, sort_keys=True)}`",
                "",
                "The package ranks only fresh-stream confirmed results. Conceptual source adaptations and",
                "non-exact SB/DER/GEM proxies remain explicitly labeled, and equivalent executable",
                "mechanisms are deduplicated before selection.",
                "",
            )
        ),
        encoding="utf-8",
    )
    (output_dir / "next_direction_recommendation.md").write_text(
        f"# Next Direction Recommendation\n\n`{recommendation}`\n\n"
        "This is a bounded selection decision. It does not authorize launching the selected full campaign.\n",
        encoding="utf-8",
    )
    (output_dir / "SELECTED_CONTINUAL_DIRECTION_FULL_CAMPAIGN_DRAFT.md").write_text(
        "# Selected Continual Direction Full Campaign Draft\n\n"
        f"Status: draft only\n\nSelected recommendation: `{recommendation}`\n\n"
        "A separate protocol and explicit user authorization are required before execution.\n",
        encoding="utf-8",
    )


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
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    state = read_json(STATE_ROOT / "campaign_state.json")
    if state.get("analysis_500_used") or state.get("final_test_used"):
        raise RuntimeError("Locked split use detected")
    pending_conditionals = [
        track
        for track, status in state["track_status"].items()
        if track in {"C10", "C11", "C12", "C13", "C14"}
        and status == "triggered_pending_pilot"
    ]
    if pending_conditionals:
        raise RuntimeError(f"Conditional tracks are not terminal: {pending_conditionals}")
    confirmation = read_json(CONFIRMATION_ROOT / "report_summary.json")
    if not confirmation.get("all_eligible_tracks_terminal"):
        raise RuntimeError("Confirmation is not terminal")

    pilots = candidate_rows()
    confirmed = confirmed_rows()
    mechanism_count = sum(_truthy(row.get("mechanism_signal_pass")) for row in pilots)
    recommendation, selected, claim = select_recommendation(
        confirmed, mechanism_signal_count=mechanism_count
    )
    track_matrix = []
    details = state.get("track_details", {})
    for index in range(15):
        track = f"C{index}"
        track_matrix.append(
            {
                "track_id": track,
                "status": state["track_status"].get(track),
                "nominated_candidate": details.get(track, {}).get("nominated_candidate"),
                "report_path": details.get(track, {}).get("report_path"),
                "rescue_used": state.get("rescues_used", {}).get(track, 0),
            }
        )
    write_csv(args.output_dir / "direction_selection_matrix.csv", track_matrix)
    tables = collect_table_rows(pilots)
    write_csv(
        args.output_dir / "sequential_retention_table.csv",
        tables["retention"],
        ("track_id", "method", "block_index", "num_seen_edits", "current_rewrite_exact", "current_paraphrase_exact", "past_rewrite_retention", "same_subject_tfpr", "near_tfpr", "far_tfpr", "malformed_rate", "base_retention_exact", "base_retention_agreement", "base_retention_loss_fraction", "protected_kl"),
    )
    write_csv(
        args.output_dir / "forgetting_curves.csv",
        tables["curves"],
        ("track_id", "method", "evaluation_block", "source_block", "rewrite_exact", "paraphrase_exact"),
    )
    write_csv(
        args.output_dir / "same_subject_table.csv",
        [
            {
                "track_id": row["track_id"],
                "method": row["method"],
                "same_subject_tfpr": row.get("same_subject_tfpr"),
                "near_tfpr": row.get("near_tfpr"),
                "far_tfpr": row.get("far_tfpr"),
                "confirmation_eligible": row.get("confirmation_eligible"),
            }
            for row in pilots
        ],
    )
    write_csv(
        args.output_dir / "base_retention_table.csv",
        [
            {
                "track_id": row["track_id"],
                "method": row["method"],
                "protected_kl": row.get("protected_kl"),
                "base_retention_loss_fraction": row.get("base_retention_loss_fraction"),
            }
            for row in pilots
        ],
    )
    write_csv(
        args.output_dir / "multi_token_table.csv",
        [
            {
                "track_id": row["track_id"],
                "method": row["method"],
                "status": "not_evaluated_no_multi_token_claim",
                "kamel_confirmation_used": False,
            }
            for row in confirmed
        ],
        ("track_id", "method", "status", "kamel_confirmation_used"),
    )
    write_csv(
        args.output_dir / "compute_storage_table.csv",
        [
            {
                "track_id": row["track_id"],
                "method": row["method"],
                "storage_mb_per_edit": row.get("storage_mb_per_edit"),
                "inference_overhead_fraction": row.get("inference_overhead_fraction"),
                "implementation_equivalence_class": row.get("implementation_equivalence_class"),
            }
            for row in pilots
        ],
    )
    bootstrap_rows = []
    for index in range(1, 10):
        track = f"C{index}"
        bootstrap_rows.extend(
            read_csv(PILOT_ROOT / "track_reports" / f"{track}_pilot_v1" / "paired_bootstrap.csv")
        )
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    write_markdown(args.output_dir, recommendation, selected, claim, track_matrix)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "I_final_package",
        "created_at_utc": now_utc(),
        "campaign_status": "terminal_pending_pod_stop",
        "recommendation": recommendation,
        "claim_classification": claim,
        "selected_track": selected.get("track_id") if selected else None,
        "selected_method": selected.get("method") if selected else None,
        "num_confirmed_methods": len(confirmed),
        "num_mechanism_signals": mechanism_count,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)

    artifact_inputs = [
        CAMPAIGN_ROOT / "protocol_v1" / "report_summary.json",
        CAMPAIGN_ROOT / "C0_common_baselines_v1" / "report_summary.json",
        PILOT_ROOT / "report_summary.json",
        CONFIRMATION_ROOT / "report_summary.json",
        CONDITIONAL_ROOT / "report_summary.json",
    ]
    availability = [
        {
            "path": str(path.relative_to(ROOT)),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in artifact_inputs
    ]
    write_json(
        args.output_dir / "artifact_availability_manifest.json",
        {"campaign_id": CAMPAIGN_ID, "artifacts": availability},
    )
    write_json(args.output_dir / "track_status_registry.json", state)
    write_json(
        args.output_dir / "reproducibility_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "input_hashes": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in artifact_inputs
                if path.is_file()
            },
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    write_json(
        args.output_dir / "terminal_package_validation.json",
        {"acceptance_pass": False, "status": "validation_in_progress"},
    )
    existing = {name: (args.output_dir / name).is_file() for name in REQUIRED_FILES}
    all_required = all(existing.values())
    all_inputs = all(row["exists"] for row in availability)
    if not all_required or not all_inputs:
        write_json(
            args.output_dir / "terminal_package_validation.json",
            {
                "required_files": existing,
                "all_required_files_present": all_required,
                "all_required_inputs_present": all_inputs,
                "acceptance_pass": False,
            },
        )
        raise RuntimeError("Terminal package validation failed")

    started_at = now_utc()
    record_stage(
        "H_final_selection",
        status="passed",
        acceptance_pass=True,
        output_dir=args.output_dir,
        started_at_utc=started_at,
        notes=f"recommendation={recommendation}; claim={claim}",
        next_stage="I_final_package",
    )
    record_stage(
        "I_final_package",
        status="passed",
        acceptance_pass=True,
        output_dir=args.output_dir,
        started_at_utc=started_at,
        notes="terminal package validated; pod stop pending",
        next_stage=None,
    )
    state = read_json(STATE_ROOT / "campaign_state.json")
    state["campaign_status"] = "terminal"
    state["current_stage"] = "I_final_package"
    state["next_stage"] = None
    state["pod_status"] = "running_pending_verified_stop"
    state["updated_at_utc"] = now_utc()
    write_json(STATE_ROOT / "campaign_state.json", state)
    write_json(args.output_dir / "track_status_registry.json", state)
    report["campaign_status"] = "terminal"
    report["pod_stop_pending"] = True
    write_json(args.output_dir / "report_summary.json", report)
    package_hashes = {
        name: sha256_file(args.output_dir / name)
        for name in REQUIRED_FILES
        if name not in {"terminal_package_validation.json", "reproducibility_manifest.json"}
    }
    write_json(
        args.output_dir / "reproducibility_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "input_hashes": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in artifact_inputs
            },
            "package_hashes": package_hashes,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    write_json(
        args.output_dir / "terminal_package_validation.json",
        {
            "required_files": {name: (args.output_dir / name).is_file() for name in REQUIRED_FILES},
            "all_required_files_present": True,
            "all_required_inputs_present": True,
            "hashes_recorded": len(package_hashes),
            "analysis_500_used": False,
            "final_test_used": False,
            "pod_stop_pending": True,
            "acceptance_pass": True,
        },
    )
    register_artifacts("I_final_package", args.output_dir.iterdir())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
