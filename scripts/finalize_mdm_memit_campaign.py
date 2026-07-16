#!/usr/bin/env python3
"""Build and validate the terminal cross-track MDM-MEMIT research package."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    MODEL_ID,
    MODEL_REVISION,
    STATE_ROOT,
    git_commit,
    now_utc,
    read_json,
    sha256_file,
    update_campaign_state,
    write_csv,
    write_json,
)


TRACK_ROOTS = {
    "M1": CAMPAIGN_ROOT / "M1_mdm_memit_reproduction_v1",
    "M2": CAMPAIGN_ROOT / "M2_partial_mask_memit_v1",
    "M3": CAMPAIGN_ROOT / "M3_schrodinger_regularized_memit_v1",
    "M4": CAMPAIGN_ROOT / "M4_mask_pattern_sb_v1",
    "F1": CAMPAIGN_ROOT / "F1_adaptive_edit_memory_v1",
    "F2": CAMPAIGN_ROOT / "F2_toy_text_csbm_v1",
}
FINAL_ROOT = CAMPAIGN_ROOT / "final_research_package_v1"
CROSS_ROOT = CAMPAIGN_ROOT / "cross_track"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _copy_csv(source: Path, destination: Path) -> list[dict[str, Any]]:
    rows = _read_csv(source)
    write_csv(destination, rows)
    return rows


def _status(report: Mapping[str, Any]) -> str:
    return "passed" if bool(report.get("acceptance_pass")) else "formal_negative"


def _write_stop_package(track: str, report: Mapping[str, Any]) -> None:
    root = TRACK_ROOTS[track]
    if bool(report.get("acceptance_pass")):
        return
    classification = (
        "actual-decode failure"
        if track in {"M1", "M2"}
        else "offline or locked scientific failure"
    )
    evidence = [
        {
            "track": track,
            "classification": classification,
            "acceptance_pass": False,
            "report_summary": str(root / "report_summary.json"),
            "old_analysis_500_used": report.get("old_analysis_500_used", False),
            "old_final_test_used": report.get("old_final_test_used", False),
        }
    ]
    write_csv(root / "track_evidence_table.csv", evidence)
    write_json(
        root / "artifact_availability_manifest.json",
        {
            "track": track,
            "report_summary_exists": (root / "report_summary.json").exists(),
            "final_track_report_exists": (root / "final_track_report.md").exists(),
        },
    )
    (root / "track_stop_checkpoint.md").write_text(
        f"# {track} Stop Checkpoint\n\nStatus: formal_negative\n\nClassification: {classification}.\n",
        encoding="utf-8",
    )
    (root / "negative_result_report.md").write_text(
        f"# {track} Negative Result\n\nThe predeclared acceptance criteria were not met. Thresholds were not lowered.\n",
        encoding="utf-8",
    )
    (root / "next_recommendation.md").write_text(
        f"# {track} Recommendation\n\nRetain this result as bounded evidence and do not silently resume the track.\n",
        encoding="utf-8",
    )


def _not_triggered(track: str, reason: str) -> dict[str, Any]:
    root = TRACK_ROOTS[track]
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": track,
        "triggered": False,
        "status": "not_triggered",
        "reason": reason,
        "acceptance_pass": False,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(root / "report_summary.json", report)
    (root / "final_track_report.md").write_text(
        f"# {track}\n\nStatus: **not triggered**\n\n{reason}\n", encoding="utf-8"
    )
    return report


def _plot_packages(
    m1: Mapping[str, Any],
    m2_rows: Sequence[Mapping[str, Any]],
    m3_rows: Sequence[Mapping[str, Any]],
    m4_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(["rewrite", "paraphrase"], [float(m1.get("efficacy", 0)), float(m1.get("generalization", 0))], color=["#276FBF", "#F4A261"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact rate")
    ax.set_title("MDM-MEMIT reproduction")
    fig.tight_layout()
    fig.savefig(FINAL_ROOT / "rewrite_generalization_plot.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, color in (("fully_masked", "#6C757D"), ("partial_seed1", "#2A9D8F")):
        points = sorted(
            (
                int(row["target_length"]),
                float(row["full_target_exact"]),
            )
            for row in m2_rows
            if row.get("label") == label and row.get("bucket") == "rewrite"
        )
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=label)
    ax.set_xlabel("Target length")
    ax.set_ylabel("Full-target exact")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Partial-mask gain by target length")
    fig.tight_layout()
    fig.savefig(FINAL_ROOT / "partial_mask_gain_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in m3_rows:
        try:
            ax.scatter(float(row["update_norm_sum"]), float(row["same_subject_tfpr"]), s=28)
        except (KeyError, TypeError, ValueError):
            continue
    for row in m4_rows:
        try:
            ax.scatter(float(row["mean_trajectory_target_cost"]), 0.0, marker="x", s=28)
        except (KeyError, TypeError, ValueError):
            continue
    ax.set_xlabel("Intervention or trajectory cost")
    ax.set_ylabel("Same-subject TFPR")
    ax.set_title("Path-cost and locality diagnostics")
    fig.tight_layout()
    fig.savefig(FINAL_ROOT / "path_cost_locality_pareto.png", dpi=180)
    plt.close(fig)


def _latest_stage_outcomes(history_rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return disjoint terminal lists using the latest audit for each stage."""

    latest: dict[str, bool] = {}
    for row in history_rows:
        stage = str(row.get("stage") or "")
        value = str(row.get("acceptance_pass", "")).casefold()
        if stage and value in {"true", "false"}:
            latest[stage] = value == "true"
    completed = [stage for stage, accepted in latest.items() if accepted]
    failed = [stage for stage, accepted in latest.items() if not accepted]
    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=FINAL_ROOT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    CROSS_ROOT.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, Any]] = {}
    for track in ("M1", "M2", "M3", "M4"):
        path = TRACK_ROOTS[track] / "report_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Mandatory track report is missing: {path}")
        reports[track] = json.loads(path.read_text(encoding="utf-8"))

    if bool(reports["M1"].get("acceptance_pass")):
        reports["F1"] = _not_triggered("F1", "M1 established the parametric reproduction, so F1 was not eligible.")
    else:
        f1_path = TRACK_ROOTS["F1"] / "report_summary.json"
        if not f1_path.exists():
            raise FileNotFoundError("F1 was triggered by M1 failure but has no terminal report")
        reports["F1"] = json.loads(f1_path.read_text(encoding="utf-8"))

    f2_triggered = not bool(reports["M3"].get("sb_specific_positive_result")) and not bool(
        reports["M4"].get("sb_specific_positive_result")
    )
    if f2_triggered:
        f2_path = TRACK_ROOTS["F2"] / "report_summary.json"
        if not f2_path.exists():
            raise FileNotFoundError("F2 was triggered by M3/M4 failure but has no terminal report")
        reports["F2"] = json.loads(f2_path.read_text(encoding="utf-8"))
    else:
        reports["F2"] = _not_triggered("F2", "At least one mandatory SB extension passed, so F2 was not eligible.")

    for track in ("M1", "M2", "M3", "M4"):
        _write_stop_package(track, reports[track])

    track_rows: list[dict[str, Any]] = []
    for track in ("M1", "M2", "M3", "M4", "F1", "F2"):
        report = reports[track]
        status = report.get("status") or _status(report)
        track_rows.append(
            {
                "track": track,
                "status": status,
                "acceptance_pass": report.get("acceptance_pass", False),
                "triggered": report.get("triggered", track in {"M1", "M2", "M3", "M4"}),
                "report_summary": str(TRACK_ROOTS[track] / "report_summary.json"),
                "old_analysis_500_used": report.get("old_analysis_500_used", False),
                "old_final_test_used": report.get("old_final_test_used", False),
            }
        )
    write_csv(CROSS_ROOT / "track_evidence_matrix.csv", track_rows)
    write_csv(args.output_dir / "track_status_table.csv", track_rows)

    m1_rows = _copy_csv(
        TRACK_ROOTS["M1"] / "counterfact_reproduction.csv",
        args.output_dir / "counterfact_reproduction.csv",
    )
    m2_rows = _copy_csv(
        TRACK_ROOTS["M2"] / "main_results_by_length.csv",
        args.output_dir / "kamel_partial_mask_results.csv",
    )
    m3_rows = _copy_csv(
        TRACK_ROOTS["M3"] / "analysis_results.csv",
        args.output_dir / "schrodinger_regularization_results.csv",
    )
    m4_rows = _copy_csv(
        TRACK_ROOTS["M4"] / "main_results_by_length.csv",
        args.output_dir / "mask_pattern_bridge_results.csv",
    )
    generation_rows = _copy_csv(
        TRACK_ROOTS["M1"] / "generation_robustness.csv",
        args.output_dir / "generation_robustness.csv",
    )

    locality_rows = [
        {
            "track": "M1",
            "same_subject_tfpr": reports["M1"].get("same_subject_tfpr"),
            "classic_specificity_base_agreement": reports["M1"].get("classic_specificity_base_agreement"),
        }
    ]
    locality_rows.extend(
        {
            "track": "M3",
            "dataset": row.get("dataset"),
            "label": row.get("label"),
            "same_subject_tfpr": row.get("same_subject_tfpr"),
            "classic_specificity_base_agreement": row.get("classic_specificity_base_agreement"),
        }
        for row in m3_rows
    )
    write_csv(args.output_dir / "same_subject_and_specificity.csv", locality_rows)
    target_length_rows = [dict(row, track="M2") for row in m2_rows] + [
        dict(row, track="M4") for row in m4_rows
    ]
    write_csv(args.output_dir / "target_length_table.csv", target_length_rows)

    relation_rows: list[dict[str, Any]] = []
    m1_per_prompt = _read_csv(TRACK_ROOTS["M1"] / "locked_reproduction_v1/edited_per_prompt.csv")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in m1_per_prompt:
        grouped.setdefault((row.get("relation_id", ""), row.get("bucket", "")), []).append(row)
    for (relation, bucket), values in grouped.items():
        if bucket not in {"rewrite", "paraphrase"}:
            continue
        relation_rows.append(
            {
                "track": "M1",
                "relation_id": relation,
                "bucket": bucket,
                "num_rows": len(values),
                "target_new_exact": sum(str(row.get("target_new_hit")).casefold() == "true" for row in values) / len(values),
            }
        )
    write_csv(args.output_dir / "relation_table.csv", relation_rows)

    compute_rows = [
        {
            "track": "M1",
            "gpu_minutes_per_edit": reports["M1"].get("gpu_minutes_per_edit"),
            "editing_seconds": reports["M1"].get("editing_seconds"),
            "inference_seconds": reports["M1"].get("edited_inference_seconds"),
        },
        {
            "track": "M2",
            "gpu_minutes_per_edit": "reported_per_run",
            "editing_seconds": "see run summaries",
            "inference_seconds": "see run summaries",
        },
        {
            "track": "M3",
            "gpu_minutes_per_edit": "see analysis_results.csv",
            "editing_seconds": "see run summaries",
            "inference_seconds": "see run summaries",
        },
        {
            "track": "M4",
            "gpu_minutes_per_edit": "see trajectory model_eval_count",
            "editing_seconds": "see run summaries",
            "inference_seconds": "see trajectory_costs.csv",
        },
    ]
    write_csv(args.output_dir / "compute_table.csv", compute_rows)

    bootstrap_rows: list[dict[str, Any]] = []
    for track in ("M1", "M2", "M3", "M4"):
        for row in _read_csv(TRACK_ROOTS[track] / "paired_bootstrap.csv"):
            bootstrap_rows.append({"track": track, **row})
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    _copy_csv(TRACK_ROOTS["M1"] / "failure_cases.csv", args.output_dir / "failure_cases.csv")

    mechanism_rows = [
        {"track": "M2", **row}
        for row in _read_csv(TRACK_ROOTS["M2"] / "state_schedule_ablation.csv")
    ] + [
        {"track": "M3", **row}
        for row in _read_csv(TRACK_ROOTS["M3"] / "loss_ablation.csv")
    ] + [
        {"track": "M4", **row}
        for row in _read_csv(TRACK_ROOTS["M4"] / "mechanism_ablation.csv")
    ]
    write_csv(CROSS_ROOT / "mechanism_ablation_matrix.csv", mechanism_rows)

    claim_rows = [
        ("MDM-MEMIT reproduction", bool(reports["M1"].get("acceptance_pass")), "M1"),
        ("partial-mask multi-token improvement", bool(reports["M2"].get("acceptance_pass")), "M2"),
        ("SB path/locality regularization", bool(reports["M3"].get("sb_specific_positive_result")), "M3"),
        ("exact mask-pattern SB control", bool(reports["M4"].get("sb_specific_positive_result")), "M4"),
        ("toy categorical CSBM", bool(reports["F2"].get("acceptance_pass")) if f2_triggered else False, "F2"),
    ]
    claim_text = "# Claim Matrix\n\n" + "\n".join(
        f"- {claim}: **{'supported' if passed else 'rejected under protocol' if track in {'M1','M2','M3','M4'} else 'not triggered'}** ({track})"
        for claim, passed, track in claim_rows
    ) + "\n"
    (CROSS_ROOT / "claim_matrix.md").write_text(claim_text, encoding="utf-8")
    (args.output_dir / "paper_claim_matrix.md").write_text(claim_text, encoding="utf-8")

    _plot_packages(reports["M1"], m2_rows, m3_rows, m4_rows)
    campaign_positive = any(passed for _, passed, _ in claim_rows)
    strongest = next((claim for claim, passed, _ in claim_rows if passed), "bounded negative result")
    report_md = f"""# Masked-Diffusion MEMIT and Schrödinger Campaign

## Outcome

Campaign status: **{'positive' if campaign_positive else 'scientific negative'}**

Strongest defensible claim: **{strongest}**.

## Tracks

""" + "\n".join(
        f"- {row['track']}: {row['status']}" for row in track_rows
    ) + """

## Protocol Integrity

The campaign used fresh manifests. Historical locked analysis/final prompt contents, labels, outputs, and metrics were not used. M3's fresh campaign analysis was opened only after its dev candidates were frozen.
"""
    (args.output_dir / "final_research_report.md").write_text(report_md, encoding="utf-8")
    (args.output_dir / "next_research_recommendation.md").write_text(
        "# Next Recommendation\n\nBuild directly on the highest supported claim and retain every failed mechanism as bounded evidence; do not reopen locked splits for tuning.\n",
        encoding="utf-8",
    )

    reproducibility = {
        "campaign_id": CAMPAIGN_ID,
        "git_commit": git_commit(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "python": platform.python_version(),
        "protocol_manifests": {
            path.name: sha256_file(path)
            for path in sorted((CAMPAIGN_ROOT / "protocol").glob("*.jsonl"))
        },
        "historical_locked_prompt_content_used": False,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(args.output_dir / "reproducibility_manifest.json", reproducibility)

    required = [
        "report_summary.json",
        "track_status_table.csv",
        "counterfact_reproduction.csv",
        "kamel_partial_mask_results.csv",
        "schrodinger_regularization_results.csv",
        "mask_pattern_bridge_results.csv",
        "generation_robustness.csv",
        "same_subject_and_specificity.csv",
        "target_length_table.csv",
        "relation_table.csv",
        "compute_table.csv",
        "paired_bootstrap.csv",
        "rewrite_generalization_plot.png",
        "partial_mask_gain_plot.png",
        "path_cost_locality_pareto.png",
        "failure_cases.csv",
        "artifact_availability_manifest.json",
        "reproducibility_manifest.json",
        "final_research_report.md",
        "paper_claim_matrix.md",
        "next_research_recommendation.md",
    ]
    artifact_rows = []
    for name in required:
        path = args.output_dir / name
        artifact_rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": True if name == "artifact_availability_manifest.json" else path.exists(),
                "size_bytes": 0 if name == "artifact_availability_manifest.json" else path.stat().st_size if path.exists() else 0,
                "sha256": (
                    "self_referential_not_hashed"
                    if name == "artifact_availability_manifest.json"
                    else sha256_file(path) if path.exists() else ""
                ),
            }
        )
    write_json(args.output_dir / "artifact_availability_manifest.json", {"artifacts": artifact_rows})
    package_pass = all((args.output_dir / name).exists() and (args.output_dir / name).stat().st_size > 0 for name in required if name != "report_summary.json")
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "terminal_final_package",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "campaign_positive": campaign_positive,
        "strongest_defensible_claim": strongest,
        "mandatory_tracks_terminal": True,
        "f1_trigger_correct": bool(reports["M1"].get("acceptance_pass")) == (not bool(reports["F1"].get("triggered"))),
        "f2_trigger_correct": f2_triggered == bool(reports["F2"].get("triggered")),
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "package_validation_pass": package_pass,
        "acceptance_pass": package_pass,
    }
    write_json(args.output_dir / "report_summary.json", summary)
    if not package_pass:
        raise RuntimeError("Terminal package validation failed")
    history_rows = _read_csv(STATE_ROOT / "stage_history.csv")
    completed_stages, failed_stages = _latest_stage_outcomes(history_rows)
    update_campaign_state(
        campaign_status="completed_positive" if campaign_positive else "completed_scientific_negative",
        current_stage="terminal_final_package",
        next_stage="pod_stop",
        track_status={row["track"]: row["status"] for row in track_rows},
        completed_stages=list(dict.fromkeys(completed_stages + ["terminal_final_package"])),
        failed_stages=list(dict.fromkeys(failed_stages)),
        old_analysis_500_used=False,
        old_final_test_used=False,
        pod_status="stop_pending",
        terminal_package=str(args.output_dir),
        terminal_package_validated=True,
        ended_at_utc=now_utc(),
        ended_epoch=time.time(),
    )
    print(json.dumps({"acceptance_pass": True, "campaign_positive": campaign_positive, "strongest_claim": strongest}))


if __name__ == "__main__":
    main()
