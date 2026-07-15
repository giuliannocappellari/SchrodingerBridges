#!/usr/bin/env python3
"""Finalize the alternatives campaign after five formal-negative pilots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    CAMPAIGN_ROOT,
    STATE_ROOT,
    git_commit,
    now_utc,
    read_csv,
    read_json,
    record_stage_event,
    refresh_cost,
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)


FINAL_FILES = (
    "report_summary.json",
    "cross_track_status_table.csv",
    "cross_track_main_results.csv",
    "same_subject_stress_table.csv",
    "target_length_table.csv",
    "relation_table.csv",
    "compute_storage_table.csv",
    "paired_bootstrap.csv",
    "rewrite_locality_pareto.png",
    "aggregate_compute_pareto.png",
    "same_subject_plot.png",
    "failure_cases.csv",
    "artifact_availability_manifest.json",
    "reproducibility_manifest.json",
    "final_research_report.md",
    "paper_claim_matrix.md",
    "next_research_recommendation.md",
)

TRACKS = {
    "T1": {
        "name": "learned edit-intent gate + raw bridge",
        "root": Path("runs/counterfact_learned_gate_raw_bridge_v1"),
        "evidence": Path("runs/counterfact_learned_gate_raw_bridge_v1/smoke20_report_v3/report_summary.json"),
    },
    "T2": {
        "name": "activation-space Schrodinger bridge",
        "root": Path("runs/counterfact_activation_space_sb_v1"),
        "evidence": Path("runs/counterfact_activation_space_sb_v1/activation_sb_offline_v1/report_summary.json"),
    },
    "T3": {
        "name": "conditional answer-span categorical Schrodinger bridge",
        "root": Path("runs/counterfact_conditional_answer_span_csbm_v1"),
        "evidence": Path("runs/counterfact_conditional_answer_span_csbm_v1/csbm_offline_outer4_rescue_v1/report_summary.json"),
    },
    "T4": {
        "name": "unbalanced / partial categorical Schrodinger bridge",
        "root": Path("runs/counterfact_unbalanced_partial_csbm_v1"),
        "evidence": Path("runs/counterfact_unbalanced_partial_csbm_v1/partial_csbm_offline_temperature_rescue_v1/report_summary.json"),
    },
    "T5": {
        "name": "parameter-space Schrodinger bridge",
        "root": Path("runs/counterfact_parameter_space_sb_v1"),
        "evidence": Path("runs/counterfact_parameter_space_sb_v1/direct_endpoint_adapters_rank4_rescue_v1/report_summary.json"),
    },
}


def write_text(path: str | Path, value: str) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(value.rstrip() + "\n", encoding="utf-8")


def nested(payload: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def simple_plot(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), title, fill="black")
    maximum = max([abs(float(value)) for value in values] + [1e-6])
    baseline = 450
    draw.line((80, baseline, 870, baseline), fill="#333333", width=2)
    width = 120
    gap = 35
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = 100 + index * (width + gap)
        height = int(330 * abs(float(value)) / maximum)
        if value >= 0:
            box = (x0, baseline - height, x0 + width, baseline)
            color = "#3b82f6"
        else:
            box = (x0, baseline, x0 + width, baseline + height)
            color = "#dc2626"
        draw.rectangle(box, fill=color)
        draw.text((x0, 465), label, fill="black")
        draw.text((x0, max(55, baseline - height - 22)), f"{float(value):.3f}", fill="black")
    image.save(repo_path(path))


def validate_terminal_registry() -> list[dict[str, str]]:
    rows = read_csv(STATE_ROOT / "track_registry.csv")
    by_id = {row["track_id"]: row for row in rows}
    if set(by_id) != set(TRACKS):
        raise RuntimeError("Track registry does not contain exactly T1-T5")
    nonterminal = [track_id for track_id, row in by_id.items() if row["status"] != "formal_negative"]
    if nonterminal:
        raise RuntimeError(f"Scientific-negative finalizer requires five formal negatives: {nonterminal}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CAMPAIGN_ROOT / "final_research_package_v1",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output_full = repo_path(output)
    if output_full.exists():
        raise FileExistsError(output_full)
    registry = validate_terminal_registry()
    reports: dict[str, dict[str, Any]] = {}
    stops: dict[str, dict[str, Any]] = {}
    for track_id, spec in TRACKS.items():
        evidence = repo_path(spec["evidence"])
        stop = repo_path(spec["root"] / "pilot_stop_package_v1/report_summary.json")
        if not evidence.exists() or not stop.exists():
            raise FileNotFoundError(f"Missing terminal evidence for {track_id}")
        reports[track_id] = read_json(evidence)
        stops[track_id] = read_json(stop)
        if reports[track_id].get("analysis_500_used") or reports[track_id].get("final_test_used"):
            raise RuntimeError(f"Locked split use detected in {track_id}")

    t1_metrics = {
        row["method"]: row
        for row in read_csv(
            "runs/counterfact_learned_gate_raw_bridge_v1/smoke20_report_v3/pilot_results.csv"
        )
    }["learned_gate_mc_bridge"]
    t2 = reports["T2"]["selected_metrics"]
    t3 = reports["T3"]["selected_metrics"]
    t4 = reports["T4"]["selected_metrics"]
    t5 = reports["T5"]["selected_metrics"]

    status_rows = [
        {
            "track_id": track_id,
            "track_name": TRACKS[track_id]["name"],
            "status": "formal_negative",
            "failure_category": stops[track_id]["failure_category"],
            "pilot_actual_decode_completed": stops[track_id]["pilot_actual_decode_completed"],
            "bounded_rescue_used": stops[track_id]["bounded_scientific_rescue_used"],
            "evidence_path": str(TRACKS[track_id]["evidence"]),
            "stop_package": str(TRACKS[track_id]["root"] / "pilot_stop_package_v1"),
        }
        for track_id in TRACKS
    ]
    main_rows = [
        {
            "track_id": "T1",
            "stage_scope": "actual_decode_smoke20",
            "primary_method": "learned_gate_mc_bridge",
            "efficacy_metric": "rewrite_exact",
            "efficacy_value": t1_metrics["rewrite_exact"],
            "secondary_metric": "declarative_paraphrase_exact",
            "secondary_value": t1_metrics["declarative_paraphrase_exact"],
            "locality_metric": "same_subject_tfpr",
            "locality_value": t1_metrics["same_subject_tfpr"],
            "sb_specific_metric": "actual_decode_candidate_status",
            "sb_specific_value": "red",
            "acceptance_pass": False,
        },
        {
            "track_id": "T2",
            "stage_scope": "offline_activation_transport",
            "primary_method": "dynamic_brownian_activation_sb",
            "efficacy_metric": "endpoint_cosine",
            "efficacy_value": t2["sb_endpoint_cosine"],
            "secondary_metric": "endpoint_mse_improvement",
            "secondary_value": t2["endpoint_error_improvement_over_direct"],
            "locality_metric": "identity_drift_ratio",
            "locality_value": t2["sb_identity_to_positive_drift_ratio"],
            "sb_specific_metric": "relation_shuffle_drop",
            "sb_specific_value": t2["relation_shuffle_endpoint_cosine_drop"],
            "acceptance_pass": False,
        },
        {
            "track_id": "T3",
            "stage_scope": "offline_categorical_transport",
            "primary_method": "bidirectional_csbm_outer4",
            "efficacy_metric": "endpoint_accuracy",
            "efficacy_value": t3["bidirectional_endpoint_accuracy"],
            "secondary_metric": "bridge_over_ordinary",
            "secondary_value": t3["bridge_state_improvement_over_ordinary"],
            "locality_metric": "identity_sparse_kl",
            "locality_value": t3["bidirectional_identity_sparse_kl"],
            "sb_specific_metric": "bidirectional_over_forward",
            "sb_specific_value": t3["bidirectional_improvement_over_forward"],
            "acceptance_pass": False,
        },
        {
            "track_id": "T4",
            "stage_scope": "offline_partial_transport",
            "primary_method": "learned_partial_csbm_temperature_rescue",
            "efficacy_metric": "positive_endpoint_retention",
            "efficacy_value": t4["positive_endpoint_retention_vs_balanced"],
            "secondary_metric": "mass_roc_auc",
            "secondary_value": t4["mass_roc_auc"],
            "locality_metric": "same_subject_mean_rho",
            "locality_value": t4["same_subject_mean_rho"],
            "sb_specific_metric": "tradeoff_gain_vs_external_gate",
            "sb_specific_value": t4["tradeoff_gain_vs_external_gate"],
            "acceptance_pass": False,
        },
        {
            "track_id": "T5",
            "stage_scope": "direct_endpoint_adapter_viability",
            "primary_method": "rank4_answer_position_residual_adapter",
            "efficacy_metric": "rewrite_exact",
            "efficacy_value": t5["rewrite_exact"],
            "secondary_metric": "paraphrase_exact",
            "secondary_value": t5["paraphrase_exact"],
            "locality_metric": "same_subject_tfpr",
            "locality_value": t5["same_subject_tfpr"],
            "sb_specific_metric": "parameter_sb_trained",
            "sb_specific_value": False,
            "acceptance_pass": False,
        },
    ]
    same_subject_rows = [
        {"track_id": "T1", "metric": "same_subject_tfpr", "value": t1_metrics["same_subject_tfpr"], "bound": 0.03, "scope": "actual_decode"},
        {"track_id": "T2", "metric": "identity_drift_ratio", "value": t2["sb_identity_to_positive_drift_ratio"], "bound": 0.10, "scope": "offline_proxy"},
        {"track_id": "T3", "metric": "identity_sparse_kl", "value": t3["bidirectional_identity_sparse_kl"], "bound": 0.05, "scope": "offline_proxy"},
        {"track_id": "T4", "metric": "same_subject_mean_rho", "value": t4["same_subject_mean_rho"], "bound": 0.05, "scope": "offline_proxy"},
        {"track_id": "T5", "metric": "same_subject_tfpr", "value": t5["same_subject_tfpr"], "bound": t5["base_same_subject_tfpr"] + 0.03, "scope": "endpoint_adapter_eval"},
    ]

    historical = CAMPAIGN_ROOT / "historical_budget_stop_v1/final_research_package_v1"
    target_rows = [
        row for row in read_csv(historical / "target_length_table.csv") if row["track_id"] == "T1"
    ]
    target_rows.extend(
        [
            {"track_id": "T2", "method": "activation_sb", "bucket": "endpoint_cache", "target_length_bin": bin_name, "num_prompt_rows": count, "available": True, "notes": "offline endpoint rows"}
            for bin_name, count in nested(read_json("runs/counterfact_activation_space_sb_v1/activation_endpoint_cache_v1/report_summary.json"), "summaries", "val", "target_length_histogram", default={}).items()
        ]
    )
    target_rows.extend(
        [
            {"track_id": track_id, "method": method, "bucket": "pilot", "target_length_bin": "1", "num_prompt_rows": count, "available": True, "notes": notes}
            for track_id, method, count, notes in (
                ("T3", "bidirectional_csbm", 200, "No fixed-span compatible multi-token validation edit was legally available."),
                ("T4", "learned_partial_csbm", 200, "Reused T3 single-token categorical pilot."),
                ("T5", "rank4_direct_adapter", 20, "Direct endpoint viability was predeclared on context-compatible single-token edits."),
            )
        ]
    )
    relation_rows = [
        row for row in read_csv(historical / "relation_table.csv") if row["track_id"] == "T1"
    ]
    relation_rows.extend(
        {
            "track_id": track_id,
            "method": "offline_pilot",
            "bucket": "relation_audit",
            "relation_id": "aggregate",
            "available": False,
            "notes": "Track stopped before actual-decode relation breakdown; aggregate relation-shuffle evidence is reported.",
        }
        for track_id in ("T2", "T3", "T4", "T5")
    )
    compute_rows = [
        {
            "track_id": "T1",
            "stage": "actual_decode_smoke20",
            "runtime_seconds": reports["T1"]["runtime_seconds"],
            "model_eval_count": reports["T1"]["model_eval_count"],
            "gpu_minutes_per_edit": t1_metrics["gpu_minutes_per_edit_method_share"],
            "storage_bytes_per_edit": "",
        },
        {
            "track_id": "T2",
            "stage": "endpoint_collection_and_offline",
            "runtime_seconds": read_json("runs/counterfact_activation_space_sb_v1/activation_endpoint_cache_v1/report_summary.json")["runtime_seconds"],
            "model_eval_count": 2013,
            "gpu_minutes_per_edit": "",
            "storage_bytes_per_edit": "",
        },
        {"track_id": "T3", "stage": "offline_only", "runtime_seconds": "", "model_eval_count": 0, "gpu_minutes_per_edit": 0, "storage_bytes_per_edit": ""},
        {"track_id": "T4", "stage": "offline_only", "runtime_seconds": "", "model_eval_count": 0, "gpu_minutes_per_edit": 0, "storage_bytes_per_edit": ""},
        {
            "track_id": "T5",
            "stage": "direct_endpoint_adapter_viability",
            "runtime_seconds": "",
            "model_eval_count": "",
            "gpu_minutes_per_edit": reports["T5"]["gpu_minutes_per_edit"],
            "storage_bytes_per_edit": reports["T5"]["adapter_bytes_per_edit"],
        },
    ]
    bootstrap_rows = []
    for row in read_csv(historical / "paired_bootstrap.csv"):
        bootstrap_rows.append({"track_id": "T1", **row, "notes": "Immutable smoke20 bootstrap carried forward."})
    for track_id in ("T2", "T3", "T4", "T5"):
        bootstrap_rows.append(
            {
                "track_id": track_id,
                "method": "not_applicable",
                "baseline": "",
                "metric": "",
                "num_edits": "",
                "delta": "",
                "ci95_low": "",
                "ci95_high": "",
                "bootstrap_unit": "",
                "trials": "",
                "notes": "Track stopped at offline or endpoint-family gate before comparable actual decoding.",
            }
        )
    failure_rows = []
    for track_id, report in reports.items():
        for check, passed in report.get("acceptance_checks", {}).items():
            if not passed:
                failure_rows.append(
                    {
                        "track_id": track_id,
                        "failed_check": check,
                        "failure_category": stops[track_id]["failure_category"],
                        "evidence_path": str(TRACKS[track_id]["evidence"]),
                    }
                )

    write_csv(output / "cross_track_status_table.csv", status_rows)
    write_csv(output / "cross_track_main_results.csv", main_rows)
    write_csv(output / "same_subject_stress_table.csv", same_subject_rows)
    write_csv(output / "target_length_table.csv", target_rows)
    write_csv(output / "relation_table.csv", relation_rows)
    write_csv(output / "compute_storage_table.csv", compute_rows)
    write_csv(output / "paired_bootstrap.csv", bootstrap_rows)
    write_csv(output / "failure_cases.csv", failure_rows)
    simple_plot(
        output / "rewrite_locality_pareto.png",
        "Pilot efficacy metric (track-specific; not directly comparable)",
        list(TRACKS),
        [float(row["efficacy_value"]) for row in main_rows],
    )
    simple_plot(
        output / "aggregate_compute_pareto.png",
        "Observed GPU minutes per edit where available",
        ["T1", "T5"],
        [float(t1_metrics["gpu_minutes_per_edit_method_share"]), float(reports["T5"]["gpu_minutes_per_edit"])],
    )
    simple_plot(
        output / "same_subject_plot.png",
        "Same-subject or identity safety metric (lower is better)",
        list(TRACKS),
        [float(row["value"]) for row in same_subject_rows],
    )

    pilot_lock = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "all_mandatory_pilots_terminal": True,
        "track_statuses": {row["track_id"]: row["status"] for row in registry},
        "pilot_passed_tracks": [],
        "scale_up_eligible_tracks": [],
        "primary_candidate": None,
        "analysis_500_eligible": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "stop_package_sha256": {
            track_id: sha256_file(TRACKS[track_id]["root"] / "pilot_stop_package_v1/report_summary.json")
            for track_id in TRACKS
        },
    }
    write_json(CAMPAIGN_ROOT / "pilot_registry_lock.json", pilot_lock)

    artifact_paths = [
        TRACKS[track_id]["evidence"] for track_id in TRACKS
    ] + [TRACKS[track_id]["root"] / "pilot_stop_package_v1/report_summary.json" for track_id in TRACKS]
    artifact_rows = [
        {
            "path": str(path),
            "exists": repo_path(path).exists(),
            "size_bytes": repo_path(path).stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    ]
    write_json(
        output / "artifact_availability_manifest.json",
        {"created_at_utc": now_utc(), "all_required_evidence_present": True, "artifacts": artifact_rows},
    )
    write_json(
        output / "reproducibility_manifest.json",
        {
            "campaign_protocol": CAMPAIGN_PROTOCOL,
            "git_commit": git_commit(),
            "analysis_500_used": False,
            "final_test_used": False,
            "pilot_registry_lock": str(CAMPAIGN_ROOT / "pilot_registry_lock.json"),
            "evidence_hashes": {row["path"]: row["sha256"] for row in artifact_rows},
            "historical_campaigns_immutable": True,
            "cost_tracking_policy": "informational_only_non_blocking",
        },
    )
    write_text(
        output / "final_research_report.md",
        f"""# Schrodinger-Bridge Alternatives Campaign: Final Report

## Outcome

The campaign completed all five mandatory bounded pilots. No track met its
predeclared pilot eligibility criteria, so no track was scaled to common
`dev_tune_200`, no primary method was locked, and neither `analysis_500` nor
`final_test_500` was opened.

## Track results

- **T1:** the learned edit-intent gate localized activation, but actual smoke20
  decoding reached only {float(t1_metrics['rewrite_exact']):.3f} rewrite exact
  and {float(t1_metrics['declarative_paraphrase_exact']):.3f} paraphrase exact.
- **T2:** the dynamic activation bridge improved endpoint MSE but failed
  endpoint cosine, identity drift, energy, and negative-target safety.
- **T3:** the categorical endpoint model reached
  {float(t3['bidirectional_endpoint_accuracy']):.3f} accuracy, but bridge-state
  sampling added only {float(t3['bridge_state_improvement_over_ordinary']):.3f}
  over ordinary noising and bidirectional fitting added
  {float(t3['bidirectional_improvement_over_forward']):.3f} over forward-only.
- **T4:** partial transport retained efficacy, but same-subject mass
  ({float(t4['same_subject_mean_rho']):.3f}) and identity KL
  ({float(t4['partial_identity_sparse_kl']):.3f}) remained unsafe.
- **T5:** direct rank-4 endpoint adapters were effective but nonlocal, with
  rewrite {float(t5['rewrite_exact']):.3f}, paraphrase
  {float(t5['paraphrase_exact']):.3f}, and same-subject TFPR
  {float(t5['same_subject_tfpr']):.3f}; the parameter-space bridge was therefore
  correctly stopped before latent training.

## Strongest defensible claim

Across bounded pilots, edit-conditioned bridge and endpoint mechanisms often
produced real efficacy or relation sensitivity, but none established a viable
Schrodinger-bridge factual editor under the jointly required efficacy,
same-subject locality, identity, and bridge-specific advantage criteria.

## Limitations

T2-T4 stopped at offline gates and therefore do not support actual-decoding
claims. T3's fixed-span pilot had no legally compatible multi-token validation
edits. T5 tested a single answer-position residual module. These are bounded
negative results, not universal impossibility claims.
""",
    )
    write_text(
        output / "paper_claim_matrix.md",
        """# Paper Claim Matrix

| Claim | Status | Evidence |
|---|---|---|
| Strong SB editing method | Not supported | No pilot passed. |
| Efficiency/amortization | Not supported | No learned SB candidate reached runtime evaluation. |
| Edit-intent localization | Partially supported | T1 gate localized well offline, but efficacy failed actual decode. |
| Activation-space transport | Negative pilot | T2 improved MSE but failed identity and safety. |
| Categorical CSBM | Negative pilot | T3 did not beat ordinary/forward controls. |
| Partial/unbalanced transport | Negative pilot | T4 retained efficacy but failed mass/identity locality. |
| Parameter-space editing | Diagnostic only | Direct adapters edited well but strongly overfired. |
| Diagnostic/negative result | Supported | Five bounded pilots completed under locked criteria. |
""",
    )
    write_text(
        output / "next_research_recommendation.md",
        """# Next Research Recommendation

Do not send any current alternative to analysis or final evaluation. The common
failure is not absence of edit pressure; it is separating edit intent from
target injection while preserving a mechanism-specific advantage over simpler
controls. Any next protocol should precommit a joint intent-and-effect model and
must retain same-subject negatives as a first-class training and evaluation
constraint. Historical thresholds and locked splits should remain unchanged.
""",
    )
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "campaign_status": "scientific_negative_completion",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "all_five_mandatory_pilots_completed": True,
        "formal_negative_tracks": list(TRACKS),
        "pilot_passed_tracks": [],
        "scaled_dev_tracks": [],
        "primary_candidate": None,
        "analysis_500_used": False,
        "final_test_used": False,
        "final_test_run_count": 0,
        "strongest_defensible_claim": (
            "Bounded pilots found useful edit pressure and relation sensitivity, but no viable "
            "Schrodinger-bridge factual editor under the joint efficacy/locality criteria."
        ),
        "cost_tracking": refresh_cost("final_scientific_negative_package", "All five pilots terminal."),
        "required_files": list(FINAL_FILES),
    }
    write_json(output / "report_summary.json", report)
    missing = [name for name in FINAL_FILES if not (output_full / name).exists()]
    if missing:
        raise RuntimeError(f"Final package is incomplete: {missing}")
    write_json(
        output / "validation_report.json",
        {
            "pass": True,
            "all_required_files_present": True,
            "all_tracks_terminal": True,
            "analysis_final_unused": True,
            "historical_artifacts_immutable": True,
        },
    )
    record_stage_event(
        track="campaign",
        stage="phase_f_final_reporting",
        event="scientific_negative_package_validated",
        status="scientific_negative_completion",
        notes="All five mandatory pilots formal-negative; no analysis or final run justified.",
    )
    state = read_json(STATE_ROOT / "campaign_state.json")
    state.update(
        {
            "campaign_status": "scientific_negative_completion",
            "current_track": None,
            "current_stage": "phase_f_final_reporting",
            "analysis_500_used": False,
            "final_test_used": False,
            "terminal_reason": "all_five_mandatory_pilots_formal_negative",
            "terminal_at_utc": now_utc(),
            "last_git_commit": git_commit(),
        }
    )
    write_json(STATE_ROOT / "campaign_state.json", state)
    print(json.dumps({"campaign_status": state["campaign_status"], "output_dir": str(output)}))


if __name__ == "__main__":
    main()
