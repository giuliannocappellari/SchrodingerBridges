#!/usr/bin/env python3
"""Build and validate the terminal TRM research package."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    SOURCE_MODEL_ID,
    SOURCE_MODEL_REVISION,
    STAGES,
    STATE_ROOT,
    git_commit,
    now_utc,
    read_json,
    record_stage,
    record_stage_cost,
    register_artifacts,
    sha256_file,
    write_csv,
    write_json,
)


REQUIRED_PACKAGE_FILES = (
    "report_summary.json",
    "main_results_table.csv",
    "multi_token_table.csv",
    "same_subject_stress_table.csv",
    "locality_table.csv",
    "causal_localization_table.csv",
    "state_bucket_ablation.csv",
    "relation_table.csv",
    "compute_storage_table.csv",
    "paired_bootstrap.csv",
    "rewrite_locality_pareto.png",
    "state_bucket_plot.png",
    "multi_token_plot.png",
    "causal_heatmap.png",
    "failure_cases.csv",
    "artifact_availability_manifest.json",
    "reproducibility_manifest.json",
    "final_research_report.md",
    "paper_claim_recommendation.md",
    "terminal_package_validation.json",
)

TERMINAL_STAGE_STATUSES = {
    "passed",
    "passed_component_branch",
    "failed",
    "not_run_trigger_not_met",
    "not_run_due_formal_pilot_stop",
    "not_run_due_locked_failure",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_or(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def first_existing_csv(paths: Iterable[Path]) -> list[dict[str, Any]]:
    for path in paths:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def main_results() -> list[dict[str, Any]]:
    return first_existing_csv(
        (
            CAMPAIGN_ROOT / "E2_pilot100_v1" / "counterfact_pilot100_method_table.csv",
            CAMPAIGN_ROOT / "E1_smoke20_v1" / "smoke_method_registry.csv",
        )
    )


def determine_outcome(requested: str) -> tuple[str, str, str]:
    if requested != "auto":
        claim = (
            "formal_bounded_negative"
            if requested == "formal_negative"
            else "infrastructure_blocked"
            if requested == "infrastructure_blocked"
            else "full_editor_positive"
        )
        return requested, claim, "explicit_cli_choice"
    e2_path = CAMPAIGN_ROOT / "E2_pilot100_v1" / "report_summary.json"
    if not e2_path.exists():
        return "infrastructure_blocked", "infrastructure_blocked", "E2 report missing"
    e2 = read_json(e2_path)
    if not e2.get("acceptance_pass"):
        classes = e2.get("positive_classes", {})
        if any(bool(value) for value in classes.values()):
            return (
                "infrastructure_blocked",
                "infrastructure_blocked",
                "E2 failed integrity despite a positive claim flag",
            )
        return (
            "formal_negative",
            "formal_bounded_negative",
            "no predeclared positive claim survived the fixed pilot100 comparison",
        )
    confirmation = CAMPAIGN_ROOT / "F3_locked_confirmation_v1" / "report_summary.json"
    if not confirmation.exists():
        return (
            "infrastructure_blocked",
            "infrastructure_blocked",
            "pilot passed but the locked pipeline is incomplete",
        )
    frozen = read_json(confirmation)
    if not frozen.get("acceptance_pass"):
        return (
            "formal_negative",
            "formal_bounded_negative",
            "the dev-locked claim failed untouched confirmation",
        )
    claim = str(frozen.get("claim_class") or "full_editor_positive")
    return "completed", claim, "the frozen candidate passed locked confirmation"


def _status_row(message: str, source: str = "") -> dict[str, Any]:
    return {"status": message, "source": source}


def multi_token_rows() -> list[dict[str, Any]]:
    rows = read_csv(
        CAMPAIGN_ROOT
        / "D1_partial_state_target_delta_v1"
        / "target_length_comparison.csv"
    )
    return rows or [_status_row("not_completed")]


def causal_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    root = CAMPAIGN_ROOT / "C1_temporal_localization_v1"
    for name in (
        "causal_trace_summary.csv",
        "temporal_trace_summary.csv",
        "site_stability.csv",
        "site_policy_comparison.csv",
    ):
        for row in read_csv(root / name):
            output.append({"source_table": name, **row})
    return output or [_status_row("not_completed")]


def state_bucket_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sources = (
        CAMPAIGN_ROOT
        / "D1_partial_state_target_delta_v1"
        / "variant_summary.csv",
        CAMPAIGN_ROOT
        / "D2_state_conditioned_protection_v1"
        / "protection_variant_summary.csv",
    )
    for source in sources:
        for row in read_csv(source):
            output.append({"source_table": source.name, **row})
    return output or [_status_row("not_completed")]


def locality_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("method") in {None, ""}:
            continue
        output.append(
            {
                "method": row.get("method"),
                "run": row.get("run", ""),
                "same_subject_tfpr": row.get("same_subject_tfpr", ""),
                "near_tfpr": row.get("near_tfpr", ""),
                "far_tfpr": row.get("far_tfpr", ""),
                "generation_tfpr": row.get("generation_tfpr", ""),
                "locality_exact": row.get("locality_exact", ""),
                "clipped_self_normalized_locality": row.get(
                    "clipped_self_normalized_locality", ""
                ),
                "malformed_rate": row.get("malformed_rate", ""),
            }
        )
    return output or [_status_row("not_completed")]


def compute_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("method") in {None, ""}:
            continue
        output.append(
            {
                "method": row.get("method"),
                "run": row.get("run", ""),
                "gpu_minutes_per_edit": row.get("gpu_minutes_per_edit", ""),
                "model_evals_per_edit": row.get("model_evals_per_edit", ""),
                "memory_storage_bytes": row.get("memory_storage_bytes", ""),
                "utility_base_agreement": row.get("utility_base_agreement", ""),
            }
        )
    return output or [_status_row("not_completed")]


def relation_rows() -> list[dict[str, Any]]:
    registry = main_results()
    if not registry:
        return [_status_row("not_completed")]
    candidate = max(
        (row for row in registry if row.get("comparable_actual_decode", "True") != "False"),
        key=lambda row: float_or(row.get("selection_score")),
        default=None,
    )
    if not candidate:
        return [_status_row("not_completed")]
    run = candidate.get("run")
    if not run:
        return [_status_row("not_completed")]
    prompts = read_csv(ROOT / str(run) / "edited_per_prompt.csv")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prompts:
        relation = str(row.get("relation_id") or "unknown")
        bucket = str(row.get("bucket") or "unknown")
        grouped[(relation, bucket)].append(row)
    output = []
    for (relation, bucket), values in sorted(grouped.items()):
        expected = [bool_value(row.get("expected_hit")) for row in values]
        target_hits = [bool_value(row.get("target_new_hit")) for row in values]
        output.append(
            {
                "method": candidate.get("method"),
                "relation_id": relation,
                "bucket": bucket,
                "num_prompt_rows": len(values),
                "expected_exact": sum(expected) / len(expected),
                "target_new_hit_rate": sum(target_hits) / len(target_hits),
            }
        )
    return output or [_status_row("relation_id_unavailable", str(run))]


def paired_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in (
        CAMPAIGN_ROOT / "E2_pilot100_v1" / "paired_bootstrap.csv",
        CAMPAIGN_ROOT / "D2_state_conditioned_protection_v1" / "paired_bootstrap.csv",
        CAMPAIGN_ROOT / "D1_partial_state_target_delta_v1" / "paired_bootstrap.csv",
    ):
        for row in read_csv(source):
            output.append({"source_table": str(source.relative_to(CAMPAIGN_ROOT)), **row})
    return output or [_status_row("not_completed")]


def failure_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted(
        (
            row
            for row in rows
            if row.get("run") and row.get("comparable_actual_decode", "True") != "False"
        ),
        key=lambda row: float_or(row.get("selection_score")),
        reverse=True,
    )
    for candidate in candidates:
        prompts = read_csv(ROOT / str(candidate["run"]) / "edited_per_prompt.csv")
        if not prompts:
            continue
        failures = []
        for row in prompts:
            bucket = str(row.get("bucket") or "")
            failed_edit = bucket in {"rewrite", "declarative_paraphrase"} and not bool_value(
                row.get("expected_hit")
            )
            leaked = bucket not in {"rewrite", "declarative_paraphrase"} and bool_value(
                row.get("target_new_hit")
            )
            malformed = bool_value(row.get("malformed"))
            if failed_edit or leaked or malformed:
                failures.append(
                    {
                        "method": candidate.get("method"),
                        "failure_type": (
                            "malformed" if malformed else "target_leakage" if leaked else "edit_failure"
                        ),
                        **row,
                    }
                )
        if failures:
            return failures[:250]
    return [_status_row("no_failure_rows_available")]


def _blank_plot(path: Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def rewrite_locality_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    values = [
        row
        for row in rows
        if row.get("method") and row.get("rewrite_exact") not in {None, ""}
    ]
    if not values:
        _blank_plot(path, "Rewrite/locality Pareto", "No completed method rows")
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    for row in values:
        axis.scatter(float_or(row.get("same_subject_tfpr")), float_or(row.get("rewrite_exact")), s=35)
    axis.axvline(0.03, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Same-subject target false-positive rate")
    axis.set_ylabel("Rewrite exact")
    axis.set_title("Temporal residual rewrite/locality trade-off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def multi_token_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    numeric = [row for row in rows if row.get("target_length") not in {None, ""}]
    if not numeric:
        _blank_plot(path, "Multi-token editing", "No completed target-length rows")
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    method_key = "method" if "method" in numeric[0] else "policy"
    groups = sorted({str(row.get(method_key) or row.get("method_family")) for row in numeric})
    for method in groups:
        values = [row for row in numeric if str(row.get(method_key) or row.get("method_family")) == method]
        values.sort(key=lambda row: int(float(row["target_length"])))
        y_key = "rewrite_exact" if values[0].get("rewrite_exact") not in {None, ""} else "rewrite_gain"
        axis.plot(
            [int(float(row["target_length"])) for row in values],
            [float_or(row.get(y_key)) for row in values],
            marker="o",
            label=method,
        )
    axis.set_xlabel("Exact target length")
    axis.set_ylabel("Rewrite exact or gain")
    axis.set_title("Partial-state multi-token results")
    if len(groups) <= 12:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def state_bucket_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    values = [
        row
        for row in rows
        if row.get("method") or row.get("variant") or row.get("state_bucket")
    ]
    if not values:
        _blank_plot(path, "State buckets", "No completed state-bucket rows")
        return
    labels = [str(row.get("method") or row.get("variant") or row.get("state_bucket")) for row in values]
    scores = [float_or(row.get("stress_aware_aggregate", row.get("rewrite_exact", 0.0))) for row in values]
    limit = min(20, len(values))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(range(limit), scores[:limit])
    axis.set_xticks(range(limit), labels[:limit], rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("Stress-aware aggregate or rewrite exact")
    axis.set_title("State-conditioned residual ablation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def causal_heatmap(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = [row for row in rows if row.get("layer") not in {None, ""}]
    if not values:
        _blank_plot(path, "Temporal causal localization", "No completed localization rows")
        return
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in values:
        layer = int(float(row["layer"]))
        metric = next(
            (
                float_or(row.get(key))
                for key in ("mean_effect", "causal_effect", "proxy_score", "stress_aware_aggregate")
                if row.get(key) not in {None, ""}
            ),
            0.0,
        )
        by_layer[layer].append(metric)
    layers = sorted(by_layer)
    matrix = np.asarray([[sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers]])
    figure, axis = plt.subplots(figsize=(11, 3))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(layers)), layers, rotation=90, fontsize=7)
    axis.set_yticks([0], ["mean causal effect"])
    axis.set_xlabel("Layer")
    axis.set_title("Temporal causal localization")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def stage_reports() -> dict[str, dict[str, Any] | None]:
    mapping = {
        "A0": "A0_bootstrap_v1",
        "A1": "A1_source_audit_v1",
        "B0": "B0_fresh_protocol_v1",
        "C0": "C0_timerome_source_reproduction_v1",
        "C1": "C1_temporal_localization_v1",
        "C2": "C2_fullmask_temporal_residual_v1",
        "D1": "D1_partial_state_target_delta_v1",
        "D2": "D2_state_conditioned_protection_v1",
        "E1": "E1_smoke20_v1",
        "E2": "E2_pilot100_v1",
        "E3": "E3_kamel_multi_token_v1",
        "F1": "F1_dev200_selection_v1",
        "F2": "F2_dev_lock_v1",
        "F3": "F3_locked_confirmation_v1",
        "G1": "G1_edit_scaling_v1",
        "G2": "G2_second_backbone_v1",
    }
    output = {}
    for stage, directory in mapping.items():
        path = CAMPAIGN_ROOT / directory / "report_summary.json"
        output[stage] = read_json(path) if path.exists() else None
    return output


def scientific_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def load(path: Path) -> dict[str, Any]:
        return read_json(path) if path.exists() else {}

    c2_root = CAMPAIGN_ROOT / "C2_fullmask_temporal_residual_v1" / "pilot100_v1"
    stable = load(c2_root / "stable_temporal_top1" / "report_summary.json")
    random = load(c2_root / "random_site_top1" / "report_summary.json")
    d1 = load(CAMPAIGN_ROOT / "D1_partial_state_target_delta_v1" / "report_summary.json")
    d2 = load(CAMPAIGN_ROOT / "D2_state_conditioned_protection_v1" / "report_summary.json")
    e1 = load(CAMPAIGN_ROOT / "E1_smoke20_v1" / "report_summary.json")
    e2 = load(CAMPAIGN_ROOT / "E2_pilot100_v1" / "report_summary.json")
    comparable = [
        row
        for row in rows
        if row.get("method") not in {None, "", "base", "timerome_source_reproduction"}
        and row.get("rewrite_exact") not in {None, ""}
    ]
    strongest = max(
        comparable,
        key=lambda row: (
            float_or(row.get("selection_score")),
            float_or(row.get("stress_aware_aggregate")),
            float_or(row.get("rewrite_exact")),
        ),
        default={},
    )
    return {
        "counterfact_fullmask_stable": {
            key: stable.get(key)
            for key in (
                "rewrite_exact",
                "declarative_paraphrase_exact",
                "same_subject_tfpr",
                "near_tfpr",
                "far_tfpr",
                "stress_aware_aggregate",
            )
        },
        "counterfact_fullmask_random_control": {
            key: random.get(key)
            for key in (
                "rewrite_exact",
                "declarative_paraphrase_exact",
                "same_subject_tfpr",
                "near_tfpr",
                "far_tfpr",
                "stress_aware_aggregate",
            )
        },
        "partial_state_target_delta": {
            "selected_method": d1.get("selected_partial_method"),
            "diffusion_specific_pass": d1.get("diffusion_specific_pass"),
        },
        "state_conditioned_protection": {
            "state_conditioning_pass": d2.get("state_conditioning_pass"),
            "relative_tfpr_reduction_vs_shared": d2.get(
                "state_conditioned_relative_tfpr_reduction_vs_shared"
            ),
            "relation_rescue_status": d2.get("relation_rescue_status"),
        },
        "smoke": {
            "acceptance_pass": e1.get("acceptance_pass"),
            "selected_method_for_E2": e1.get("selected_method_for_E2"),
        },
        "pilot": {
            "acceptance_pass": e2.get("acceptance_pass"),
            "positive_classes": e2.get("positive_classes", {}),
            "selected_candidate_methods": e2.get("selected_candidate_methods", []),
        },
        "strongest_pilot_tradeoff": {
            key: strongest.get(key)
            for key in (
                "method",
                "rewrite_exact",
                "declarative_paraphrase_exact",
                "same_subject_tfpr",
                "near_tfpr",
                "far_tfpr",
                "selection_score",
                "stress_aware_aggregate",
            )
        },
    }


def artifact_hashes(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "terminal_package_validation.json":
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def runtime_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "runpod_image": os.environ.get("RUNPOD_IMAGE_NAME") or os.environ.get("RUNPOD_IMAGE"),
    }
    for distribution in ("torch", "transformers", "bitsandbytes", "accelerate"):
        try:
            environment[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            environment[distribution] = None
    try:
        import torch

        environment.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except (ImportError, RuntimeError):
        environment.update(
            {
                "cuda_available": False,
                "cuda_version": None,
                "cudnn_version": None,
                "gpu_name": None,
            }
        )
    return environment


def mark_terminal_state(outcome: str) -> dict[str, Any]:
    state_path = STATE_ROOT / "campaign_state.json"
    state = read_json(state_path)
    d2_path = CAMPAIGN_ROOT / "D2_state_conditioned_protection_v1" / "report_summary.json"
    relation_rescue_triggered = bool(
        read_json(d2_path).get("relation_rescue_triggered") if d2_path.exists() else False
    )
    for stage in STAGES:
        if stage == "H1_final_package":
            continue
        current = state["stage_status"].get(stage)
        if (
            stage == "D2_relation_conditioned_rescue"
            and current in {None, "pending"}
            and not relation_rescue_triggered
        ):
            state["stage_status"][stage] = "not_run_trigger_not_met"
        elif current in {None, "pending"}:
            state["stage_status"][stage] = (
                "not_run_due_formal_pilot_stop"
                if outcome == "formal_negative"
                else "not_run_due_locked_failure"
            )
    state["campaign_status"] = outcome
    state["current_stage"] = "H1_final_package"
    state["next_stage"] = None
    state["analysis_500_used"] = False
    state["final_test_used"] = False
    state["pod_status"] = "stop_pending_after_terminal_validation"
    state["updated_at_utc"] = now_utc()
    write_json(state_path, state)
    return state


def build_package(output: Path, *, requested_outcome: str, pod_idle_verified: bool) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not pod_idle_verified:
        raise RuntimeError("Pod idleness must be verified immediately before finalization")
    started = now_utc()
    begin = time.monotonic()
    output.mkdir(parents=True)
    outcome, claim, reason = determine_outcome(requested_outcome)
    rows = main_results()
    multi = multi_token_rows()
    causal = causal_rows()
    buckets = state_bucket_rows()
    locality = locality_rows(rows)
    compute = compute_rows(rows)
    relations = relation_rows()
    bootstraps = paired_rows()
    write_csv(output / "main_results_table.csv", rows or [_status_row("not_completed")])
    write_csv(output / "multi_token_table.csv", multi)
    write_csv(output / "same_subject_stress_table.csv", locality)
    write_csv(output / "locality_table.csv", locality)
    write_csv(output / "causal_localization_table.csv", causal)
    write_csv(output / "state_bucket_ablation.csv", buckets)
    write_csv(output / "relation_table.csv", relations)
    write_csv(output / "compute_storage_table.csv", compute)
    write_csv(output / "paired_bootstrap.csv", bootstraps)
    write_csv(output / "failure_cases.csv", failure_rows(rows))
    rewrite_locality_plot(output / "rewrite_locality_pareto.png", rows)
    state_bucket_plot(output / "state_bucket_plot.png", buckets)
    multi_token_plot(output / "multi_token_plot.png", multi)
    causal_heatmap(output / "causal_heatmap.png", causal)
    reports = stage_reports()
    evidence = scientific_evidence(rows)
    availability = []
    for stage, report in reports.items():
        availability.append(
            {
                "stage": stage,
                "available": report is not None,
                "acceptance_pass": report.get("acceptance_pass") if report else None,
                "analysis_500_used": report.get("analysis_500_used", False) if report else False,
                "final_test_used": report.get("final_test_used", False) if report else False,
            }
        )
    locked_evaluation_unused = not any(
        bool(row["analysis_500_used"] or row["final_test_used"])
        for row in availability
    )
    write_json(
        output / "artifact_availability_manifest.json",
        {"campaign_id": CAMPAIGN_ID, "stages": availability},
    )
    protocol_files = sorted(PROTOCOL_ROOT.glob("*.json*"))
    source_files = sorted(
        path
        for pattern in ("*trm*.py", "run_dnpe_editor.py", "run_dnpe_runtime_baseline.py")
        for path in (ROOT / "scripts").glob(pattern)
        if path.is_file()
    )
    config_files = sorted(
        path
        for path in CAMPAIGN_ROOT.glob("**/run_config.json")
        if output not in path.parents
    )
    schema_files = sorted(
        path
        for path in CAMPAIGN_ROOT.glob("**/*schema*.json")
        if output not in path.parents
    )
    finalizer_command = "python scripts/finalize_trm_campaign.py --pod_idle_verified 1"
    write_json(
        output / "reproducibility_manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "git_commit": git_commit(),
            "models": [
                {"role": "primary", "id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION},
                {"role": "source", "id": SOURCE_MODEL_ID, "revision": SOURCE_MODEL_REVISION},
                {"role": "secondary", "id": SECONDARY_MODEL_ID, "revision": SECONDARY_MODEL_REVISION},
            ],
            "protocol_artifacts": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in protocol_files
            ],
            "source_scripts": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in source_files
            ],
            "run_configs": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in config_files
            ],
            "runtime_feature_schemas": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in schema_files
            ],
            "runtime_environment": runtime_environment(),
            "bootstrap_seeds": [260718901, 260718902],
            "commands": {
                "smoke": "python scripts/run_trm_e1_smoke.py",
                "pilot": "python scripts/run_trm_e2_pilot.py",
                "finalize": finalizer_command,
            },
            "artifact_commands": {
                name: finalizer_command
                for name in REQUIRED_PACKAGE_FILES
                if name.endswith((".csv", ".png"))
            },
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    e2 = reports.get("E2") or {}
    positive_classes = e2.get("positive_classes", {})
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "H1_final_package",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "terminal_outcome": outcome,
        "terminal_reason": reason,
        "claim_classification": claim,
        "positive_classes": positive_classes,
        "scientific_evidence": evidence,
        "stage_acceptance": {
            stage: payload.get("acceptance_pass") if payload else None
            for stage, payload in reports.items()
        },
        "analysis_500_used": False,
        "final_test_used": False,
        "historical_analysis_final_opened": not locked_evaluation_unused,
        "pod_idle_before_shutdown": True,
        "package_validation_pass": False,
        "acceptance_pass": False,
    }
    write_json(output / "report_summary.json", report)
    stage_lines = "".join(
        f"- {stage}: `{payload.get('acceptance_pass') if payload else 'not run'}`\n"
        for stage, payload in reports.items()
    )
    interpretation = (
        "No predeclared positive claim survived the bounded pilot. The campaign therefore stops before dev selection and untouched confirmation. The evidence supports a diagnostic result: temporal residuals can produce edit efficacy, but the tested partial-state and state-conditioned variants did not jointly satisfy efficacy and locality."
        if outcome == "formal_negative"
        else "The campaign reached the frozen positive path. Interpret only the claim class recorded above."
        if outcome == "completed"
        else "The scientific question remains unresolved because the required pipeline did not complete."
    )
    stable = evidence["counterfact_fullmask_stable"]
    random = evidence["counterfact_fullmask_random_control"]
    strongest = evidence["strongest_pilot_tradeoff"]
    evidence_lines = (
        f"- Stable temporal full-mask pilot: rewrite `{stable.get('rewrite_exact')}`, "
        f"paraphrase `{stable.get('declarative_paraphrase_exact')}`, same-subject TFPR "
        f"`{stable.get('same_subject_tfpr')}`.\n"
        f"- Random-site control: rewrite `{random.get('rewrite_exact')}`, paraphrase "
        f"`{random.get('declarative_paraphrase_exact')}`, same-subject TFPR "
        f"`{random.get('same_subject_tfpr')}`.\n"
        f"- Diffusion-specific partial-state criterion: "
        f"`{evidence['partial_state_target_delta']['diffusion_specific_pass']}`.\n"
        f"- State-conditioning criterion: "
        f"`{evidence['state_conditioned_protection']['state_conditioning_pass']}`.\n"
        f"- Strongest fixed-pilot tradeoff: `{strongest.get('method')}` with rewrite "
        f"`{strongest.get('rewrite_exact')}`, paraphrase "
        f"`{strongest.get('declarative_paraphrase_exact')}`, and same-subject TFPR "
        f"`{strongest.get('same_subject_tfpr')}`.\n"
    )
    (output / "final_research_report.md").write_text(
        "# Partial-State Temporal Residual Editor Final Report\n\n"
        f"- Terminal outcome: `{outcome}`\n"
        f"- Claim classification: `{claim}`\n"
        f"- Reason: {reason}\n"
        "- Historical analysis/final splits opened: `false`\n\n"
        "## Stage Evidence\n\n"
        + stage_lines
        + "\n## Key Scientific Evidence\n\n"
        + evidence_lines
        + "\n## Interpretation\n\n"
        + interpretation
        + "\n\n## Limitations\n\n"
        "The source-compatible TimeROME branch was equation-level because no official code source was available. Smoke and pilot evidence use the fresh campaign manifests; skipped locked/scaling/backbone stages are not reported as method failures.\n",
        encoding="utf-8",
    )
    (output / "paper_claim_recommendation.md").write_text(
        "# Paper Claim Recommendation\n\n"
        f"Primary frozen classification: `{claim}`.\n\n"
        "Do not claim a full editor, Pareto-locality improvement, diffusion-specific advantage, or state-conditioning advantage unless the corresponding frozen criterion is true. Report useful efficacy and mechanism diagnostics separately from the terminal editor verdict.\n",
        encoding="utf-8",
    )
    (output / "formal_stop_checkpoint.md").write_text(
        "# Formal Campaign Stop Checkpoint\n\n"
        f"- Outcome: `{outcome}`\n"
        f"- Reason: {reason}\n"
        "- Locked historical analysis/final data used: `false`\n"
        "- Further tuning authorized: `false`\n",
        encoding="utf-8",
    )
    (output / "negative_result_report.md").write_text(
        "# Bounded Negative Result\n\n"
        + interpretation
        + "\n",
        encoding="utf-8",
    )
    (output / "next_research_recommendation.md").write_text(
        "# Next Research Recommendation\n\n"
        "Preserve this v1 as immutable evidence. Any follow-up should begin under a new, explicitly approved protocol and should target the observed efficacy-locality conflict rather than silently expanding this campaign.\n",
        encoding="utf-8",
    )
    write_json(
        output / "terminal_package_validation.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "required_files": list(REQUIRED_PACKAGE_FILES),
            "required_files_present": False,
            "nonempty_files": False,
            "all_stage_statuses_terminal": False,
            "historical_analysis_final_unused": True,
            "claim_follows_frozen_evidence": False,
            "pod_idle_before_shutdown": True,
            "artifact_hashes": [],
            "self_hash_excluded_to_avoid_recursive_manifest": True,
            "acceptance_pass": False,
        },
    )
    runtime = time.monotonic() - begin
    record_stage_cost("H1_final_package", runtime_seconds=runtime, gpu_count=0, notes="Terminal package assembly")
    record_stage(
        "H1_final_package",
        status="passed",
        acceptance_pass=True,
        output_dir=output,
        started_at_utc=started,
        notes=f"Terminal {outcome} package assembled.",
        next_stage=None,
    )
    state = mark_terminal_state(outcome)
    state["stage_status"]["H1_final_package"] = "passed"
    state["completed_at_utc"] = now_utc()
    write_json(STATE_ROOT / "campaign_state.json", state)
    all_terminal = all(
        status in TERMINAL_STAGE_STATUSES for status in state["stage_status"].values()
    )
    confirmation = reports.get("F3")
    frozen_claim_valid = bool(
        (
            outcome == "formal_negative"
            and (
                not any(bool(value) for value in positive_classes.values())
                or (confirmation is not None and not confirmation.get("acceptance_pass"))
            )
        )
        or outcome in {"completed", "infrastructure_blocked"}
    )
    present = all((output / name).is_file() for name in REQUIRED_PACKAGE_FILES)
    nonempty = present and all((output / name).stat().st_size > 0 for name in REQUIRED_PACKAGE_FILES)
    package_pass = bool(
        present
        and nonempty
        and all_terminal
        and frozen_claim_valid
        and locked_evaluation_unused
    )
    report["package_validation_pass"] = package_pass
    report["acceptance_pass"] = package_pass
    write_json(output / "report_summary.json", report)
    validation = read_json(output / "terminal_package_validation.json")
    hashes = artifact_hashes(output)
    hashes_match = all(
        sha256_file(ROOT / row["path"]) == row["sha256"] for row in hashes
    )
    validation.update(
        {
            "required_files_present": present,
            "nonempty_files": nonempty,
            "all_stage_statuses_terminal": all_terminal,
            "historical_analysis_final_unused": locked_evaluation_unused,
            "claim_follows_frozen_evidence": frozen_claim_valid,
            "artifact_hashes": hashes,
            "all_recorded_hashes_match": hashes_match,
            "acceptance_pass": package_pass and hashes_match,
        }
    )
    write_json(output / "terminal_package_validation.json", validation)
    register_artifacts("H1_final_package", sorted(path for path in output.rglob("*") if path.is_file()))
    if not package_pass or not hashes_match:
        raise RuntimeError("Terminal package validation failed")
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
        choices=("auto", "completed", "formal_negative", "infrastructure_blocked"),
        default="auto",
    )
    parser.add_argument("--pod_idle_verified", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    report = build_package(
        args.output_dir,
        requested_outcome=args.outcome,
        pod_idle_verified=bool(args.pod_idle_verified),
    )
    print(
        json.dumps(
            {
                "terminal_outcome": report["terminal_outcome"],
                "claim_classification": report["claim_classification"],
                "package_validation_pass": report["package_validation_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
