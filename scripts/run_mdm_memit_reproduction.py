#!/usr/bin/env python3
"""Run the frozen 500-edit MDM-MEMIT reproduction and robustness checks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mdm_memit_editor import MemitConfig, apply_memit_batch
from scripts.run_mdm_memit_stage import (
    add_base_agreement,
    aggregate,
    evaluate_rows,
    load_covariance,
    load_model,
)


M1_ROOT = CAMPAIGN_ROOT / "M1_mdm_memit_reproduction_v1"
DEFAULT_OUTPUT = M1_ROOT / "locked_reproduction_v1"


def _mean_by_case(
    rows: Sequence[Mapping[str, Any]], bucket: str, field: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["bucket"] == bucket:
            grouped[str(row["case_id"])].append(float(bool(row[field])))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _paired_bootstrap(
    base_rows: Sequence[Mapping[str, Any]],
    edited_rows: Sequence[Mapping[str, Any]],
    *,
    trials: int = 2000,
    seed: int = 260603924,
) -> list[dict[str, Any]]:
    specs = [
        ("rewrite", "target_new_hit"),
        ("paraphrase", "target_new_hit"),
        ("same_subject_stress", "target_new_hit"),
    ]
    output: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for bucket, field in specs:
        left = _mean_by_case(base_rows, bucket, field)
        right = _mean_by_case(edited_rows, bucket, field)
        case_ids = sorted(set(left) & set(right))
        if not case_ids:
            continue
        observed = sum(right[c] - left[c] for c in case_ids) / len(case_ids)
        samples: list[float] = []
        for _ in range(trials):
            chosen = [case_ids[rng.randrange(len(case_ids))] for _ in case_ids]
            samples.append(sum(right[c] - left[c] for c in chosen) / len(chosen))
        samples.sort()
        lo = samples[int(0.025 * (trials - 1))]
        hi = samples[int(0.975 * (trials - 1))]
        output.append(
            {
                "bucket": bucket,
                "metric": field,
                "edited_minus_base": observed,
                "ci95_low": lo,
                "ci95_high": hi,
                "num_edits": len(case_ids),
                "bootstrap_trials": trials,
            }
        )
    return output


def _summary_rows(
    method: str,
    summary: Mapping[str, Mapping[str, Any]],
    agreement: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket, values in summary.items():
        rows.append(
            {
                "method": method,
                "bucket": bucket,
                **dict(values),
                "exact_base_agreement": (
                    agreement.get(bucket, {}).get("exact_base_agreement", "")
                    if agreement
                    else ""
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROTOCOL_ROOT / "cf_repro_main_500.jsonl")
    parser.add_argument("--selection", type=Path, default=M1_ROOT / "layer_selection_v1/selected_layer_window.json")
    parser.add_argument("--covariance_dir", type=Path, default=CAMPAIGN_ROOT / "covariance_cache_v1")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = now_utc()
    overall_start = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = read_jsonl(args.manifest)
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 locked reproduction rows, got {len(rows)}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    layers = tuple(map(int, selection["layers"]))
    if len(layers) != 4:
        raise RuntimeError(f"Expected a four-layer frozen window, got {layers}")

    model, tokenizer = load_model(
        "GSAI-ML/LLaDA-8B-Instruct",
        "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "float16",
    )
    config = MemitConfig(layers=layers)
    inference_start = time.monotonic()
    base_results = evaluate_rows(model, tokenizer, rows, include_locality=True)
    base_robustness_results = {
        fixed_length: evaluate_rows(
            model,
            tokenizer,
            rows,
            include_locality=True,
            fixed_length=fixed_length,
            fixed_steps=fixed_length,
        )
        for fixed_length in (8, 16, 32)
    }
    base_inference_seconds = time.monotonic() - inference_start
    edit_start = time.monotonic()
    rollback, diagnostics = apply_memit_batch(
        model,
        tokenizer,
        rows,
        config,
        lambda layer: load_covariance(args.covariance_dir, layer),
        target_cache_dir=args.output_dir / "target_value_cache",
    )
    editing_seconds = time.monotonic() - edit_start
    inference_start = time.monotonic()
    edited_results = evaluate_rows(model, tokenizer, rows, include_locality=True)
    edited_inference_seconds = time.monotonic() - inference_start
    edited_results, agreement = add_base_agreement(base_results, edited_results)

    robustness_rows: list[dict[str, Any]] = []
    for fixed_length in (8, 16, 32):
        base_fixed = base_robustness_results[fixed_length]
        edited_fixed = evaluate_rows(
            model,
            tokenizer,
            rows,
            include_locality=True,
            fixed_length=fixed_length,
            fixed_steps=fixed_length,
        )
        edited_fixed, fixed_agreement = add_base_agreement(base_fixed, edited_fixed)
        summary = aggregate(edited_fixed)
        for bucket, values in summary.items():
            robustness_rows.append(
                {
                    "fixed_length": fixed_length,
                    "fixed_steps": fixed_length,
                    "bucket": bucket,
                    **dict(values),
                    "exact_base_agreement": fixed_agreement.get(bucket, {}).get(
                        "exact_base_agreement", ""
                    ),
                }
            )

    rollback.rollback()
    rollback_pass = rollback.checksum_matches(atol=0.0)
    if not rollback_pass:
        raise RuntimeError("Locked reproduction rollback checksum failed")

    base_summary = aggregate(base_results)
    edited_summary = aggregate(edited_results)
    efficacy = float(edited_summary["rewrite"]["target_new_exact"])
    generalization = float(edited_summary["paraphrase"]["target_new_exact"])
    pre_edit_efficacy = float(base_summary["rewrite"]["target_new_exact"])
    malformed = max(float(values["malformed_rate"]) for values in edited_summary.values())
    reproduction_pass = (
        efficacy >= 0.75
        and generalization >= 0.40
        and pre_edit_efficacy <= 0.10
        and malformed <= 0.05
        and rollback_pass
    )
    strong_reproduction = efficacy >= 0.85 and generalization >= 0.50
    robustness_rewrite = {
        int(row["fixed_length"]): float(row["target_new_exact"])
        for row in robustness_rows
        if row["bucket"] == "rewrite"
    }
    generation_robustness_pass = (
        set(robustness_rewrite) == {8, 16, 32}
        and all(value >= efficacy - 0.15 for value in robustness_rewrite.values())
    )

    write_csv(args.output_dir / "base_per_prompt.csv", base_results)
    write_csv(args.output_dir / "edited_per_prompt.csv", edited_results)
    result_rows = _summary_rows("base", base_summary) + _summary_rows(
        "mdm_memit", edited_summary, agreement
    )
    write_csv(args.output_dir / "counterfact_reproduction.csv", result_rows)
    write_csv(M1_ROOT / "counterfact_reproduction.csv", result_rows)
    write_csv(args.output_dir / "generation_robustness.csv", robustness_rows)
    write_csv(M1_ROOT / "generation_robustness.csv", robustness_rows)
    bootstrap = _paired_bootstrap(base_results, edited_results)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap)
    write_csv(M1_ROOT / "paired_bootstrap.csv", bootstrap)
    failures = [
        row
        for row in edited_results
        if row["bucket"] in {"rewrite", "paraphrase"} and not row["target_new_hit"]
    ]
    write_csv(args.output_dir / "failure_cases.csv", failures[:1000])
    write_csv(M1_ROOT / "failure_cases.csv", failures[:1000])
    write_json(args.output_dir / "target_value_diagnostics.json", diagnostics)
    write_json(
        args.output_dir / "implementation_config.json",
        {
            "model_id": "GSAI-ML/LLaDA-8B-Instruct",
            "model_revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
            "layers": list(layers),
            "memit": config.to_dict(),
            "selection_sha256": sha256_file(args.selection),
            "manifest_sha256": sha256_file(args.manifest),
            "quantized": False,
        },
    )
    covariance_manifest = {
        str(layer): {
            "path": str(args.covariance_dir / f"layer_{layer}_covariance.pt"),
            "sha256": sha256_file(args.covariance_dir / f"layer_{layer}_covariance.pt"),
        }
        for layer in layers
    }
    write_json(args.output_dir / "covariance_manifest.json", covariance_manifest)
    report = {
        "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
        "track": "M1",
        "stage": "M1_locked_reproduction_500",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "num_edits": len(rows),
        "selected_layers": list(layers),
        "pre_edit_target_new_efficacy": pre_edit_efficacy,
        "efficacy": efficacy,
        "generalization": generalization,
        "classic_specificity_base_agreement": agreement.get("classic_specificity", {}).get("exact_base_agreement"),
        "same_subject_tfpr": edited_summary.get("same_subject_stress", {}).get("target_new_exact"),
        "malformed_rate": malformed,
        "editing_seconds": editing_seconds,
        "base_inference_seconds": base_inference_seconds,
        "edited_inference_seconds": edited_inference_seconds,
        "gpu_minutes_per_edit": (time.monotonic() - overall_start) / 60.0 / len(rows),
        "rollback_checksum_pass": rollback_pass,
        "reproduction_pass": reproduction_pass,
        "strong_reproduction": strong_reproduction,
        "generation_robustness_pass": generation_robustness_pass,
        "generation_robustness_rewrite": robustness_rewrite,
        "acceptance_pass": reproduction_pass,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(M1_ROOT / "report_summary.json", report)
    final_report = f"""# M1 MDM-MEMIT Reproduction

Status: **{'passed_reproduction' if reproduction_pass else 'partial_reproduction'}**

- Frozen layers: {list(layers)}
- Efficacy: {efficacy:.4f}
- Generalization: {generalization:.4f}
- Pre-edit target-new efficacy: {pre_edit_efficacy:.4f}
- Same-subject TFPR: {report['same_subject_tfpr']}
- Malformed rate: {malformed:.4f}
- Strong reproduction: {strong_reproduction}

The run used fresh campaign manifests and did not inspect historical locked analysis or final splits.
"""
    (args.output_dir / "final_track_report.md").write_text(final_report, encoding="utf-8")
    (M1_ROOT / "final_track_report.md").write_text(final_report, encoding="utf-8")
    record_stage(
        stage="M1_locked_reproduction_500",
        track="M1",
        status="passed" if reproduction_pass else "failed",
        output_dir=args.output_dir,
        acceptance_pass=reproduction_pass,
        started_at_utc=started,
        notes=f"efficacy={efficacy:.4f}; generalization={generalization:.4f}",
    )
    print(json.dumps({"acceptance_pass": reproduction_pass, "efficacy": efficacy, "generalization": generalization}))


if __name__ == "__main__":
    main()
