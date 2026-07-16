#!/usr/bin/env python3
"""Execute M3, the bounded path-KL/identity-regularized MEMIT track."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
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
    write_jsonl,
)
from scripts.run_partial_mask_memit_track import _augment_locality


M1_ROOT = CAMPAIGN_ROOT / "M1_mdm_memit_reproduction_v1"
M2_ROOT = CAMPAIGN_ROOT / "M2_partial_mask_memit_v1"
M3_ROOT = CAMPAIGN_ROOT / "M3_schrodinger_regularized_memit_v1"
PATH_GRID = (0.01, 0.05, 0.1, 0.25)
IDENTITY_GRID = (0.25, 0.5, 1.0)
WEIGHT_GRID = (0.0, 0.001, 0.01)


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _identity_bank(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        pair = (str(row["relation_id"]), str(row["rewrite_template"]))
        if pair not in by_source[source]:
            by_source[source].append(pair)
    return by_source


def _attach_identity_prompts(
    rows: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Sequence[tuple[str, str]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        candidates = [
            (relation, template)
            for relation, template in bank[str(row.get("source_dataset") or "unknown")]
            if relation != str(row["relation_id"]) and "{}" in template
        ]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{row['case_id']}::{item[0]}::{item[1]}".encode("utf-8")
            ).hexdigest()
        )
        row["identity_prompts"] = [
            template.format(row["subject"]) for _, template in candidates[:2]
        ]
        row["identity_prompt_provenance"] = "frozen_dev_rewrite_template_bank_different_relation"
        if len(row["identity_prompts"]) < 2:
            raise RuntimeError(f"Insufficient identity templates for {row['case_id']}")
        output.append(row)
    return output


def _run_or_reuse(
    *,
    output_dir: Path,
    manifest: Path,
    layers: Sequence[int],
    schedule: str,
    reveal_policy: str,
    lambda_path: float,
    lambda_identity: float,
    lambda_weight: float,
    include_locality: bool,
) -> dict[str, Any]:
    report_path = output_dir / "report_summary.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if output_dir.exists():
        raise RuntimeError(f"Partial M3 output requires audit: {output_dir}")
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
        str(CAMPAIGN_ROOT / "covariance_cache_v1"),
        "--layers",
        ",".join(map(str, layers)),
        "--partial_mask_schedule",
        schedule,
        "--reveal_policy",
        reveal_policy,
        "--lambda_path",
        str(lambda_path),
        "--lambda_identity",
        str(lambda_identity),
        "--lambda_weight",
        str(lambda_weight),
        "--include_locality",
        "1" if include_locality else "0",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_metrics(
    label: str,
    output_dir: Path,
    lambda_path: float,
    lambda_identity: float,
    lambda_weight: float,
) -> dict[str, Any]:
    report = json.loads((output_dir / "report_summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (output_dir / "target_value_diagnostics.json").read_text(encoding="utf-8")
    )
    updates = diagnostics["layer_updates"]
    histories = [item["history"] for item in diagnostics["target_optimization"]]
    path_values = [step.get("path_kl_loss", 0.0) for history in histories for step in history]
    identity_values = [step.get("identity_loss", 0.0) for history in histories for step in history]
    same_subject = report["edited_summary"].get("same_subject_stress", {})
    classic = report.get("base_agreement_summary", {}).get("classic_specificity", {})
    return {
        "label": label,
        "lambda_path": lambda_path,
        "lambda_identity": lambda_identity,
        "lambda_weight": lambda_weight,
        "num_edits": report["num_edits"],
        "rewrite_exact": report["rewrite_exact"],
        "paraphrase_exact": report["paraphrase_exact"],
        "same_subject_tfpr": same_subject.get("target_new_exact", 0.0),
        "classic_specificity_base_agreement": classic.get("exact_base_agreement", ""),
        "malformed_rate": report["malformed_rate"],
        "update_norm_sum": sum(float(row["update_norm"]) for row in updates),
        "mean_path_kl_training": sum(path_values) / len(path_values) if path_values else 0.0,
        "mean_identity_loss_training": sum(identity_values) / len(identity_values) if identity_values else 0.0,
        "gpu_minutes_per_edit": report["gpu_minutes_per_edit"],
        "output_dir": str(output_dir),
    }


def _candidate_score(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    rewrite = float(row["rewrite_exact"])
    para = float(row["paraphrase_exact"])
    harmonic = 0.0 if rewrite <= 0 or para <= 0 else 2 * rewrite * para / (rewrite + para)
    safety_gain = max(0.0, float(baseline["same_subject_tfpr"]) - float(row["same_subject_tfpr"]))
    update_gain = max(0.0, 1.0 - float(row["update_norm_sum"]) / max(float(baseline["update_norm_sum"]), 1e-8))
    return harmonic + 0.25 * safety_gain + 0.10 * update_gain


def _retains_dev_efficacy(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return (
        float(row["rewrite_exact"]) >= float(baseline["rewrite_exact"]) - 0.05
        and float(row["paraphrase_exact"]) >= float(baseline["paraphrase_exact"]) - 0.05
        and float(row["malformed_rate"]) <= 0.05
    )


def _nearest_lower_path_weight(value: float) -> float | None:
    lower = [candidate for candidate in PATH_GRID if candidate < value]
    return max(lower) if lower else None


def _make_manifests(output_dir: Path) -> tuple[Path, Path, Path]:
    cf_dev = read_jsonl(PROTOCOL_ROOT / "cf_sb_dev_200.jsonl")
    kamel_dev = _augment_locality(read_jsonl(PROTOCOL_ROOT / "kamel_dev_50_per_length.jsonl"))
    dev_bank = _identity_bank(cf_dev + kamel_dev)
    dev_rows = _attach_identity_prompts(cf_dev + kamel_dev, dev_bank)
    dev_path = output_dir / "m3_dev_combined_with_frozen_identity.jsonl"
    if not dev_path.exists():
        write_jsonl(dev_path, dev_rows)

    cf_analysis = read_jsonl(PROTOCOL_ROOT / "cf_sb_analysis_200.jsonl")
    cf_analysis_rows = _attach_identity_prompts(cf_analysis, dev_bank)
    cf_analysis_path = output_dir / "m3_cf_analysis_with_frozen_identity.jsonl"
    if not cf_analysis_path.exists():
        write_jsonl(cf_analysis_path, cf_analysis_rows)

    kamel_main = _augment_locality(read_jsonl(PROTOCOL_ROOT / "kamel_repro_200_per_length.jsonl"))
    kamel_main_rows = _attach_identity_prompts(kamel_main, dev_bank)
    kamel_main_path = output_dir / "m3_kamel_main_with_frozen_identity.jsonl"
    if not kamel_main_path.exists():
        write_jsonl(kamel_main_path, kamel_main_rows)
    write_json(
        output_dir / "identity_bank_manifest.json",
        {
            "source": "fresh campaign dev rewrite templates only",
            "templates_by_source": {key: list(value) for key, value in dev_bank.items()},
            "dev_sha256": sha256_file(dev_path),
            "cf_analysis_sha256": sha256_file(cf_analysis_path),
            "kamel_main_sha256": sha256_file(kamel_main_path),
            "analysis_templates_added_to_bank": False,
            "evaluation_identity_prompts_used_for_efficacy": False,
        },
    )
    return dev_path, cf_analysis_path, kamel_main_path


def _paired_bootstrap(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    dataset: str,
    candidate_label: str,
    trials: int = 2000,
) -> list[dict[str, Any]]:
    def load(path: Path, bucket: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["bucket"] == bucket:
                    grouped[row["case_id"]].append(
                        float(str(row["target_new_hit"]).casefold() == "true")
                    )
        return {key: sum(values) / len(values) for key, values in grouped.items()}

    rng = random.Random(260603924)
    output: list[dict[str, Any]] = []
    for bucket in ("rewrite", "paraphrase", "same_subject_stress"):
        left = load(baseline_dir / "edited_per_prompt.csv", bucket)
        right = load(candidate_dir / "edited_per_prompt.csv", bucket)
        cases = sorted(set(left) & set(right))
        if not cases:
            continue
        observed = sum(right[c] - left[c] for c in cases) / len(cases)
        draws = []
        for _ in range(trials):
            sample = [cases[rng.randrange(len(cases))] for _ in cases]
            draws.append(sum(right[c] - left[c] for c in sample) / len(sample))
        draws.sort()
        output.append(
            {
                "dataset": dataset,
                "candidate_label": candidate_label,
                "baseline_label": "partial_baseline",
                "bucket": bucket,
                "candidate_minus_baseline": observed,
                "ci95_low": draws[int(0.025 * (trials - 1))],
                "ci95_high": draws[int(0.975 * (trials - 1))],
                "num_edits": len(cases),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=M3_ROOT)
    args = parser.parse_args()
    started = now_utc()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    m1_selection = json.loads(
        (M1_ROOT / "layer_selection_v1/selected_layer_window.json").read_text(encoding="utf-8")
    )
    m2_report = json.loads((M2_ROOT / "report_summary.json").read_text(encoding="utf-8"))
    layers = list(map(int, m1_selection["layers"]))
    schedule = str(m2_report["selected_schedule"])
    reveal = str(m2_report["selected_reveal_policy"])
    dev_manifest, cf_analysis_manifest, kamel_main_manifest = _make_manifests(args.output_dir)

    runs: dict[str, tuple[Path, float, float, float]] = {}

    def ensure(label: str, lp: float, li: float, lw: float) -> dict[str, Any]:
        if label in runs:
            return _run_metrics(label, *runs[label])
        path = args.output_dir / f"dev_{label}_v1"
        _run_or_reuse(
            output_dir=path,
            manifest=dev_manifest,
            layers=layers,
            schedule=schedule,
            reveal_policy=reveal,
            lambda_path=lp,
            lambda_identity=li,
            lambda_weight=lw,
            include_locality=True,
        )
        runs[label] = (path, lp, li, lw)
        return _run_metrics(label, path, lp, li, lw)

    baseline = ensure("partial_baseline", 0.0, 0.0, 0.0)
    l2_baseline = ensure("l2_only_0p001", 0.0, 0.0, 0.001)
    identity_only = ensure("identity_only_0p5", 0.0, 0.5, 0.0)
    stage1 = [
        ensure(f"path_{_slug(lp)}_identity_0p5_weight_0p001", lp, 0.5, 0.001)
        for lp in PATH_GRID
    ]
    top_path = sorted(stage1, key=lambda row: -_candidate_score(row, baseline))[:2]
    stage2: list[dict[str, Any]] = []
    for parent in top_path:
        lp = float(parent["lambda_path"])
        for li in IDENTITY_GRID:
            stage2.append(
                ensure(
                    f"path_{_slug(lp)}_identity_{_slug(li)}_weight_0p001",
                    lp,
                    li,
                    0.001,
                )
            )
    unique_stage2 = {row["label"]: row for row in stage2}
    top_identity = sorted(
        unique_stage2.values(), key=lambda row: -_candidate_score(row, baseline)
    )[:2]
    stage3: list[dict[str, Any]] = []
    for parent in top_identity:
        lp = float(parent["lambda_path"])
        li = float(parent["lambda_identity"])
        for lw in WEIGHT_GRID:
            stage3.append(
                ensure(
                    f"path_{_slug(lp)}_identity_{_slug(li)}_weight_{_slug(lw)}",
                    lp,
                    li,
                    lw,
                )
            )
    unique_stage3 = {row["label"]: row for row in stage3}
    eligible = [row for row in unique_stage3.values() if _retains_dev_efficacy(row, baseline)]
    bounded_rescue_used = False
    rescue_record: dict[str, Any] | None = None
    if not eligible:
        anchor = sorted(
            unique_stage3.values(), key=lambda row: -_candidate_score(row, baseline)
        )[0]
        rescue_path = _nearest_lower_path_weight(float(anchor["lambda_path"]))
        if rescue_path is not None:
            bounded_rescue_used = True
            rescue_record = ensure(
                f"bounded_rescue_path_{_slug(rescue_path)}_identity_{_slug(float(anchor['lambda_identity']))}_weight_{_slug(float(anchor['lambda_weight']))}",
                rescue_path,
                float(anchor["lambda_identity"]),
                float(anchor["lambda_weight"]),
            )
            if _retains_dev_efficacy(rescue_record, baseline):
                eligible.append(rescue_record)
    candidates = sorted(eligible, key=lambda row: -_candidate_score(row, baseline))[:2]
    for candidate in candidates:
        ensure(
            f"path_only_{_slug(float(candidate['lambda_path']))}",
            float(candidate["lambda_path"]),
            0.0,
            0.0,
        )

    dev_rows = [_run_metrics(label, *spec) for label, spec in runs.items()]
    write_csv(args.output_dir / "dev_grid.csv", dev_rows)
    write_json(
        args.output_dir / "pareto_candidates.json",
        {
            "selection_split": "fresh cf_sb_dev_200 plus kamel_dev_50_per_length",
            "selected": candidates,
            "bounded_rescue_used": bounded_rescue_used,
            "bounded_rescue_candidate": rescue_record,
            "analysis_not_opened_during_selection": True,
        },
    )

    if not candidates:
        write_csv(
            args.output_dir / "analysis_results.csv",
            [],
            fieldnames=("dataset", "label", "rewrite_exact", "paraphrase_exact"),
        )
        write_csv(
            args.output_dir / "loss_ablation.csv",
            [baseline, l2_baseline, identity_only],
        )
        write_csv(
            args.output_dir / "identity_stress.csv",
            [],
            fieldnames=("dataset", "label", "same_subject_tfpr"),
        )
        write_csv(
            args.output_dir / "path_kl_table.csv",
            [
                {
                    "dataset": "dev",
                    "label": row["label"],
                    "mean_path_kl_training": row["mean_path_kl_training"],
                    "update_norm_sum": row["update_norm_sum"],
                }
                for row in dev_rows
            ],
        )
        write_csv(
            args.output_dir / "paired_bootstrap.csv",
            [],
            fieldnames=("dataset", "candidate_label", "baseline_label", "bucket"),
        )
        report = {
            "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
            "track": "M3",
            "stage": "M3_complete",
            "created_at_utc": now_utc(),
            "git_commit": git_commit(),
            "selected_layers": layers,
            "partial_mask_schedule": schedule,
            "reveal_policy": reveal,
            "dev_candidates": [],
            "analysis_candidate_checks": [],
            "sb_specific_positive_result": False,
            "acceptance_pass": False,
            "bounded_rescue_used": bounded_rescue_used,
            "bounded_rescue_candidate": rescue_record,
            "failure_reason": "efficacy_collapsed_on_dev_after_bounded_rescue",
            "old_analysis_500_used": False,
            "old_final_test_used": False,
            "fresh_campaign_analysis_used": False,
        }
        write_json(args.output_dir / "report_summary.json", report)
        (args.output_dir / "final_track_report.md").write_text(
            "# M3 Path-KL/Identity-Regularized MEMIT\n\n"
            "Status: **formal_negative**\n\n"
            "No candidate retained dev efficacy after the single predeclared nearest-lower "
            "path-KL rescue. Fresh campaign analysis was not opened.\n",
            encoding="utf-8",
        )
        record_stage(
            stage="M3_complete",
            track="M3",
            status="failed",
            output_dir=args.output_dir,
            acceptance_pass=False,
            started_at_utc=started,
            notes="dev efficacy collapsed after bounded rescue; analysis not opened",
        )
        print(json.dumps({"acceptance_pass": False, "selected_candidates": []}))
        return

    analysis_rows: list[dict[str, Any]] = []
    analysis_specs = [baseline, l2_baseline] + candidates
    for dataset_name, manifest in (
        ("cf_sb_analysis_200", cf_analysis_manifest),
        ("kamel_repro_200_per_length", kamel_main_manifest),
    ):
        for spec in analysis_specs:
            label = str(spec["label"])
            run_dir = args.output_dir / f"analysis_{dataset_name}_{label}_v1"
            _run_or_reuse(
                output_dir=run_dir,
                manifest=manifest,
                layers=layers,
                schedule=schedule,
                reveal_policy=reveal,
                lambda_path=float(spec["lambda_path"]),
                lambda_identity=float(spec["lambda_identity"]),
                lambda_weight=float(spec["lambda_weight"]),
                include_locality=True,
            )
            row = _run_metrics(label, run_dir, float(spec["lambda_path"]), float(spec["lambda_identity"]), float(spec["lambda_weight"]))
            row["dataset"] = dataset_name
            analysis_rows.append(row)
    write_csv(args.output_dir / "analysis_results.csv", analysis_rows)

    def lookup(dataset: str, label: str) -> Mapping[str, Any]:
        return next(row for row in analysis_rows if row["dataset"] == dataset and row["label"] == label)

    positive_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        label = str(candidate["label"])
        checks: list[dict[str, Any]] = []
        for dataset in ("cf_sb_analysis_200", "kamel_repro_200_per_length"):
            base = lookup(dataset, "partial_baseline")
            l2 = lookup(dataset, "l2_only_0p001")
            row = lookup(dataset, label)
            efficacy_ok = (
                float(row["rewrite_exact"]) >= float(base["rewrite_exact"]) - 0.05
                and float(row["paraphrase_exact"]) >= float(base["paraphrase_exact"]) - 0.05
            )
            tfpr_reduction = (
                (float(base["same_subject_tfpr"]) - float(row["same_subject_tfpr"]))
                / max(float(base["same_subject_tfpr"]), 1e-8)
            )
            cost_reduction = 1.0 - float(row["update_norm_sum"]) / max(float(base["update_norm_sum"]), 1e-8)
            beats_l2 = (
                float(row["same_subject_tfpr"]) < float(l2["same_subject_tfpr"])
                or float(row["update_norm_sum"]) < float(l2["update_norm_sum"])
            )
            checks.append(
                {
                    "dataset": dataset,
                    "efficacy_ok": efficacy_ok,
                    "same_subject_tfpr_relative_reduction": tfpr_reduction,
                    "intervention_cost_relative_reduction": cost_reduction,
                    "beats_l2": beats_l2,
                }
            )
        positive = all(check["efficacy_ok"] for check in checks) and any(
            check["same_subject_tfpr_relative_reduction"] >= 0.25
            or check["intervention_cost_relative_reduction"] >= 0.25
            for check in checks
        ) and any(check["beats_l2"] for check in checks)
        positive_candidates.append({"label": label, "positive": positive, "checks": checks})

    acceptance = any(row["positive"] for row in positive_candidates)
    ablation_labels = {
        "partial_baseline",
        "l2_only_0p001",
        "identity_only_0p5",
        *(f"path_only_{_slug(float(row['lambda_path']))}" for row in candidates),
        *(str(row["label"]) for row in candidates),
    }
    write_csv(
        args.output_dir / "loss_ablation.csv",
        [row for row in dev_rows if row["label"] in ablation_labels],
    )
    write_csv(
        args.output_dir / "identity_stress.csv",
        [
            {
                "dataset": row["dataset"],
                "label": row["label"],
                "same_subject_tfpr": row["same_subject_tfpr"],
                "classic_specificity_base_agreement": row["classic_specificity_base_agreement"],
            }
            for row in analysis_rows
        ],
    )
    write_csv(
        args.output_dir / "path_kl_table.csv",
        [
            {
                "dataset": row.get("dataset", "dev"),
                "label": row["label"],
                "mean_path_kl_training": row["mean_path_kl_training"],
                "update_norm_sum": row["update_norm_sum"],
            }
            for row in dev_rows + analysis_rows
        ],
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for dataset in ("cf_sb_analysis_200", "kamel_repro_200_per_length"):
        baseline_dir = Path(str(lookup(dataset, "partial_baseline")["output_dir"]))
        for candidate in candidates:
            row = lookup(dataset, str(candidate["label"]))
            bootstrap_rows.extend(
                _paired_bootstrap(
                    baseline_dir,
                    Path(str(row["output_dir"])),
                    dataset=dataset,
                    candidate_label=str(candidate["label"]),
                )
            )
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    report = {
        "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
        "track": "M3",
        "stage": "M3_complete",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "selected_layers": layers,
        "partial_mask_schedule": schedule,
        "reveal_policy": reveal,
        "dev_candidates": candidates,
        "analysis_candidate_checks": positive_candidates,
        "sb_specific_positive_result": acceptance,
        "acceptance_pass": acceptance,
        "bounded_rescue_used": bounded_rescue_used,
        "bounded_rescue_candidate": rescue_record,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "fresh_campaign_analysis_used": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    final = f"""# M3 Path-KL/Identity-Regularized MEMIT

Status: **{'passed' if acceptance else 'formal_negative'}**

The staged predeclared grid selected {len(candidates)} candidate(s) on fresh dev data. The frozen candidates were then evaluated once on the fresh campaign analysis and KAMEL main manifests.

An SB-specific positive result was {'established' if acceptance else 'not established'} under the efficacy-retention, 25% safety/path improvement, and ordinary-L2 comparison criteria.
"""
    (args.output_dir / "final_track_report.md").write_text(final, encoding="utf-8")
    record_stage(
        stage="M3_complete",
        track="M3",
        status="passed" if acceptance else "failed",
        output_dir=args.output_dir,
        acceptance_pass=acceptance,
        started_at_utc=started,
        notes=f"selected={len(candidates)}; sb_positive={acceptance}",
    )
    print(json.dumps({"acceptance_pass": acceptance, "selected_candidates": [row["label"] for row in candidates]}))


if __name__ == "__main__":
    main()
