#!/usr/bin/env python3
"""Fit and evaluate the N4 pre-edit selective conformal wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    git_commit,
    now_utc,
    read_csv,
    sha256_file,
    update_track,
    write_csv,
    write_json,
)
from scripts.nds_methods import (
    binomial_upper_bound,
    calibrate_selective_threshold,
    fit_logistic_risk,
    validate_runtime_feature_schema,
)


FEATURES = (
    "base_target_rank",
    "base_target_margin",
    "base_target_probability",
    "target_length",
)


def per_case_outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    output = {}
    for case_id, values in grouped.items():
        rewrite = [row for row in values if row["bucket"] == "rewrite"]
        paraphrase = [row for row in values if row["bucket"] == "declarative_paraphrase"]
        same = [row for row in values if row["bucket"] == "same_subject"]
        malformed = any(str(row.get("malformed", "")).casefold() == "true" for row in values)
        rewrite_exact = sum(str(row["expected_hit"]).casefold() == "true" for row in rewrite) / max(len(rewrite), 1)
        paraphrase_exact = sum(str(row["expected_hit"]).casefold() == "true" for row in paraphrase) / max(len(paraphrase), 1)
        same_tfpr = sum(str(row["target_new_hit"]).casefold() == "true" for row in same) / max(len(same), 1)
        unsafe = rewrite_exact < 1.0 or same_tfpr > 0.0 or malformed
        output[case_id] = {
            "rewrite_exact": rewrite_exact,
            "paraphrase_exact": paraphrase_exact,
            "same_subject_tfpr": same_tfpr,
            "malformed": malformed,
            "unsafe": unsafe,
        }
    return output


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], torch.Tensor]:
    index = {str(row["case_id"]): row for row in rows}
    case_ids = sorted(index)
    matrix = torch.tensor(
        [[float(index[case_id][name]) for name in FEATURES] for case_id in case_ids],
        dtype=torch.float32,
    )
    return case_ids, matrix


def isotonic_fit(scores: Sequence[float], labels: Sequence[bool]) -> list[dict[str, float]]:
    pairs = sorted(zip(map(float, scores), map(float, labels)))
    blocks = [
        {"low": score, "high": score, "sum": label, "count": 1.0}
        for score, label in pairs
    ]
    index = 0
    while index < len(blocks) - 1:
        left = blocks[index]["sum"] / blocks[index]["count"]
        right = blocks[index + 1]["sum"] / blocks[index + 1]["count"]
        if left <= right:
            index += 1
            continue
        merged = {
            "low": blocks[index]["low"],
            "high": blocks[index + 1]["high"],
            "sum": blocks[index]["sum"] + blocks[index + 1]["sum"],
            "count": blocks[index]["count"] + blocks[index + 1]["count"],
        }
        blocks[index : index + 2] = [merged]
        index = max(0, index - 1)
    return [
        {**block, "risk": block["sum"] / block["count"]} for block in blocks
    ]


def isotonic_predict(blocks: Sequence[Mapping[str, float]], scores: Sequence[float]) -> list[float]:
    output = []
    for score in scores:
        closest = min(
            blocks,
            key=lambda block: 0.0
            if float(block["low"]) <= float(score) <= float(block["high"])
            else min(abs(float(score) - float(block["low"])), abs(float(score) - float(block["high"]))),
        )
        output.append(float(closest["risk"]))
    return output


def accepted_metrics(
    outcomes: Mapping[str, Mapping[str, Any]], accepted: set[str]
) -> dict[str, float | int]:
    values = [outcomes[case_id] for case_id in sorted(accepted) if case_id in outcomes]
    if not values:
        return {
            "accepted": 0,
            "accepted_rewrite_exact": 0.0,
            "accepted_paraphrase_exact": 0.0,
            "accepted_same_subject_tfpr": 1.0,
            "accepted_malformed_rate": 1.0,
            "failures": 0,
            "risk_upper_bound": 1.0,
        }
    failures = sum(bool(row["unsafe"]) for row in values)
    return {
        "accepted": len(values),
        "accepted_rewrite_exact": sum(float(row["rewrite_exact"]) for row in values) / len(values),
        "accepted_paraphrase_exact": sum(float(row["paraphrase_exact"]) for row in values) / len(values),
        "accepted_same_subject_tfpr": sum(float(row["same_subject_tfpr"]) for row in values) / len(values),
        "accepted_malformed_rate": sum(bool(row["malformed"]) for row in values) / len(values),
        "failures": failures,
        "risk_upper_bound": binomial_upper_bound(failures, len(values), 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics_dir", type=Path, required=True)
    parser.add_argument("--calibration_dir", type=Path, required=True)
    parser.add_argument("--pilot_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "N4_selective_conformal_pilot_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    validate_runtime_feature_schema(FEATURES)
    directories = {
        "statistics_train": args.statistics_dir,
        "calibration": args.calibration_dir,
        "pilot": args.pilot_dir,
    }
    feature_rows = {
        split: read_csv(path / "pre_edit_features.csv") for split, path in directories.items()
    }
    outcomes = {
        split: per_case_outcomes(read_csv(path / "edited_per_prompt.csv"))
        for split, path in directories.items()
    }
    ids = {}
    matrices = {}
    for split in directories:
        ids[split], matrices[split] = feature_matrix(feature_rows[split])
        if set(ids[split]) != set(outcomes[split]):
            raise RuntimeError(f"feature/outcome case mismatch for {split}")
    train_labels = torch.tensor(
        [float(outcomes["statistics_train"][case_id]["unsafe"]) for case_id in ids["statistics_train"]]
    )
    model = fit_logistic_risk(
        matrices["statistics_train"],
        train_labels,
        feature_names=FEATURES,
        steps=500,
    )
    train_scores = model.predict(matrices["statistics_train"]).tolist()
    calibration_scores = model.predict(matrices["calibration"]).tolist()
    pilot_scores = model.predict(matrices["pilot"]).tolist()
    calibration_failures = [
        bool(outcomes["calibration"][case_id]["unsafe"])
        for case_id in ids["calibration"]
    ]
    logistic_threshold = calibrate_selective_threshold(
        calibration_scores,
        calibration_failures,
        maximum_upper_bound=0.05,
    )
    isotonic = isotonic_fit(train_scores, train_labels.bool().tolist())
    isotonic_calibration = isotonic_predict(isotonic, calibration_scores)
    isotonic_pilot = isotonic_predict(isotonic, pilot_scores)
    isotonic_threshold = calibrate_selective_threshold(
        isotonic_calibration,
        calibration_failures,
        maximum_upper_bound=0.05,
    )
    candidates = [
        {"calibrator": "logistic", **logistic_threshold},
        {"calibrator": "isotonic", **isotonic_threshold},
    ]
    selected = max(candidates, key=lambda row: (float(row["coverage"]), -float(row["risk_upper_bound"])))
    if selected["threshold"] is None:
        accepted: set[str] = set()
    else:
        scores = pilot_scores if selected["calibrator"] == "logistic" else isotonic_pilot
        accepted = {
            case_id
            for case_id, risk in zip(ids["pilot"], scores)
            if float(risk) <= float(selected["threshold"])
        }
    metrics = accepted_metrics(outcomes["pilot"], accepted)
    coverage = len(accepted) / len(ids["pilot"])
    success = (
        coverage >= 0.50
        and float(metrics["accepted_rewrite_exact"]) >= 0.85
        and float(metrics["accepted_paraphrase_exact"]) >= 0.45
        and float(metrics["accepted_same_subject_tfpr"]) <= 0.03
        and float(metrics["risk_upper_bound"]) <= 0.05
        and float(metrics["accepted_malformed_rate"]) <= 0.05
    )
    strong = success and coverage >= 0.60 and float(metrics["risk_upper_bound"]) <= 0.03
    rows = []
    scores = pilot_scores if selected["calibrator"] == "logistic" else isotonic_pilot
    for case_id, risk in zip(ids["pilot"], scores):
        rows.append(
            {
                "case_id": case_id,
                "predicted_risk": risk,
                "accepted": case_id in accepted,
                **outcomes["pilot"][case_id],
            }
        )
    write_csv(args.output_dir / "selective_per_edit.csv", rows)
    write_csv(args.output_dir / "calibration_candidates.csv", candidates)
    write_json(
        args.output_dir / "risk_model.json",
        {
            "feature_names": list(FEATURES),
            "mean": model.mean.tolist(),
            "scale": model.scale.tolist(),
            "weights": model.weights.tolist(),
            "bias": model.bias,
            "isotonic_blocks": isotonic,
            "selected_calibrator": selected["calibrator"],
            "threshold": selected["threshold"],
            "runtime_schema_deployable": True,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N4",
        "stage": "pilot100",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "underlying_method": read_csv(args.pilot_dir / "edited_per_prompt.csv")[0].get("method", "frozen_editor"),
        "runtime_feature_schema": list(FEATURES),
        "forbidden_runtime_features_used": False,
        "selected_calibrator": selected["calibrator"],
        "calibration_threshold": selected,
        "coverage": coverage,
        **metrics,
        "selective_safe_success": bool(success),
        "strong_selective_success": bool(strong),
        "pilot_pass": bool(success),
        "success_class": "B" if success else None,
        "rescue_used": selected["calibrator"] == "isotonic",
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": bool(success),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "run_config.json",
        {
            "statistics_dir": str(args.statistics_dir),
            "calibration_dir": str(args.calibration_dir),
            "pilot_dir": str(args.pilot_dir),
            "statistics_features_sha256": sha256_file(args.statistics_dir / "pre_edit_features.csv"),
            "calibration_features_sha256": sha256_file(args.calibration_dir / "pre_edit_features.csv"),
            "pilot_features_sha256": sha256_file(args.pilot_dir / "pre_edit_features.csv"),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    if success:
        write_json(
            args.output_dir / "confirmation_candidate_lock.json",
            {
                "track_id": "N4",
                "underlying_editor_dir": str(args.pilot_dir),
                "risk_model_sha256": sha256_file(args.output_dir / "risk_model.json"),
                "selected_calibrator": selected["calibrator"],
                "threshold": selected["threshold"],
                "frozen_before_confirmation": True,
            },
        )
    else:
        (args.output_dir / "track_stop_checkpoint.md").write_text(
            "# N4 Track Stop Checkpoint\n\nThe selective wrapper failed its frozen coverage-risk-efficacy criteria.\n",
            encoding="utf-8",
        )
        (args.output_dir / "negative_result_report.md").write_text(
            f"# N4 Bounded Negative Result\n\nCoverage was `{coverage:.4f}` and the exact 95% upper risk bound was `{float(metrics['risk_upper_bound']):.4f}`.\n",
            encoding="utf-8",
        )
        write_csv(args.output_dir / "track_evidence_table.csv", rows)
        write_json(args.output_dir / "artifact_availability_manifest.json", {"artifacts": [path.name for path in args.output_dir.iterdir()]})
        (args.output_dir / "next_recommendation.md").write_text("# Next Recommendation\n\nContinue to N5.\n", encoding="utf-8")
    update_track(
        "N4",
        status="pilot_passed" if success else "pilot_failed",
        mechanism_pass=bool(selected["threshold"] is not None),
        pilot_pass=bool(success),
        candidate_id=f"selective_{selected['calibrator']}",
        success_class="B" if success else None,
        output_dir=args.output_dir,
        notes="Pre-edit selective wrapper evaluated with exact one-sided risk control.",
        rescue_used=selected["calibrator"] == "isotonic",
    )
    print(json.dumps({"pilot_pass": success, "coverage": coverage, "risk_upper_bound": metrics["risk_upper_bound"]}))


if __name__ == "__main__":
    main()
