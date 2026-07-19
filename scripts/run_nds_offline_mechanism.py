#!/usr/bin/env python3
"""Run frozen calibration-only mechanism gates for N1, N2, or N3."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    sha256_file,
    update_track,
    write_csv,
    write_json,
)
from scripts.nds_editor import (
    RelationKeyStatistics,
    fixed_penalty_update,
    low_rank_fisher_update,
    primal_dual_update,
    residualize_runtime_keys,
)
from scripts.nds_methods import (
    fisher_discriminant_ratio,
    fisher_quadratic,
    mean_row_cosine,
    protected_response,
)


def _payload(root: Path, split: str, layer: int) -> dict[str, Any]:
    return torch.load(
        root / split / f"layer_{layer}_measurements.pt",
        map_location="cpu",
        weights_only=True,
    )


def _relation_stats(payload: Mapping[str, Any]) -> RelationKeyStatistics:
    global_mean = payload["relation_global_mean"].float()
    return RelationKeyStatistics(
        global_mean,
        {str(key): value.float() for key, value in payload["relation_means"].items()},
        global_mean.clone(),
    )


def nearest_relation_accuracy(
    train_keys: torch.Tensor,
    train_labels: Sequence[str],
    test_keys: torch.Tensor,
    test_labels: Sequence[str],
) -> dict[str, float | int]:
    labels = sorted(set(map(str, train_labels)) & set(map(str, test_labels)))
    indices = [index for index, label in enumerate(test_labels) if str(label) in labels]
    if not labels or not indices:
        return {"accuracy": 0.0, "chance": 1.0, "num_rows": 0, "num_classes": 0}
    centroids = torch.stack(
        [
            train_keys[
                [index for index, value in enumerate(train_labels) if str(value) == label]
            ].float().mean(dim=0)
            for label in labels
        ]
    )
    normalized_centroids = F.normalize(centroids, dim=1)
    test = F.normalize(test_keys[indices].float(), dim=1)
    predictions = [labels[index] for index in (test @ normalized_centroids.T).argmax(dim=1)]
    truth = [str(test_labels[index]) for index in indices]
    accuracy = sum(left == right for left, right in zip(predictions, truth)) / len(truth)
    counts = Counter(truth)
    return {
        "accuracy": accuracy,
        "chance": max(counts.values()) / len(truth),
        "num_rows": len(truth),
        "num_classes": len(labels),
    }


def n1_report(train: Mapping[str, Any], calibration: Mapping[str, Any], layer: int) -> dict[str, Any]:
    raw = calibration["edit_keys"].float()
    anchors = calibration["subject_anchor_keys"].float()
    relations = list(map(str, calibration["relation_ids"]))
    residual, transform = residualize_runtime_keys(
        raw,
        relations,
        _relation_stats(train),
        subject_anchor_keys=anchors,
        mode="full",
    )
    rescue, rescue_transform = residualize_runtime_keys(
        raw,
        relations,
        _relation_stats(train),
        subject_anchor_keys=anchors,
        mode="full",
        shrinkage=0.5,
    )
    raw_ratio = fisher_discriminant_ratio(raw, anchors)
    residual_ratio = fisher_discriminant_ratio(residual, anchors)
    raw_conflict = abs(mean_row_cosine(raw, anchors))
    residual_conflict = abs(mean_row_cosine(residual, anchors))
    classification = nearest_relation_accuracy(
        train["edit_keys"],
        train["relation_ids"],
        residual,
        relations,
    )
    checks = {
        "fisher_discriminant_improvement": (
            residual_ratio / max(raw_ratio, 1e-12) - 1.0
        )
        >= 0.25,
        "cosine_conflict_reduction": (
            1.0 - residual_conflict / max(raw_conflict, 1e-12)
        )
        >= 0.20,
        "relation_classification_above_chance": classification["accuracy"]
        > classification["chance"],
        "cross_fitted": True,
    }
    base_passes = sum(bool(checks[key]) for key in list(checks)[:3])
    selected = "relation_full"
    rescue_used = False
    if base_passes < 2:
        rescue_ratio = fisher_discriminant_ratio(rescue, anchors)
        rescue_conflict = abs(mean_row_cosine(rescue, anchors))
        rescue_classification = nearest_relation_accuracy(
            train["edit_keys"], train["relation_ids"], rescue, relations
        )
        rescue_checks = {
            "fisher_discriminant_improvement": rescue_ratio / max(raw_ratio, 1e-12) - 1.0
            >= 0.25,
            "cosine_conflict_reduction": 1.0 - rescue_conflict / max(raw_conflict, 1e-12)
            >= 0.20,
            "relation_classification_above_chance": rescue_classification["accuracy"]
            > rescue_classification["chance"],
        }
        if sum(rescue_checks.values()) >= 2:
            selected = "relation_full_shrinkage"
            rescue_used = True
            residual_ratio = rescue_ratio
            residual_conflict = rescue_conflict
            classification = rescue_classification
            checks.update(rescue_checks)
    mechanism_pass = sum(bool(checks[key]) for key in list(checks)[:3]) >= 2
    return {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N1",
        "stage": "offline_mechanism",
        "layer": layer,
        "raw_fisher_discriminant_ratio": raw_ratio,
        "residual_fisher_discriminant_ratio": residual_ratio,
        "fisher_discriminant_relative_improvement": residual_ratio / max(raw_ratio, 1e-12) - 1.0,
        "raw_gradient_cosine_conflict": raw_conflict,
        "residual_gradient_cosine_conflict": residual_conflict,
        "cosine_conflict_relative_reduction": 1.0 - residual_conflict / max(raw_conflict, 1e-12),
        "relation_classification": classification,
        "transform": rescue_transform if rescue_used else transform,
        "checks": checks,
        "mechanism_pass": mechanism_pass,
        "rescue_used": rescue_used,
        "selected_candidate": selected if mechanism_pass else None,
    }


def n2_report(train: Mapping[str, Any], calibration: Mapping[str, Any], layer: int) -> dict[str, Any]:
    edit_direction = (calibration["edit_keys"] - calibration["subject_anchor_keys"]).float().mean(dim=0)
    euclidean = edit_direction.unsqueeze(0)
    keys = calibration["edit_keys"].float()
    residuals = torch.ones(keys.shape[0], 1)
    fisher = train["fisher_diagonal"].float()
    euclidean_gain = float(edit_direction @ euclidean[0])
    euclidean_quadratic = fisher_quadratic(euclidean, fisher)
    candidates = []
    damping_values = (1e-3, 1e-4, 1e-2)
    ranks = (64, 32, 128)
    for damping in damping_values:
        for rank in ranks:
            effective = min(rank, int(train["fisher_basis"].shape[1]))
            natural, transform = low_rank_fisher_update(
                euclidean,
                train["fisher_basis"][:, :effective],
                train["fisher_eigenvalues"][:effective],
                damping,
                keys,
                residuals,
            )
            gain = float(edit_direction @ natural[0])
            quadratic = fisher_quadratic(natural, fisher)
            signal_sensitivity = abs(gain) / math.sqrt(max(quadratic, 1e-12))
            base_ratio = abs(euclidean_gain) / math.sqrt(max(euclidean_quadratic, 1e-12))
            candidates.append(
                {
                    "damping": damping,
                    "rank": effective,
                    "gain": gain,
                    "fisher_quadratic": quadratic,
                    "signal_protected_sensitivity_ratio": signal_sensitivity,
                    "ratio_improvement": signal_sensitivity / max(base_ratio, 1e-12) - 1.0,
                    "quadratic_reduction": 1.0 - quadratic / max(euclidean_quadratic, 1e-12),
                    "finite": bool(torch.isfinite(natural).all()),
                    "transform": transform,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            bool(row["finite"]),
            min(float(row["ratio_improvement"]), float(row["quadratic_reduction"])),
        ),
    )
    psd_finite = bool(torch.isfinite(fisher).all() and (fisher > 0).all())
    checks = {
        "signal_protected_sensitivity_ratio_improves_20pct": selected["ratio_improvement"] >= 0.20,
        "protected_fisher_quadratic_falls_20pct": selected["quadratic_reduction"] >= 0.20,
        "fisher_finite_psd": psd_finite,
    }
    mechanism_pass = all(checks.values())
    return {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N2",
        "stage": "offline_mechanism",
        "layer": layer,
        "euclidean_gain": euclidean_gain,
        "euclidean_fisher_quadratic": euclidean_quadratic,
        "selected": selected,
        "bounded_grid": candidates,
        "checks": checks,
        "mechanism_pass": mechanism_pass,
        "rescue_used": bool(selected["damping"] != 1e-3 or selected["rank"] != min(64, train["fisher_basis"].shape[1])),
        "selected_candidate": "fisher_lowrank" if mechanism_pass else None,
    }


def _row_satisfaction(update: torch.Tensor, keys: torch.Tensor, limit: float) -> float:
    responses = (keys.float() @ update.float().T).square().mean(dim=1)
    return float((responses <= float(limit)).float().mean())


def n3_report(train: Mapping[str, Any], calibration: Mapping[str, Any], layer: int) -> dict[str, Any]:
    update = (calibration["edit_keys"] - calibration["subject_anchor_keys"]).float().mean(dim=0, keepdim=True)
    families = {name: value.float() for name, value in calibration["protected_keys"].items()}
    baseline_responses = {name: protected_response(update, keys) for name, keys in families.items()}
    limits = {name: value * 0.8 for name, value in baseline_responses.items()}
    fixed, fixed_report = fixed_penalty_update(update, families, 0.05)
    grids = []
    for multiplier_step in (0.05, 0.01, 0.1):
        for penalty_growth in (1.5, 2.0):
            candidate, report = primal_dual_update(
                update,
                families,
                limits,
                multiplier_step=multiplier_step,
                penalty_growth=penalty_growth,
                iterations=30,
            )
            satisfaction = sum(
                _row_satisfaction(candidate, families[name], limits[name])
                for name in families
            ) / len(families)
            fixed_satisfaction = sum(
                _row_satisfaction(fixed, families[name], limits[name])
                for name in families
            ) / len(families)
            first_violation = float(report["trajectory"][0]["maximum_violation"])
            last_violation = float(report["trajectory"][-1]["maximum_violation"])
            grids.append(
                {
                    "multiplier_step": multiplier_step,
                    "penalty_growth": penalty_growth,
                    "satisfaction": satisfaction,
                    "fixed_satisfaction": fixed_satisfaction,
                    "satisfaction_advantage": satisfaction - fixed_satisfaction,
                    "first_maximum_violation": first_violation,
                    "last_maximum_violation": last_violation,
                    "violation_decreased": last_violation < first_violation,
                    "finite": bool(report["finite"]),
                    "all_constraints_satisfied": bool(report["all_constraints_satisfied"]),
                    "report": report,
                }
            )
    selected = max(
        grids,
        key=lambda row: (
            bool(row["finite"]),
            float(row["satisfaction"]),
            float(row["satisfaction_advantage"]),
        ),
    )
    checks = {
        "constraint_violation_decreased": bool(selected["violation_decreased"]),
        "no_multiplier_divergence_or_nan": bool(selected["finite"]),
        "calibration_constraint_satisfaction_80pct": selected["satisfaction"] >= 0.80,
        "advantage_over_fixed_penalty_15pp": selected["satisfaction_advantage"] >= 0.15,
    }
    mechanism_pass = all(checks.values())
    return {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N3",
        "stage": "offline_mechanism",
        "layer": layer,
        "baseline_responses": baseline_responses,
        "limits": limits,
        "fixed_penalty": fixed_report,
        "selected": selected,
        "bounded_grid": grids,
        "checks": checks,
        "mechanism_pass": mechanism_pass,
        "rescue_used": bool(
            selected["multiplier_step"] != 0.05 or selected["penalty_growth"] != 1.5
        ),
        "selected_candidate": "primal_dual" if mechanism_pass else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("N1", "N2", "N3"), required=True)
    parser.add_argument(
        "--measurement_dir", type=Path, default=CAMPAIGN_ROOT / "S1_shared_measurements_v1"
    )
    parser.add_argument(
        "--calibration_manifest",
        type=Path,
        default=CAMPAIGN_ROOT / "protocol_v1" / "cf_nds_calibration_200.jsonl",
    )
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or CAMPAIGN_ROOT / f"{args.track}_offline_mechanism_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if "calibration" not in args.calibration_manifest.name.casefold():
        raise PermissionError("offline mechanism gate requires fresh calibration only")
    rows = read_jsonl(args.calibration_manifest)
    train = _payload(args.measurement_dir, "statistics_train", args.layer)
    calibration = _payload(args.measurement_dir, "calibration", args.layer)
    if len(rows) != len(calibration["case_ids"]):
        raise RuntimeError("calibration manifest/cache count mismatch")
    if args.track == "N1":
        report = n1_report(train, calibration, args.layer)
    elif args.track == "N2":
        report = n2_report(train, calibration, args.layer)
    else:
        report = n3_report(train, calibration, args.layer)
    report.update(
        {
            "created_at_utc": now_utc(),
            "git_commit": git_commit(),
            "calibration_manifest": str(args.calibration_manifest),
            "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
            "num_calibration_edits": len(rows),
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": bool(report["mechanism_pass"]),
        }
    )
    write_json(output / "report_summary.json", report)
    write_json(
        output / "candidate_lock.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "track_id": args.track,
            "candidate": report["selected_candidate"],
            "mechanism_pass": report["mechanism_pass"],
            "layer": args.layer,
            "rescue_used": report["rescue_used"],
            "frozen_before_smoke": True,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    grid = report.get("bounded_grid") or []
    if grid:
        write_csv(output / "bounded_grid.csv", grid)
    update_track(
        args.track,
        status="running" if report["mechanism_pass"] else "pilot_failed",
        mechanism_pass=bool(report["mechanism_pass"]),
        candidate_id=report["selected_candidate"],
        output_dir=output,
        notes=(
            "Offline mechanism gate passed; frozen candidate may proceed to smoke."
            if report["mechanism_pass"]
            else "Offline mechanism gate failed after the bounded rescue."
        ),
        rescue_used=bool(report["rescue_used"]),
    )
    print(json.dumps({"track": args.track, "mechanism_pass": report["mechanism_pass"], "candidate": report["selected_candidate"]}))


if __name__ == "__main__":
    main()
