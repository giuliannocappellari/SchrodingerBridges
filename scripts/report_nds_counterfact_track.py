#!/usr/bin/env python3
"""Validate one N1-N3 CounterFact pilot against its frozen criteria."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    git_commit,
    now_utc,
    read_csv,
    read_json,
    sha256_file,
    update_track,
    write_csv,
    write_json,
)
from scripts.nds_methods import paired_bootstrap_delta


def case_metric(
    rows: Sequence[Mapping[str, Any]],
    buckets: set[str],
    field: str,
    *,
    invert: bool = False,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("bucket")) not in buckets:
            continue
        value = row.get(field)
        if value in {None, ""}:
            continue
        numeric = float(str(value).casefold() == "true") if str(value).casefold() in {"true", "false"} else float(value)
        grouped[str(row["case_id"])].append(1.0 - numeric if invert else numeric)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _relative_reduction(candidate: float, baseline: float) -> float:
    return 1.0 - float(candidate) / max(float(baseline), 1e-12)


def _stop_package(
    output: Path,
    track: str,
    report: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(output / "track_evidence_table.csv", evidence)
    write_json(
        output / "artifact_availability_manifest.json",
        {
            "track_id": track,
            "artifacts": [
                {"path": path.name, "exists": path.is_file()}
                for path in sorted(output.iterdir())
                if path.is_file()
            ],
        },
    )
    (output / "track_stop_checkpoint.md").write_text(
        f"# {track} Track Stop Checkpoint\n\n"
        f"Status: `pilot_failed`\n\n"
        f"The frozen candidate did not satisfy its predeclared pilot criteria. "
        f"No threshold was lowered and no confirmation split was opened.\n",
        encoding="utf-8",
    )
    (output / "negative_result_report.md").write_text(
        f"# {track} Bounded Negative Result\n\n"
        f"Mechanism pass: `{report['mechanism_pass']}`. Full success: "
        f"`{report['full_success']}`. Pareto/constraint success: "
        f"`{report['secondary_success']}`. This is a bounded pilot result, not a "
        f"universal impossibility claim.\n",
        encoding="utf-8",
    )
    (output / "next_recommendation.md").write_text(
        "# Next Recommendation\n\nContinue breadth-first to the next mandatory track.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("N1", "N2", "N3", "N6"), required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--smoke_dir", type=Path, required=True)
    parser.add_argument("--mechanism_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    baseline = read_json(args.baseline_dir / "report_summary.json")
    candidate = read_json(args.candidate_dir / "report_summary.json")
    smoke = read_json(args.smoke_dir / "report_summary.json")
    mechanism = read_json(args.mechanism_dir / "report_summary.json")
    if not mechanism.get("mechanism_pass"):
        raise RuntimeError("actual pilot report cannot follow a failed mechanism gate")
    baseline_rows = read_csv(args.baseline_dir / "edited_per_prompt.csv")
    candidate_rows = read_csv(args.candidate_dir / "edited_per_prompt.csv")
    same_boot = paired_bootstrap_delta(
        case_metric(candidate_rows, {"same_subject"}, "target_new_hit"),
        case_metric(baseline_rows, {"same_subject"}, "target_new_hit"),
        trials=2000,
        seed=260719401,
    )
    baseline_kl_rows = read_csv(args.baseline_dir / "protected_distribution_per_prompt.csv")
    candidate_kl_rows = read_csv(args.candidate_dir / "protected_distribution_per_prompt.csv")
    kl_boot = paired_bootstrap_delta(
        case_metric(candidate_kl_rows, {"same_subject", "near_locality", "far_locality", "generation", "attribute"}, "protected_kl"),
        case_metric(baseline_kl_rows, {"same_subject", "near_locality", "far_locality", "generation", "attribute"}, "protected_kl"),
        trials=2000,
        seed=260719402,
    )
    candidate_locality = case_metric(
        candidate_rows,
        {"same_subject", "near_locality", "far_locality"},
        "target_new_hit",
        invert=True,
    )
    baseline_locality = case_metric(
        baseline_rows,
        {"same_subject", "near_locality", "far_locality"},
        "target_new_hit",
        invert=True,
    )
    locality_improvement = (
        sum(candidate_locality.values()) / max(len(candidate_locality), 1)
        - sum(baseline_locality.values()) / max(len(baseline_locality), 1)
    )
    rewrite = float(candidate["rewrite_exact"])
    paraphrase = float(candidate["declarative_paraphrase_exact"])
    same = float(candidate["same_subject_tfpr"])
    near = float(candidate["near_tfpr"])
    far = float(candidate["far_tfpr"])
    malformed = float(candidate["malformed_rate"])
    full_success = (
        rewrite >= 0.85
        and paraphrase >= 0.45
        and same <= 0.03
        and near <= float(baseline["near_tfpr"]) + 0.03
        and far <= float(baseline["far_tfpr"]) + 0.03
        and malformed <= 0.05
    )
    rewrite_loss = float(baseline["rewrite_exact"]) - rewrite
    paraphrase_loss = float(baseline["declarative_paraphrase_exact"]) - paraphrase
    same_reduction = _relative_reduction(same, float(baseline["same_subject_tfpr"]))
    kl_reduction = _relative_reduction(
        float(candidate["protected_distributional_kl"]),
        float(baseline["protected_distributional_kl"]),
    )
    if args.track in {"N1", "N6"}:
        secondary_success = (
            rewrite_loss <= 0.02
            and paraphrase_loss <= 0.02
            and same_reduction >= 0.25
            and kl_reduction >= 0.20
            and float(same_boot["ci_high"]) < 0.0
        )
    elif args.track == "N2":
        secondary_success = (
            rewrite_loss <= 0.02
            and paraphrase_loss <= 0.02
            and same_reduction >= 0.20
            and kl_reduction >= 0.20
            and float(kl_boot["ci_high"]) < 0.0
        )
    else:
        secondary_success = (
            rewrite >= 0.80
            and paraphrase >= 0.40
            and locality_improvement >= 0.20
            and same_reduction >= 0.25
            and float(same_boot["ci_high"]) < 0.0
        )
    pilot_pass = bool(full_success or secondary_success)
    evidence = [
        {
            "metric": key,
            "baseline": baseline.get(key),
            "candidate": candidate.get(key),
        }
        for key in (
            "rewrite_exact",
            "declarative_paraphrase_exact",
            "same_subject_tfpr",
            "near_tfpr",
            "far_tfpr",
            "protected_distributional_kl",
            "malformed_rate",
            "gpu_minutes_per_edit",
        )
    ]
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": args.track,
        "stage": "pilot100",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "candidate_id": candidate["method"],
        "mechanism_pass": True,
        "smoke_integration_pass": bool(
            smoke.get("rollback_checksum_pass") and smoke.get("all_metrics_finite")
        ),
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "same_subject_tfpr": same,
        "near_tfpr": near,
        "far_tfpr": far,
        "malformed_rate": malformed,
        "protected_distributional_kl": candidate["protected_distributional_kl"],
        "rewrite_loss_vs_baseline": rewrite_loss,
        "paraphrase_loss_vs_baseline": paraphrase_loss,
        "same_subject_relative_reduction": same_reduction,
        "protected_kl_relative_reduction": kl_reduction,
        "heldout_locality_satisfaction_improvement": locality_improvement,
        "same_subject_paired_bootstrap": same_boot,
        "protected_kl_paired_bootstrap": kl_boot,
        "full_success": bool(full_success),
        "secondary_success": bool(secondary_success),
        "pilot_pass": pilot_pass,
        "success_class": (
            "A" if full_success else "C" if secondary_success else None
        ),
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": pilot_pass,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "run_config.json",
        {
            "baseline_dir": str(args.baseline_dir),
            "candidate_dir": str(args.candidate_dir),
            "smoke_dir": str(args.smoke_dir),
            "mechanism_dir": str(args.mechanism_dir),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    write_csv(
        args.output_dir / "paired_bootstrap.csv",
        [
            {"metric": "same_subject_tfpr", **same_boot},
            {"metric": "protected_kl", **kl_boot},
        ],
    )
    write_csv(args.output_dir / "pilot_metrics.csv", evidence)
    if pilot_pass:
        write_json(
            args.output_dir / "confirmation_candidate_lock.json",
            {
                "track_id": args.track,
                "candidate_id": candidate["method"],
                "candidate_run_config_sha256": sha256_file(
                    args.candidate_dir / "run_config.json"
                ),
                "pilot_manifest_sha256": candidate["manifest_sha256"],
                "frozen_before_confirmation": True,
                "analysis_500_used": False,
                "final_test_used": False,
            },
        )
    else:
        _stop_package(args.output_dir, args.track, report, evidence)
    update_track(
        args.track,
        status="pilot_passed" if pilot_pass else "pilot_failed",
        mechanism_pass=True,
        pilot_pass=pilot_pass,
        candidate_id=candidate["method"],
        success_class=report["success_class"],
        output_dir=args.output_dir,
        notes="Frozen pilot candidate validated against the predeclared track criteria.",
    )
    print(json.dumps({"track": args.track, "pilot_pass": pilot_pass, "success_class": report["success_class"]}))


if __name__ == "__main__":
    main()
