#!/usr/bin/env python3
"""Confirm a frozen CounterFact N1/N2/N3/N6 candidate without retuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import (
    CAMPAIGN_ID,
    git_commit,
    now_utc,
    read_csv,
    read_json,
    update_track,
    write_csv,
    write_json,
)
from scripts.nds_methods import paired_bootstrap_delta
from scripts.report_nds_counterfact_track import case_metric


def relative_reduction(candidate: float, baseline: float) -> float:
    return 1.0 - float(candidate) / max(float(baseline), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("N1", "N2", "N3", "N6"), required=True)
    parser.add_argument("--pilot_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pilot = read_json(args.pilot_dir / "report_summary.json")
    baseline = read_json(args.baseline_dir / "report_summary.json")
    candidate = read_json(args.candidate_dir / "report_summary.json")
    if not pilot.get("pilot_pass"):
        raise RuntimeError("confirmation requires a pilot-passed frozen candidate")
    lock = read_json(args.pilot_dir / "confirmation_candidate_lock.json")
    if str(lock["candidate_id"]) != str(candidate["method"]):
        raise RuntimeError("confirmation method does not match the pilot lock")
    if "confirmation" not in Path(candidate["manifest"]).name.casefold():
        raise RuntimeError("confirmation report requires the fresh confirmation manifest")
    baseline_rows = read_csv(args.baseline_dir / "edited_per_prompt.csv")
    candidate_rows = read_csv(args.candidate_dir / "edited_per_prompt.csv")
    same_boot = paired_bootstrap_delta(
        case_metric(candidate_rows, {"same_subject"}, "target_new_hit"),
        case_metric(baseline_rows, {"same_subject"}, "target_new_hit"),
        trials=10000,
        seed=260719601,
    )
    baseline_kl = read_csv(args.baseline_dir / "protected_distribution_per_prompt.csv")
    candidate_kl = read_csv(args.candidate_dir / "protected_distribution_per_prompt.csv")
    kl_boot = paired_bootstrap_delta(
        case_metric(candidate_kl, {"same_subject", "near_locality", "far_locality", "generation", "attribute"}, "protected_kl"),
        case_metric(baseline_kl, {"same_subject", "near_locality", "far_locality", "generation", "attribute"}, "protected_kl"),
        trials=10000,
        seed=260719602,
    )
    rewrite = float(candidate["rewrite_exact"])
    paraphrase = float(candidate["declarative_paraphrase_exact"])
    same = float(candidate["same_subject_tfpr"])
    near = float(candidate["near_tfpr"])
    far = float(candidate["far_tfpr"])
    malformed = float(candidate["malformed_rate"])
    full = (
        rewrite >= 0.85
        and paraphrase >= 0.45
        and same <= 0.03
        and near <= float(baseline["near_tfpr"]) + 0.03
        and far <= float(baseline["far_tfpr"]) + 0.03
        and malformed <= 0.05
    )
    rewrite_loss = float(baseline["rewrite_exact"]) - rewrite
    paraphrase_loss = float(baseline["declarative_paraphrase_exact"]) - paraphrase
    same_reduction = relative_reduction(same, float(baseline["same_subject_tfpr"]))
    kl_reduction = relative_reduction(
        float(candidate["protected_distributional_kl"]),
        float(baseline["protected_distributional_kl"]),
    )
    if args.track == "N2":
        pareto = (
            rewrite_loss <= 0.02
            and paraphrase_loss <= 0.02
            and same_reduction >= 0.20
            and kl_reduction >= 0.20
            and float(kl_boot["ci_high"]) < 0.0
        )
    else:
        pareto = (
            rewrite_loss <= 0.02
            and paraphrase_loss <= 0.02
            and same_reduction >= 0.25
            and kl_reduction >= 0.20
            and float(same_boot["ci_high"]) < 0.0
        )
    expected_class = pilot.get("success_class")
    confirmation_pass = bool(full if expected_class == "A" else pareto)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": args.track,
        "stage": "fresh_confirmation_200",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "candidate_id": candidate["method"],
        "pilot_success_class": expected_class,
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "same_subject_tfpr": same,
        "near_tfpr": near,
        "far_tfpr": far,
        "malformed_rate": malformed,
        "same_subject_relative_reduction": same_reduction,
        "protected_kl_relative_reduction": kl_reduction,
        "same_subject_paired_bootstrap": same_boot,
        "protected_kl_paired_bootstrap": kl_boot,
        "full_success": bool(full),
        "pareto_success": bool(pareto),
        "confirmation_pass": confirmation_pass,
        "success_class": expected_class if confirmation_pass else None,
        "candidate_frozen_before_confirmation": True,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": confirmation_pass,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_csv(
        args.output_dir / "paired_bootstrap.csv",
        [
            {"metric": "same_subject_tfpr", **same_boot},
            {"metric": "protected_kl", **kl_boot},
        ],
    )
    write_json(
        args.output_dir / "validation_report.json",
        {
            "candidate_lock_match": True,
            "bootstrap_trials": 10000,
            "confirmation_used_for_tuning": False,
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": confirmation_pass,
        },
    )
    if not confirmation_pass:
        (args.output_dir / "track_stop_checkpoint.md").write_text(
            f"# {args.track} Confirmation Stop\n\nThe pilot direction did not persist on fresh confirmation.\n",
            encoding="utf-8",
        )
        (args.output_dir / "negative_result_report.md").write_text(
            f"# {args.track} Confirmation Failure\n\nThe frozen candidate was not retuned or replaced.\n",
            encoding="utf-8",
        )
        write_csv(args.output_dir / "track_evidence_table.csv", [{"metric": key, "value": report[key]} for key in ("rewrite_exact", "declarative_paraphrase_exact", "same_subject_tfpr", "near_tfpr", "far_tfpr")])
        write_json(args.output_dir / "artifact_availability_manifest.json", {"artifacts": [path.name for path in args.output_dir.iterdir()]})
        (args.output_dir / "next_recommendation.md").write_text("# Next Recommendation\n\nExclude this candidate from final direction selection.\n", encoding="utf-8")
    update_track(
        args.track,
        status="confirmation_passed" if confirmation_pass else "confirmation_failed",
        candidate_id=candidate["method"],
        confirmation_pass=confirmation_pass,
        success_class=report["success_class"],
        output_dir=args.output_dir,
        notes="Frozen candidate evaluated once on fresh confirmation without tuning.",
    )
    print(json.dumps({"track": args.track, "confirmation_pass": confirmation_pass, "success_class": report["success_class"]}))


if __name__ == "__main__":
    main()
