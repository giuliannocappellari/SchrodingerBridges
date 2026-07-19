#!/usr/bin/env python3
"""Evaluate the frozen N4 selective policy on fresh confirmation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import CAMPAIGN_ID, git_commit, now_utc, read_csv, read_json, sha256_file, update_track, write_csv, write_json
from scripts.run_nds_selective_conformal import accepted_metrics, feature_matrix, isotonic_predict, per_case_outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", type=Path, required=True)
    parser.add_argument("--confirmation_editor_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pilot = read_json(args.pilot_dir / "report_summary.json")
    if not pilot.get("pilot_pass"):
        raise RuntimeError("N4 confirmation requires a pilot-passed policy")
    risk = read_json(args.pilot_dir / "risk_model.json")
    editor = read_json(args.confirmation_editor_dir / "report_summary.json")
    if "confirmation" not in Path(editor["manifest"]).name.casefold():
        raise RuntimeError("N4 confirmation editor did not use the fresh confirmation manifest")
    feature_rows = read_csv(args.confirmation_editor_dir / "pre_edit_features.csv")
    case_ids, features = feature_matrix(feature_rows)
    mean = torch.tensor(risk["mean"])
    scale = torch.tensor(risk["scale"])
    weights = torch.tensor(risk["weights"])
    scores = torch.sigmoid(((features - mean) / scale) @ weights + float(risk["bias"])).tolist()
    if risk["selected_calibrator"] == "isotonic":
        scores = isotonic_predict(risk["isotonic_blocks"], scores)
    threshold = float(risk["threshold"])
    accepted = {case_id for case_id, score in zip(case_ids, scores) if float(score) <= threshold}
    outcomes = per_case_outcomes(read_csv(args.confirmation_editor_dir / "edited_per_prompt.csv"))
    metrics = accepted_metrics(outcomes, accepted)
    coverage = len(accepted) / len(case_ids)
    passed = (
        coverage >= 0.50
        and float(metrics["accepted_rewrite_exact"]) >= 0.85
        and float(metrics["accepted_paraphrase_exact"]) >= 0.45
        and float(metrics["accepted_same_subject_tfpr"]) <= 0.03
        and float(metrics["risk_upper_bound"]) <= 0.05
        and float(metrics["accepted_malformed_rate"]) <= 0.05
    )
    per_edit = [
        {
            "case_id": case_id,
            "predicted_risk": score,
            "accepted": case_id in accepted,
            **outcomes[case_id],
        }
        for case_id, score in zip(case_ids, scores)
    ]
    write_csv(args.output_dir / "selective_per_edit.csv", per_edit)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N4",
        "stage": "fresh_confirmation_200",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "risk_model_sha256": sha256_file(args.pilot_dir / "risk_model.json"),
        "risk_model_frozen_before_confirmation": True,
        "coverage": coverage,
        **metrics,
        "confirmation_pass": bool(passed),
        "success_class": "B" if passed else None,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": bool(passed),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", {"policy_retuned": False, "confirmation_prompt_used_for_training": False, "acceptance_pass": bool(passed)})
    if not passed:
        (args.output_dir / "track_stop_checkpoint.md").write_text("# N4 Confirmation Stop\n\nThe frozen selective policy failed fresh confirmation.\n", encoding="utf-8")
        (args.output_dir / "negative_result_report.md").write_text("# N4 Confirmation Failure\n\nNo threshold or feature was changed.\n", encoding="utf-8")
        write_csv(args.output_dir / "track_evidence_table.csv", per_edit)
        write_json(args.output_dir / "artifact_availability_manifest.json", {"artifacts": [path.name for path in args.output_dir.iterdir()]})
        (args.output_dir / "next_recommendation.md").write_text("# Next Recommendation\n\nExclude N4 from final selection.\n", encoding="utf-8")
    update_track("N4", status="confirmation_passed" if passed else "confirmation_failed", candidate_id=f"selective_{risk['selected_calibrator']}", confirmation_pass=bool(passed), success_class="B" if passed else None, output_dir=args.output_dir)
    print(json.dumps({"confirmation_pass": passed, "coverage": coverage, "risk_upper_bound": metrics["risk_upper_bound"]}))


if __name__ == "__main__":
    main()
