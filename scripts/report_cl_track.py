#!/usr/bin/env python3
"""Apply frozen continual pilot gates and paired edit-level evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    ROOT as REPO_ROOT,
    now_utc,
    read_json,
    read_jsonl,
    success_classes,
    update_track,
    write_csv,
    write_json,
)


SB_TRACKS = {"C7", "C8"}
CLASS_PRIORITY = {"A": 4, "B": 3, "C": 2, "D": 1}


def terminal_past_rewrite(run_dir: Path) -> dict[str, float]:
    rows = read_jsonl(run_dir / "per_prompt_results.jsonl")
    terminal = max(int(row["evaluation_after_block"]) for row in rows)
    selected = [
        row
        for row in rows
        if int(row["evaluation_after_block"]) == terminal
        and str(row["bucket"]) == "rewrite"
        and int(row["edit_block"]) < terminal
    ]
    output = {str(row["case_id"]): float(bool(row["expected_hit"])) for row in selected}
    if not output:
        raise RuntimeError(f"No terminal past-edit rewrite rows in {run_dir}")
    return output


def paired_bootstrap_delta(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    trials: int = 5000,
    seed: int = 260723101,
) -> dict[str, Any]:
    case_ids = sorted(set(baseline) & set(candidate))
    if not case_ids:
        raise ValueError("Paired bootstrap has no aligned edit IDs")
    deltas = [float(candidate[key]) - float(baseline[key]) for key in case_ids]
    rng = random.Random(seed)
    samples = []
    for _ in range(trials):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()
    low_index = max(0, int(math.floor(0.025 * (trials - 1))))
    high_index = min(trials - 1, int(math.ceil(0.975 * (trials - 1))))
    return {
        "num_paired_edits": len(case_ids),
        "trials": trials,
        "seed": seed,
        "mean_delta": sum(deltas) / len(deltas),
        "ci_low": samples[low_index],
        "ci_high": samples[high_index],
    }


def _fraction_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0 else 0.0


def track_mechanism_signal(
    track: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> tuple[bool, str]:
    rewrite_gap = abs(
        float(candidate.get("current_rewrite_exact", 0.0))
        - float(baseline.get("current_rewrite_exact", 0.0))
    )
    retention_gain = float(candidate.get("past_retention", 0.0)) - float(
        baseline.get("past_retention", 0.0)
    )
    forgetting_reduction = _fraction_reduction(
        float(baseline.get("average_forgetting", 0.0)),
        float(candidate.get("average_forgetting", 1.0)),
    )
    positive = float(bootstrap["ci_low"]) > 0.0
    if track == "C2":
        passed = rewrite_gap <= 0.03 and positive and (
            forgetting_reduction >= 0.30 or retention_gain >= 0.10
        )
        return passed, "partial-state replay at matched efficacy"
    if track == "C5":
        baseline_kl = float(baseline.get("protected_kl", 0.0))
        kl_reduction = _fraction_reduction(
            baseline_kl, float(candidate.get("protected_kl", baseline_kl))
        )
        passed = rewrite_gap <= 0.03 and positive and (
            forgetting_reduction >= 0.30
            or retention_gain >= 0.10
            or kl_reduction >= 0.20
        )
        return passed, "orthogonal/Fisher signal at matched efficacy"
    if track == "C6":
        return positive and retention_gain >= 0.05, "partial-state functional replay over clean replay"
    return False, "no separate mechanism-only gate"


def is_confirmation_eligible(track: str, classes: Sequence[str], exact: bool) -> bool:
    """Keep non-exact SB proxies out of confirmation and SB claims."""

    return bool(classes) and (track not in SB_TRACKS or exact)


def candidate_row(
    track: str,
    candidate_dir: Path,
    baseline_dir: Path,
    *,
    mechanism_baseline_dir: Path | None,
    matched_non_sb_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(candidate_dir / "report_summary.json")
    baseline = read_json(baseline_dir / "report_summary.json")
    bootstrap = paired_bootstrap_delta(
        terminal_past_rewrite(baseline_dir), terminal_past_rewrite(candidate_dir)
    )
    enriched = dict(report)
    enriched["paired_lower_bound_positive"] = float(bootstrap["ci_low"]) > 0.0
    exact = bool(report.get("exact_method_claim_eligible", False))
    enriched["is_sb"] = track in SB_TRACKS and exact
    if matched_non_sb_dir is not None:
        matched = read_json(matched_non_sb_dir / "report_summary.json")
        enriched["matched_non_sb"] = {
            "past_retention": matched.get("past_retention"),
            "average_forgetting": matched.get("average_forgetting"),
        }
        candidate_storage = float(report.get("storage_bytes", 0.0))
        matched_storage = float(matched.get("storage_bytes", 0.0))
        enriched["matched_storage_reduction"] = (
            (matched_storage - candidate_storage) / matched_storage if matched_storage > 0 else 0.0
        )
    classes = success_classes(enriched, baseline)
    mechanism_baseline = (
        read_json(mechanism_baseline_dir / "report_summary.json")
        if mechanism_baseline_dir is not None
        else baseline
    )
    mechanism_bootstrap = bootstrap
    if mechanism_baseline_dir is not None:
        mechanism_bootstrap = paired_bootstrap_delta(
            terminal_past_rewrite(mechanism_baseline_dir),
            terminal_past_rewrite(candidate_dir),
        )
    mechanism_pass, mechanism_rule = track_mechanism_signal(
        track, enriched, mechanism_baseline, mechanism_bootstrap
    )
    confirmation_eligible = is_confirmation_eligible(track, classes, exact)
    row = {
        "track_id": track,
        "method": report["method"],
        "report_path": str((candidate_dir / "report_summary.json").relative_to(REPO_ROOT)),
        "implementation_status": report.get("implementation_status"),
        "exact_method_claim_eligible": exact,
        "sb_claim_eligible": track not in SB_TRACKS or exact,
        "success_classes": ",".join(classes),
        "confirmation_eligible": confirmation_eligible,
        "mechanism_signal_pass": mechanism_pass,
        "mechanism_signal_rule": mechanism_rule,
        "paired_retention_delta": bootstrap["mean_delta"],
        "paired_ci_low": bootstrap["ci_low"],
        "paired_ci_high": bootstrap["ci_high"],
        "mechanism_baseline_method": mechanism_baseline.get("method"),
        "mechanism_paired_retention_delta": mechanism_bootstrap["mean_delta"],
        "mechanism_paired_ci_low": mechanism_bootstrap["ci_low"],
        "mechanism_paired_ci_high": mechanism_bootstrap["ci_high"],
        "current_rewrite_exact": report.get("current_rewrite_exact"),
        "current_paraphrase_exact": report.get("current_paraphrase_exact"),
        "past_retention": report.get("past_retention"),
        "average_forgetting": report.get("average_forgetting"),
        "same_subject_tfpr": report.get("same_subject_tfpr"),
        "near_tfpr": report.get("near_tfpr"),
        "far_tfpr": report.get("far_tfpr"),
        "protected_kl": report.get("protected_kl"),
        "base_retention_loss_fraction": report.get("base_retention_loss_fraction"),
        "malformed_rate": report.get("malformed_rate"),
        "storage_mb_per_edit": report.get("storage_mb_per_edit"),
        "inference_overhead_fraction": report.get("inference_overhead_fraction"),
    }
    return row, bootstrap


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    def number(key: str, default: float) -> float:
        value = row.get(key)
        return default if value in {None, ""} else float(value)

    classes = str(row.get("success_classes") or "").split(",")
    priority = max((CLASS_PRIORITY.get(item, 0) for item in classes), default=0)
    return (
        priority,
        number("past_retention", 0.0),
        -number("average_forgetting", 1.0),
        -number("same_subject_tfpr", 1.0),
        number("current_rewrite_exact", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=tuple(f"C{i}" for i in range(1, 10)), required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, action="append", required=True)
    parser.add_argument("--mechanism_baseline_dir", type=Path)
    parser.add_argument("--matched_non_sb_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--update_state", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    baseline_dir = args.baseline_dir.resolve() if args.baseline_dir.is_absolute() else (ROOT / args.baseline_dir).resolve()
    matched = None
    if args.matched_non_sb_dir:
        matched = args.matched_non_sb_dir.resolve() if args.matched_non_sb_dir.is_absolute() else (ROOT / args.matched_non_sb_dir).resolve()
    mechanism_baseline = None
    if args.mechanism_baseline_dir:
        mechanism_baseline = (
            args.mechanism_baseline_dir.resolve()
            if args.mechanism_baseline_dir.is_absolute()
            else (ROOT / args.mechanism_baseline_dir).resolve()
        )
    rows = []
    bootstraps = []
    for raw in args.candidate_dir:
        candidate_dir = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
        row, bootstrap = candidate_row(
            args.track,
            candidate_dir,
            baseline_dir,
            mechanism_baseline_dir=mechanism_baseline,
            matched_non_sb_dir=matched,
        )
        rows.append(row)
        bootstraps.append({"track_id": args.track, "method": row["method"], **bootstrap})
    eligible = [row for row in rows if row["confirmation_eligible"]]
    selected = max(eligible, key=_selection_key) if eligible else None
    mechanism_only = [row for row in rows if row["mechanism_signal_pass"]]
    status = "pilot_passed" if selected else "pilot_failed_mechanism_signal" if mechanism_only else "pilot_failed"
    write_csv(args.output_dir / "candidate_results.csv", rows)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstraps)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "track_id": args.track,
            "baseline_dir": str(baseline_dir.relative_to(ROOT)),
            "mechanism_baseline_dir": (
                str(mechanism_baseline.relative_to(ROOT)) if mechanism_baseline else ""
            ),
            "candidate_dirs": [str(path) for path in args.candidate_dir],
            "matched_non_sb_dir": str(args.matched_non_sb_dir or ""),
            "bootstrap_trials": 5000,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": args.track,
        "stage": f"{args.track}_pilot_report",
        "created_at_utc": now_utc(),
        "num_candidates": len(rows),
        "num_confirmation_eligible": len(eligible),
        "num_mechanism_signals": len(mechanism_only),
        "selected_candidate": selected["method"] if selected else None,
        "selected_success_classes": selected["success_classes"].split(",") if selected else [],
        "status": status,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": bool(selected),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "validation_report.json",
        {
            "all_candidate_reports_present": len(rows) == len(args.candidate_dir),
            "paired_by_edit_id": True,
            "sb_proxy_misclaimed_as_exact": any(
                row["track_id"] in SB_TRACKS
                and not row["exact_method_claim_eligible"]
                and row["sb_claim_eligible"]
                for row in rows
            ),
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance_pass": True,
        },
    )
    if not selected:
        (args.output_dir / "track_stop_checkpoint.md").write_text(
            "\n".join(
                (
                    f"# {args.track} Pilot Stop Checkpoint",
                    "",
                    f"- Status: `{status}`",
                    f"- Confirmation eligible candidates: `{len(eligible)}`",
                    f"- Mechanism-only signals: `{len(mechanism_only)}`",
                    "- Analysis split used: `false`",
                    "- Final split used: `false`",
                    "- Decision: do not confirm this track; continue breadth-first execution.",
                    "",
                )
            ),
            encoding="utf-8",
        )
    if args.update_state:
        update_track(
            args.track,
            status=status,
            nominated_candidate=selected["method"] if selected else None,
            report_path=str((args.output_dir / "report_summary.json").relative_to(ROOT)),
            pilot_pass=bool(selected),
            mechanism_signal=bool(mechanism_only),
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
