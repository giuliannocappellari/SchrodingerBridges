#!/usr/bin/env python3
"""Create the bounded T1 smoke or confirmation decision report."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    record_stage_event,
    refresh_budget,
    repo_path,
    write_csv,
    write_json,
)


T1_ROOT = Path("runs/counterfact_learned_gate_raw_bridge_v1")
LEARNED_TO_BASELINE = {
    "learned_gate_myopic": "myopic_score",
    "learned_gate_no_rollout": "no_rollout_bridge",
    "learned_gate_mc_bridge": "mc_bridge",
}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: Sequence[Mapping[str, Any]], runtime_seconds: float) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    original_edits = {
        str(row["edit_id"]).replace("__same_subject_stress", "") for row in rows
    }
    output: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        standard = [row for row in method_rows if "same_subject_stress" not in str(row.get("split_role"))]
        stress = [row for row in method_rows if "same_subject_stress" in str(row.get("split_role"))]
        by_bucket: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in standard:
            by_bucket[str(row["bucket"])].append(row)
        output.append(
            {
                "method": method,
                "rewrite_exact": mean([float(row["exact_rate"]) for row in by_bucket["rewrite"]]),
                "declarative_paraphrase_exact": mean(
                    [float(row["exact_rate"]) for row in by_bucket["declarative_paraphrases"]]
                ),
                "near_tfpr": mean(
                    [float(row["target_false_positive_rate"]) for row in by_bucket["near_locality"]]
                ),
                "far_tfpr": mean(
                    [float(row["target_false_positive_rate"]) for row in by_bucket["far_locality"]]
                ),
                "same_subject_tfpr": mean([float(row["exact_rate"]) for row in stress]),
                "malformed_rate": max(
                    [float(row["malformed_rate"]) for row in method_rows], default=0.0
                ),
                "mean_sparse_guidance_kl": mean(
                    [float(row["sparse_guidance_kl"]) for row in method_rows]
                ),
                "gate_activation_rewrite": mean(
                    [
                        float(row["gate_activation_rate"])
                        for row in by_bucket["rewrite"]
                        if row.get("gate_activation_rate") is not None
                    ]
                ),
                "gate_activation_paraphrase": mean(
                    [
                        float(row["gate_activation_rate"])
                        for row in by_bucket["declarative_paraphrases"]
                        if row.get("gate_activation_rate") is not None
                    ]
                ),
                "gate_activation_same_subject": mean(
                    [
                        float(row["gate_activation_rate"])
                        for row in stress
                        if row.get("gate_activation_rate") is not None
                    ]
                ),
                "gpu_minutes_per_edit_method_share": runtime_seconds
                / 60.0
                / max(1, len(original_edits))
                / max(1, len(methods)),
                "num_prompt_rows": len(method_rows),
            }
        )
    return output


def classify(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in rows}
    base = by_method["base"]
    for method, baseline_method in LEARNED_TO_BASELINE.items():
        if method not in by_method:
            continue
        row = by_method[method]
        baseline = by_method[baseline_method]
        row["rewrite_gain_vs_base"] = row["rewrite_exact"] - base["rewrite_exact"]
        row["paraphrase_gain_vs_base"] = (
            row["declarative_paraphrase_exact"] - base["declarative_paraphrase_exact"]
        )
        row["same_subject_budget"] = base["same_subject_tfpr"] + 0.03
        row["near_budget"] = base["near_tfpr"] + 0.03
        row["far_budget"] = base["far_tfpr"] + 0.03
        row["safety_reduction_vs_ungated"] = (
            1.0
            if baseline["same_subject_tfpr"] <= 0.0 and row["same_subject_tfpr"] <= 0.0
            else 1.0 - row["same_subject_tfpr"] / max(baseline["same_subject_tfpr"], 1e-12)
        )
        common_safe = (
            row["same_subject_tfpr"] <= row["same_subject_budget"]
            and row["near_tfpr"] <= row["near_budget"]
            and row["far_tfpr"] <= row["far_budget"]
            and row["malformed_rate"] <= 0.05
        )
        green = (
            row["rewrite_gain_vs_base"] >= 0.15
            and row["paraphrase_gain_vs_base"] >= 0.10
            and common_safe
            and row["safety_reduction_vs_ungated"] >= 0.50
        )
        yellow = (
            row["rewrite_gain_vs_base"] > 0.0
            and row["paraphrase_gain_vs_base"] > 0.0
            and row["same_subject_tfpr"] <= 0.10
            and row["malformed_rate"] <= 0.05
        )
        row["pilot_color"] = "green" if green else "yellow" if yellow else "red"
        row["common_hard_constraints_pass"] = common_safe
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode_dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke20", "confirmation30"), required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = args.output_dir or (T1_ROOT / f"{args.stage}_report_v1")
    output_dir = repo_path(output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    decode_dir = repo_path(args.decode_dir)
    summary = read_json(decode_dir / "summary.json")
    rows = read_jsonl(decode_dir / "per_case_results.jsonl")
    aggregate_rows = classify(
        aggregate(rows, float(summary["efficiency"]["wall_time_seconds"]))
    )
    write_csv(output_dir / "pilot_results.csv", aggregate_rows)
    learned_rows = [row for row in aggregate_rows if row["method"] in LEARNED_TO_BASELINE]
    candidates = [row for row in learned_rows if row.get("pilot_color") in {"green", "yellow"}]
    selected = max(
        candidates,
        key=lambda row: (
            row["pilot_color"] == "green",
            float(row["rewrite_exact"]) + float(row["declarative_paraphrase_exact"]),
            -float(row["same_subject_tfpr"]),
        ),
        default=None,
    )
    budget = refresh_budget(f"T1_{args.stage}_decode", "Actual LLaDA pilot decode and report.")
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_learned_gate_raw_bridge_v1",
        "stage": f"T1 {args.stage} actual decode decision",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "decode_dir": str(args.decode_dir),
        "num_result_rows": len(rows),
        "runtime_seconds": float(summary["efficiency"]["wall_time_seconds"]),
        "model_eval_count": int(summary["efficiency"]["model_eval_count"]),
        "selected_candidate": selected,
        "green_candidate_count": sum(row.get("pilot_color") == "green" for row in learned_rows),
        "yellow_candidate_count": sum(row.get("pilot_color") == "yellow" for row in learned_rows),
        "confirmation_required": bool(selected) and args.stage == "smoke20",
        "acceptance_pass": bool(selected),
        "budget_state": {
            "estimated_spend_usd": budget["estimated_spend_usd"],
            "remaining_budget_usd": budget["remaining_budget_usd"],
        },
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T1",
        stage=f"T1_{args.stage}",
        event="actual_decode_reported",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=(f"selected={selected['method']} color={selected['pilot_color']}" if selected else "no viable candidate"),
    )
    print(f"acceptance_pass={report['acceptance_pass']}")
    print(f"selected_candidate={selected['method'] if selected else None}")


if __name__ == "__main__":
    main()
