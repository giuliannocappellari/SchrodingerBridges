#!/usr/bin/env python3
"""P6 editor-generality test under the frozen reveal controller."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    HISTORICAL_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    record_stage,
    write_csv,
    write_json,
)
from scripts.mask_pattern_publication_runtime import (
    PlannerSpec,
    build_full_cost_tables,
    build_prompt_items,
    decode_with_planner,
    planner_spec_from_label,
)
from scripts.mask_pattern_publication_stats import paired_bootstrap, paired_values
from scripts.mdm_memit_editor import MemitConfig, apply_memit_batch
from scripts.run_mdm_memit_stage import load_covariance, load_model
from scripts.run_publication_locked_confirmation import (
    RANDOM_SEEDS,
    _aggregate,
    _attach_base,
    _seed_rows,
)


CONDITIONS = {
    "ordinary_fully_masked": "ordinary",
    "paper_matched_partial_state": "partial",
}


def _prefix(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        raw_family = str(item["family"])
        item["raw_family"] = raw_family
        item["editor_condition"] = condition
        item["family"] = f"{condition}::{raw_family}"
        output.append(item)
    return output


def _read_p4_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _specs(lock: Mapping[str, Any], n: int) -> list[PlannerSpec]:
    finite = str(lock["finite_controller_label"])
    baseline = str(lock["best_non_sb_planner"])
    labels = ["default_confidence", "one_step_myopic", "deterministic_global", finite]
    if baseline not in labels and baseline != "uniform_random":
        labels.append(baseline)
    specs = [
        planner_spec_from_label(
            label,
            n=n,
            fixed_order=lock["fixed_orders"][str(n)],
            seed=260_717_901,
        )
        for label in labels
    ]
    if baseline == "uniform_random":
        specs.extend(
            PlannerSpec(f"uniform_random_seed{seed}", "uniform_random", seed=seed)
            for seed in RANDOM_SEEDS
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "editor_generality_v1"
    )
    parser.add_argument("--lengths", default="2,3,4,5,6")
    parser.add_argument("--limit_per_length", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    if args.output_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    wall_start = time.monotonic()
    lengths = tuple(sorted({int(value) for value in args.lengths.split(",")}))
    lock = read_json(CAMPAIGN_ROOT / "dev_method_lock.json")
    if not lock.get("validation_pass"):
        raise RuntimeError("P6 requires the frozen dev method lock")
    p4_dir = CAMPAIGN_ROOT / "llada_locked_confirmation_v1"
    p4_report = read_json(p4_dir / "report_summary.json")
    if p4_report.get("locked_outcomes_used_for_tuning"):
        raise RuntimeError("P4 provenance is not eligible for editor generality")
    p4_rows = _read_p4_rows(p4_dir / "per_prompt_results.jsonl.gz")
    finite = str(lock["finite_controller_label"])
    baseline = str(lock["best_non_sb_planner"])
    partial_keep = {
        "partial_memit_default",
        "one_step_myopic",
        "deterministic_global",
        finite,
        baseline,
        "uniform_random",
    }
    partial_rows = [
        dict(row)
        for row in p4_rows
        if str(row["family"]) in partial_keep
        and int(row["target_length"]) in lengths
    ]
    if args.limit_per_length:
        allowed: dict[int, set[str]] = {}
        for n in lengths:
            allowed[n] = set(
                sorted(
                    {
                        str(row["case_id"])
                        for row in partial_rows
                        if int(row["target_length"]) == n
                    }
                )[: args.limit_per_length]
            )
        partial_rows = [
            row
            for row in partial_rows
            if str(row["case_id"]) in allowed[int(row["target_length"])]
        ]
    all_rows = _prefix(partial_rows, "paper_matched_partial_state")

    model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, args.dtype)
    covariance_dir = HISTORICAL_ROOT / "covariance_cache_v1"
    compute_rows = []
    for n in lengths:
        manifest_rows = read_jsonl(PROTOCOL_ROOT / f"kamel_pub_locked_n{n}.jsonl")
        rows = manifest_rows[: args.limit_per_length] if args.limit_per_length else manifest_rows
        items = build_prompt_items(rows, include_stress=True)
        base_tables, base_account = build_full_cost_tables(model, tokenizer, items)
        base = decode_with_planner(
            model,
            tokenizer,
            items,
            base_tables,
            PlannerSpec("base_default_confidence", "default_confidence"),
        )
        base_seeded = _seed_rows(base)
        config = MemitConfig(
            layers=tuple(map(int, lock["layers"])),
            learning_rate=float(lock["target_value_config"]["learning_rate"]),
            target_optimization_steps=int(lock["target_value_config"]["steps"]),
            clamp_norm_factor=float(lock["target_value_config"]["clamp_norm_factor"]),
            kl_factor=float(lock["target_value_config"]["kl_factor"]),
            partial_mask_schedule="fully_masked",
            reveal_policy="random",
            seed=260_717_902,
        )
        rollback, _ = apply_memit_batch(
            model,
            tokenizer,
            rows,
            config,
            lambda layer: load_covariance(covariance_dir, layer),
            target_cache_dir=args.output_dir / "target_value_cache" / f"ordinary_n{n}",
        )
        try:
            tables, edited_account = build_full_cost_tables(model, tokenizer, items)
            decoded_rows = []
            for spec in _specs(lock, n):
                decoded_rows.extend(
                    _attach_base(
                        _seed_rows(decode_with_planner(model, tokenizer, items, tables, spec)),
                        base_seeded,
                    )
                )
            all_rows.extend(_prefix(decoded_rows, "ordinary_fully_masked"))
            compute_rows.append(
                {
                    "editor_condition": "ordinary_fully_masked",
                    "target_length": n,
                    "num_edits": len(rows),
                    "base_cost_table": json.dumps(base_account, sort_keys=True),
                    "edited_cost_table": json.dumps(edited_account, sort_keys=True),
                }
            )
        finally:
            rollback.rollback()
        if not rollback.checksum_matches(atol=0.0):
            raise RuntimeError(f"P6 ordinary editor rollback failed at N={n}")

    aggregate = _aggregate(all_rows)
    bootstrap_rows = []
    safety_rows = []
    positive_conditions = 0
    significant_conditions = 0
    for condition in CONDITIONS:
        left = f"{condition}::{finite}"
        right = f"{condition}::{baseline}"
        result = paired_bootstrap(
            paired_values(
                all_rows,
                left=left,
                right=right,
                bucket="rewrite",
                metric="full_target_exact",
                lengths={3, 4},
            ),
            resamples=10_000,
            seed=260_717_910 + len(bootstrap_rows),
        )
        bootstrap_rows.append(
            {"editor_condition": condition, "left": left, "right": right, **result}
        )
        positive_conditions += int(float(result["mean_delta"]) > 0)
        significant_conditions += int(float(result["ci95_low"]) > 0)
        # P4/P6 base outputs are editor-independent. Compute the absolute safety
        # rate and compare it with the P4 frozen base rate below.
        finite_values = [
            row
            for row in aggregate
            if row["family"] == left and row["bucket"] == "same_subject_stress"
        ]
        finite_stress = sum(float(row["full_target_exact"]) for row in finite_values) / len(finite_values)
        base_stress = float(p4_report["base_same_subject_tfpr"])
        stress_delta = finite_stress - base_stress
        malformed = max(
            float(row["malformed"])
            for row in aggregate
            if row["family"] == left
        )
        safety_rows.append(
            {
                "editor_condition": condition,
                "same_subject_tfpr": finite_stress,
                "base_same_subject_tfpr": base_stress,
                "same_subject_tfpr_delta": stress_delta,
                "malformed_rate": malformed,
                "safety_pass": stress_delta <= 0.03 and malformed <= 0.05,
            }
        )
    acceptance = (
        positive_conditions >= 2
        and significant_conditions >= 1
        and all(bool(row["safety_pass"]) for row in safety_rows)
    )
    decision = (
        "editor_general_effect"
        if acceptance
        else "editor_specific_effect"
        if positive_conditions == 1
        else "editor_generality_not_established"
    )
    write_csv(args.output_dir / "editor_condition_results.csv", aggregate)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    write_csv(args.output_dir / "safety_table.csv", safety_rows)
    write_csv(args.output_dir / "cost_table.csv", compute_rows)
    (args.output_dir / "editor_generality_decision.md").write_text(
        f"# P6 Editor Generality\n\nDecision: `{decision}`.\n\n"
        f"Positive conditions: {positive_conditions}; conditions with paired lower bound above zero: "
        f"{significant_conditions}.\n",
        encoding="utf-8",
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P6",
        "stage": "P6_editor_generality",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "conditions": list(CONDITIONS),
        "finite_controller": finite,
        "frozen_non_sb_baseline": baseline,
        "partial_condition_reused_from_p4": True,
        "positive_condition_count": positive_conditions,
        "significant_condition_count": significant_conditions,
        "decision": decision,
        "locked_outcomes_used_for_tuning": False,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "runtime_seconds": time.monotonic() - wall_start,
        "acceptance_pass": acceptance,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="P6_editor_generality",
        track="P6",
        status=decision,
        output_dir=args.output_dir,
        acceptance_pass=acceptance,
        started_at_utc=started,
        notes=f"decision={decision}; positive={positive_conditions}; significant={significant_conditions}",
        next_stage="P7_approximate_solver",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
