#!/usr/bin/env python3
"""P7 bounded KL-beam approximation and exact target-length scaling audit."""

from __future__ import annotations

import argparse
import gzip
import json
import math
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
    build_prompt_items,
    decode_with_planner,
    planner_spec_from_label,
)
from scripts.mdm_memit_editor import MemitConfig, apply_memit_batch
from scripts.run_mdm_memit_stage import load_covariance, load_model
from scripts.run_publication_locked_confirmation import (
    RANDOM_SEEDS,
    _aggregate,
    _attach_base,
    _seed_rows,
)


def _read_tables(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row.pop("item_key"))] = row
    return output


def _read_prompt_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _baseline_specs(lock: Mapping[str, Any], n: int) -> list[PlannerSpec]:
    baseline = str(lock["best_non_sb_planner"])
    if baseline == "uniform_random":
        return [
            PlannerSpec(f"uniform_random_seed{seed}", "uniform_random", seed=seed)
            for seed in RANDOM_SEEDS
        ]
    return [
        planner_spec_from_label(
            baseline,
            n=n,
            fixed_order=lock["fixed_orders"][str(n)],
            seed=260_717_950,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "approximate_solver_v1"
    )
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
    p3_dir = CAMPAIGN_ROOT / "planner_baselines_dev_v1"
    lock = read_json(CAMPAIGN_ROOT / "dev_method_lock.json")
    p3_report = read_json(p3_dir / "report_summary.json")
    if not lock.get("validation_pass") or p3_report.get("locked_confirmation_opened"):
        raise RuntimeError("P7 requires the frozen, development-only P3 artifacts")
    finite = str(lock["finite_controller_label"])
    baseline = str(lock["best_non_sb_planner"])
    reference = str(lock["reference_process"])
    beta = float(lock["beta"])

    planner_rows = list(__import__("csv").DictReader((p3_dir / "planner_results.csv").open(newline="")))
    exact_scaling = []
    for n in (2, 3, 4, 5, 6):
        finite_rows = [
            row
            for row in planner_rows
            if row["family"] == finite and int(row["target_length"]) == n
        ]
        planner_cpu = (
            sum(float(row["mean_planner_cpu_seconds"]) for row in finite_rows)
            / len(finite_rows)
            if finite_rows
            else math.nan
        )
        states = (1 << n) - 1
        transitions = n * (1 << (n - 1))
        exact_scaling.append(
            {
                "target_length": n,
                "mask_states": states,
                "transitions": transitions,
                "cost_table_forward_evaluations_per_prompt": states,
                "mean_planner_cpu_seconds": planner_cpu,
                "estimated_cost_table_bytes_float64": transitions * 8,
                "estimated_peak_planner_memory_bytes": states * 24 + transitions * 24,
                "complexity": "O(N*2^N) transitions and O(2^N) states",
            }
        )

    model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, args.dtype)
    covariance_dir = HISTORICAL_ROOT / "covariance_cache_v1"
    p3_prompt_rows = _read_prompt_rows(p3_dir / "per_prompt_results.jsonl.gz")
    base_stress_by_length = {}
    for n in (5, 6):
        base_rows = [
            row
            for row in p3_prompt_rows
            if row["family"] == "base_default_confidence"
            and row["bucket"] == "same_subject_stress"
            and int(row["target_length"]) == n
        ]
        base_stress_by_length[n] = sum(float(row["full_target_exact"]) for row in base_rows) / len(base_rows)

    all_rows: list[dict[str, Any]] = []
    for n in (5, 6):
        rows = read_jsonl(PROTOCOL_ROOT / f"kamel_pub_dev_n{n}.jsonl")
        if args.limit_per_length:
            rows = rows[: args.limit_per_length]
        items = build_prompt_items(rows, include_stress=True)
        tables = _read_tables(p3_dir / f"edited_cost_tables_n{n}.jsonl.gz")
        editor = lock["editor"]
        config = MemitConfig(
            layers=tuple(map(int, lock["layers"])),
            learning_rate=float(lock["target_value_config"]["learning_rate"]),
            target_optimization_steps=int(lock["target_value_config"]["steps"]),
            clamp_norm_factor=float(lock["target_value_config"]["clamp_norm_factor"]),
            kl_factor=float(lock["target_value_config"]["kl_factor"]),
            partial_mask_schedule=str(editor["partial_mask_schedule"]),
            reveal_policy=str(editor["reveal_policy"]),
            seed=260_717_951,
        )
        target_cache = p3_dir / f"target_value_cache_n{n}"
        rollback, _ = apply_memit_batch(
            model,
            tokenizer,
            rows,
            config,
            lambda layer: load_covariance(covariance_dir, layer),
            target_cache_dir=target_cache,
        )
        try:
            specs = [
                PlannerSpec(finite, "finite_beta", beta=beta, reference=reference),
                *_baseline_specs(lock, n),
            ]
            specs.extend(
                PlannerSpec(
                    f"approx_kl_beam_{multiplier}n",
                    "bounded_kl_beam",
                    beta=beta,
                    reference=reference,
                    beam_width=8,
                    query_budget=min((1 << n) - 1, multiplier * n),
                    regime="bounded_approximation",
                )
                for multiplier in (2, 4, 8)
            )
            for spec in specs:
                decoded = decode_with_planner(model, tokenizer, items, tables, spec)
                all_rows.extend(_attach_base(_seed_rows(decoded), []))
        finally:
            rollback.rollback()
        if not rollback.checksum_matches(atol=0.0):
            raise RuntimeError(f"P7 rollback failed at N={n}")

    aggregate = _aggregate(all_rows)
    comparisons = []
    for n in (5, 6):
        def exact_for(family: str, bucket: str = "rewrite") -> float:
            selected = [
                row
                for row in aggregate
                if row["family"] == family
                and row["bucket"] == bucket
                and int(row["target_length"]) == n
            ]
            return sum(float(row["full_target_exact"]) for row in selected) / len(selected)

        exact_value = exact_for(finite)
        baseline_value = exact_for(baseline)
        exact_gain = exact_value - baseline_value
        exact_queries = (1 << n) - 1
        for multiplier in (2, 4, 8):
            family = f"approx_kl_beam_{multiplier}n"
            approx_value = exact_for(family)
            approx_gain = approx_value - baseline_value
            rows_for_family = [
                row
                for row in aggregate
                if row["family"] == family
                and row["bucket"] == "rewrite"
                and int(row["target_length"]) == n
            ]
            query_count = sum(float(row["unique_state_queries"]) for row in rows_for_family) / len(rows_for_family)
            stress = exact_for(family, "same_subject_stress")
            malformed = max(
                float(row["malformed"])
                for row in aggregate
                if row["family"] == family and int(row["target_length"]) == n
            )
            comparisons.append(
                {
                    "target_length": n,
                    "approximation": family,
                    "exact_rewrite": exact_value,
                    "baseline_rewrite": baseline_value,
                    "approximate_rewrite": approx_value,
                    "exact_gain": exact_gain,
                    "approximate_gain": approx_gain,
                    "gain_retention": approx_gain / exact_gain if exact_gain > 0 else "",
                    "exact_unique_state_queries": exact_queries,
                    "approximate_unique_state_queries": query_count,
                    "query_ratio": query_count / exact_queries,
                    "same_subject_tfpr": stress,
                    "base_same_subject_tfpr": base_stress_by_length[n],
                    "same_subject_tfpr_delta": stress - base_stress_by_length[n],
                    "malformed_rate": malformed,
                    "safety_pass": stress - base_stress_by_length[n] <= 0.03 and malformed <= 0.05,
                }
            )
    candidates = []
    for multiplier in (2, 4, 8):
        family = f"approx_kl_beam_{multiplier}n"
        rows = [row for row in comparisons if row["approximation"] == family]
        retention_values = [float(row["gain_retention"]) for row in rows if row["gain_retention"] != ""]
        mean_retention = (
            sum(retention_values) / len(retention_values) if retention_values else None
        )
        candidates.append(
            {
                "approximation": family,
                "mean_gain_retention": mean_retention,
                "mean_query_ratio": sum(float(row["query_ratio"]) for row in rows) / len(rows),
                "safety_pass": all(bool(row["safety_pass"]) for row in rows),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            bool(row["safety_pass"]),
            (
                float(row["mean_gain_retention"])
                if row["mean_gain_retention"] is not None
                else -math.inf
            ),
            -float(row["mean_query_ratio"]),
        ),
    )
    strong = (
        bool(selected["safety_pass"])
        and selected["mean_gain_retention"] is not None
        and float(selected["mean_gain_retention"]) >= 0.80
        and float(selected["mean_query_ratio"]) <= 0.50
    )
    minimum = (
        bool(selected["safety_pass"])
        and selected["mean_gain_retention"] is not None
        and float(selected["mean_gain_retention"]) >= 0.70
        and float(selected["mean_query_ratio"]) <= 0.50
    )
    decision = (
        "strong_approximation"
        if strong
        else "minimum_useful_approximation"
        if minimum
        else "approximation_scalability_limitation"
    )
    write_csv(args.output_dir / "exact_scaling.csv", exact_scaling)
    write_csv(args.output_dir / "approximate_vs_exact.csv", comparisons)
    write_csv(
        args.output_dir / "long_target_results.csv",
        [
            {
                "target_length_scope": "7-10",
                "status": "not_available_in_fresh_frozen_protocol",
                "reason": "No source manifests were precommitted for N=7 through N=10.",
            }
        ],
    )
    write_csv(args.output_dir / "query_budget_table.csv", candidates)
    write_csv(args.output_dir / "runtime_memory_table.csv", exact_scaling)
    (args.output_dir / "approximation_decision.md").write_text(
        f"# P7 Approximation Decision\n\nDecision: `{decision}`.\n\n"
        f"Selected bounded configuration: `{selected['approximation']}`. This analysis uses the "
        "same finite-beta objective, reference process, and transition cost as the exact controller.\n",
        encoding="utf-8",
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P7",
        "stage": "P7_approximate_solver",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "finite_controller": finite,
        "reference_process": reference,
        "beta": beta,
        "development_grid": ["2N", "4N", "8N"],
        "selected_approximation": selected,
        "decision": decision,
        "strong_approximation_pass": strong,
        "minimum_approximation_pass": minimum,
        "configuration_repair_used": False,
        "locked_confirmation_opened_for_tuning": False,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "runtime_seconds": time.monotonic() - wall_start,
        "acceptance_pass": minimum,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="P7_approximate_solver",
        track="P7",
        status=decision,
        output_dir=args.output_dir,
        acceptance_pass=minimum,
        started_at_utc=started,
        notes=f"decision={decision}; selected={selected['approximation']}",
        next_stage="P8_publication_package",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
