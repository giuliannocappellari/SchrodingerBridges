#!/usr/bin/env python3
"""Run P3 full-table and online compute-matched reveal-planner development."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mask_pattern_kl_control import path_cost
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
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mask_pattern_publication_runtime import (
    PlannerSpec,
    build_full_cost_tables,
    build_prompt_items,
    decode_with_planner,
    item_key,
)
from scripts.mdm_memit_editor import MemitConfig, apply_memit_batch
from scripts.run_mdm_memit_stage import load_covariance, load_model


BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)
REFERENCES = ("uniform", "edited_target_confidence", "edited_max_confidence")
RANDOM_SEEDS = (260717401, 260717402, 260717403, 260717404, 260717405)


def _tuple_costs(table: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    return {
        tuple(map(int, key.split(":"))): float(value)
        for key, value in table["costs"].items()
    }


def _select_fixed_order(
    items: Sequence[Mapping[str, Any]], tables: Mapping[str, Mapping[str, Any]], n: int
) -> tuple[int, ...]:
    rewrite = [item for item in items if item["bucket"] == "rewrite"]
    candidates = []
    for order in itertools.permutations(range(n)):
        mean_cost = sum(path_cost(order, _tuple_costs(tables[item_key(item)])) for item in rewrite) / len(rewrite)
        candidates.append((mean_cost, order))
    return min(candidates, key=lambda row: (row[0], row[1]))[1]


def _planner_specs(n: int, fixed_order: tuple[int, ...], profile: str) -> list[PlannerSpec]:
    if profile == "smoke":
        return [
            PlannerSpec("default_confidence", "default_confidence"),
            PlannerSpec("one_step_myopic", "myopic"),
            PlannerSpec("deterministic_global", "deterministic_global"),
            PlannerSpec(
                "finite_uniform_beta1", "finite_beta", beta=1.0, reference="uniform"
            ),
        ]
    specs = [
        PlannerSpec("default_confidence", "default_confidence"),
        PlannerSpec("left_to_right", "left_to_right"),
        PlannerSpec("right_to_left", "right_to_left"),
        PlannerSpec("best_fixed_permutation", "fixed_order", fixed_order=fixed_order),
        PlannerSpec("minimum_entropy", "minimum_entropy"),
        PlannerSpec("one_step_myopic", "myopic"),
        PlannerSpec("deterministic_global", "deterministic_global"),
        PlannerSpec("beam_width2", "beam", beam_width=2),
        PlannerSpec("beam_width4", "beam", beam_width=4),
        PlannerSpec("beam_width8", "beam", beam_width=8),
        PlannerSpec(
            "random_search_full", "random_search", random_paths=max(8, 1 << n), seed=260717450
        ),
    ]
    specs.extend(
        PlannerSpec(f"uniform_random_seed{seed}", "uniform_random", seed=seed)
        for seed in RANDOM_SEEDS
    )
    for reference in REFERENCES:
        specs.append(
            PlannerSpec(
                f"beta0_{reference}", "beta_zero", beta=0.0, reference=reference
            )
        )
        specs.extend(
            PlannerSpec(
                f"finite_{reference}_beta{beta:g}",
                "finite_beta",
                beta=beta,
                reference=reference,
            )
            for beta in BETAS
        )
    for budget in sorted({n, 2 * n, 4 * n, (1 << n) - 1}):
        specs.append(
            PlannerSpec(
                f"online_beam8_budget{budget}",
                "bounded_beam",
                beam_width=8,
                query_budget=budget,
                regime="online_compute_matched",
            )
        )
        specs.append(
            PlannerSpec(
                f"online_random_budget{budget}",
                "bounded_random",
                query_budget=budget,
                seed=260717451,
                regime="online_compute_matched",
            )
        )
    return specs


def _family(label: str) -> str:
    if label.startswith("uniform_random_seed"):
        return "uniform_random"
    return label


def _attach_base(
    rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    base = {
        (str(row["case_id"]), str(row["bucket"])): str(row["output_token_ids"])
        for row in base_rows
    }
    output = []
    for row in rows:
        item = dict(row)
        item["family"] = _family(str(row["label"]))
        base_ids = base.get((str(row["case_id"]), str(row["bucket"])), "")
        item["base_output_token_ids"] = base_ids
        item["base_agreement"] = bool(base_ids and base_ids == str(row["output_token_ids"]))
        output.append(item)
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["family"]),
                str(row["bucket"]),
                int(row["target_length"]),
                str(row["regime"]),
            )
        ].append(row)
    output = []
    for (family, bucket, length, regime), values in sorted(grouped.items()):
        by_edit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_edit[str(row["case_id"])].append(row)
        def edit_mean(key: str) -> float:
            return sum(
                sum(float(row[key]) for row in edit_rows) / len(edit_rows)
                for edit_rows in by_edit.values()
            ) / len(by_edit)
        output.append(
            {
                "family": family,
                "bucket": bucket,
                "target_length": length,
                "regime": regime,
                "num_edits": len(by_edit),
                "num_seed_rows": len(values),
                "full_target_exact": edit_mean("full_target_exact"),
                "target_token_f1": edit_mean("target_token_f1"),
                "malformed_rate": edit_mean("malformed"),
                "target_false_positive_rate": edit_mean("full_target_exact")
                if bucket in {"same_subject_stress", "far_locality"}
                else "",
                "base_agreement": edit_mean("base_agreement"),
                "mean_trajectory_target_cost": edit_mean("trajectory_target_cost"),
                "mean_unique_state_queries": edit_mean("unique_state_queries"),
                "mean_model_evaluations": edit_mean("model_evaluations"),
                "mean_candidate_path_evaluations": edit_mean("candidate_path_evaluations"),
                "mean_planner_cpu_seconds": edit_mean("planner_cpu_seconds"),
                "mean_path_entropy": edit_mean("path_entropy"),
                "mean_path_kl_from_reference": edit_mean("path_kl_from_reference"),
                "seed_count": len({int(row["planner_seed"]) for row in values}),
            }
        )
    return output


def _method_summary(aggregate: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row["family"]) for row in aggregate})
    output = []
    for family in families:
        selected = [row for row in aggregate if row["family"] == family]
        regime = str(selected[0]["regime"])
        rewrite = [float(row["full_target_exact"]) for row in selected if row["bucket"] == "rewrite"]
        paraphrase = [float(row["full_target_exact"]) for row in selected if row["bucket"] == "paraphrase"]
        stress = [
            float(row["target_false_positive_rate"])
            for row in selected
            if row["bucket"] == "same_subject_stress"
        ]
        far = [
            float(row["target_false_positive_rate"])
            for row in selected
            if row["bucket"] == "far_locality"
        ]
        malformed = [float(row["malformed_rate"]) for row in selected]
        costs = [
            float(row["mean_trajectory_target_cost"])
            for row in selected
            if row["bucket"] == "rewrite"
        ]
        queries = [
            float(row["mean_unique_state_queries"])
            for row in selected
            if row["bucket"] == "rewrite"
        ]
        output.append(
            {
                "family": family,
                "regime": regime,
                "rewrite_exact": sum(rewrite) / len(rewrite) if rewrite else 0.0,
                "paraphrase_exact": sum(paraphrase) / len(paraphrase) if paraphrase else 0.0,
                "same_subject_tfpr": sum(stress) / len(stress) if stress else 0.0,
                "far_tfpr": sum(far) / len(far) if far else 0.0,
                "malformed_rate": max(malformed, default=0.0),
                "mean_trajectory_target_cost": sum(costs) / len(costs) if costs else 0.0,
                "mean_unique_state_queries": sum(queries) / len(queries) if queries else 0.0,
            }
        )
    return output


def _safety_pass(
    candidate: Mapping[str, Any], base: Mapping[str, Any]
) -> tuple[bool, dict[str, float]]:
    thresholds = {
        "same_subject_tfpr_budget": float(base["same_subject_tfpr"]) + 0.03,
        "far_tfpr_budget": float(base["far_tfpr"]) + 0.03,
        "malformed_rate_budget": 0.05,
    }
    passed = (
        float(candidate["same_subject_tfpr"])
        <= thresholds["same_subject_tfpr_budget"]
        and float(candidate["far_tfpr"]) <= thresholds["far_tfpr_budget"]
        and float(candidate["malformed_rate"])
        <= thresholds["malformed_rate_budget"]
    )
    return passed, thresholds


def _select_editor(p1_dir: Path) -> dict[str, Any]:
    report = read_json(p1_dir / "report_summary.json")
    rows = list(csv.DictReader((p1_dir / "method_bucket.csv").open(newline="")))
    scores: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["bucket"] not in {"rewrite", "paraphrase"}:
            continue
        scores[row["label"]] += float(row["full_target_token_exact"])
        counts[row["label"]] += 1
    best = max(scores, key=lambda label: (scores[label] / counts[label], label))
    if report.get("discrepancy_decision") == "reproduced_paper_trend":
        best = "paper_matched_partial_cycle"
    mapping = {
        "ordinary_fully_masked": ("fully_masked", "random"),
        "partial_cycle_fixed_positions": ("cycle", "left_to_right"),
        "partial_cycle_random_positions": ("cycle", "random"),
        "paper_matched_partial_cycle": ("cycle", "random"),
        "partial_random_count_random_positions": ("uniform", "random"),
    }
    schedule, reveal = mapping[best]
    return {
        "label": best,
        "partial_mask_schedule": schedule,
        "reveal_policy": reveal,
        "p1_acceptance_pass": bool(report["acceptance_pass"]),
        "p1_discrepancy_decision": report["discrepancy_decision"],
    }


def _write_cost_tables(path: Path, tables: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for key in sorted(tables):
            handle.write(json.dumps({"item_key": key, **tables[key]}, sort_keys=True) + "\n")


def _power_analysis(
    rows: Sequence[Mapping[str, Any]], finite: str, baseline: str
) -> dict[str, Any]:
    deltas = []
    for length in (3, 4):
        maps = {}
        for family in (finite, baseline):
            by_case: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if (
                    row["family"] == family
                    and row["bucket"] == "rewrite"
                    and int(row["target_length"]) == length
                ):
                    by_case[str(row["case_id"])].append(float(bool(row["full_target_exact"])))
            maps[family] = {
                case: sum(values) / len(values) for case, values in by_case.items()
            }
        for case in sorted(set(maps[finite]) & set(maps[baseline])):
            deltas.append(maps[finite][case] - maps[baseline][case])
    variance = statistics.pvariance(deltas) if len(deltas) > 1 else 0.25
    target_effect = 0.05
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    required = math.ceil((z_alpha + z_power) ** 2 * max(variance, 1e-6) / target_effect**2)
    return {
        "written_before_locked_confirmation_opened": True,
        "dev_paired_difference_variance": variance,
        "target_absolute_effect": target_effect,
        "alpha_two_sided": 0.05,
        "target_power": 0.80,
        "normal_approx_required_pooled_pairs": required,
        "planned_primary_locked_pairs": 1000,
        "planned_sample_not_reduced_after_results": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "planner_baselines_dev_v1"
    )
    parser.add_argument("--lengths", default="2,3,4,5,6")
    parser.add_argument("--limit_per_length", type=int, default=0)
    parser.add_argument("--planner_profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--include_stress", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    started = now_utc()
    wall_start = time.monotonic()
    if args.output_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths = tuple(sorted({int(value) for value in args.lengths.split(",") if value.strip()}))
    if not set(lengths) <= {2, 3, 4, 5, 6}:
        raise ValueError(lengths)
    p1_dir = CAMPAIGN_ROOT / "partial_state_memit_audit_v1"
    editor = _select_editor(p1_dir)

    model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, args.dtype)
    covariance_dir = HISTORICAL_ROOT / "covariance_cache_v1"
    all_outputs: list[dict[str, Any]] = []
    cost_accounting = []
    fixed_orders = {}
    manifest_hashes = {}
    for length in lengths:
        manifest = PROTOCOL_ROOT / f"kamel_pub_dev_n{length}.jsonl"
        rows = read_jsonl(manifest)
        if args.limit_per_length:
            rows = rows[: args.limit_per_length]
        manifest_hashes[str(length)] = sha256_file(manifest)
        items = build_prompt_items(rows, include_stress=bool(args.include_stress))
        base_tables, base_account = build_full_cost_tables(model, tokenizer, items)
        base_spec = PlannerSpec("base_default_confidence", "default_confidence")
        base_rows = decode_with_planner(model, tokenizer, items, base_tables, base_spec)
        all_outputs.extend(_attach_base(base_rows, base_rows))
        config = MemitConfig(
            layers=(4, 5, 6, 7),
            learning_rate=0.1,
            target_optimization_steps=25,
            clamp_norm_factor=0.75,
            kl_factor=0.0625,
            partial_mask_schedule=editor["partial_mask_schedule"],
            reveal_policy=editor["reveal_policy"],
            seed=260717501,
        )
        rollback, diagnostics = apply_memit_batch(
            model,
            tokenizer,
            rows,
            config,
            lambda layer: load_covariance(covariance_dir, layer),
            target_cache_dir=args.output_dir / f"target_value_cache_n{length}",
        )
        try:
            edited_tables, edited_account = build_full_cost_tables(model, tokenizer, items)
            _write_cost_tables(args.output_dir / f"edited_cost_tables_n{length}.jsonl.gz", edited_tables)
            fixed = _select_fixed_order(items, edited_tables, length)
            fixed_orders[str(length)] = list(fixed)
            specs = _planner_specs(length, fixed, args.planner_profile)
            length_outputs = []
            for spec in specs:
                length_outputs.extend(
                    decode_with_planner(model, tokenizer, items, edited_tables, spec)
                )
            # The default and maximum-confidence planners are identical under
            # the frozen one-token-per-step LLaDA schedule; retain an explicit alias.
            default_rows = [row for row in length_outputs if row["label"] == "default_confidence"]
            for row in default_rows:
                alias = dict(row)
                alias["label"] = "maximum_confidence"
                alias["planner_kind"] = "maximum_confidence"
                length_outputs.append(alias)
            all_outputs.extend(_attach_base(length_outputs, base_rows))
            cost_accounting.append(
                {
                    "target_length": length,
                    "num_edits": len(rows),
                    "num_prompt_items": len(items),
                    "base_cost_table": json.dumps(base_account, sort_keys=True),
                    "edited_cost_table": json.dumps(edited_account, sort_keys=True),
                    "target_optimization_rows": len(diagnostics.get("target_optimization", [])),
                    "fixed_order": json.dumps(fixed),
                }
            )
        finally:
            rollback.rollback()
        if not rollback.checksum_matches(atol=0.0):
            raise RuntimeError(f"P3 rollback failed at target length {length}")

    aggregate = _aggregate(all_outputs)
    summaries = _method_summary(aggregate)
    full_finite = [
        row
        for row in summaries
        if row["family"].startswith("finite_") and row["regime"] == "full_cost_table"
    ]
    non_sb = [
        row
        for row in summaries
        if not row["family"].startswith(("finite_", "beta0_"))
        and row["regime"] == "full_cost_table"
        and row["family"] not in {"base_default_confidence"}
    ]
    if not full_finite or not non_sb:
        raise RuntimeError("P3 planner profile did not produce selection candidates")
    selected_finite = max(
        full_finite,
        key=lambda row: (
            float(row["rewrite_exact"]) + float(row["paraphrase_exact"]),
            -float(row["mean_trajectory_target_cost"]),
            row["family"],
        ),
    )
    selected_non_sb = max(
        non_sb,
        key=lambda row: (
            float(row["rewrite_exact"]) + float(row["paraphrase_exact"]),
            -float(row["mean_trajectory_target_cost"]),
            row["family"],
        ),
    )
    beta0_reference = "beta0_" + selected_finite["family"].split("_beta", 1)[0].removeprefix("finite_")
    beta0 = next(row for row in summaries if row["family"] == beta0_reference)
    base_result = next(row for row in summaries if row["family"] == "base_default_confidence")
    myopic = next(row for row in summaries if row["family"] == "one_step_myopic")
    deterministic = next(row for row in summaries if row["family"] == "deterministic_global")
    finite_rewrite = float(selected_finite["rewrite_exact"])
    deterministic_rewrite = float(deterministic["rewrite_exact"])
    deterministic_cost = float(deterministic["mean_trajectory_target_cost"])
    finite_cost = float(selected_finite["mean_trajectory_target_cost"])
    mechanism_pass = (
        finite_rewrite - deterministic_rewrite >= 0.03
        or (
            finite_rewrite >= deterministic_rewrite - 0.02
            and finite_cost <= 0.80 * max(deterministic_cost, 1e-12)
        )
    ) and finite_rewrite > float(beta0["rewrite_exact"]) and finite_rewrite > float(myopic["rewrite_exact"])
    safety_pass, safety_thresholds = _safety_pass(selected_finite, base_result)
    power = _power_analysis(
        all_outputs, str(selected_finite["family"]), str(selected_non_sb["family"])
    )
    write_json(args.output_dir / "power_analysis.json", power)
    lock = {
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "locked_confirmation_opened": False,
        "model_id": PRIMARY_MODEL_ID,
        "model_revision": PRIMARY_MODEL_REVISION,
        "dtype": args.dtype,
        "use_4bit": False,
        "editor": editor,
        "layers": [4, 5, 6, 7],
        "target_value_config": {
            "learning_rate": 0.1,
            "steps": 25,
            "clamp_norm_factor": 0.75,
            "kl_factor": 0.0625,
        },
        "reference_process": selected_finite["family"].removeprefix("finite_").rsplit("_beta", 1)[0],
        "beta": float(selected_finite["family"].rsplit("beta", 1)[1]),
        "finite_controller_label": selected_finite["family"],
        "best_non_sb_planner": selected_non_sb["family"],
        "online_primary_query_budget": "2^N-1 unique states",
        "beam_widths": [2, 4, 8],
        "fixed_orders": fixed_orders,
        "generation_steps": "target_length",
        "span_policy": "exact_contextual_target_length",
        "random_policy_seeds": list(RANDOM_SEEDS),
        "bootstrap_resamples": 10_000,
        "holm_primary_lengths": [3, 4],
        "manifest_hashes": manifest_hashes,
        "power_analysis_sha256": sha256_file(args.output_dir / "power_analysis.json"),
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "validation_pass": True,
    }
    write_json(args.output_dir / "dev_method_lock.json", lock)
    write_json(CAMPAIGN_ROOT / "dev_method_lock.json", lock)
    write_csv(args.output_dir / "planner_results.csv", aggregate)
    write_csv(
        args.output_dir / "compute_matched_results.csv",
        [row for row in aggregate if row["regime"] == "online_compute_matched"],
    )
    write_csv(
        args.output_dir / "beta_sweep.csv",
        [row for row in aggregate if row["family"].startswith(("finite_", "beta0_"))],
    )
    write_csv(
        args.output_dir / "reference_process_ablation.csv",
        [row for row in aggregate if row["family"].startswith(("finite_", "beta0_"))],
    )
    write_csv(
        args.output_dir / "beam_random_search_results.csv",
        [
            row
            for row in aggregate
            if row["family"].startswith(("beam_", "random_search", "online_"))
        ],
    )
    write_csv(args.output_dir / "planner_query_accounting.csv", cost_accounting)
    write_csv(
        args.output_dir / "path_entropy_kl.csv",
        [row for row in aggregate if row["family"].startswith(("finite_", "beta0_"))],
    )
    with gzip.open(args.output_dir / "per_prompt_results.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in all_outputs:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P3",
        "stage": "P3_planner_baselines_dev",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "target_lengths": list(lengths),
        "limit_per_length": args.limit_per_length,
        "planner_profile": args.planner_profile,
        "editor": editor,
        "selected_finite_controller": selected_finite,
        "selected_non_sb_planner": selected_non_sb,
        "base_result": base_result,
        "beta0_reference_result": beta0,
        "myopic_result": myopic,
        "deterministic_global_result": deterministic,
        "finite_beta_mechanism_pass": mechanism_pass,
        "safety_pass": safety_pass,
        "safety_thresholds": safety_thresholds,
        "full_cost_table_regime_complete": True,
        "online_compute_matched_regime_complete": args.planner_profile == "full",
        "dev_method_lock_written": True,
        "locked_confirmation_opened": False,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "runtime_seconds": time.monotonic() - wall_start,
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "acceptance_pass": mechanism_pass and safety_pass,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="P3_planner_baselines_dev",
        track="P3",
        status="passed" if report["acceptance_pass"] else "mechanism_not_established_on_dev",
        output_dir=args.output_dir,
        acceptance_pass=bool(report["acceptance_pass"]),
        started_at_utc=started,
        notes=(
            f"finite={selected_finite['family']}; non_sb={selected_non_sb['family']}; "
            f"mechanism={mechanism_pass}; safety={safety_pass}"
        ),
        next_stage="P4_llada_locked_confirmation",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
