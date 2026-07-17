#!/usr/bin/env python3
"""Run the frozen P4 LLaDA confirmation exactly once on fresh locked KAMEL data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    protocol_split_summary,
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
    planner_spec_from_label,
)
from scripts.mask_pattern_publication_stats import holm_adjust, paired_bootstrap, paired_values
from scripts.mdm_memit_editor import MemitConfig, apply_memit_batch
from scripts.run_mdm_memit_stage import load_covariance, load_model


GENERATION_SEEDS = (260_717_701, 260_717_702, 260_717_703)
RANDOM_SEEDS = (260_717_711, 260_717_712, 260_717_713, 260_717_714, 260_717_715)


def _family(label: str) -> str:
    if label.startswith("uniform_random_seed"):
        return "uniform_random"
    return label


def _seed_rows(
    rows: Sequence[Mapping[str, Any]], generation_seeds: Sequence[int] = GENERATION_SEEDS
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        for seed in generation_seeds:
            item = dict(row)
            item["generation_seed"] = seed
            item["family"] = _family(str(item["label"]))
            output.append(item)
    return output


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
        key = (str(item["case_id"]), str(item["bucket"]))
        item["base_output_token_ids"] = base.get(key, "")
        item["base_agreement"] = item["base_output_token_ids"] == str(
            item["output_token_ids"]
        )
        output.append(item)
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["bucket"]), int(row["target_length"]))].append(row)
    output = []
    metrics = (
        "full_target_exact",
        "target_token_f1",
        "old_target_suppression",
        "malformed",
        "base_agreement",
        "trajectory_target_cost",
        "unique_state_queries",
        "model_evaluations",
        "planner_cpu_seconds",
        "path_entropy",
        "path_kl_from_reference",
    )
    for (family, bucket, length), values in sorted(grouped.items()):
        by_edit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_edit[str(row["case_id"])].append(row)
        item: dict[str, Any] = {
            "family": family,
            "bucket": bucket,
            "target_length": length,
            "num_edits": len(by_edit),
            "num_seed_rows": len(values),
        }
        for metric in metrics:
            item[metric] = sum(
                sum(float(row[metric]) for row in edit_rows) / len(edit_rows)
                for edit_rows in by_edit.values()
            ) / len(by_edit)
        item["target_false_positive_rate"] = (
            item["full_target_exact"]
            if bucket in {"same_subject_stress", "far_locality"}
            else ""
        )
        item["planner_seed_count"] = len({int(row["planner_seed"]) for row in values})
        item["generation_seed_count"] = len(
            {int(row["generation_seed"]) for row in values}
        )
        output.append(item)
    return output


def _planner_specs(lock: Mapping[str, Any], n: int) -> list[PlannerSpec]:
    fixed = lock["fixed_orders"][str(n)]
    finite = str(lock["finite_controller_label"])
    non_sb = str(lock["best_non_sb_planner"])
    search = str(lock["best_full_table_beam_or_random_planner"])
    beta0 = "beta0_" + finite.removeprefix("finite_").rsplit("_beta", 1)[0]
    labels = [
        "left_to_right",
        "right_to_left",
        "best_fixed_permutation",
        "one_step_myopic",
        "deterministic_global",
        beta0,
        finite,
    ]
    if non_sb not in labels and non_sb != "uniform_random":
        labels.append(non_sb)
    if search not in labels and search != "uniform_random":
        labels.append(search)
    specs = [
        planner_spec_from_label(label, n=n, fixed_order=fixed, seed=260_717_720)
        for label in labels
    ]
    specs.extend(
        PlannerSpec(
            f"uniform_random_seed{seed}", "uniform_random", seed=seed
        )
        for seed in RANDOM_SEEDS
    )
    return specs


def _bootstrap_table(
    rows: Sequence[Mapping[str, Any]], finite: str, baseline: str
) -> list[dict[str, Any]]:
    output = []
    for bucket in ("rewrite", "paraphrase"):
        for metric in ("full_target_exact", "target_token_f1"):
            for label, lengths in (
                ("pooled_n3_n4", {3, 4}),
                ("n3", {3}),
                ("n4", {4}),
            ):
                result = paired_bootstrap(
                    paired_values(
                        rows,
                        left=finite,
                        right=baseline,
                        bucket=bucket,
                        metric=metric,
                        lengths=lengths,
                    ),
                    resamples=10_000,
                    seed=260_717_730 + len(output),
                )
                output.append(
                    {
                        "left": finite,
                        "right": baseline,
                        "bucket": bucket,
                        "metric": metric,
                        "scope": label,
                        **result,
                    }
                )
    return output


def _mean_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    bucket: str,
    metric: str,
    lengths: set[int],
) -> tuple[float, float, float]:
    pairs = paired_values(
        rows,
        left=left,
        right=right,
        bucket=bucket,
        metric=metric,
        lengths=lengths,
    )
    if not pairs:
        raise RuntimeError(f"No matched rows for {left} vs {right}: {bucket}/{metric}")
    left_mean = sum(row[1] for row in pairs) / len(pairs)
    right_mean = sum(row[2] for row in pairs) / len(pairs)
    return left_mean - right_mean, left_mean, right_mean


def _write_cost_tables(path: Path, tables: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for key in sorted(tables):
            handle.write(json.dumps({"item_key": key, **tables[key]}, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "llada_locked_confirmation_v1"
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
    lock_path = CAMPAIGN_ROOT / "dev_method_lock.json"
    lock = read_json(lock_path)
    if not lock.get("validation_pass") or lock.get("locked_confirmation_opened"):
        raise RuntimeError("P4 requires an unopened, validated dev method lock")
    if int(lock.get("bootstrap_resamples", 0)) != 10_000:
        raise RuntimeError("Dev lock does not freeze the required bootstrap count")
    if lock.get("controller_action_rule") != "greedy_argmax_of_exact_controlled_transition":
        raise RuntimeError("Dev lock has an unknown finite-controller action rule")
    p3_report = read_json(CAMPAIGN_ROOT / "planner_baselines_dev_v1" / "report_summary.json")
    if p3_report.get("planner_profile") != "full":
        raise RuntimeError("P4 requires the complete P3 planner suite, not a smoke lock")

    manifests = {
        length: PROTOCOL_ROOT / f"kamel_pub_locked_n{length}.jsonl" for length in lengths
    }
    protocol_report = read_json(PROTOCOL_ROOT / "report_summary.json")
    expected_hashes = {
        length: str(
            protocol_split_summary(
                protocol_report, f"kamel_pub_locked_n{length}"
            )["sha256"]
        )
        for length in lengths
    }
    for length, path in manifests.items():
        if sha256_file(path) != expected_hashes[length]:
            raise RuntimeError(f"Locked manifest hash mismatch for N={length}")
    write_json(
        args.output_dir / "locked_open_record.json",
        {
            "created_at_utc": now_utc(),
            "dev_lock_sha256": sha256_file(lock_path),
            "locked_manifest_hashes": expected_hashes,
            "selection_complete_before_open": True,
            "no_locked_outcome_used_for_tuning": True,
        },
    )

    model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, args.dtype)
    covariance_dir = HISTORICAL_ROOT / "covariance_cache_v1"
    all_rows: list[dict[str, Any]] = []
    compute_rows = []
    manifest_counts = {}
    for length, manifest in manifests.items():
        source_rows = read_jsonl(manifest)
        rows = source_rows[: args.limit_per_length] if args.limit_per_length else source_rows
        manifest_counts[str(length)] = len(rows)
        items = build_prompt_items(rows, include_stress=True)
        base_tables, base_account = build_full_cost_tables(model, tokenizer, items)
        base_decoded = decode_with_planner(
            model,
            tokenizer,
            items,
            base_tables,
            PlannerSpec("base_default_confidence", "default_confidence"),
        )
        base_seeded = _seed_rows(base_decoded)
        all_rows.extend(_attach_base(base_seeded, base_seeded))

        ordinary_config = MemitConfig(
            layers=(4, 5, 6, 7),
            learning_rate=0.1,
            target_optimization_steps=25,
            clamp_norm_factor=0.75,
            kl_factor=0.0625,
            partial_mask_schedule="fully_masked",
            reveal_policy="random",
            seed=260_717_740,
        )
        rollback, _ = apply_memit_batch(
            model,
            tokenizer,
            rows,
            ordinary_config,
            lambda layer: load_covariance(covariance_dir, layer),
            target_cache_dir=args.output_dir / "target_value_cache" / f"ordinary_n{length}",
        )
        try:
            ordinary_tables, ordinary_account = build_full_cost_tables(model, tokenizer, items)
            ordinary = decode_with_planner(
                model,
                tokenizer,
                items,
                ordinary_tables,
                PlannerSpec("ordinary_memit_default", "default_confidence"),
            )
            all_rows.extend(_attach_base(_seed_rows(ordinary), base_seeded))
        finally:
            rollback.rollback()
        if not rollback.checksum_matches(atol=0.0):
            raise RuntimeError(f"Ordinary editor rollback failed at N={length}")

        editor = lock["editor"]
        partial_config = MemitConfig(
            layers=tuple(map(int, lock["layers"])),
            learning_rate=float(lock["target_value_config"]["learning_rate"]),
            target_optimization_steps=int(lock["target_value_config"]["steps"]),
            clamp_norm_factor=float(lock["target_value_config"]["clamp_norm_factor"]),
            kl_factor=float(lock["target_value_config"]["kl_factor"]),
            partial_mask_schedule=str(editor["partial_mask_schedule"]),
            reveal_policy=str(editor["reveal_policy"]),
            seed=260_717_741,
        )
        rollback, _ = apply_memit_batch(
            model,
            tokenizer,
            rows,
            partial_config,
            lambda layer: load_covariance(covariance_dir, layer),
            target_cache_dir=args.output_dir / "target_value_cache" / f"partial_n{length}",
        )
        try:
            edited_tables, edited_account = build_full_cost_tables(model, tokenizer, items)
            _write_cost_tables(
                args.output_dir / f"edited_cost_tables_n{length}.jsonl.gz", edited_tables
            )
            partial_default = decode_with_planner(
                model,
                tokenizer,
                items,
                edited_tables,
                PlannerSpec("partial_memit_default", "default_confidence"),
            )
            all_rows.extend(_attach_base(_seed_rows(partial_default), base_seeded))
            for spec in _planner_specs(lock, length):
                decoded = decode_with_planner(model, tokenizer, items, edited_tables, spec)
                all_rows.extend(_attach_base(_seed_rows(decoded), base_seeded))
            compute_rows.append(
                {
                    "target_length": length,
                    "num_edits": len(rows),
                    "num_prompt_items": len(items),
                    "base_cost_table": json.dumps(base_account, sort_keys=True),
                    "ordinary_cost_table": json.dumps(ordinary_account, sort_keys=True),
                    "partial_cost_table": json.dumps(edited_account, sort_keys=True),
                }
            )
        finally:
            rollback.rollback()
        if not rollback.checksum_matches(atol=0.0):
            raise RuntimeError(f"Partial editor rollback failed at N={length}")

    aggregate = _aggregate(all_rows)
    finite = str(lock["finite_controller_label"])
    baseline = str(lock["best_non_sb_planner"])
    bootstrap = _bootstrap_table(all_rows, finite, baseline)
    primary_tests = [
        row
        for row in bootstrap
        if row["bucket"] == "rewrite"
        and row["metric"] == "full_target_exact"
        and row["scope"] in {"n3", "n4"}
    ]
    holm = holm_adjust(primary_tests)
    pooled = next(
        row
        for row in bootstrap
        if row["bucket"] == "rewrite"
        and row["metric"] == "full_target_exact"
        and row["scope"] == "pooled_n3_n4"
    )
    by_length = {row["scope"]: row for row in primary_tests}
    cost_delta, finite_cost, baseline_cost = _mean_delta(
        all_rows,
        left=finite,
        right=baseline,
        bucket="rewrite",
        metric="trajectory_target_cost",
        lengths={3, 4},
    )
    cost_reduction = -cost_delta / max(baseline_cost, 1e-12)
    f1_delta, _, _ = _mean_delta(
        all_rows,
        left=finite,
        right=baseline,
        bucket="rewrite",
        metric="target_token_f1",
        lengths={3, 4},
    )
    stress_delta, finite_stress, base_stress = _mean_delta(
        all_rows,
        left=finite,
        right="base_default_confidence",
        bucket="same_subject_stress",
        metric="full_target_exact",
        lengths={3, 4},
    )
    finite_malformed = max(
        float(row["malformed"])
        for row in aggregate
        if row["family"] == finite and int(row["target_length"]) in {3, 4}
    )
    one_length_strong = any(
        float(by_length[scope]["mean_delta"]) >= 0.05
        and float(by_length[scope]["ci95_low"]) > 0
        for scope in ("n3", "n4")
    )
    minimum_pass = (
        float(pooled["mean_delta"]) >= 0.05
        and float(pooled["ci95_low"]) > 0
        and any(bool(row["holm_reject_0_05"]) for row in holm)
        and all(float(by_length[scope]["mean_delta"]) >= 0 for scope in ("n3", "n4"))
        and one_length_strong
        and cost_reduction >= 0.15
        and finite_malformed <= 0.05
        and stress_delta <= 0.03
        and f1_delta >= -0.02
    )
    strong_pass = minimum_pass and all(
        float(by_length[scope]["ci95_low"]) > 0 for scope in ("n3", "n4")
    )
    classification = (
        "strong_locked_pass"
        if strong_pass
        else "minimum_credible_locked_pass"
        if minimum_pass
        else "fresh_confirmation_failed"
    )

    failure_cases = []
    finite_by_case = {
        str(row["case_id"]): row
        for row in all_rows
        if row["family"] == finite
        and row["bucket"] == "rewrite"
        and int(row["generation_seed"]) == GENERATION_SEEDS[0]
    }
    baseline_by_case = {
        str(row["case_id"]): row
        for row in all_rows
        if row["family"] == baseline
        and row["bucket"] == "rewrite"
        and int(row["generation_seed"]) == GENERATION_SEEDS[0]
    }
    for case_id in sorted(set(finite_by_case) & set(baseline_by_case)):
        left, right = finite_by_case[case_id], baseline_by_case[case_id]
        if not left["full_target_exact"] or left["malformed"]:
            failure_cases.append(
                {
                    "case_id": case_id,
                    "target_length": left["target_length"],
                    "finite_output": left["output_text"],
                    "baseline_output": right["output_text"],
                    "finite_exact": left["full_target_exact"],
                    "baseline_exact": right["full_target_exact"],
                    "finite_trajectory": left["trajectory"],
                    "baseline_trajectory": right["trajectory"],
                }
            )

    write_csv(args.output_dir / "main_results.csv", aggregate)
    write_csv(args.output_dir / "target_length_results.csv", aggregate)
    write_csv(
        args.output_dir / "compute_matched_results.csv",
        [row for row in aggregate if row["family"] in {finite, baseline}],
    )
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap)
    write_csv(args.output_dir / "holm_corrected_tests.csv", holm)
    write_csv(
        args.output_dir / "random_seed_summary.csv",
        [row for row in aggregate if row["family"] == "uniform_random"],
    )
    write_csv(
        args.output_dir / "same_subject_stress.csv",
        [row for row in aggregate if row["bucket"] == "same_subject_stress"],
    )
    write_csv(
        args.output_dir / "locality_malformed.csv",
        [
            row
            for row in aggregate
            if row["bucket"] in {"same_subject_stress", "far_locality"}
            or float(row["malformed"]) > 0
        ],
    )
    write_csv(
        args.output_dir / "trajectory_cost_table.csv",
        [row for row in aggregate if row["bucket"] == "rewrite"],
    )
    write_csv(args.output_dir / "failure_cases.csv", failure_cases[:300])
    write_csv(args.output_dir / "compute_accounting.csv", compute_rows)
    with gzip.open(args.output_dir / "per_prompt_results.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    interpretation = f"""# P4 Locked LLaDA Confirmation

Classification: `{classification}`.

The frozen finite controller `{finite}` was compared against the dev-selected
compute-matched non-SB planner `{baseline}`. The pooled N=3/N=4 rewrite delta
was {float(pooled['mean_delta']):.6f} with 95% CI
[{float(pooled['ci95_low']):.6f}, {float(pooled['ci95_high']):.6f}]. Locked
outcomes were not used to alter the method, editor, beta, reference process,
or thresholds.
"""
    (args.output_dir / "locked_result_interpretation.md").write_text(
        interpretation, encoding="utf-8"
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P4",
        "stage": "P4_llada_locked_confirmation",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": PRIMARY_MODEL_ID,
        "model_revision": PRIMARY_MODEL_REVISION,
        "dev_lock_sha256": sha256_file(lock_path),
        "p3_dev_acceptance_pass": bool(p3_report["acceptance_pass"]),
        "locked_manifest_hashes": expected_hashes,
        "manifest_counts": manifest_counts,
        "limit_per_length": args.limit_per_length,
        "generation_seeds": list(GENERATION_SEEDS),
        "random_policy_seeds": list(RANDOM_SEEDS),
        "finite_controller": finite,
        "controller_action_rule": lock["controller_action_rule"],
        "compute_matched_baseline": baseline,
        "pooled_primary_bootstrap": pooled,
        "holm_tests": holm,
        "trajectory_target_cost_reduction": cost_reduction,
        "finite_mean_trajectory_cost": finite_cost,
        "baseline_mean_trajectory_cost": baseline_cost,
        "target_token_f1_delta": f1_delta,
        "same_subject_tfpr_delta": stress_delta,
        "finite_same_subject_tfpr": finite_stress,
        "base_same_subject_tfpr": base_stress,
        "finite_malformed_rate": finite_malformed,
        "classification": classification,
        "minimum_credible_pass": minimum_pass,
        "strong_pass": strong_pass,
        "locked_outcomes_used_for_tuning": False,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "runtime_seconds": time.monotonic() - wall_start,
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "acceptance_pass": minimum_pass,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="P4_llada_locked_confirmation",
        track="P4",
        status="passed" if minimum_pass else "fresh_confirmation_failed",
        output_dir=args.output_dir,
        acceptance_pass=minimum_pass,
        started_at_utc=started,
        notes=(
            f"classification={classification}; pooled_delta={pooled['mean_delta']}; "
            f"cost_reduction={cost_reduction}"
        ),
        next_stage="P5_dream_confirmation",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
