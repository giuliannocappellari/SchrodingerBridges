#!/usr/bin/env python3
"""Aggregate runtime editor outputs for ``counterfact_direction1_v1``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROTOCOL_VERSION = "counterfact_direction1_v1"


def balance_unit_id(row: Dict[str, Any]) -> str:
    """Return the edit-level unit used for aggregation/bootstrap."""

    return str(row.get("edit_id", row.get("case_id", "")))


def display_method(row: Dict[str, Any]) -> str:
    """Return the report label for a method/run variant."""

    return str(row.get("method_variant") or row.get("method") or "")


def harmonic_mean(values: Sequence[float], eps: float = 1e-12) -> float:
    clean = [max(float(v), eps) for v in values]
    if not clean:
        return 0.0
    return float(len(clean) / sum(1.0 / v for v in clean))


def paired_bootstrap_delta_by_case(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate_method: str,
    baseline_method: str,
    bucket: str,
    metric: str,
    samples: int = 10_000,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    by_unit: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        if row.get("bucket") != bucket:
            continue
        method = display_method(row)
        if method not in {candidate_method, baseline_method}:
            continue
        value = row.get(metric)
        if value is None:
            continue
        unit = balance_unit_id(row)
        by_unit.setdefault(unit, {}).setdefault(method, []).append(float(value))
    paired = [
        (
            sum(values[candidate_method]) / len(values[candidate_method]),
            sum(values[baseline_method]) / len(values[baseline_method]),
        )
        for values in by_unit.values()
        if candidate_method in values and baseline_method in values
    ]
    if not paired:
        return None
    deltas = [cand - base for cand, base in paired]
    mean_delta = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return {
            "num_edits": 1,
            "num_cases": 1,
            "mean_delta": float(mean_delta),
            "ci_low": float(mean_delta),
            "ci_high": float(mean_delta),
        }
    rng = random.Random(seed)
    draws: List[float] = []
    n = len(deltas)
    for _ in range(samples):
        draw = [deltas[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(draw) / n)
    draws.sort()
    low_idx = int(0.025 * (len(draws) - 1))
    high_idx = int(0.975 * (len(draws) - 1))
    return {
        "num_edits": len(deltas),
        "num_cases": len(deltas),
        "mean_delta": float(mean_delta),
        "ci_low": float(draws[low_idx]),
        "ci_high": float(draws[high_idx]),
    }


def self_normalized_locality(
    locality: float,
    selfloc_base: float,
    epsilon: float = 1e-6,
    clip: bool = False,
) -> float:
    value = float(locality) / max(float(selfloc_base), epsilon)
    if clip:
        value = min(value, 1.0)
    return float(value)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_summary_or_rows(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return load_jsonl(path), {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    per_case_path = payload.get("per_case_results_path")
    rows = load_jsonl(per_case_path) if per_case_path and os.path.exists(per_case_path) else []
    return rows, payload


def group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(display_method(row) if key == "method" else str(row.get(key, "")) for key in keys)
        grouped.setdefault(group_key, []).append(row)
    return grouped


def aggregate_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped = group_rows(rows, keys)
    out: List[Dict[str, Any]] = []
    metric_keys = [
        "exact_rate",
        "greedy_exact",
        "token_f1",
        "malformed_rate",
        "target_false_positive_rate",
        "sparse_guidance_kl",
        "base_margin",
        "gate_activation_rate",
    ]
    for group_key, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        units = sorted({balance_unit_id(item) for item in items})
        row["num_edits"] = len(units)
        row["num_prompt_rows"] = len(items)
        # Kept for compatibility with old readers; this now means edit units.
        row["num_cases"] = len(units)
        for metric in metric_keys:
            by_unit: Dict[str, List[float]] = {}
            for item in items:
                value = item.get(metric)
                if value is None:
                    continue
                by_unit.setdefault(balance_unit_id(item), []).append(float(value))
            unit_means = [
                sum(values) / len(values)
                for values in by_unit.values()
                if values
            ]
            row[f"mean_{metric}"] = mean_or_none(unit_means)
        out.append(row)
    return out


def selection_scores(
    aggregate: Sequence[Dict[str, Any]],
    *,
    selfloc_base: float,
    thresholds: Optional[Dict[str, Any]] = None,
    method_efficiency: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    constraints_enabled: bool = True,
) -> List[Dict[str, Any]]:
    by_method_bucket = {
        (row.get("method"), row.get("bucket")): row
        for row in aggregate
        if "method" in row and "bucket" in row
    }
    thresholds = thresholds or {}
    method_efficiency = method_efficiency or {}
    tfpr_budget = thresholds.get("target_false_positive_rate_budget_by_bucket", {})
    malformed_budget = parse_optional_float(thresholds.get("malformed_span_rate_budget"))
    if malformed_budget is None:
        malformed_budget = 0.05
    gpu_budget = parse_optional_float(thresholds.get("gpu_minutes_per_edit_budget"))
    if gpu_budget is None:
        gpu_budget = 2.0
    methods = sorted({row.get("method") for row in aggregate if row.get("method")})
    scored: List[Dict[str, Any]] = []
    for method in methods:
        rewrite = by_method_bucket.get((method, "rewrite"), {})
        declarative = by_method_bucket.get((method, "declarative_paraphrases"), {})
        qa = by_method_bucket.get((method, "qa_format_generalization"), {})
        locality_rows = [
            by_method_bucket.get((method, "near_locality"), {}),
            by_method_bucket.get((method, "far_locality"), {}),
        ]
        rewrite_exact = rewrite.get("mean_exact_rate")
        paraphrase_exact = declarative.get("mean_exact_rate")
        qa_exact = qa.get("mean_exact_rate")
        locality_exact = mean_or_none(row.get("mean_exact_rate") for row in locality_rows)
        if rewrite_exact is None or paraphrase_exact is None or locality_exact is None:
            score = None
            clipped_self_norm = None
        else:
            clipped_self_norm = self_normalized_locality(locality_exact, selfloc_base, clip=True)
            score = harmonic_mean([rewrite_exact, paraphrase_exact, clipped_self_norm])
        near = by_method_bucket.get((method, "near_locality"), {})
        far = by_method_bucket.get((method, "far_locality"), {})
        near_tfpr = parse_optional_float(near.get("mean_target_false_positive_rate"))
        far_tfpr = parse_optional_float(far.get("mean_target_false_positive_rate"))
        near_tfpr_budget = parse_optional_float(tfpr_budget.get("near_locality"))
        far_tfpr_budget = parse_optional_float(tfpr_budget.get("far_locality"))
        malformed_values = [
            parse_optional_float(row.get("mean_malformed_rate"))
            for row in by_method_bucket.values()
            if row.get("method") == method
        ]
        malformed_values = [value for value in malformed_values if value is not None]
        max_malformed = max(malformed_values) if malformed_values else None
        efficiency = method_efficiency.get(str(method), {})
        gpu_minutes_per_edit = parse_optional_float(efficiency.get("gpu_minutes_per_edit"))

        violations: List[str] = []
        if constraints_enabled:
            if near_tfpr is not None and near_tfpr_budget is not None and near_tfpr > near_tfpr_budget:
                violations.append(f"near_tfpr>{near_tfpr_budget:.6g}")
            if far_tfpr is not None and far_tfpr_budget is not None and far_tfpr > far_tfpr_budget:
                violations.append(f"far_tfpr>{far_tfpr_budget:.6g}")
            if max_malformed is not None and max_malformed > malformed_budget:
                violations.append(f"malformed>{malformed_budget:.6g}")
            if gpu_minutes_per_edit is None:
                violations.append("gpu_minutes_per_edit_missing")
            elif gpu_minutes_per_edit > gpu_budget:
                violations.append(f"gpu_minutes_per_edit>{gpu_budget:.6g}")
        constraint_pass = constraints_enabled and not violations
        feasible_score = score if constraint_pass else None
        scored.append(
            {
                "method": method,
                "rewrite_exact": rewrite_exact,
                "declarative_paraphrases_exact": paraphrase_exact,
                "paraphrase_exact_primary": paraphrase_exact,
                "qa_format_generalization_exact": qa_exact,
                "locality_exact": locality_exact,
                "selfloc_base": selfloc_base,
                "clipped_self_normalized_locality": clipped_self_norm,
                "selection_score": score,
                "near_locality_tfpr": near_tfpr,
                "near_locality_tfpr_budget": near_tfpr_budget,
                "far_locality_tfpr": far_tfpr,
                "far_locality_tfpr_budget": far_tfpr_budget,
                "max_malformed_rate": max_malformed,
                "malformed_span_rate_budget": malformed_budget,
                "gpu_minutes_per_edit": gpu_minutes_per_edit,
                "gpu_minutes_per_edit_budget": gpu_budget,
                "kl_constraint_status": "pending",
                "constraint_pass": constraint_pass,
                "constraint_violations": ";".join(violations),
                "feasible_selection_score": feasible_score,
            }
        )
    return scored


def aggregate_coverage(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    grouped = group_rows(rows, [key])
    out: List[Dict[str, Any]] = []
    coverage_keys = [
        "target_new_first_token_in_base_topk",
        "all_target_new_tokens_in_base_topk",
        "target_true_first_token_in_base_topk",
        "all_target_true_tokens_in_base_topk",
        "all_target_new_tokens_after_candidate_insert",
    ]
    for (group_value,), items in sorted(grouped.items()):
        row = {key: group_value, "num_edits": len(items)}
        for coverage_key in coverage_keys:
            row[coverage_key] = sum(1 for item in items if item.get(coverage_key)) / max(1, len(items))
        out.append(row)
    return out


def gate_activation_summary(aggregate_method_bucket: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = [
        "rewrite",
        "declarative_paraphrases",
        "qa_format_generalization",
        "near_locality",
        "far_locality",
    ]
    methods = sorted({row.get("method") for row in aggregate_method_bucket if row.get("method")})
    by_method_bucket = {
        (row.get("method"), row.get("bucket")): row
        for row in aggregate_method_bucket
    }
    out: List[Dict[str, Any]] = []
    for method in methods:
        row: Dict[str, Any] = {"method": method}
        has_gate = False
        for bucket in buckets:
            value = by_method_bucket.get((method, bucket), {}).get("mean_gate_activation_rate")
            row[f"gate_activation_rate_{bucket}"] = value
            if value is not None:
                has_gate = True
        if has_gate:
            out.append(row)
    return out


def constraint_summary_rows(scores: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [
        "method",
        "constraint_pass",
        "constraint_violations",
        "near_locality_tfpr",
        "near_locality_tfpr_budget",
        "far_locality_tfpr",
        "far_locality_tfpr_budget",
        "max_malformed_rate",
        "malformed_span_rate_budget",
        "gpu_minutes_per_edit",
        "gpu_minutes_per_edit_budget",
        "kl_constraint_status",
    ]
    return [{key: row.get(key) for key in keys} for row in scores]


def load_thresholds(path: str) -> Dict[str, Any]:
    if not path:
        return {
            "target_false_positive_rate_budget_by_bucket": {},
            "malformed_span_rate_budget": 0.05,
            "gpu_minutes_per_edit_budget": 2.0,
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_selfloc_base(cli_selfloc_base: float, thresholds: Dict[str, Any]) -> float:
    threshold_value = parse_optional_float(thresholds.get("selfloc_base"))
    if threshold_value is not None:
        if not (0.0 < threshold_value <= 1.0):
            raise ValueError(f"thresholds selfloc_base must be in (0, 1], got {threshold_value}")
        return float(threshold_value)
    if not (0.0 < float(cli_selfloc_base) <= 1.0):
        raise ValueError(f"selfloc_base must be in (0, 1], got {cli_selfloc_base}")
    return float(cli_selfloc_base)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True, help="summary.json or per_case_results.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--selfloc_base", type=float, default=1.0)
    parser.add_argument("--bootstrap_baseline", type=str, default="base")
    parser.add_argument("--bootstrap_candidates", type=str, nargs="*", default=["mc_bridge", "raw_bridge_gated_subject"])
    parser.add_argument("--bootstrap_pairs", type=str, nargs="*", default=[])
    parser.add_argument("--bootstrap_metric", type=str, default="exact_rate")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--thresholds_path", type=str, default="")
    parser.add_argument("--constraints_enabled", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: List[Dict[str, Any]] = []
    summary_payloads: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    method_efficiency: Dict[str, Dict[str, Optional[float]]] = {}
    for path in args.input:
        rows, payload = load_summary_or_rows(path)
        all_rows.extend(rows)
        if payload:
            summary_payloads.append(payload)
            efficiency = payload.get("efficiency") or {}
            wall_seconds = parse_optional_float(efficiency.get("wall_time_seconds"))
            if rows and wall_seconds is not None:
                methods = sorted({display_method(row) for row in rows if display_method(row)})
                for method in methods:
                    units = {
                        balance_unit_id(row)
                        for row in rows
                        if display_method(row) == method
                    }
                    if units:
                        method_efficiency[method] = {
                            "gpu_minutes_per_edit": wall_seconds / 60.0 / max(1, len(units)) / max(1, len(methods)),
                            "model_evals_per_edit": parse_optional_float(efficiency.get("model_evals_per_edit")),
                        }
            coverage_path = payload.get("candidate_coverage_path")
            if coverage_path and os.path.exists(coverage_path):
                coverage_rows.extend(load_jsonl(coverage_path))

    aggregate_method_bucket = aggregate_rows(all_rows, ["method", "bucket"])
    aggregate_target_length = aggregate_rows(all_rows, ["method", "target_length_bin", "bucket"])
    aggregate_relation = aggregate_rows(all_rows, ["method", "relation_id", "bucket"])
    thresholds = load_thresholds(args.thresholds_path)
    selfloc_base = resolve_selfloc_base(args.selfloc_base, thresholds)
    scores = selection_scores(
        aggregate_method_bucket,
        selfloc_base=selfloc_base,
        thresholds=thresholds,
        method_efficiency=method_efficiency,
        constraints_enabled=bool(args.constraints_enabled),
    )
    feasible_scores = [row for row in scores if row.get("constraint_pass")]
    constraint_summary = constraint_summary_rows(scores)
    gate_summary = gate_activation_summary(aggregate_method_bucket)

    bootstrap: List[Dict[str, Any]] = []
    comparisons: List[Tuple[str, str]] = [
        (candidate, args.bootstrap_baseline)
        for candidate in args.bootstrap_candidates
    ]
    for pair in args.bootstrap_pairs:
        if ":" in pair:
            candidate, baseline = pair.split(":", 1)
        elif ">" in pair:
            candidate, baseline = pair.split(">", 1)
        else:
            raise ValueError(f"Unsupported bootstrap pair format: {pair!r}; use candidate:baseline")
        comparisons.append((candidate.strip(), baseline.strip()))
    comparisons = list(dict.fromkeys(comparisons))
    for candidate, baseline in comparisons:
        for bucket in ["rewrite", "declarative_paraphrases", "near_locality", "far_locality"]:
            stats = paired_bootstrap_delta_by_case(
                all_rows,
                candidate_method=candidate,
                baseline_method=baseline,
                bucket=bucket,
                metric=args.bootstrap_metric,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            bootstrap.append(
                {
                    "candidate_method": candidate,
                    "baseline_method": baseline,
                    "bucket": bucket,
                    "metric": args.bootstrap_metric,
                    **(stats or {}),
                }
            )

    coverage_by_length = aggregate_coverage(coverage_rows, "target_length_bin") if coverage_rows else []
    coverage_by_relation = aggregate_coverage(coverage_rows, "relation_id") if coverage_rows else []

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "aggregate_method_bucket.csv"), aggregate_method_bucket)
    write_csv(os.path.join(args.output_dir, "aggregate_target_length.csv"), aggregate_target_length)
    write_csv(os.path.join(args.output_dir, "aggregate_relation.csv"), aggregate_relation)
    write_csv(os.path.join(args.output_dir, "selection_scores.csv"), scores)
    write_csv(os.path.join(args.output_dir, "selection_scores_feasible.csv"), feasible_scores)
    write_csv(os.path.join(args.output_dir, "constraint_summary.csv"), constraint_summary)
    write_csv(os.path.join(args.output_dir, "paired_bootstrap.csv"), bootstrap)
    write_csv(os.path.join(args.output_dir, "candidate_coverage_by_length.csv"), coverage_by_length)
    write_csv(os.path.join(args.output_dir, "candidate_coverage_by_relation.csv"), coverage_by_relation)
    write_csv(os.path.join(args.output_dir, "gate_activation_summary.csv"), gate_summary)
    write_json(
        os.path.join(args.output_dir, "report_summary.json"),
        {
            "protocol_version": PROTOCOL_VERSION,
            "num_case_rows": len(all_rows),
            "num_summary_payloads": len(summary_payloads),
            "primary_paraphrase_bucket": "declarative_paraphrases",
            "qa_format_generalization": "secondary_diagnostic",
            "constraint_filtering": bool(args.constraints_enabled),
            "constraints_enabled": bool(args.constraints_enabled),
            "analysis_500_used": False,
            "final_test_used": False,
            "bootstrap_unit": "edit_id_if_available_else_case_id",
            "selfloc_base": selfloc_base,
            "thresholds_path": args.thresholds_path,
            "method_efficiency": method_efficiency,
            "artifacts": {
                "aggregate_method_bucket": os.path.join(args.output_dir, "aggregate_method_bucket.csv"),
                "aggregate_target_length": os.path.join(args.output_dir, "aggregate_target_length.csv"),
                "aggregate_relation": os.path.join(args.output_dir, "aggregate_relation.csv"),
                "selection_scores": os.path.join(args.output_dir, "selection_scores.csv"),
                "selection_scores_feasible": os.path.join(args.output_dir, "selection_scores_feasible.csv"),
                "constraint_summary": os.path.join(args.output_dir, "constraint_summary.csv"),
                "paired_bootstrap": os.path.join(args.output_dir, "paired_bootstrap.csv"),
                "gate_activation_summary": os.path.join(args.output_dir, "gate_activation_summary.csv"),
            },
        },
    )
    print(f"[INFO] Wrote report artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
