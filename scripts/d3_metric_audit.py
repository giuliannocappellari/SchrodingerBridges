#!/usr/bin/env python3
"""Audit Direction 3 deployable replay metrics at candidate-group level.

The offline replay summary reports global correlations across all candidates.
For a runtime top-k controller, the meaningful ranking unit is one cached
candidate group: ``(edit_id, prompt_id, step_index, selected_mask_position)``.
This script recomputes value and gate diagnostics by group without loading
LLaDA or touching locked analysis/final splits.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, git_commit, mean, now_utc, read_json, read_jsonl, repo_path, softmax, spearman, write_csv, write_json
from scripts.eval_d3_offline_replay import base_scores, candidate_prediction_scores, gate_score, model_parts, row_target_score, target_rank
from scripts.train_d3_bridge_controller import VALUE_FEATURE_NAMES, annotate_gate_context, candidate_features, finite_scores, target_candidate_positions


DEFAULT_CACHE_DIR = D3_ROOT / "teacher_cache_train100_val50_v1"
DEFAULT_CONTROLLER_DIR = D3_ROOT / "offline_train_deployable_value_gate_train100_val50_v2"
DEFAULT_OUTPUT_DIR = D3_ROOT / "offline_replay_deployable_train100_val50_v2_metric_audit"
POSITIVE_PROMPT_TYPES = {"rewrite", "declarative_paraphrase"}
SAME_SUBJECT_PROMPT_TYPES = {"same_subject_different_relation", "same_subject_template"}
LOCALITY_PROMPT_TYPES = {"near_locality", "far_locality"}


def rank_order(values: Sequence[float]) -> List[int]:
    return sorted(range(len(values)), key=lambda idx: float(values[idx]), reverse=True)


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else 0.0


def pairwise_accuracy(teacher: Sequence[float], student: Sequence[float]) -> float:
    total = 0
    correct = 0
    for i in range(len(teacher)):
        for j in range(i + 1, len(teacher)):
            dt = (teacher[i] > teacher[j]) - (teacher[i] < teacher[j])
            ds = (student[i] > student[j]) - (student[i] < student[j])
            if dt == 0:
                continue
            total += 1
            if dt == ds:
                correct += 1
    return correct / total if total else 0.0


def ndcg_at_k(teacher: Sequence[float], student: Sequence[float], k: int = 8) -> float:
    min_score = min(float(x) for x in teacher)
    relevance = [max(0.0, float(score) - min_score) for score in teacher]
    if max(relevance, default=0.0) == 0.0:
        relevance = [1.0 for _ in teacher]

    def dcg(order: Sequence[int]) -> float:
        total = 0.0
        for rank, idx in enumerate(order[:k], start=1):
            total += relevance[idx] / math.log2(rank + 1)
        return total

    ideal = dcg(rank_order(relevance))
    return dcg(rank_order(student)) / ideal if ideal else 0.0


def centered_softmax(scores: Sequence[float], tau: float) -> List[float]:
    mu = mean([float(x) for x in scores])
    return softmax([(float(x) - mu) / max(float(tau), 1e-6) for x in scores])


def kl_div(p: Sequence[float], q: Sequence[float]) -> float:
    eps = 1e-12
    return sum(float(pi) * math.log((float(pi) + eps) / (float(qi) + eps)) for pi, qi in zip(p, q))


def js_div(p: Sequence[float], q: Sequence[float]) -> float:
    m = [(float(pi) + float(qi)) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def target_rank_from_scores(row: Mapping[str, Any], scores: Sequence[float]) -> int:
    return target_rank(row, scores)


def ablated_candidate_scores(row: Mapping[str, Any], model: Mapping[str, Any], ablate_target_indicator: bool) -> List[float]:
    value_weights = model.get("value_weights")
    if value_weights is None:
        return base_scores(row)
    scores: List[float] = []
    target_idx = VALUE_FEATURE_NAMES.index("candidate_is_target_new_token") if "candidate_is_target_new_token" in VALUE_FEATURE_NAMES else -1
    for candidate_index in range(len(base_scores(row))):
        features = list(candidate_features(row, candidate_index))
        if ablate_target_indicator and target_idx >= 0:
            features[target_idx] = 0.0
        scores.append(sum(float(w) * float(x) for w, x in zip(value_weights, features)))
    if model.get("gate_weights") is not None and model.get("controller_type") == "value_gate":
        gate = gate_score(row, model, scores)
        scores = [score * gate for score in scores]
    return scores


def group_metrics(row: Mapping[str, Any], model: Mapping[str, Any], controller: str, tau: float) -> Dict[str, Any]:
    teacher = finite_scores(row, ["raw_bridge_scores_top_k", "raw_bridge_scores"])
    student = candidate_prediction_scores(row, model)
    base = base_scores(row)
    teacher_dist = centered_softmax(teacher, tau)
    student_dist = centered_softmax(student, tau)
    teacher_top = rank_order(teacher)[0]
    student_top = rank_order(student)[0]
    teacher_top3 = set(rank_order(teacher)[:3])
    student_top3 = set(rank_order(student)[:3])
    edit_id = str(row.get("edit_id") or row.get("case_id"))
    group_key = "|".join(
        [
            edit_id,
            str(row.get("prompt_id")),
            str(row.get("step_index")),
            str(row.get("selected_mask_position")),
        ]
    )
    return {
        "controller": controller,
        "split": str(row.get("split_role")),
        "group_key": group_key,
        "edit_id": edit_id,
        "prompt_id": row.get("prompt_id"),
        "prompt_type": row.get("prompt_type"),
        "step_index": row.get("step_index"),
        "selected_mask_position": row.get("selected_mask_position"),
        "target_length_bin": row.get("target_length_bin"),
        "groupwise_spearman": spearman(teacher, student),
        "kendall_tau": kendall_tau(teacher, student),
        "pairwise_ranking_accuracy": pairwise_accuracy(teacher, student),
        "ndcg_at_8": ndcg_at_k(teacher, student, k=8),
        "teacher_student_kl": kl_div(teacher_dist, student_dist),
        "teacher_student_js": js_div(teacher_dist, student_dist),
        "top1_agreement": float(teacher_top == student_top),
        "top3_overlap": len(teacher_top3 & student_top3) / max(1, len(teacher_top3)),
        "target_rank_student": target_rank_from_scores(row, student),
        "target_rank_base": target_rank_from_scores(row, base),
        "target_top3_student": float(target_rank_from_scores(row, student) <= 3),
        "target_top3_base": float(target_rank_from_scores(row, base) <= 3),
    }


def summarize(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field) for field in group_fields)].append(row)
    metric_names = [
        "groupwise_spearman",
        "kendall_tau",
        "pairwise_ranking_accuracy",
        "ndcg_at_8",
        "teacher_student_kl",
        "teacher_student_js",
        "top1_agreement",
        "top3_overlap",
        "target_top3_student",
        "target_top3_base",
    ]
    out: List[Dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=lambda item: str(item[0])):
        payload = {field: value for field, value in zip(group_fields, key)}
        payload["num_groups"] = len(items)
        for metric in metric_names:
            values = [float(row[metric]) for row in items]
            payload[f"mean_{metric}"] = mean(values)
            payload[f"median_{metric}"] = sorted(values)[len(values) // 2] if values else 0.0
        payload["target_top3_improvement_over_base"] = payload["mean_target_top3_student"] - payload["mean_target_top3_base"]
        out.append(payload)
    return out


def bootstrap_ci(rows: Sequence[Mapping[str, Any]], metric: str, trials: int, seed: int) -> Dict[str, float]:
    by_edit: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_edit[str(row["edit_id"])].append(row)
    edit_ids = sorted(by_edit)
    rng = random.Random(seed)
    values: List[float] = []
    for _ in range(trials):
        sample_rows: List[Mapping[str, Any]] = []
        for _ in edit_ids:
            sample_rows.extend(by_edit[rng.choice(edit_ids)])
        values.append(mean([float(row[metric]) for row in sample_rows]))
    values.sort()
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    low_idx = int(0.025 * (len(values) - 1))
    high_idx = int(0.975 * (len(values) - 1))
    return {
        "mean": mean([float(row[metric]) for row in rows]),
        "ci_low": values[low_idx],
        "ci_high": values[high_idx],
    }


def target_indicator_ablation_rows(rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any], controller: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        normal = ablated_candidate_scores(row, model, ablate_target_indicator=False)
        ablated = ablated_candidate_scores(row, model, ablate_target_indicator=True)
        base = base_scores(row)
        buckets[str(row.get("prompt_type"))].append(
            {
                "normal_target_top3": float(target_rank_from_scores(row, normal) <= 3),
                "ablated_target_top3": float(target_rank_from_scores(row, ablated) <= 3),
                "base_target_top3": float(target_rank_from_scores(row, base) <= 3),
            }
        )
    out: List[Dict[str, Any]] = []
    for prompt_type, items in sorted(buckets.items()):
        normal = mean([item["normal_target_top3"] for item in items])
        ablated = mean([item["ablated_target_top3"] for item in items])
        base = mean([item["base_target_top3"] for item in items])
        out.append(
            {
                "controller": controller,
                "prompt_type": prompt_type,
                "num_groups": len(items),
                "normal_target_top3": normal,
                "ablated_target_top3": ablated,
                "base_target_top3": base,
                "normal_minus_ablated": normal - ablated,
                "ablated_minus_base": ablated - base,
            }
        )
    return out


def gate_error_rows(rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any], controller: str) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for row in rows:
        scores = candidate_prediction_scores(row, model)
        label = int(str(row.get("prompt_type")) in POSITIVE_PROMPT_TYPES or int(row.get("label", 0)) == 1)
        scored.append(
            {
                "controller": controller,
                "edit_id": row.get("edit_id") or row.get("case_id"),
                "prompt_id": row.get("prompt_id"),
                "prompt_type": row.get("prompt_type"),
                "label": label,
                "gate_score": gate_score(row, model, scores),
            }
        )
    return sorted(scored, key=lambda row: abs(float(row["label"]) - float(row["gate_score"])), reverse=True)[:200]


def negative_guidance_rows(rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any], controller: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        scores = candidate_prediction_scores(row, model)
        g = gate_score(row, model, scores)
        prompt_type = str(row.get("prompt_type"))
        guidance = abs(row_target_score(row, scores) * g)
        buckets[prompt_type].append(guidance)
    out = []
    pos = [v for pt, vals in buckets.items() if pt in POSITIVE_PROMPT_TYPES for v in vals]
    pos_mean = mean(pos)
    for prompt_type, vals in sorted(buckets.items()):
        out.append(
            {
                "controller": controller,
                "prompt_type": prompt_type,
                "num_groups": len(vals),
                "mean_abs_guidance": mean(vals),
                "ratio_to_positive": (mean(vals) / pos_mean) if pos_mean else 0.0,
            }
        )
    return out


def write_metric_definition(path: Path) -> None:
    text = """# Direction 3 Stage 1B.2A Metric Definition Audit

This audit recomputes value-controller ranking metrics within each top-k
candidate group instead of flattening unrelated groups.

Candidate group key:

```text
edit_id | prompt_id | step_index | selected_mask_position
```

Primary teacher scores:

```text
raw_bridge_scores_top_k
```

Distributional metrics use within-group centered softmax:

```text
p(v | x_t) = softmax((score(v) - mean(score)) / tau)
```

Reported groupwise metrics:

- Spearman over the 8 candidates
- Kendall tau
- pairwise ranking accuracy
- NDCG@8
- teacher/student KL and JS divergence
- top-1 agreement
- top-3 overlap
- target-token rank

The audit is diagnostic only. It does not override the prior offline
scientific acceptance result, because the deployable v2 gate and negative
guidance criteria still have to pass before any GPU decode.
"""
    repo_path(path).write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--controller_dir", type=Path, default=DEFAULT_CONTROLLER_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--controller", type=str, default="value_gate")
    parser.add_argument("--tau", type=float, default=8.0)
    parser.add_argument("--bootstrap_trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    repo_path(out).mkdir(parents=True, exist_ok=True)
    cache_summary = read_json(args.teacher_cache_dir / "report_summary.json")
    controller_summary = read_json(args.controller_dir / "report_summary.json")
    weights = read_json(args.controller_dir / "controller_weights.json")
    train_rows = annotate_gate_context(read_jsonl(args.teacher_cache_dir / "teacher_states_train.jsonl"))
    val_rows = annotate_gate_context(read_jsonl(args.teacher_cache_dir / "teacher_states_val.jsonl"))
    models = model_parts(weights)
    if args.controller not in models:
        raise AssertionError(f"Missing controller {args.controller}; available={sorted(models)}")
    model = models[args.controller]
    all_group_rows: List[Dict[str, Any]] = []
    for split, rows in [("train", train_rows), ("val", val_rows)]:
        for row in rows:
            metric = group_metrics({**row, "split_role": split}, model, args.controller, tau=float(args.tau))
            metric["split"] = split
            all_group_rows.append(metric)
    val_group_rows = [row for row in all_group_rows if row["split"] == "val"]
    group_summary = summarize(all_group_rows, ["controller", "split"])
    prompt_summary = summarize(all_group_rows, ["controller", "split", "prompt_type"])
    step_summary = summarize(all_group_rows, ["controller", "split", "step_index"])
    length_summary = summarize(all_group_rows, ["controller", "split", "target_length_bin"])
    bootstrap = {
        metric: bootstrap_ci(val_group_rows, metric, int(args.bootstrap_trials), int(args.seed))
        for metric in ["groupwise_spearman", "kendall_tau", "pairwise_ranking_accuracy", "ndcg_at_8", "top1_agreement", "top3_overlap"]
    }
    target_ablation = target_indicator_ablation_rows(val_rows, model, args.controller)
    gate_errors = gate_error_rows(val_rows, model, args.controller)
    negative_guidance = negative_guidance_rows(val_rows, model, args.controller)
    report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 Stage 1B.2A metric and shortcut audit",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "teacher_cache_dir": str(args.teacher_cache_dir),
        "controller_dir": str(args.controller_dir),
        "controller": args.controller,
        "fake_model": False,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "source_cache_analysis_500_used": bool(cache_summary.get("analysis_500_used", False)),
        "source_cache_final_test_used": bool(cache_summary.get("final_test_used", False)),
        "source_controller_analysis_500_used": bool(controller_summary.get("analysis_500_used", False)),
        "source_controller_final_test_used": bool(controller_summary.get("final_test_used", False)),
        "num_train_groups": len(train_rows),
        "num_val_groups": len(val_rows),
        "macro_groupwise_summary": group_summary,
        "val_bootstrap_ci_by_edit_id": bootstrap,
        "decision_note": "Diagnostic only; does not authorize Stage 2A GPU decode.",
        "artifacts": {
            "groupwise_ranking_metrics": str(out / "groupwise_ranking_metrics.csv"),
            "metric_definition_audit": str(out / "metric_definition_audit.md"),
            "target_indicator_ablation": str(out / "target_indicator_ablation.csv"),
            "per_prompt_type_metrics": str(out / "per_prompt_type_metrics.csv"),
            "per_step_metrics": str(out / "per_step_metrics.csv"),
            "per_target_length_metrics": str(out / "per_target_length_metrics.csv"),
            "gate_error_analysis": str(out / "gate_error_analysis.csv"),
            "negative_guidance_analysis": str(out / "negative_guidance_analysis.csv"),
        },
    }
    write_csv(out / "groupwise_ranking_metrics.csv", all_group_rows)
    write_csv(out / "target_indicator_ablation.csv", target_ablation)
    write_csv(out / "per_prompt_type_metrics.csv", prompt_summary)
    write_csv(out / "per_step_metrics.csv", step_summary)
    write_csv(out / "per_target_length_metrics.csv", length_summary)
    write_csv(out / "gate_error_analysis.csv", gate_errors)
    write_csv(out / "negative_guidance_analysis.csv", negative_guidance)
    write_metric_definition(out / "metric_definition_audit.md")
    write_json(out / "report_summary.json", report)
    print(f"[INFO] Wrote D3 metric audit to {out}")


if __name__ == "__main__":
    main()
