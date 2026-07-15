#!/usr/bin/env python3
"""Evaluate Direction 3 controller outputs in offline replay.

This script consumes cached teacher rows and controller weights only. It does
not import or load LLaDA, including when ``--fake_model 0`` is used for a real
teacher-cache smoke.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import (
    D3_PROTOCOL_VERSION,
    D3_ROOT,
    auc_score,
    git_commit,
    mean,
    now_utc,
    read_json,
    read_jsonl,
    repo_path,
    spearman,
    write_csv,
    write_json,
)
from scripts.train_d3_bridge_controller import (
    GATE_FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    annotate_gate_context,
    array_value,
    candidate_features,
    candidate_ids,
    finite_scores,
    gate_probability,
    row_gate_features,
    sigmoid,
    predict,
    target_candidate_positions,
)


POSITIVE_PROMPT_TYPES = {"rewrite", "declarative_paraphrase"}
SAME_SUBJECT_PROMPT_TYPES = {"same_subject_different_relation", "same_subject_template"}
LOCALITY_PROMPT_TYPES = {"near_locality", "far_locality"}
THRESHOLDS = [round(i / 20.0, 2) for i in range(21)]


def dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(float(w) * float(x) for w, x in zip(weights, features))


def model_parts(weights_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    controllers = weights_payload.get("controllers")
    if isinstance(controllers, dict) and controllers:
        return {str(key): dict(value) for key, value in controllers.items()}
    # Backwards compatibility for earlier single-controller fake-mode artifacts.
    return {
        "value_gate": {
            "controller_type": "value_gate",
            "value_feature_names": weights_payload.get("feature_names", VALUE_FEATURE_NAMES),
            "value_weights": weights_payload["weights"],
        }
    }


def candidate_prediction_scores(row: Mapping[str, Any], model: Mapping[str, Any]) -> List[float]:
    value_weights = model.get("value_weights")
    gate_weights = model.get("gate_weights")
    if value_weights is not None:
        scores = [dot(value_weights, candidate_features(row, i)) for i in range(len(candidate_ids(row)))]
    else:
        scores = finite_scores(row, ["base_logits_top_k", "base_logits"])
    if gate_weights is not None and model.get("controller_type") == "value_gate":
        gate = gate_probability(row, gate_weights)
        scores = [score * gate for score in scores]
    return [float(score) for score in scores]


def gate_score(row: Mapping[str, Any], model: Mapping[str, Any], pred_scores: Sequence[float]) -> float:
    gate_weights = model.get("gate_weights")
    if gate_weights is not None:
        return gate_probability(row, gate_weights)
    positions = target_candidate_positions(row)
    if positions:
        return sigmoid(max(pred_scores[idx] for idx in positions if idx < len(pred_scores)))
    return sigmoid(max(pred_scores) if pred_scores else 0.0)


def target_rank(row: Mapping[str, Any], scores: Sequence[float]) -> int:
    target_positions = set(target_candidate_positions(row))
    if not target_positions:
        return len(scores) + 1
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(order, start=1):
        if idx in target_positions:
            return rank
    return len(scores) + 1


def row_target_score(row: Mapping[str, Any], scores: Sequence[float]) -> float:
    positions = target_candidate_positions(row)
    if not positions:
        return max(scores) if scores else 0.0
    return max(scores[idx] for idx in positions)


def base_scores(row: Mapping[str, Any]) -> List[float]:
    return finite_scores(row, ["base_logits_top_k", "base_logits"])


def eval_rows(rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any]) -> Dict[str, Any]:
    raw_all_scores: List[float] = []
    pred_all_scores: List[float] = []
    top1_agreements: List[float] = []
    target_top3: List[float] = []
    teacher_top3_overlap: List[float] = []
    base_target_top3: List[float] = []
    gate_labels: List[int] = []
    gate_scores: List[float] = []
    same_subject_gate_labels: List[int] = []
    same_subject_gate_scores: List[float] = []
    negative_guidance: List[float] = []
    positive_guidance_abs: List[float] = []
    negative_guidance_abs: List[float] = []
    same_subject_negative_guidance: List[float] = []
    locality_negative_guidance: List[float] = []
    target_advantages_same_subject: List[float] = []

    for row in rows:
        pred_scores = candidate_prediction_scores(row, model)
        raw_scores = [
            array_value(row, ["raw_bridge_scores_top_k", "raw_bridge_scores"], idx)
            for idx in range(len(pred_scores))
        ]
        base = base_scores(row)
        raw_top = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        pred_top = max(range(len(pred_scores)), key=lambda i: pred_scores[i])
        raw_top3 = set(sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:3])
        pred_top3 = set(sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)[:3])
        raw_all_scores.extend(float(value) for value in raw_scores)
        pred_all_scores.extend(float(value) for value in pred_scores)
        top1_agreements.append(float(raw_top == pred_top))
        teacher_top3_overlap.append(len(raw_top3 & pred_top3) / max(1, len(raw_top3)))
        target_top3.append(float(target_rank(row, pred_scores) <= 3))
        base_target_top3.append(float(target_rank(row, base) <= 3))
        label = int(str(row.get("prompt_type")) in POSITIVE_PROMPT_TYPES or int(row.get("label", 0)) == 1)
        prompt_type = str(row.get("prompt_type"))
        gscore = gate_score(row, model, pred_scores)
        gate_labels.append(label)
        gate_scores.append(gscore)
        if label or prompt_type in SAME_SUBJECT_PROMPT_TYPES:
            same_subject_gate_labels.append(label)
            same_subject_gate_scores.append(gscore)
        if label == 0:
            guidance = row_target_score(row, pred_scores) * gscore
            negative_guidance.append(guidance)
            negative_guidance_abs.append(abs(guidance))
            if prompt_type in SAME_SUBJECT_PROMPT_TYPES:
                same_subject_negative_guidance.append(guidance)
                target_advantages_same_subject.append(row_target_score(row, pred_scores) - row_target_score(row, base))
            if prompt_type in LOCALITY_PROMPT_TYPES:
                locality_negative_guidance.append(guidance)
        else:
            positive_guidance_abs.append(abs(row_target_score(row, pred_scores) * gscore))

    pos_guidance = mean(positive_guidance_abs)
    neg_guidance = mean(negative_guidance_abs)
    return {
        "num_rows": len(rows),
        "bridge_score_spearman": spearman(raw_all_scores, pred_all_scores),
        "top1_agreement_with_raw_bridge": mean(top1_agreements),
        "teacher_top3_overlap": mean(teacher_top3_overlap),
        "target_token_ranked_top3_rate": mean(target_top3),
        "base_target_token_ranked_top3_rate": mean(base_target_top3),
        "target_token_top3_improvement_over_base": mean(target_top3) - mean(base_target_top3),
        "same_subject_gate_auc": auc_score(same_subject_gate_labels, same_subject_gate_scores),
        "all_prompt_gate_auc": auc_score(gate_labels, gate_scores),
        "locality_negative_average_guidance": mean(locality_negative_guidance),
        "all_negative_average_guidance": mean(negative_guidance),
        "same_subject_negative_average_guidance": mean(same_subject_negative_guidance),
        "same_subject_target_advantage_vs_base": mean(target_advantages_same_subject),
        "mean_abs_guidance_positive": pos_guidance,
        "mean_abs_guidance_negative": neg_guidance,
        "negative_to_positive_guidance_ratio": (neg_guidance / pos_guidance) if pos_guidance else 0.0,
        "offline_replay_rewrite_paraphrase_gain_proxy": mean(target_top3),
        "offline_replay_target_false_positive_proxy": mean(negative_guidance),
    }


def grouped_metrics(
    *,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    models: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metric_rows: List[Dict[str, Any]] = []
    agreement_rows: List[Dict[str, Any]] = []
    negative_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    for controller_name, model in models.items():
        for split, rows in rows_by_split.items():
            metrics = eval_rows(rows, model)
            metric_rows.append({"controller": controller_name, "split": split, **metrics})
            by_prompt: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                by_prompt[str(row.get("prompt_type"))].append(row)
            for prompt_type, prompt_rows in sorted(by_prompt.items()):
                prompt_metrics = eval_rows(prompt_rows, model)
                agreement_rows.append(
                    {
                        "controller": controller_name,
                        "split": split,
                        "prompt_type": prompt_type,
                        "num_rows": prompt_metrics["num_rows"],
                        "bridge_score_spearman": prompt_metrics["bridge_score_spearman"],
                        "top1_agreement_with_raw_bridge": prompt_metrics["top1_agreement_with_raw_bridge"],
                        "teacher_top3_overlap": prompt_metrics["teacher_top3_overlap"],
                    }
                )
                ranking_rows.append(
                    {
                        "controller": controller_name,
                        "split": split,
                        "prompt_type": prompt_type,
                        "num_rows": prompt_metrics["num_rows"],
                        "target_token_ranked_top3_rate": prompt_metrics["target_token_ranked_top3_rate"],
                        "base_target_token_ranked_top3_rate": prompt_metrics["base_target_token_ranked_top3_rate"],
                        "target_token_top3_improvement_over_base": prompt_metrics["target_token_top3_improvement_over_base"],
                    }
                )
                if prompt_type not in POSITIVE_PROMPT_TYPES:
                    negative_rows.append(
                        {
                            "controller": controller_name,
                            "split": split,
                            "prompt_type": prompt_type,
                            "num_rows": prompt_metrics["num_rows"],
                            "negative_average_guidance": prompt_metrics["all_negative_average_guidance"],
                            "same_subject_negative_average_guidance": prompt_metrics["same_subject_negative_average_guidance"],
                            "locality_negative_average_guidance": prompt_metrics["locality_negative_average_guidance"],
                            "same_subject_target_advantage_vs_base": prompt_metrics["same_subject_target_advantage_vs_base"],
                        }
                    )
    return metric_rows, agreement_rows, negative_rows, ranking_rows


def gate_threshold_rows(rows: Sequence[Mapping[str, Any]], models: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for controller_name, model in models.items():
        scored = []
        for row in rows:
            pred_scores = candidate_prediction_scores(row, model)
            label = int(str(row.get("prompt_type")) in POSITIVE_PROMPT_TYPES or int(row.get("label", 0)) == 1)
            prompt_type = str(row.get("prompt_type"))
            scored.append((label, prompt_type, gate_score(row, model, pred_scores)))
        positives = [item for item in scored if item[0] == 1]
        rewrites = [item for item in scored if item[1] == "rewrite"]
        paraphrases = [item for item in scored if item[1] == "declarative_paraphrase"]
        same_subject = [item for item in scored if item[1] in SAME_SUBJECT_PROMPT_TYPES]
        generation = [item for item in scored if item[1] == "generation"]
        locality = [item for item in scored if item[1] in LOCALITY_PROMPT_TYPES]
        near = [item for item in scored if item[1] == "near_locality"]
        far = [item for item in scored if item[1] == "far_locality"]
        for threshold in THRESHOLDS:
            def activation(items: Sequence[Tuple[int, str, float]]) -> float:
                return mean([float(score >= threshold) for _, _, score in items])

            out.append(
                {
                    "controller": controller_name,
                    "threshold": threshold,
                    "positive_activation": activation(positives),
                    "rewrite_activation": activation(rewrites),
                    "paraphrase_activation": activation(paraphrases),
                    "same_subject_negative_activation": activation(same_subject),
                    "generation_activation": activation(generation),
                    "locality_activation": activation(locality),
                    "near_locality_activation": activation(near),
                    "far_locality_activation": activation(far),
                    "num_positive": len(positives),
                    "num_rewrite": len(rewrites),
                    "num_paraphrase": len(paraphrases),
                    "num_same_subject_negative": len(same_subject),
                    "num_generation": len(generation),
                    "num_locality": len(locality),
                }
            )
    return out


def select_gate_threshold(rows: Sequence[Mapping[str, Any]], controller: str) -> Dict[str, Any]:
    candidates = [row for row in rows if row["controller"] == controller]
    passing = [
        row
        for row in candidates
        if float(row.get("rewrite_activation", 0.0)) >= 0.90
        and float(row.get("paraphrase_activation", 0.0)) >= 0.85
        and float(row.get("same_subject_negative_activation", 1.0)) <= 0.05
        and float(row.get("near_locality_activation", 1.0)) <= 0.02
        and float(row.get("far_locality_activation", 1.0)) <= 0.02
    ]
    if passing:
        return max(passing, key=lambda row: (float(row["paraphrase_activation"]), float(row["rewrite_activation"]), -float(row["threshold"])))
    if not candidates:
        return {
            "controller": controller,
            "threshold": "",
            "gate_threshold_acceptance_pass": False,
            "reason": "no_threshold_rows",
        }
    best = min(
        candidates,
        key=lambda row: (
            max(0.0, 0.90 - float(row.get("rewrite_activation", 0.0)))
            + max(0.0, 0.85 - float(row.get("paraphrase_activation", 0.0)))
            + max(0.0, float(row.get("same_subject_negative_activation", 1.0)) - 0.05)
            + max(0.0, float(row.get("near_locality_activation", 1.0)) - 0.02)
            + max(0.0, float(row.get("far_locality_activation", 1.0)) - 0.02)
        ),
    )
    return {**best, "gate_threshold_acceptance_pass": False, "reason": "no_threshold_met_all_constraints"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_cache_dir", type=Path, default=D3_ROOT / "fake_teacher_cache_v1")
    parser.add_argument("--controller_dir", type=Path, default=D3_ROOT / "fake_controller_train_v1")
    parser.add_argument("--output_dir", type=Path, default=D3_ROOT / "fake_offline_replay_v1")
    parser.add_argument("--fake_model", type=int, default=0)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    weights_payload = read_json(args.controller_dir / "controller_weights.json")
    if weights_payload.get("value_feature_names", weights_payload.get("feature_names")) != VALUE_FEATURE_NAMES:
        raise AssertionError("Controller feature names do not match replay code")
    if weights_payload.get("gate_feature_names", GATE_FEATURE_NAMES) != GATE_FEATURE_NAMES:
        raise AssertionError("Gate feature names do not match replay code")
    models = model_parts(weights_payload)
    train_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_train.jsonl")
    val_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_val.jsonl")
    train_rows = annotate_gate_context(train_rows)
    val_rows = annotate_gate_context(val_rows)
    rows_by_split = {"train": train_rows, "val": val_rows}
    metric_rows, agreement_rows, negative_rows, ranking_rows = grouped_metrics(rows_by_split=rows_by_split, models=models)
    threshold_rows = gate_threshold_rows(val_rows, models)
    default_controller = str(weights_payload.get("default_controller") or ("value_gate" if "value_gate" in models else next(iter(models))))
    train_metrics = next(row for row in metric_rows if row["controller"] == default_controller and row["split"] == "train")
    val_metrics = next(row for row in metric_rows if row["controller"] == default_controller and row["split"] == "val")
    selected_threshold = select_gate_threshold(threshold_rows, default_controller)
    gate_threshold_pass = bool(selected_threshold.get("gate_threshold_acceptance_pass", True)) if "gate_threshold_acceptance_pass" in selected_threshold else True

    pass_criteria = {
        "bridge_score_spearman_ge_0_4": val_metrics["bridge_score_spearman"] >= 0.4,
        "teacher_top1_agreement_ge_0_4": val_metrics["top1_agreement_with_raw_bridge"] >= 0.4,
        "teacher_top3_overlap_ge_0_65": val_metrics["teacher_top3_overlap"] >= 0.65,
        "target_top3_improvement_ge_0_15": val_metrics["target_token_top3_improvement_over_base"] >= 0.15,
        "same_subject_gate_auc_ge_0_85": val_metrics["same_subject_gate_auc"] >= 0.85,
        "gate_threshold_acceptance_pass": gate_threshold_pass,
        "negative_guidance_ratio_le_0_15": val_metrics["negative_to_positive_guidance_ratio"] <= 0.15,
        "same_subject_target_advantage_nonpositive": val_metrics["same_subject_target_advantage_vs_base"] <= 0.0,
    }
    fake_model = bool(args.fake_model)
    stage = "Direction 3 fake offline replay" if fake_model else "Direction 3 Stage 1B real-cache offline replay"
    payload = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": stage,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": fake_model,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "teacher_cache_dir": str(args.teacher_cache_dir),
        "controller_dir": str(args.controller_dir),
        "controllers": sorted(models),
        "default_controller": default_controller,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "selected_gate_threshold": selected_threshold,
        "pass_criteria": pass_criteria,
        "acceptance_pass": all(pass_criteria.values()),
        "artifacts": {
            "offline_replay_metrics": str(args.output_dir / "offline_replay_metrics.csv"),
            "gate_threshold_sweep": str(args.output_dir / "gate_threshold_sweep.csv"),
            "controller_candidate_agreement": str(args.output_dir / "controller_candidate_agreement.csv"),
            "negative_guidance_diagnostics": str(args.output_dir / "negative_guidance_diagnostics.csv"),
            "target_token_ranking": str(args.output_dir / "target_token_ranking.csv"),
        },
    }
    write_csv(args.output_dir / "offline_replay_metrics.csv", metric_rows)
    write_csv(args.output_dir / "gate_threshold_sweep.csv", threshold_rows)
    write_csv(args.output_dir / "controller_candidate_agreement.csv", agreement_rows)
    write_csv(args.output_dir / "negative_guidance_diagnostics.csv", negative_rows)
    write_csv(args.output_dir / "target_token_ranking.csv", ranking_rows)
    write_json(args.output_dir / "offline_replay_metrics.json", payload)
    write_json(args.output_dir / "report_summary.json", payload)
    print(f"[INFO] Wrote Direction 3 offline replay output to {args.output_dir}")


if __name__ == "__main__":
    main()
