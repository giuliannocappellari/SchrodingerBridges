#!/usr/bin/env python3
"""Evaluate fake Direction 3 controller outputs in offline replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, auc_score, git_commit, mean, now_utc, read_json, read_jsonl, repo_path, spearman, write_json
from scripts.train_d3_bridge_controller import FEATURE_NAMES, predict, row_features


def candidate_prediction_scores(row: Dict[str, Any], weights: Sequence[float]) -> List[float]:
    scores: List[float] = []
    for i in range(len(row["top_k_candidate_token_ids"])):
        pseudo = dict(row)
        pseudo["base_logits_top_k"] = [row["base_logits_top_k"][i]]
        pseudo["myopic_scores_top_k"] = [row["myopic_scores_top_k"][i]]
        pseudo["no_rollout_scores_top_k"] = [row["no_rollout_scores_top_k"][i]]
        scores.append(predict(weights, row_features(pseudo)))
    return scores


def target_rank(scores: Sequence[float]) -> int:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order.index(0) + 1


def eval_rows(rows: Sequence[Dict[str, Any]], weights: Sequence[float]) -> Dict[str, Any]:
    raw_top_scores: List[float] = []
    pred_top_scores: List[float] = []
    top1_agreements: List[float] = []
    target_top3: List[float] = []
    gate_labels: List[int] = []
    gate_scores: List[float] = []
    negative_guidance: List[float] = []

    for row in rows:
        pred_scores = candidate_prediction_scores(row, weights)
        raw_scores = row["raw_bridge_scores_top_k"]
        raw_top = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        pred_top = max(range(len(pred_scores)), key=lambda i: pred_scores[i])
        raw_top_scores.append(float(raw_scores[0]))
        pred_top_scores.append(float(pred_scores[0]))
        top1_agreements.append(float(raw_top == pred_top))
        target_top3.append(float(target_rank(pred_scores) <= 3))
        label = int(row.get("label", 0))
        gate_labels.append(label)
        gate_scores.append(pred_scores[0])
        if label == 0:
            negative_guidance.append(pred_scores[0])

    return {
        "num_rows": len(rows),
        "bridge_score_spearman": spearman(raw_top_scores, pred_top_scores),
        "top1_agreement_with_raw_bridge": mean(top1_agreements),
        "target_token_ranked_top3_rate": mean(target_top3),
        "same_subject_gate_auc": auc_score(gate_labels, gate_scores),
        "locality_negative_average_guidance": mean(negative_guidance),
        "offline_replay_rewrite_paraphrase_gain_proxy": mean(target_top3),
        "offline_replay_target_false_positive_proxy": mean(negative_guidance),
    }


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
    if not bool(args.fake_model):
        raise SystemExit("Real offline replay requires a trained controller artifact. Re-run with --fake_model 1 locally.")
    repo_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    weights_payload = read_json(args.controller_dir / "controller_weights.json")
    if weights_payload.get("feature_names") != FEATURE_NAMES:
        raise AssertionError("Controller feature names do not match replay code")
    weights = [float(x) for x in weights_payload["weights"]]
    train_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_train.jsonl")
    val_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_val.jsonl")
    train_metrics = eval_rows(train_rows, weights)
    val_metrics = eval_rows(val_rows, weights)

    pass_criteria = {
        "bridge_score_spearman_gt_0_4": val_metrics["bridge_score_spearman"] > 0.4,
        "target_top3_rate_gt_0_5": val_metrics["target_token_ranked_top3_rate"] > 0.5,
        "same_subject_gate_auc_gt_0_85": val_metrics["same_subject_gate_auc"] > 0.85,
        "locality_negative_average_guidance_lt_0_5": val_metrics["locality_negative_average_guidance"] < 0.5,
    }
    payload = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 fake offline replay",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": True,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "pass_criteria": pass_criteria,
        "acceptance_pass": all(pass_criteria.values()),
        "artifacts": {
            "offline_replay_metrics": str(args.output_dir / "offline_replay_metrics.json"),
        },
    }
    write_json(args.output_dir / "offline_replay_metrics.json", payload)
    write_json(args.output_dir / "report_summary.json", payload)
    print(f"[INFO] Wrote fake Direction 3 offline replay output to {args.output_dir}")


if __name__ == "__main__":
    main()
