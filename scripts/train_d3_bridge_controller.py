#!/usr/bin/env python3
"""Train a tiny fake Direction 3 bridge controller.

This local scaffold intentionally implements only fake mode. It proves the
training interface, loss accounting, and output summaries without loading
LLaDA or requiring GPU.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, git_commit, mean, now_utc, read_jsonl, repo_path, write_json


FEATURE_NAMES = ["bias", "base", "myopic", "no_rollout", "step", "active_masks", "is_positive_prompt"]


def row_features(row: Dict[str, Any]) -> List[float]:
    return [
        1.0,
        float(row["base_logits_top_k"][0]),
        float(row["myopic_scores_top_k"][0]),
        float(row["no_rollout_scores_top_k"][0]),
        float(row["step_index"]) / 3.0,
        float(row["active_mask_count"]) / 4.0,
        float(row.get("label", 0)),
    ]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict(weights: Sequence[float], features: Sequence[float]) -> float:
    return sigmoid(sum(w * x for w, x in zip(weights, features)))


def loss_for_rows(rows: Sequence[Dict[str, Any]], weights: Sequence[float], l2: float = 0.001) -> Dict[str, float]:
    eps = 1e-8
    bce_terms: List[float] = []
    ranking_terms: List[float] = []
    locality_terms: List[float] = []
    for row in rows:
        y = float(row.get("final_edit_success") or row.get("label") or 0.0)
        pred = predict(weights, row_features(row))
        bce_terms.append(-(y * math.log(pred + eps) + (1.0 - y) * math.log(1.0 - pred + eps)))
        bridge_target = float(row["raw_bridge_scores_top_k"][0] > max(row["raw_bridge_scores_top_k"][1:]))
        ranking_terms.append(abs(pred - bridge_target))
        if int(row.get("label", 0)) == 0:
            locality_terms.append(pred)
    l2_term = l2 * sum(w * w for w in weights)
    return {
        "bridge_distillation_loss": mean(bce_terms),
        "ranking_loss": mean(ranking_terms),
        "locality_kl_proxy_loss": mean(locality_terms),
        "l2_correction_loss": l2_term,
        "total_loss": mean(bce_terms) + 0.25 * mean(ranking_terms) + 0.5 * mean(locality_terms) + l2_term,
    }


def train_fake(rows: Sequence[Dict[str, Any]], epochs: int, lr: float) -> Tuple[List[float], List[Dict[str, float]]]:
    weights = [0.0 for _ in FEATURE_NAMES]
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        for row in rows:
            features = row_features(row)
            y = float(row.get("final_edit_success") or row.get("label") or 0.0)
            pred = predict(weights, features)
            grad_scale = pred - y
            for i, feature in enumerate(features):
                weights[i] -= lr * grad_scale * feature
        metrics = loss_for_rows(rows, weights)
        metrics["epoch"] = float(epoch)
        history.append(metrics)
    return weights, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_cache_dir", type=Path, default=D3_ROOT / "fake_teacher_cache_v1")
    parser.add_argument("--output_dir", type=Path, default=D3_ROOT / "fake_controller_train_v1")
    parser.add_argument("--fake_model", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not bool(args.fake_model):
        raise SystemExit("Real controller training requires a later approved path. Re-run with --fake_model 1 locally.")
    repo_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_train.jsonl")
    val_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_val.jsonl")
    if not train_rows or not val_rows:
        raise AssertionError("Teacher cache train/val rows are required")

    weights, history = train_fake(train_rows, args.epochs, args.lr)
    val_metrics = loss_for_rows(val_rows, weights)
    train_metrics = history[-1]
    payload = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 fake bridge controller training",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": True,
        "llada_loaded": False,
        "analysis_500_used": False,
        "final_test_used": False,
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
        "artifacts": {
            "controller_weights": str(args.output_dir / "controller_weights.json"),
            "train_metrics": str(args.output_dir / "train_metrics.json"),
        },
    }
    write_json(args.output_dir / "controller_weights.json", {"feature_names": FEATURE_NAMES, "weights": weights})
    write_json(args.output_dir / "train_metrics.json", payload)
    write_json(args.output_dir / "report_summary.json", payload)
    print(f"[INFO] Wrote fake Direction 3 controller training output to {args.output_dir}")


if __name__ == "__main__":
    main()
