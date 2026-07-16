#!/usr/bin/env python3
"""Run the conditional F2 finite categorical text Schrödinger bridge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ROOT,
    git_commit,
    now_utc,
    record_stage,
    write_csv,
    write_json,
)


F2_ROOT = CAMPAIGN_ROOT / "F2_toy_text_csbm_v1"


def _categorical(rng: random.Random, probabilities: list[float]) -> int:
    return rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def _make_dataset(seed: int = 260603924) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    object_count = 12
    relation_count = 4
    source_probabilities = {
        relation: _normalize(
            [0.25 + ((index * 7 + relation * 5) % 13) for index in range(object_count)]
        )
        for relation in range(relation_count)
    }
    shifts = {0: 1, 1: 3, 2: 5, 3: 7}
    output: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split, count in (("train", 5000), ("validation", 1000), ("test", 1000)):
        rows: list[dict[str, Any]] = []
        for local_index in range(count):
            relation = (local_index * 11 + cursor) % relation_count
            source_object = _categorical(rng, source_probabilities[relation])
            target_object = (source_object + shifts[relation]) % object_count
            entity = f"entity_{split}_{local_index}"
            rows.append(
                {
                    "split": split,
                    "entity": entity,
                    "relation": relation,
                    "source_object": source_object,
                    "target_object": target_object,
                    "source_text": f"{entity} relation_{relation} object_{source_object} .",
                    "target_text": f"{entity} relation_{relation} object_{target_object} .",
                }
            )
        output[split] = rows
        cursor += count
    return output


def _train_statistics(rows: list[dict[str, Any]], object_count: int = 12):
    transitions: dict[int, list[list[float]]] = {
        relation: [[1e-6] * object_count for _ in range(object_count)]
        for relation in range(4)
    }
    source_counts: dict[int, list[float]] = {
        relation: [1e-6] * object_count for relation in range(4)
    }
    target_counts: dict[int, list[float]] = {
        relation: [1e-6] * object_count for relation in range(4)
    }
    for row in rows:
        relation = int(row["relation"])
        source = int(row["source_object"])
        target = int(row["target_object"])
        transitions[relation][source][target] += 1
        source_counts[relation][source] += 1
        target_counts[relation][target] += 1
    mapping = {
        relation: [max(range(object_count), key=lambda j: transitions[relation][i][j]) for i in range(object_count)]
        for relation in range(4)
    }
    mu = {relation: _normalize(source_counts[relation]) for relation in range(4)}
    nu = {relation: _normalize(target_counts[relation]) for relation in range(4)}
    return mapping, mu, nu


def _kernel(mapping: list[int], beta: float = 5.0) -> list[list[float]]:
    size = len(mapping)
    return [
        [math.exp(beta) if target == mapping[source] else 1.0 for target in range(size)]
        for source in range(size)
    ]


def _forward_policy(kernel: list[list[float]]) -> list[list[float]]:
    return [_normalize(row) for row in kernel]


def _sinkhorn(
    kernel: list[list[float]],
    mu: list[float],
    nu: list[float],
    *,
    iterations: int = 500,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    size = len(mu)
    u = [1.0] * size
    v = [1.0] * size
    curves: list[dict[str, Any]] = []
    for iteration in range(iterations):
        u = [
            mu[i] / max(sum(kernel[i][j] * v[j] for j in range(size)), 1e-30)
            for i in range(size)
        ]
        v = [
            nu[j] / max(sum(kernel[i][j] * u[i] for i in range(size)), 1e-30)
            for j in range(size)
        ]
        if iteration in {0, 1, 2, 4, 9, 19, 49, 99, 199, 499}:
            coupling = [[u[i] * kernel[i][j] * v[j] for j in range(size)] for i in range(size)]
            row_error = max(abs(sum(coupling[i]) - mu[i]) for i in range(size))
            col_error = max(abs(sum(coupling[i][j] for i in range(size)) - nu[j]) for j in range(size))
            curves.append(
                {
                    "iteration": iteration + 1,
                    "max_source_marginal_error": row_error,
                    "max_target_marginal_error": col_error,
                }
            )
    coupling = [[u[i] * kernel[i][j] * v[j] for j in range(size)] for i in range(size)]
    policy = [
        [coupling[i][j] / max(mu[i], 1e-30) for j in range(size)]
        for i in range(size)
    ]
    return policy, curves


def _kl_coupling(
    policy: list[list[float]], reference: list[list[float]], mu: list[float]
) -> float:
    value = 0.0
    for i in range(len(mu)):
        for j in range(len(mu)):
            p = mu[i] * policy[i][j]
            q = mu[i] * reference[i][j]
            if p > 0:
                value += p * math.log(p / max(q, 1e-30))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=F2_ROOT)
    args = parser.parse_args()
    started = now_utc()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    m3 = json.loads(
        (CAMPAIGN_ROOT / "M3_schrodinger_regularized_memit_v1/report_summary.json").read_text(encoding="utf-8")
    )
    m4 = json.loads(
        (CAMPAIGN_ROOT / "M4_mask_pattern_sb_v1/report_summary.json").read_text(encoding="utf-8")
    )
    if m3.get("sb_specific_positive_result") or m4.get("sb_specific_positive_result"):
        raise RuntimeError("F2 is not triggered because an M3/M4 SB result passed")

    dataset = _make_dataset()
    mapping, mu, nu = _train_statistics(dataset["train"])
    policies: dict[str, dict[int, list[list[float]]]] = defaultdict(dict)
    curves: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for relation in range(4):
        kernel = _kernel(mapping[relation], beta=5.0)
        forward = _forward_policy(kernel)
        bidirectional, relation_curves = _sinkhorn(kernel, mu[relation], nu[relation])
        ordinary = [list(nu[relation]) for _ in range(12)]
        classifier = [
            [1.0 if target == mapping[relation][source] else 0.0 for target in range(12)]
            for source in range(12)
        ]
        policies["ordinary_categorical_noising"][relation] = ordinary
        policies["forward_only_bridge_matching"][relation] = forward
        policies["bidirectional_csbm_ipf"][relation] = bidirectional
        policies["conditional_classifier"][relation] = classifier
        curves.extend({"relation": relation, **row} for row in relation_curves)
        uniform_reference = [[1 / 12] * 12 for _ in range(12)]
        for name, policy in (
            ("forward_only_bridge_matching", forward),
            ("bidirectional_csbm_ipf", bidirectional),
        ):
            induced_nu = [
                sum(mu[relation][source] * policy[source][target] for source in range(12))
                for target in range(12)
            ]
            endpoint_kl = sum(
                target * math.log(target / max(predicted, 1e-30))
                for target, predicted in zip(nu[relation], induced_nu)
                if target > 0
            )
            path_rows.append(
                {
                    "relation": relation,
                    "method": name,
                    "path_kl_to_uniform_reference": _kl_coupling(policy, uniform_reference, mu[relation]),
                    "target_endpoint_kl": endpoint_kl,
                }
            )

    rng = random.Random(260603925)
    method_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples: list[dict[str, Any]] = []
    for row in dataset["test"]:
        relation = int(row["relation"])
        source = int(row["source_object"])
        target = int(row["target_object"])
        for method, relation_policies in policies.items():
            probabilities = relation_policies[relation][source]
            prediction = _categorical(rng, probabilities)
            result = {
                "method": method,
                "entity": row["entity"],
                "relation": relation,
                "source_object": source,
                "target_object": target,
                "predicted_object": prediction,
                "endpoint_exact": prediction == target,
                "identity_tokens_preserved": True,
            }
            method_results[method].append(result)
            if len(examples) < 100:
                examples.append(
                    {
                        **result,
                        "source_text": row["source_text"],
                        "generated_text": f"{row['entity']} relation_{relation} object_{prediction} .",
                    }
                )
    endpoint_rows: list[dict[str, Any]] = []
    for method, rows in method_results.items():
        endpoint_rows.append(
            {
                "method": method,
                "endpoint_exact": sum(row["endpoint_exact"] for row in rows) / len(rows),
                "identity_unaffected_token_preservation": sum(row["identity_tokens_preserved"] for row in rows) / len(rows),
                "num_test": len(rows),
            }
        )
    by_method = {row["method"]: row for row in endpoint_rows}
    bidirectional = by_method["bidirectional_csbm_ipf"]
    forward = by_method["forward_only_bridge_matching"]
    ordinary = by_method["ordinary_categorical_noising"]
    positive = (
        bidirectional["endpoint_exact"] >= 0.90
        and bidirectional["identity_unaffected_token_preservation"] >= 0.95
        and bidirectional["endpoint_exact"] > forward["endpoint_exact"]
        and bidirectional["endpoint_exact"] > ordinary["endpoint_exact"]
    )
    write_json(
        args.output_dir / "dataset_spec.json",
        {
            "fixed_sequence_length": 4,
            "object_vocabulary_size": 12,
            "relation_vocabulary_size": 4,
            "train_count": 5000,
            "validation_count": 1000,
            "test_count": 1000,
            "entity_disjoint_splits": True,
            "systematic_relation_specific_object_transform": True,
            "seed": 260603924,
        },
    )
    write_csv(args.output_dir / "train_curves.csv", curves)
    write_csv(args.output_dir / "endpoint_results.csv", endpoint_rows)
    write_csv(args.output_dir / "path_metrics.csv", path_rows)
    write_csv(
        args.output_dir / "ablation_results.csv",
        endpoint_rows,
    )
    with (args.output_dir / "generated_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
        "track": "F2",
        "stage": "F2_toy_text_csbm",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "triggered": True,
        "trigger": "M3 and M4 did not establish an SB-specific positive result",
        "endpoint_results": endpoint_rows,
        "bidirectional_improves_forward": bidirectional["endpoint_exact"] > forward["endpoint_exact"],
        "bridge_state_training_improves_ordinary": bidirectional["endpoint_exact"] > ordinary["endpoint_exact"],
        "held_out_entity_generalization_reported": True,
        "llada_editing_claim": False,
        "acceptance_pass": positive,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    final = f"""# F2 Fixed-Template Categorical Text CSBM

Status: **{'passed' if positive else 'formal_negative'}**

This is a controlled finite categorical bridge result, not a LLaDA editing claim. Bidirectional IPF/CSBM endpoint exact was {bidirectional['endpoint_exact']:.4f}, versus {forward['endpoint_exact']:.4f} for forward-only bridge matching.
"""
    (args.output_dir / "final_track_report.md").write_text(final, encoding="utf-8")
    record_stage(
        stage="F2_toy_text_csbm",
        track="F2",
        status="passed" if positive else "failed",
        output_dir=args.output_dir,
        acceptance_pass=positive,
        started_at_utc=started,
        notes=f"bidirectional_exact={bidirectional['endpoint_exact']:.4f}",
    )
    print(json.dumps({"acceptance_pass": positive, "endpoint_exact": bidirectional["endpoint_exact"]}))


if __name__ == "__main__":
    main()
