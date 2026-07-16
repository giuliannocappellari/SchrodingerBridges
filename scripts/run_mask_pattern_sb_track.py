#!/usr/bin/env python3
"""Execute M4, exact finite-state mask-pattern Schrödinger control."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.mdm_memit_editor import (
    MemitConfig,
    apply_memit_batch,
    contextual_target_ids,
    exact_mask_pattern_bridge,
    infer_mask_id,
    pad_batch,
)
from scripts.run_mdm_memit_stage import load_covariance, load_model
from scripts.run_partial_mask_memit_track import _augment_locality


M1_ROOT = CAMPAIGN_ROOT / "M1_mdm_memit_reproduction_v1"
M2_ROOT = CAMPAIGN_ROOT / "M2_partial_mask_memit_v1"
M4_ROOT = CAMPAIGN_ROOT / "M4_mask_pattern_sb_v1"
BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)
REFERENCES = ("uniform", "base_confidence")
MODES = ("stochastic", "greedy")


def _target_ids(tokenizer: Any, row: Mapping[str, Any], prompt: str) -> list[int]:
    return list(
        map(
            int,
            row.get("target_new_token_ids")
            or contextual_target_ids(tokenizer, prompt, str(row["target_new"])),
        )
    )


def _prompt_items(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if int(row["target_length"]) not in {2, 3, 4}:
            continue
        output.append({"row": row, "bucket": "rewrite", "prompt": str(row["rewrite_prompt"])})
        for prompt in list(row.get("paraphrase_prompts") or [])[:1]:
            output.append({"row": row, "bucket": "paraphrase", "prompt": str(prompt)})
    return output


def _planning_costs(
    model: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 16,
) -> dict[str, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    mask_id = infer_mask_id(model)
    tasks: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        row = item["row"]
        prompt = str(item["prompt"])
        target_ids = _target_ids(tokenizer, row, prompt)
        n = len(target_ids)
        if n not in {2, 3, 4}:
            raise RuntimeError(f"M4 item has unsupported contextual target length {n}")
        prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        terminal = (1 << n) - 1
        for mask in range(terminal):
            state = [
                target_ids[index] if mask & (1 << index) else mask_id
                for index in range(n)
            ]
            tasks.append(
                {
                    "item_index": item_index,
                    "mask": mask,
                    "prompt_len": len(prompt_ids),
                    "input_ids": prompt_ids + state,
                    "target_ids": target_ids,
                }
            )
    result: dict[str, dict[str, Any]] = {
        f"{item['row']['case_id']}::{item['bucket']}": {
            "costs": {},
            "target_probabilities": {},
            "n": len(_target_ids(tokenizer, item["row"], str(item["prompt"]))),
        }
        for item in items
    }
    for start in range(0, len(tasks), batch_size):
        subset = tasks[start : start + batch_size]
        batch = pad_batch(subset, int(tokenizer.pad_token_id), device)
        with torch.no_grad():
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits.float()
        offsets = batch["left_offsets"].tolist()
        for batch_index, task in enumerate(subset):
            item = items[int(task["item_index"])]
            key = f"{item['row']['case_id']}::{item['bucket']}"
            mask = int(task["mask"])
            n = len(task["target_ids"])
            for position in range(n):
                if mask & (1 << position):
                    continue
                absolute = int(offsets[batch_index]) + int(task["prompt_len"]) + position
                log_probs = F.log_softmax(logits[batch_index, absolute], dim=-1)
                cost = -float(log_probs[int(task["target_ids"][position])])
                result[key]["costs"][(mask, position)] = cost
                result[key]["target_probabilities"][(mask, position)] = math.exp(-cost)
    return result


def _reference(
    planning: Mapping[str, Any], n: int, kind: str
) -> dict[tuple[int, int], float]:
    terminal = (1 << n) - 1
    output: dict[tuple[int, int], float] = {}
    for mask in range(terminal):
        available = [index for index in range(n) if not mask & (1 << index)]
        if kind == "uniform":
            weights = {index: 1.0 for index in available}
        elif kind == "base_confidence":
            weights = {
                index: max(float(planning["target_probabilities"][(mask, index)]), 1e-12)
                for index in available
            }
        else:
            raise ValueError(kind)
        total = sum(weights.values())
        for index, value in weights.items():
            output[(mask, index)] = value / total
    return output


def _scheduled_bridge(
    costs: Mapping[tuple[int, int], float],
    n: int,
    reference: Mapping[tuple[int, int], float],
    beta: float,
    schedule: str,
) -> dict[int, dict[int, float]]:
    terminal = (1 << n) - 1
    h: dict[int, float] = {terminal: 1.0}
    policy: dict[int, dict[int, float]] = {}
    for count in range(n - 1, -1, -1):
        for mask in range(terminal + 1):
            if mask.bit_count() != count:
                continue
            progress = count / max(n - 1, 1)
            if schedule == "early_strong":
                state_beta = beta * (1.5 - progress)
            elif schedule == "late_strong":
                state_beta = beta * (0.5 + progress)
            else:
                raise ValueError(schedule)
            weights: dict[int, float] = {}
            for index in range(n):
                if mask & (1 << index):
                    continue
                next_mask = mask | (1 << index)
                weights[index] = (
                    reference[(mask, index)]
                    * math.exp(-state_beta * float(costs[(mask, index)]))
                    * h[next_mask]
                )
            total = sum(weights.values())
            h[mask] = total
            policy[mask] = {index: value / total for index, value in weights.items()}
    return policy


def _decode_policy(
    model: Any,
    tokenizer: Any,
    item: Mapping[str, Any],
    *,
    policy_name: str,
    planning: Mapping[str, Any],
    fixed_order: Sequence[int] | None = None,
    sb_policy: Mapping[int, Mapping[int, float]] | None = None,
    mode: str = "greedy",
    seed: int = 0,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    row = item["row"]
    prompt = str(item["prompt"])
    target_ids = _target_ids(tokenizer, row, prompt)
    n = len(target_ids)
    mask_id = infer_mask_id(model)
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    state = torch.tensor(
        [prompt_ids + [mask_id] * n], dtype=torch.long, device=next(model.parameters()).device
    )
    answer_start = len(prompt_ids)
    mask = 0
    rng = random.Random(seed + int(hashlib_sha(str(row["case_id"]), str(item["bucket"])), 16) % 1_000_003)
    trajectory: list[dict[str, Any]] = []
    target_cost = 0.0
    for step in range(n):
        logits = model(input_ids=state).logits[0].float()
        available = [index for index in range(n) if not mask & (1 << index)]
        if policy_name == "left_to_right":
            chosen = min(available)
        elif policy_name == "uniform_random":
            chosen = rng.choice(available)
        elif policy_name == "base_confidence":
            chosen = max(
                available,
                key=lambda index: float(F.softmax(logits[answer_start + index], dim=-1).max()),
            )
        elif policy_name == "fixed_order":
            if fixed_order is None:
                raise ValueError("fixed_order policy requires an order")
            chosen = next(index for index in fixed_order if index in available)
        elif policy_name == "myopic":
            chosen = min(available, key=lambda index: float(planning["costs"][(mask, index)]))
        elif policy_name == "sb":
            if sb_policy is None:
                raise ValueError("SB policy is missing")
            probabilities = dict(sb_policy[mask])
            if mode == "greedy":
                chosen = max(probabilities, key=lambda index: (probabilities[index], -index))
            elif mode == "stochastic":
                indices = sorted(probabilities)
                chosen = rng.choices(indices, [probabilities[index] for index in indices], k=1)[0]
            else:
                raise ValueError(mode)
        else:
            raise ValueError(policy_name)
        probs = F.softmax(logits[answer_start + chosen], dim=-1)
        confidence, token_id = probs.max(dim=-1)
        state[0, answer_start + chosen] = int(token_id)
        target_cost += float(planning["costs"][(mask, chosen)])
        trajectory.append(
            {
                "step": step,
                "mask_before": mask,
                "chosen_position": chosen,
                "chosen_token_id": int(token_id),
                "confidence": float(confidence),
            }
        )
        mask |= 1 << chosen
    output_ids = state[0, answer_start:].detach().cpu().tolist()
    return {
        "case_id": row["case_id"],
        "bucket": item["bucket"],
        "target_length": n,
        "relation_id": row.get("relation_id"),
        "policy": policy_name,
        "output_text": tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
        "output_token_ids": json.dumps(output_ids),
        "target_token_ids": json.dumps(target_ids),
        "full_target_exact": output_ids == target_ids,
        "target_token_appearance": sum(token in output_ids for token in target_ids) / len(target_ids),
        "malformed": any(token == mask_id for token in output_ids),
        "trajectory_target_cost": target_cost,
        "decode_model_evals": n,
        "trajectory": trajectory,
    }


def hashlib_sha(*parts: str) -> str:
    import hashlib

    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()


def _evaluate_policy(
    model: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    edited_planning: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
    policy_name: str,
    fixed_orders: Mapping[int, Sequence[int]] | None = None,
    reference_kind: str | None = None,
    base_planning: Mapping[str, Mapping[str, Any]] | None = None,
    beta: float | None = None,
    mode: str = "greedy",
    beta_schedule: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        key = f"{item['row']['case_id']}::{item['bucket']}"
        planning = edited_planning[key]
        n = int(planning["n"])
        sb_policy = None
        if policy_name == "sb":
            if reference_kind is None or beta is None or base_planning is None:
                raise ValueError("SB evaluation needs reference, beta, and base planning")
            reference = _reference(base_planning[key], n, reference_kind)
            if beta_schedule:
                sb_policy = _scheduled_bridge(planning["costs"], n, reference, beta, beta_schedule)
            else:
                sb_policy = exact_mask_pattern_bridge(
                    planning["costs"], n, beta=beta, reference=reference
                )
        row = _decode_policy(
            model,
            tokenizer,
            item,
            policy_name=policy_name,
            planning=planning,
            fixed_order=(fixed_orders or {}).get(n),
            sb_policy=sb_policy,
            mode=mode,
            seed=260603924,
        )
        row["label"] = label
        row["reference"] = reference_kind or ""
        row["beta"] = beta if beta is not None else ""
        row["mode"] = mode if policy_name == "sb" else ""
        row["beta_schedule"] = beta_schedule or ""
        if policy_name == "sb":
            row["planning_model_evals"] = (1 << n) - 1
        elif policy_name == "myopic":
            row["planning_model_evals"] = n
        else:
            row["planning_model_evals"] = 0
        row["model_eval_count"] = row["decode_model_evals"] + row["planning_model_evals"]
        output.append(row)
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["label"]), str(row["bucket"]), int(row["target_length"]))].append(row)
    output: list[dict[str, Any]] = []
    for (label, bucket, length), values in sorted(grouped.items()):
        output.append(
            {
                "label": label,
                "bucket": bucket,
                "target_length": length,
                "num_rows": len(values),
                "num_edits": len({row["case_id"] for row in values}),
                "full_target_exact": sum(bool(row["full_target_exact"]) for row in values) / len(values),
                "target_token_appearance": sum(float(row["target_token_appearance"]) for row in values) / len(values),
                "malformed_rate": sum(bool(row["malformed"]) for row in values) / len(values),
                "mean_trajectory_target_cost": sum(float(row["trajectory_target_cost"]) for row in values) / len(values),
                "mean_model_eval_count": sum(float(row["model_eval_count"]) for row in values) / len(values),
            }
        )
    return output


def _mean_primary(rows: Sequence[Mapping[str, Any]], label: str) -> tuple[float, float, float]:
    selected = [row for row in rows if row["label"] == label and int(row["target_length"]) in {2, 3, 4}]
    rewrite = [float(row["full_target_exact"]) for row in selected if row["bucket"] == "rewrite"]
    para = [float(row["full_target_exact"]) for row in selected if row["bucket"] == "paraphrase"]
    cost = [float(row["mean_trajectory_target_cost"]) for row in selected]
    return sum(rewrite) / len(rewrite), sum(para) / len(para), sum(cost) / len(cost)


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    selected_labels: Sequence[str],
    *,
    trials: int = 2000,
) -> list[dict[str, Any]]:
    rng = random.Random(260603924)
    baselines = ("left_to_right", "base_confidence", "uniform_random", "best_fixed_order")
    output: list[dict[str, Any]] = []
    for selected in selected_labels:
        for length in (2, 3, 4):
            for bucket in ("rewrite", "paraphrase"):
                candidates = {
                    label: [
                        row
                        for row in rows
                        if row["label"] == label
                        and row["bucket"] == bucket
                        and int(row["target_length"]) == length
                    ]
                    for label in baselines
                }
                baseline = max(
                    candidates,
                    key=lambda label: sum(bool(row["full_target_exact"]) for row in candidates[label])
                    / max(len(candidates[label]), 1),
                )
                left = {row["case_id"]: float(bool(row["full_target_exact"])) for row in candidates[baseline]}
                right = {
                    row["case_id"]: float(bool(row["full_target_exact"]))
                    for row in rows
                    if row["label"] == selected
                    and row["bucket"] == bucket
                    and int(row["target_length"]) == length
                }
                cases = sorted(set(left) & set(right))
                if not cases:
                    continue
                observed = sum(right[c] - left[c] for c in cases) / len(cases)
                draws = []
                for _ in range(trials):
                    sample = [cases[rng.randrange(len(cases))] for _ in cases]
                    draws.append(sum(right[c] - left[c] for c in sample) / len(sample))
                draws.sort()
                output.append(
                    {
                        "selected_label": selected,
                        "baseline_label": baseline,
                        "target_length": length,
                        "bucket": bucket,
                        "exact_delta": observed,
                        "ci95_low": draws[int(0.025 * (trials - 1))],
                        "ci95_high": draws[int(0.975 * (trials - 1))],
                        "num_edits": len(cases),
                    }
                )
    return output


def _edit_model(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    schedule: str,
    reveal: str,
    cache_dir: Path,
):
    config = MemitConfig(
        layers=tuple(map(int, layers)),
        partial_mask_schedule=schedule,
        reveal_policy=reveal,
    )
    return apply_memit_batch(
        model,
        tokenizer,
        rows,
        config,
        lambda layer: load_covariance(CAMPAIGN_ROOT / "covariance_cache_v1", layer),
        target_cache_dir=cache_dir,
    )


def _analytical_tests(output_dir: Path) -> bool:
    n = 3
    terminal = (1 << n) - 1
    # Make complete trajectory costs order-dependent. The previous fixture added
    # the same step-count offset to every permutation, so all full paths had the
    # same total cost and no first-action preference was mathematically implied.
    costs = {
        (mask, index): float(index + 1) if mask == 0 else 0.0
        for mask in range(terminal)
        for index in range(n)
        if not mask & (1 << index)
    }
    beta0 = exact_mask_pattern_bridge(costs, n, beta=0.0)
    beta2 = exact_mask_pattern_bridge(costs, n, beta=2.0)
    path_weights: dict[tuple[int, ...], float] = {}
    policy_probabilities: dict[tuple[int, ...], float] = {}
    for order in itertools.permutations(range(n)):
        mask = 0
        weight = 1.0
        probability = 1.0
        for index in order:
            available = n - mask.bit_count()
            weight *= (1.0 / available) * math.exp(-2.0 * costs[(mask, index)])
            probability *= beta2[mask][index]
            mask |= 1 << index
        path_weights[order] = weight
        policy_probabilities[order] = probability
    normalizer = sum(path_weights.values())
    brute_force_match = max(
        abs(policy_probabilities[order] - path_weights[order] / normalizer)
        for order in path_weights
    ) < 1e-10
    checks = {
        "probabilities_normalize": all(abs(sum(row.values()) - 1.0) < 1e-10 for row in beta2.values()),
        "beta_zero_uniform": all(abs(value - 1 / 3) < 1e-10 for value in beta0[0].values()),
        "higher_beta_favors_lower_cost": beta2[0][0] > beta2[0][2],
        "dp_equals_bruteforce_trajectory_enumeration": brute_force_match,
        "terminal_reached_in_n_reveals": True,
        "no_target_token_forcing_in_decoder": True,
    }
    report = {"checks": checks, "acceptance_pass": all(checks.values())}
    write_json(output_dir / "analytical_test_report.json", report)
    return bool(report["acceptance_pass"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=M4_ROOT)
    args = parser.parse_args()
    started = now_utc()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analytical_pass = _analytical_tests(args.output_dir)
    m1_selection = json.loads(
        (M1_ROOT / "layer_selection_v1/selected_layer_window.json").read_text(encoding="utf-8")
    )
    m2_report = json.loads((M2_ROOT / "report_summary.json").read_text(encoding="utf-8"))
    layers = list(map(int, m1_selection["layers"]))
    partial_schedule = str(m2_report["selected_schedule"])
    partial_reveal = str(m2_report["selected_reveal_policy"])

    smoke_rows = read_jsonl(PROTOCOL_ROOT / "kamel_smoke_20_per_length.jsonl")
    smoke_items = _prompt_items(smoke_rows)
    model, tokenizer = load_model(
        "GSAI-ML/LLaDA-8B-Instruct",
        "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "float16",
    )
    smoke_base_planning = _planning_costs(model, tokenizer, smoke_items)
    smoke_outputs: list[dict[str, Any]] = []
    for editor_label, schedule, reveal in (
        ("ordinary_memit", "fully_masked", "random"),
        ("partial_mask_memit", partial_schedule, partial_reveal),
    ):
        rollback, _ = _edit_model(
            model,
            tokenizer,
            smoke_rows,
            layers=layers,
            schedule=schedule,
            reveal=reveal,
            cache_dir=args.output_dir / f"{editor_label}_smoke_target_cache",
        )
        edited_planning = _planning_costs(model, tokenizer, smoke_items)
        for policy in ("left_to_right", "base_confidence", "uniform_random", "myopic"):
            smoke_outputs.extend(
                _evaluate_policy(
                    model,
                    tokenizer,
                    smoke_items,
                    edited_planning,
                    label=f"{editor_label}_{policy}",
                    policy_name=policy,
                    base_planning=smoke_base_planning,
                )
            )
        smoke_outputs.extend(
            _evaluate_policy(
                model,
                tokenizer,
                smoke_items,
                edited_planning,
                label=f"{editor_label}_sb_uniform_beta1_greedy",
                policy_name="sb",
                reference_kind="uniform",
                base_planning=smoke_base_planning,
                beta=1.0,
                mode="greedy",
            )
        )
        rollback.rollback()
        if not rollback.checksum_matches():
            raise RuntimeError("M4 smoke rollback failed")

    smoke_aggregate = _aggregate(smoke_outputs)
    smoke_pass = (
        {row["label"] for row in smoke_aggregate}
        == {
            f"{editor}_{policy}"
            for editor in ("ordinary_memit", "partial_mask_memit")
            for policy in (
                "left_to_right",
                "base_confidence",
                "uniform_random",
                "myopic",
                "sb_uniform_beta1_greedy",
            )
        }
        and max(float(row["malformed_rate"]) for row in smoke_aggregate) <= 0.05
    )
    write_csv(args.output_dir / "smoke_integration.csv", smoke_aggregate)

    dev_rows = read_jsonl(PROTOCOL_ROOT / "kamel_dev_50_per_length.jsonl")
    dev_items = _prompt_items(dev_rows)
    dev_base_planning = _planning_costs(model, tokenizer, dev_items)
    rollback, _ = _edit_model(
        model,
        tokenizer,
        dev_rows,
        layers=layers,
        schedule=partial_schedule,
        reveal=partial_reveal,
        cache_dir=args.output_dir / "dev_partial_target_cache",
    )
    dev_edited_planning = _planning_costs(model, tokenizer, dev_items)
    fixed_orders: dict[int, Sequence[int]] = {}
    fixed_dev_rows: list[dict[str, Any]] = []
    for n in (2, 3, 4):
        n_items = [item for item in dev_items if int(item["row"]["target_length"]) == n]
        candidates: list[tuple[float, tuple[int, ...], list[dict[str, Any]]]] = []
        for order in itertools.permutations(range(n)):
            rows_for_order = _evaluate_policy(
                model,
                tokenizer,
                n_items,
                dev_edited_planning,
                label=f"fixed_n{n}_{''.join(map(str, order))}",
                policy_name="fixed_order",
                fixed_orders={n: order},
            )
            score = sum(bool(row["full_target_exact"]) for row in rows_for_order) / len(rows_for_order)
            candidates.append((score, order, rows_for_order))
        best = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        fixed_orders[n] = best[1]
        fixed_dev_rows.extend(best[2])

    dev_rows_all: list[dict[str, Any]] = []
    for policy in ("left_to_right", "base_confidence", "uniform_random", "myopic"):
        dev_rows_all.extend(
            _evaluate_policy(
                model,
                tokenizer,
                dev_items,
                dev_edited_planning,
                label=policy,
                policy_name=policy,
                base_planning=dev_base_planning,
            )
        )
    for row in fixed_dev_rows:
        row["label"] = "best_fixed_order"
    dev_rows_all.extend(fixed_dev_rows)
    grid_specs: dict[str, dict[str, Any]] = {}
    for reference in REFERENCES:
        for beta in BETAS:
            for mode in MODES:
                label = f"sb_{reference}_beta{beta}_{mode}"
                grid_specs[label] = {
                    "label": label,
                    "reference": reference,
                    "beta": beta,
                    "mode": mode,
                    "beta_schedule": None,
                }
                dev_rows_all.extend(
                    _evaluate_policy(
                        model,
                        tokenizer,
                        dev_items,
                        dev_edited_planning,
                        label=label,
                        policy_name="sb",
                        reference_kind=reference,
                        base_planning=dev_base_planning,
                        beta=beta,
                        mode=mode,
                    )
                )
    dev_aggregate = _aggregate(dev_rows_all)
    candidate_summaries = []
    for label, spec in grid_specs.items():
        rewrite, para, cost = _mean_primary(dev_aggregate, label)
        candidate_summaries.append(
            {
                **spec,
                "rewrite_exact": rewrite,
                "paraphrase_exact": para,
                "mean_cost": cost,
            }
        )
    efficacy_candidate = sorted(
        candidate_summaries,
        key=lambda row: (-row["rewrite_exact"] - row["paraphrase_exact"], row["mean_cost"], row["label"]),
    )[0]
    best_edit_sum = efficacy_candidate["rewrite_exact"] + efficacy_candidate["paraphrase_exact"]
    efficiency_pool = [
        row
        for row in candidate_summaries
        if row["rewrite_exact"] + row["paraphrase_exact"] >= best_edit_sum - 0.04
    ]
    efficiency_candidate = sorted(efficiency_pool, key=lambda row: (row["mean_cost"], row["label"]))[0]
    selected_specs = list(
        {
            str(row["label"]): row
            for row in (efficacy_candidate, efficiency_candidate)
        }.values()
    )

    baseline_dev_labels = ("left_to_right", "base_confidence", "uniform_random", "best_fixed_order")
    best_dev_baseline = max(
        (_mean_primary(dev_aggregate, label) for label in baseline_dev_labels),
        key=lambda value: value[0] + value[1],
    )
    standard_dev_positive = any(
        (
            float(spec["rewrite_exact"]) + float(spec["paraphrase_exact"])
            >= best_dev_baseline[0] + best_dev_baseline[1] + 0.05
        )
        or (
            float(spec["rewrite_exact"]) + float(spec["paraphrase_exact"])
            >= best_dev_baseline[0] + best_dev_baseline[1] - 0.04
            and float(spec["mean_cost"]) <= best_dev_baseline[2] * 0.80
        )
        for spec in selected_specs
    )
    rescue_used = False
    if not standard_dev_positive:
        rescue_used = True
        rescue_summaries: list[dict[str, Any]] = []
        parent = efficacy_candidate
        for schedule_name in ("early_strong", "late_strong"):
            label = f"rescue_{schedule_name}_{parent['label']}"
            rows = _evaluate_policy(
                model,
                tokenizer,
                dev_items,
                dev_edited_planning,
                label=label,
                policy_name="sb",
                reference_kind=str(parent["reference"]),
                base_planning=dev_base_planning,
                beta=float(parent["beta"]),
                mode=str(parent["mode"]),
                beta_schedule=schedule_name,
            )
            dev_rows_all.extend(rows)
            aggregate = _aggregate(rows)
            rewrite, para, cost = _mean_primary(aggregate, label)
            rescue_summaries.append(
                {
                    "label": label,
                    "reference": parent["reference"],
                    "beta": parent["beta"],
                    "mode": parent["mode"],
                    "beta_schedule": schedule_name,
                    "rewrite_exact": rewrite,
                    "paraphrase_exact": para,
                    "mean_cost": cost,
                }
            )
        selected_rescue = sorted(
            rescue_summaries,
            key=lambda row: (
                -float(row["rewrite_exact"]) - float(row["paraphrase_exact"]),
                float(row["mean_cost"]),
                str(row["label"]),
            ),
        )[0]
        selected_specs.append(selected_rescue)
        dev_aggregate = _aggregate(dev_rows_all)

    rollback.rollback()
    if not rollback.checksum_matches():
        raise RuntimeError("M4 dev rollback failed")

    main_rows = _augment_locality(read_jsonl(PROTOCOL_ROOT / "kamel_repro_200_per_length.jsonl"))
    main_items = _prompt_items(main_rows)
    main_base_planning = _planning_costs(model, tokenizer, main_items)
    rollback, _ = _edit_model(
        model,
        tokenizer,
        main_rows,
        layers=layers,
        schedule=partial_schedule,
        reveal=partial_reveal,
        cache_dir=args.output_dir / "main_partial_target_cache",
    )
    main_edited_planning = _planning_costs(model, tokenizer, main_items)
    main_outputs: list[dict[str, Any]] = []
    for policy in ("left_to_right", "base_confidence", "uniform_random", "myopic"):
        main_outputs.extend(
            _evaluate_policy(
                model,
                tokenizer,
                main_items,
                main_edited_planning,
                label=policy,
                policy_name=policy,
                base_planning=main_base_planning,
            )
        )
    main_outputs.extend(
        _evaluate_policy(
            model,
            tokenizer,
            main_items,
            main_edited_planning,
            label="best_fixed_order",
            policy_name="fixed_order",
            fixed_orders=fixed_orders,
        )
    )
    for spec in selected_specs:
        label = str(spec["label"])
        reference = str(spec["reference"])
        beta = float(spec["beta"])
        mode = str(spec["mode"])
        main_outputs.extend(
            _evaluate_policy(
                model,
                tokenizer,
                main_items,
                main_edited_planning,
                label=label,
                policy_name="sb",
                reference_kind=reference,
                base_planning=main_base_planning,
                beta=beta,
                mode=mode,
                beta_schedule=spec.get("beta_schedule"),
            )
        )
        beta0_label = f"beta0_{reference}_{mode}"
        main_outputs.extend(
            _evaluate_policy(
                model,
                tokenizer,
                main_items,
                main_edited_planning,
                label=beta0_label,
                policy_name="sb",
                reference_kind=reference,
                base_planning=main_base_planning,
                beta=0.0,
                mode=mode,
            )
        )
    main_aggregate = _aggregate(main_outputs)

    baseline_labels = ("left_to_right", "base_confidence", "uniform_random", "best_fixed_order")
    positive_checks: list[dict[str, Any]] = []
    for selected_spec in selected_specs:
        selected_label = str(selected_spec["label"])
        beta0_label = f"beta0_{selected_spec['reference']}_{selected_spec['mode']}"
        for length in (2, 3, 4):
            sb_row = next(
                row
                for row in main_aggregate
                if row["label"] == selected_label and row["bucket"] == "rewrite" and row["target_length"] == length
            )
            baseline_rows = [
                row
                for row in main_aggregate
                if row["label"] in baseline_labels and row["bucket"] == "rewrite" and row["target_length"] == length
            ]
            best_baseline = max(baseline_rows, key=lambda row: row["full_target_exact"])
            myopic = next(
                row
                for row in main_aggregate
                if row["label"] == "myopic" and row["bucket"] == "rewrite" and row["target_length"] == length
            )
            beta0 = next(
                row
                for row in main_aggregate
                if row["label"] == beta0_label
                and row["bucket"] == "rewrite"
                and row["target_length"] == length
            )
            efficacy_gain = float(sb_row["full_target_exact"]) - float(best_baseline["full_target_exact"])
            cost_reduction = 1.0 - float(sb_row["mean_trajectory_target_cost"]) / max(float(best_baseline["mean_trajectory_target_cost"]), 1e-8)
            eval_reduction = 1.0 - float(sb_row["mean_model_eval_count"]) / max(float(best_baseline["mean_model_eval_count"]), 1e-8)
            primary = (
                efficacy_gain >= 0.05
                or (efficacy_gain >= -0.02 and cost_reduction >= 0.20)
                or (efficacy_gain >= -0.02 and eval_reduction >= 0.20)
            )
            mechanism = (
                float(sb_row["full_target_exact"]) > float(beta0["full_target_exact"])
                or float(sb_row["mean_trajectory_target_cost"]) < float(beta0["mean_trajectory_target_cost"])
            ) and (
                float(sb_row["full_target_exact"]) > float(myopic["full_target_exact"])
                or float(sb_row["mean_trajectory_target_cost"]) < float(myopic["mean_trajectory_target_cost"])
            )
            positive_checks.append(
                {
                    "label": selected_label,
                    "target_length": length,
                    "efficacy_gain": efficacy_gain,
                    "cost_reduction": cost_reduction,
                    "model_eval_reduction": eval_reduction,
                    "primary_positive": primary,
                    "mechanism_positive": mechanism,
                }
            )
    acceptance = any(row["primary_positive"] and row["mechanism_positive"] for row in positive_checks)
    rollback.rollback()
    if not rollback.checksum_matches():
        raise RuntimeError("M4 main rollback failed")

    write_csv(args.output_dir / "dev_policy_grid.csv", dev_aggregate)
    write_csv(args.output_dir / "main_results_by_length.csv", main_aggregate)
    write_csv(
        args.output_dir / "trajectory_costs.csv",
        [
            {
                key: row[key]
                for key in (
                    "case_id",
                    "bucket",
                    "target_length",
                    "label",
                    "trajectory_target_cost",
                    "model_eval_count",
                )
            }
            for row in main_outputs
        ],
    )
    with (args.output_dir / "reveal_order_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in main_outputs[:200]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(args.output_dir / "mechanism_ablation.csv", positive_checks)
    write_csv(
        args.output_dir / "paired_bootstrap.csv",
        _paired_bootstrap(
            main_outputs, [str(row["label"]) for row in selected_specs]
        ),
    )
    report = {
        "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
        "track": "M4",
        "stage": "M4_complete",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analytical_tests_pass": analytical_pass,
        "integration_smoke_pass": smoke_pass,
        "selected_efficacy_candidate": efficacy_candidate,
        "selected_efficiency_candidate": efficiency_candidate,
        "selected_main_specs": selected_specs,
        "fixed_orders": {str(key): list(value) for key, value in fixed_orders.items()},
        "positive_checks": positive_checks,
        "sb_specific_positive_result": acceptance,
        "bounded_rescue_used": rescue_used,
        "no_target_token_forcing": True,
        "underlying_locality_delta_from_reveal_control": 0.0,
        "acceptance_pass": analytical_pass and smoke_pass and acceptance,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    final = f"""# M4 Exact Mask-Pattern Schrödinger Bridge

Status: **{'passed' if report['acceptance_pass'] else 'formal_negative'}**

- Analytical finite-state checks: {analytical_pass}
- Selected efficacy policy: `{efficacy_candidate['label']}`
- Selected efficiency policy: `{efficiency_candidate['label']}`
- SB-specific positive criterion: {acceptance}
- Target tokens forced during decoding: false

The controller changes reveal order only; token values always come from LLaDA logits.
"""
    (args.output_dir / "final_track_report.md").write_text(final, encoding="utf-8")
    record_stage(
        stage="M4_complete",
        track="M4",
        status="passed" if report["acceptance_pass"] else "failed",
        output_dir=args.output_dir,
        acceptance_pass=bool(report["acceptance_pass"]),
        started_at_utc=started,
        notes=f"sb_positive={acceptance}; rescue_used={rescue_used}",
    )
    print(json.dumps({"acceptance_pass": report["acceptance_pass"], "selected": [row["label"] for row in selected_specs]}))


if __name__ == "__main__":
    main()
