"""Batched cost-table construction and reveal-policy decoding for P3-P7."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scripts.mask_pattern_kl_control import (
    beam_search_paths,
    normalized_reference,
    random_search_paths,
    solve_deterministic_global,
    solve_exact_kl_control,
    uniform_reference,
)
from scripts.mdm_memit_editor import contextual_target_ids, infer_mask_id, pad_batch


@dataclass(frozen=True)
class PlannerSpec:
    label: str
    kind: str
    seed: int = 0
    beta: float | None = None
    reference: str | None = None
    mode: str = "greedy"
    beam_width: int | None = None
    random_paths: int | None = None
    fixed_order: tuple[int, ...] | None = None
    query_budget: int | None = None
    regime: str = "full_cost_table"


def target_ids(tokenizer: Any, row: Mapping[str, Any], prompt: str) -> list[int]:
    if prompt == str(row.get("rewrite_prompt")) and row.get("target_new_token_ids"):
        return list(map(int, row["target_new_token_ids"]))
    return list(map(int, contextual_target_ids(tokenizer, prompt, str(row["target_new"]))))


def build_prompt_items(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_stress: bool,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({"row": row, "bucket": "rewrite", "prompt": str(row["rewrite_prompt"])})
        for prompt in list(row.get("paraphrase_prompts") or [])[:1]:
            output.append({"row": row, "bucket": "paraphrase", "prompt": str(prompt)})
        if include_stress:
            for prompt in list(row.get("same_subject_prompts") or [])[:1]:
                output.append(
                    {"row": row, "bucket": "same_subject_stress", "prompt": str(prompt)}
                )
            output.append(
                {
                    "row": row,
                    "bucket": "far_locality",
                    "prompt": "A distant and unrelated weather observation reports",
                    "prompt_provenance": "predeclared_synthetic_unrelated_control",
                }
            )
    return output


def item_key(item: Mapping[str, Any]) -> str:
    return f"{item['row']['case_id']}::{item['bucket']}"


def build_full_cost_tables(
    model: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 16,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Query each unique target-filled mask state exactly once per prompt."""

    import torch
    import torch.nn.functional as F

    started = time.monotonic()
    device = next(model.parameters()).device
    mask_id = infer_mask_id(model)
    tasks = []
    tables: dict[str, dict[str, Any]] = {}
    for item_index, item in enumerate(items):
        prompt = str(item["prompt"])
        ids = target_ids(tokenizer, item["row"], prompt)
        expected_n = int(item["row"]["target_length"])
        # Stress prompts are scored for over-injection of the rewrite target and
        # therefore deliberately retain its frozen answer-span tokenization.
        if len(ids) != expected_n:
            ids = list(map(int, item["row"]["target_new_token_ids"]))
        n = len(ids)
        prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        key = item_key(item)
        tables[key] = {
            "n": n,
            "target_ids": ids,
            "costs": {},
            "target_probabilities": {},
            "maximum_confidences": {},
            "entropies": {},
            "state_queries": (1 << n) - 1,
        }
        for mask in range((1 << n) - 1):
            state = [ids[index] if mask & (1 << index) else mask_id for index in range(n)]
            tasks.append(
                {
                    "item_index": item_index,
                    "key": key,
                    "mask": mask,
                    "prompt_len": len(prompt_ids),
                    "input_ids": prompt_ids + state,
                    "target_ids": ids,
                }
            )
    forward_batch_calls = 0
    for start in range(0, len(tasks), batch_size):
        subset = tasks[start : start + batch_size]
        batch = pad_batch(subset, int(tokenizer.pad_token_id), device)
        with torch.no_grad():
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits.float()
        forward_batch_calls += 1
        offsets = batch["left_offsets"].tolist()
        for batch_index, task in enumerate(subset):
            table = tables[str(task["key"])]
            mask = int(task["mask"])
            for position, target_token in enumerate(task["target_ids"]):
                if mask & (1 << position):
                    continue
                absolute = int(offsets[batch_index]) + int(task["prompt_len"]) + position
                log_probs = F.log_softmax(logits[batch_index, absolute], dim=-1)
                probs = log_probs.exp()
                cost = -float(log_probs[int(target_token)])
                transition = f"{mask}:{position}"
                table["costs"][transition] = cost
                table["target_probabilities"][transition] = math.exp(-cost)
                table["maximum_confidences"][transition] = float(probs.max())
                table["entropies"][transition] = float(-(probs * log_probs).sum())
    return tables, {
        "num_items": len(items),
        "num_unique_prompt_states": len(tasks),
        "forward_batch_calls": forward_batch_calls,
        "batch_size": batch_size,
        "wall_clock_seconds": time.monotonic() - started,
    }


def _tuple_table(values: Mapping[str, float]) -> dict[tuple[int, int], float]:
    return {
        tuple(map(int, key.split(":"))): float(value) for key, value in values.items()
    }


def reference_table(table: Mapping[str, Any], kind: str) -> dict[tuple[int, int], float]:
    n = int(table["n"])
    if kind == "uniform":
        return uniform_reference(n)
    if kind == "edited_target_confidence":
        return normalized_reference(n, _tuple_table(table["target_probabilities"]))
    if kind == "edited_max_confidence":
        return normalized_reference(n, _tuple_table(table["maximum_confidences"]))
    raise ValueError(kind)


def planner_policy(
    table: Mapping[str, Any], spec: PlannerSpec
) -> tuple[dict[int, dict[int, float]] | None, tuple[int, ...] | None, dict[str, Any]]:
    n = int(table["n"])
    costs = _tuple_table(table["costs"])
    started = time.monotonic()
    policy = None
    order = spec.fixed_order
    path_entropy = 0.0
    path_kl = 0.0
    expected_cost = None
    candidate_paths = 0
    state_queries = 0
    if spec.kind in {"finite_beta", "beta_zero"}:
        beta = 0.0 if spec.kind == "beta_zero" else float(spec.beta)
        reference = reference_table(table, str(spec.reference))
        solution = solve_exact_kl_control(costs, n, beta=beta, reference=reference)
        policy = {mask: dict(values) for mask, values in solution.policy.items()}
        path_entropy = solution.path_entropy
        path_kl = solution.kl_from_reference
        expected_cost = solution.expected_cost
        state_queries = (1 << n) - 1
    elif spec.kind == "deterministic_global":
        order, expected_cost = solve_deterministic_global(costs, n)
        state_queries = (1 << n) - 1
    elif spec.kind == "beam":
        paths = beam_search_paths(costs, n, int(spec.beam_width))
        order, expected_cost = paths[0]
        candidate_paths = len(paths)
        state_queries = (1 << n) - 1
    elif spec.kind == "random_search":
        paths = random_search_paths(
            costs, n, num_paths=int(spec.random_paths), seed=int(spec.seed)
        )
        order, expected_cost = paths[0]
        candidate_paths = len(paths)
        state_queries = (1 << n) - 1
    elif spec.kind in {"left_to_right", "right_to_left", "fixed_order"}:
        if spec.kind == "left_to_right":
            order = tuple(range(n))
        elif spec.kind == "right_to_left":
            order = tuple(reversed(range(n)))
        if order is None:
            raise ValueError(f"{spec.kind} requires an order")
        expected_cost = sum(
            costs[(sum(1 << prior for prior in order[:step]), index)]
            for step, index in enumerate(order)
        )
    elif spec.kind == "bounded_beam":
        order, state_queries, candidate_paths = bounded_beam_order(
            costs,
            n,
            beam_width=int(spec.beam_width),
            state_budget=int(spec.query_budget),
        )
        expected_cost = sum(
            costs[(sum(1 << prior for prior in order[:step]), index)]
            for step, index in enumerate(order)
        )
    elif spec.kind == "bounded_random":
        order, state_queries, candidate_paths = bounded_random_order(
            costs,
            n,
            state_budget=int(spec.query_budget),
            seed=int(spec.seed),
        )
        expected_cost = sum(
            costs[(sum(1 << prior for prior in order[:step]), index)]
            for step, index in enumerate(order)
        )
    return policy, order, {
        "planner_cpu_seconds": time.monotonic() - started,
        "expected_path_cost": expected_cost,
        "path_entropy": path_entropy,
        "path_kl_from_reference": path_kl,
        "candidate_path_evaluations": candidate_paths,
        "planner_state_queries": state_queries,
    }


def bounded_beam_order(
    costs: Mapping[tuple[int, int], float],
    n: int,
    *,
    beam_width: int,
    state_budget: int,
) -> tuple[tuple[int, ...], int, int]:
    """Beam search whose planner can reveal at most ``state_budget`` states."""

    queried: set[int] = set()
    beam: list[tuple[tuple[int, ...], int, float, int]] = [((), 0, 0.0, 0)]
    expansions = 0
    for _ in range(n):
        expanded = []
        for order, mask, known_cost, known_steps in beam:
            can_query = mask in queried or len(queried) < state_budget
            if can_query:
                queried.add(mask)
                for index in range(n):
                    if mask & (1 << index):
                        continue
                    expanded.append(
                        (
                            order + (index,),
                            mask | (1 << index),
                            known_cost + float(costs[(mask, index)]),
                            known_steps + 1,
                        )
                    )
                    expansions += 1
            else:
                remaining = tuple(index for index in range(n) if not mask & (1 << index))
                # No hidden cost is consulted for ranking the fallback completion.
                expanded.append((order + remaining, (1 << n) - 1, known_cost, known_steps))
        beam = sorted(expanded, key=lambda row: (-row[3], row[2], row[0]))[:beam_width]
        if all(len(row[0]) == n for row in beam):
            break
    return beam[0][0], len(queried), expansions


def bounded_random_order(
    costs: Mapping[tuple[int, int], float],
    n: int,
    *,
    state_budget: int,
    seed: int,
) -> tuple[tuple[int, ...], int, int]:
    """Random path search with a unique-state visibility budget."""

    rng = random.Random(seed)
    queried: set[int] = set()
    candidates: list[tuple[int, float, tuple[int, ...]]] = []
    attempts = max(5, state_budget)
    for _ in range(attempts):
        order = list(range(n))
        rng.shuffle(order)
        mask = 0
        known_cost = 0.0
        known_steps = 0
        for index in order:
            if mask not in queried and len(queried) < state_budget:
                queried.add(mask)
            if mask in queried:
                known_cost += float(costs[(mask, index)])
                known_steps += 1
            mask |= 1 << index
        candidates.append((known_steps, known_cost, tuple(order)))
    best = min(candidates, key=lambda row: (-row[0], row[1], row[2]))
    return best[2], len(queried), len(candidates)


def decode_with_planner(
    model: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    cost_tables: Mapping[str, Mapping[str, Any]],
    spec: PlannerSpec,
    *,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Decode one token per denoising step with a fixed planner specification."""

    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    mask_id = infer_mask_id(model)
    output_rows = []
    for start in range(0, len(items), batch_size):
        subset = items[start : start + batch_size]
        prompt_ids = [
            list(map(int, tokenizer(str(item["prompt"]), add_special_tokens=False)["input_ids"]))
            for item in subset
        ]
        ns = [int(cost_tables[item_key(item)]["n"]) for item in subset]
        if len(set(ns)) != 1:
            raise ValueError("Batched planner decode requires one target length")
        n = ns[0]
        batch = pad_batch(
            [{"input_ids": ids + [mask_id] * n} for ids in prompt_ids],
            int(tokenizer.pad_token_id),
            device,
        )
        state = batch["input_ids"]
        attention = batch["attention_mask"]
        offsets = batch["left_offsets"].tolist()
        answer_positions = [
            [int(offsets[index]) + len(prompt_ids[index]) + position for position in range(n)]
            for index in range(len(subset))
        ]
        masks = [0] * len(subset)
        trajectories: list[list[dict[str, Any]]] = [[] for _ in subset]
        planner_data = [planner_policy(cost_tables[item_key(item)], spec) for item in subset]
        rngs = [
            random.Random(int(spec.seed) + int(str(item["row"]["case_id"]).encode().hex()[:8], 16))
            for item in subset
        ]
        path_costs = [0.0] * len(subset)
        for step in range(n):
            with torch.no_grad():
                logits = model(input_ids=state, attention_mask=attention).logits.float()
            for row_index, item in enumerate(subset):
                mask = masks[row_index]
                table = cost_tables[item_key(item)]
                costs = _tuple_table(table["costs"])
                available = [position for position in range(n) if not mask & (1 << position)]
                policy, order, diagnostics = planner_data[row_index]
                if spec.kind in {"left_to_right", "right_to_left", "fixed_order", "deterministic_global", "beam", "random_search", "bounded_beam", "bounded_random"}:
                    assert order is not None
                    chosen = next(position for position in order if position in available)
                elif spec.kind in {"finite_beta", "beta_zero"}:
                    assert policy is not None
                    probabilities = policy[mask]
                    if spec.mode == "stochastic":
                        positions = sorted(probabilities)
                        chosen = rngs[row_index].choices(
                            positions, [probabilities[position] for position in positions], k=1
                        )[0]
                    else:
                        chosen = max(
                            probabilities,
                            key=lambda position: (probabilities[position], -position),
                        )
                elif spec.kind == "uniform_random":
                    chosen = rngs[row_index].choice(available)
                elif spec.kind == "myopic":
                    chosen = min(available, key=lambda position: (costs[(mask, position)], position))
                elif spec.kind in {"default_confidence", "maximum_confidence"}:
                    chosen = max(
                        available,
                        key=lambda position: (
                            float(
                                F.softmax(
                                    logits[row_index, answer_positions[row_index][position]], dim=-1
                                ).max()
                            ),
                            -position,
                        ),
                    )
                elif spec.kind == "minimum_entropy":
                    chosen = min(
                        available,
                        key=lambda position: (
                            float(
                                -(
                                    F.softmax(
                                        logits[row_index, answer_positions[row_index][position]], dim=-1
                                    )
                                    * F.log_softmax(
                                        logits[row_index, answer_positions[row_index][position]], dim=-1
                                    )
                                ).sum()
                            ),
                            position,
                        ),
                    )
                else:
                    raise ValueError(spec.kind)
                absolute = answer_positions[row_index][chosen]
                probs = F.softmax(logits[row_index, absolute], dim=-1)
                confidence, token = probs.max(dim=-1)
                state[row_index, absolute] = int(token)
                path_costs[row_index] += costs[(mask, chosen)]
                trajectories[row_index].append(
                    {
                        "step": step,
                        "mask_before": mask,
                        "chosen_position": chosen,
                        "chosen_token_id": int(token),
                        "confidence": float(confidence),
                        "target_transition_cost": costs[(mask, chosen)],
                    }
                )
                masks[row_index] |= 1 << chosen
        for row_index, item in enumerate(subset):
            ids = [int(state[row_index, position]) for position in answer_positions[row_index]]
            targets = list(map(int, cost_tables[item_key(item)]["target_ids"]))
            exact_positions = sum(left == right for left, right in zip(ids, targets))
            _, _, planner_diagnostics = planner_data[row_index]
            state_queries = int(planner_diagnostics.get("planner_state_queries", 0))
            if spec.kind == "myopic":
                state_queries = n
            output_rows.append(
                {
                    "case_id": item["row"]["case_id"],
                    "relation_id": item["row"].get("relation_id"),
                    "bucket": item["bucket"],
                    "target_length": n,
                    "label": spec.label,
                    "planner_kind": spec.kind,
                    "planner_seed": spec.seed,
                    "beta": "" if spec.beta is None else spec.beta,
                    "reference": spec.reference or "",
                    "mode": spec.mode,
                    "regime": spec.regime,
                    "output_text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                    "output_token_ids": json.dumps(ids),
                    "target_token_ids": json.dumps(targets),
                    "full_target_exact": ids == targets,
                    "target_token_f1": exact_positions / n,
                    "malformed": any(token == mask_id for token in ids),
                    "trajectory_target_cost": path_costs[row_index],
                    "unique_state_queries": state_queries,
                    "decode_forward_evaluations": n,
                    "model_evaluations": n + state_queries,
                    **planner_diagnostics,
                    "trajectory": json.dumps(trajectories[row_index], sort_keys=True),
                }
            )
    return output_rows
