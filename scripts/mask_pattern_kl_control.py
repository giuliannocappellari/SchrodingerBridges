"""Exact and bounded mask-pattern path-control solvers on the monotone subset DAG."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


Transition = tuple[int, int]
CostTable = Mapping[Transition, float]
ReferenceTable = Mapping[Transition, float]


@dataclass(frozen=True)
class PathSolution:
    n: int
    beta: float
    log_partition: Mapping[int, float]
    policy: Mapping[int, Mapping[int, float]]
    expected_cost: float
    path_entropy: float
    kl_from_reference: float


def available_positions(mask: int, n: int) -> list[int]:
    return [index for index in range(n) if not mask & (1 << index)]


def validate_tables(costs: CostTable, reference: ReferenceTable, n: int) -> None:
    terminal = (1 << n) - 1
    for mask in range(terminal):
        available = available_positions(mask, n)
        if not available:
            continue
        probabilities = []
        for index in available:
            key = (mask, index)
            if key not in costs or key not in reference:
                raise ValueError(f"Missing transition {key}")
            cost = float(costs[key])
            probability = float(reference[key])
            if not math.isfinite(cost):
                raise ValueError(f"Non-finite cost at {key}")
            if not math.isfinite(probability) or probability <= 0:
                raise ValueError(f"Reference must have positive finite support at {key}")
            probabilities.append(probability)
        if abs(sum(probabilities) - 1.0) > 1e-10:
            raise ValueError(f"Reference probabilities at mask {mask} sum to {sum(probabilities)}")


def uniform_reference(n: int) -> dict[Transition, float]:
    terminal = (1 << n) - 1
    output: dict[Transition, float] = {}
    for mask in range(terminal):
        available = available_positions(mask, n)
        for index in available:
            output[(mask, index)] = 1.0 / len(available)
    return output


def normalized_reference(
    n: int, weights: Mapping[Transition, float], *, epsilon: float = 1e-12
) -> dict[Transition, float]:
    terminal = (1 << n) - 1
    output: dict[Transition, float] = {}
    for mask in range(terminal):
        available = available_positions(mask, n)
        values = {index: max(float(weights.get((mask, index), 0.0)), epsilon) for index in available}
        total = sum(values.values())
        for index, value in values.items():
            output[(mask, index)] = value / total
    return output


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def solve_exact_kl_control(
    costs: CostTable, n: int, *, beta: float, reference: ReferenceTable | None = None
) -> PathSolution:
    if n <= 0:
        raise ValueError("n must be positive")
    if beta < 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    q = dict(reference or uniform_reference(n))
    validate_tables(costs, q, n)
    terminal = (1 << n) - 1
    log_z: dict[int, float] = {terminal: 0.0}
    policy: dict[int, dict[int, float]] = {}
    for count in range(n - 1, -1, -1):
        for mask in range(terminal):
            if mask.bit_count() != count:
                continue
            actions = available_positions(mask, n)
            logits = {
                index: math.log(q[(mask, index)])
                - beta * float(costs[(mask, index)])
                + log_z[mask | (1 << index)]
                for index in actions
            }
            normalizer = _logsumexp(list(logits.values()))
            log_z[mask] = normalizer
            policy[mask] = {
                index: math.exp(value - normalizer) for index, value in logits.items()
            }

    state_probability: dict[int, float] = {0: 1.0}
    expected_cost = 0.0
    path_entropy = 0.0
    kl = 0.0
    for count in range(n):
        for mask in sorted(value for value in state_probability if value.bit_count() == count):
            mass = state_probability[mask]
            for index, probability in policy[mask].items():
                transition_mass = mass * probability
                next_mask = mask | (1 << index)
                state_probability[next_mask] = state_probability.get(next_mask, 0.0) + transition_mass
                expected_cost += transition_mass * float(costs[(mask, index)])
                # Under a very large beta, a dominated transition can underflow
                # to exactly zero even though the log-space recursion is stable.
                # Its entropy and KL contributions have the limiting value zero.
                if transition_mass > 0.0 and probability > 0.0:
                    path_entropy -= transition_mass * math.log(probability)
                    kl += transition_mass * math.log(probability / q[(mask, index)])
    return PathSolution(
        n=n,
        beta=beta,
        log_partition=log_z,
        policy=policy,
        expected_cost=expected_cost,
        path_entropy=path_entropy,
        kl_from_reference=kl,
    )


def solve_deterministic_global(costs: CostTable, n: int) -> tuple[tuple[int, ...], float]:
    terminal = (1 << n) - 1
    value: dict[int, float] = {terminal: 0.0}
    action: dict[int, int] = {}
    for count in range(n - 1, -1, -1):
        for mask in range(terminal):
            if mask.bit_count() != count:
                continue
            options = [
                (float(costs[(mask, index)]) + value[mask | (1 << index)], index)
                for index in available_positions(mask, n)
            ]
            best_cost, best_index = min(options, key=lambda pair: (pair[0], pair[1]))
            value[mask] = best_cost
            action[mask] = best_index
    order = []
    mask = 0
    while mask != terminal:
        index = action[mask]
        order.append(index)
        mask |= 1 << index
    return tuple(order), value[0]


def path_cost(order: Sequence[int], costs: CostTable) -> float:
    mask = 0
    total = 0.0
    for index in order:
        total += float(costs[(mask, int(index))])
        mask |= 1 << int(index)
    return total


def enumerate_gibbs_paths(
    costs: CostTable, n: int, *, beta: float, reference: ReferenceTable | None = None
) -> dict[tuple[int, ...], float]:
    q = dict(reference or uniform_reference(n))
    validate_tables(costs, q, n)
    log_weights: dict[tuple[int, ...], float] = {}
    for order in itertools.permutations(range(n)):
        mask = 0
        value = 0.0
        for index in order:
            value += math.log(q[(mask, index)]) - beta * float(costs[(mask, index)])
            mask |= 1 << index
        log_weights[order] = value
    normalizer = _logsumexp(list(log_weights.values()))
    return {order: math.exp(value - normalizer) for order, value in log_weights.items()}


def policy_path_distribution(
    policy: Mapping[int, Mapping[int, float]], n: int
) -> dict[tuple[int, ...], float]:
    output: dict[tuple[int, ...], float] = {}
    for order in itertools.permutations(range(n)):
        mask = 0
        probability = 1.0
        for index in order:
            probability *= float(policy[mask][index])
            mask |= 1 << index
        output[order] = probability
    return output


def beam_search_paths(costs: CostTable, n: int, beam_width: int) -> list[tuple[tuple[int, ...], float]]:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    beam: list[tuple[tuple[int, ...], int, float]] = [((), 0, 0.0)]
    for _ in range(n):
        expanded = []
        for order, mask, total in beam:
            for index in available_positions(mask, n):
                expanded.append(
                    (order + (index,), mask | (1 << index), total + float(costs[(mask, index)]))
                )
        beam = sorted(expanded, key=lambda row: (row[2], row[0]))[:beam_width]
    return [(order, total) for order, _, total in beam]


def random_search_paths(
    costs: CostTable, n: int, *, num_paths: int, seed: int
) -> list[tuple[tuple[int, ...], float]]:
    if num_paths <= 0:
        raise ValueError("num_paths must be positive")
    rng = random.Random(seed)
    results = []
    for _ in range(num_paths):
        order = list(range(n))
        rng.shuffle(order)
        results.append((tuple(order), path_cost(order, costs)))
    return sorted(results, key=lambda row: (row[1], row[0]))


def query_limited_greedy(
    n: int,
    query: Callable[[int, int], float],
    *,
    budget: int,
    fallback_order: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], int]:
    """Myopic planner with an explicit unique-transition query budget."""

    fallback = list(fallback_order or range(n))
    cache: dict[Transition, float] = {}
    mask = 0
    order = []
    while len(order) < n:
        available = available_positions(mask, n)
        options = []
        for index in available:
            key = (mask, index)
            if key not in cache and len(cache) < budget:
                cache[key] = float(query(mask, index))
            if key in cache:
                options.append((cache[key], index))
        if options:
            chosen = min(options, key=lambda row: (row[0], row[1]))[1]
        else:
            chosen = next(index for index in fallback if index in available)
        order.append(chosen)
        mask |= 1 << chosen
    return tuple(order), len(cache)
