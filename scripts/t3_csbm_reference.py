#!/usr/bin/env python3
"""Finite-support reciprocal bridge for the T3/T4 categorical pilots."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

import torch


def normalize_distribution(values: Mapping[int, float]) -> dict[int, float]:
    clean = {int(key): max(0.0, float(value)) for key, value in values.items()}
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Categorical reference distribution has no finite mass")
    return {key: value / total for key, value in clean.items()}


def _tokens(x0: int, xT: int, mask_id: int, support: Sequence[int]) -> list[int]:
    return list(dict.fromkeys([int(x0), int(xT), int(mask_id), *map(int, support)]))


def _reference_transition(
    tokens: Sequence[int], mask_id: int, duration: float
) -> torch.Tensor:
    """Continuous-time mask-biased reference with small full-support mixing."""

    duration = max(0.0, float(duration))
    size = len(tokens)
    generator = torch.zeros((size, size), dtype=torch.float64)
    for source, source_token in enumerate(tokens):
        for destination, destination_token in enumerate(tokens):
            if source == destination:
                continue
            if destination_token == int(mask_id):
                rate = 2.0
            elif source_token == int(mask_id):
                rate = 0.20
            else:
                rate = 0.025
            generator[source, destination] = rate
        generator[source, source] = -generator[source].sum()
    transition = torch.matrix_exp(generator * duration)
    transition = transition.clamp_min(0.0)
    return transition / transition.sum(dim=1, keepdim=True).clamp_min(1e-15)


def _terminal_distribution(tokens: Sequence[int], xT: int, epsilon: float) -> torch.Tensor:
    eps = min(0.25, max(0.0, float(epsilon)))
    terminal = torch.full((len(tokens),), eps / len(tokens), dtype=torch.float64)
    terminal[tokens.index(int(xT))] += 1.0 - eps
    return terminal / terminal.sum()


def reciprocal_bridge_distribution(
    *,
    x0: int,
    xT: int,
    mask_id: int,
    support: Sequence[int],
    time: float,
    epsilon: float,
) -> dict[int, float]:
    """Marginal of a Doob bridge to a smoothed endpoint distribution."""

    t = min(1.0, max(0.0, float(time)))
    tokens = _tokens(x0, xT, mask_id, support)
    start_index = tokens.index(int(x0))
    terminal = _terminal_distribution(tokens, xT, epsilon)
    forward = _reference_transition(tokens, mask_id, t)[start_index]
    future = _reference_transition(tokens, mask_id, 1.0 - t) @ terminal
    values = forward * future
    if t == 0.0:
        values.zero_()
        values[start_index] = 1.0
    elif t == 1.0:
        values = terminal
    values = values / values.sum().clamp_min(1e-15)
    return {token: float(values[index]) for index, token in enumerate(tokens)}


def next_step_distribution(
    *,
    x_t: int,
    x0: int,
    xT: int,
    mask_id: int,
    support: Sequence[int],
    time: float,
    next_time: float,
    epsilon: float,
) -> dict[int, float]:
    """Doob h-transform transition conditioned on the smoothed endpoint."""

    if next_time < time:
        raise ValueError("next_time must be >= time")
    t = min(1.0, max(0.0, float(time)))
    next_t = min(1.0, max(0.0, float(next_time)))
    tokens = _tokens(x0, xT, mask_id, support)
    if int(x_t) not in tokens:
        tokens.append(int(x_t))
    current_index = tokens.index(int(x_t))
    terminal = _terminal_distribution(tokens, xT, epsilon)
    step = _reference_transition(tokens, mask_id, next_t - t)[current_index]
    future = _reference_transition(tokens, mask_id, 1.0 - next_t) @ terminal
    values = step * future
    values = values / values.sum().clamp_min(1e-15)
    return {token: float(values[index]) for index, token in enumerate(tokens)}


def seeded_sample(distribution: Mapping[int, float], seed: int) -> int:
    rng = random.Random(int(seed))
    threshold = rng.random()
    cumulative = 0.0
    for token, probability in sorted(distribution.items()):
        cumulative += float(probability)
        if threshold <= cumulative:
            return int(token)
    return int(next(reversed(sorted(distribution))))
