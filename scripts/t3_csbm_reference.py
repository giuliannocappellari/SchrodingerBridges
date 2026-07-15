#!/usr/bin/env python3
"""Finite-support categorical reciprocal reference used by T3 and T4."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence


def normalize_distribution(values: Mapping[int, float]) -> dict[int, float]:
    clean = {int(key): max(0.0, float(value)) for key, value in values.items()}
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Categorical reference distribution has no finite mass")
    return {key: value / total for key, value in clean.items()}


def reciprocal_bridge_distribution(
    *,
    x0: int,
    xT: int,
    mask_id: int,
    support: Sequence[int],
    time: float,
    epsilon: float,
) -> dict[int, float]:
    """Smoothed endpoint reciprocal interpolation with absorbing-mask noise."""

    t = min(1.0, max(0.0, float(time)))
    eps = min(0.25, max(0.0, float(epsilon)))
    tokens = list(dict.fromkeys([int(x0), int(xT), int(mask_id), *map(int, support)]))
    target = {token: eps / len(tokens) for token in tokens}
    target[int(xT)] += 1.0 - eps
    target = normalize_distribution(target)
    bridge_noise = 0.20 * 4.0 * t * (1.0 - t)
    values = {token: t * target.get(token, 0.0) for token in tokens}
    values[int(x0)] = values.get(int(x0), 0.0) + (1.0 - t)
    if bridge_noise > 0.0:
        for token in values:
            values[token] *= 1.0 - bridge_noise
        values[int(mask_id)] = values.get(int(mask_id), 0.0) + bridge_noise
    return normalize_distribution(values)


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
    if next_time < time:
        raise ValueError("next_time must be >= time")
    if int(x_t) != int(mask_id) and next_time < 1.0:
        persistence = min(0.95, max(0.0, 1.0 - (next_time - time) * 2.0))
    else:
        persistence = 0.0
    destination = reciprocal_bridge_distribution(
        x0=x0,
        xT=xT,
        mask_id=mask_id,
        support=support,
        time=next_time,
        epsilon=epsilon,
    )
    destination[int(x_t)] = destination.get(int(x_t), 0.0) + persistence
    return normalize_distribution(destination)


def seeded_sample(distribution: Mapping[int, float], seed: int) -> int:
    rng = random.Random(int(seed))
    threshold = rng.random()
    cumulative = 0.0
    for token, probability in sorted(distribution.items()):
        cumulative += float(probability)
        if threshold <= cumulative:
            return int(token)
    return int(next(reversed(sorted(distribution))))
