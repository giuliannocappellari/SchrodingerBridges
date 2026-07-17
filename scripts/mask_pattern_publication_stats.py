"""Deterministic paired statistics for the publication confirmation tracks."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence


def paired_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    bucket: str,
    metric: str,
    lengths: set[int] | None = None,
) -> list[tuple[str, float, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("bucket")) != bucket:
            continue
        if lengths is not None and int(row["target_length"]) not in lengths:
            continue
        family = str(row.get("family") or row.get("label"))
        if family not in {left, right}:
            continue
        grouped[(family, str(row["case_id"]))].append(float(row[metric]))
    output = []
    case_ids = sorted(
        {case_id for family, case_id in grouped if family == left}
        & {case_id for family, case_id in grouped if family == right}
    )
    for case_id in case_ids:
        left_values = grouped[(left, case_id)]
        right_values = grouped[(right, case_id)]
        output.append(
            (
                case_id,
                sum(left_values) / len(left_values),
                sum(right_values) / len(right_values),
            )
        )
    return output


def paired_bootstrap(
    pairs: Sequence[tuple[str, float, float]],
    *,
    resamples: int = 10_000,
    seed: int = 260_717_601,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Paired bootstrap requires at least one matched edit")
    deltas = [left - right for _, left, right in pairs]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        samples.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    samples.sort()

    def percentile(q: float) -> float:
        index = min(len(samples) - 1, max(0, int(math.floor(q * len(samples)))))
        return samples[index]

    nonpositive = (sum(value <= 0.0 for value in samples) + 1) / (resamples + 1)
    nonnegative = (sum(value >= 0.0 for value in samples) + 1) / (resamples + 1)
    return {
        "num_pairs": len(deltas),
        "mean_delta": observed,
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "p_two_sided": min(1.0, 2.0 * min(nonpositive, nonnegative)),
        "resamples": resamples,
        "seed": seed,
    }


def holm_adjust(rows: Sequence[Mapping[str, Any]], *, p_key: str = "p_two_sided") -> list[dict[str, Any]]:
    ordered = sorted(
        enumerate(rows), key=lambda pair: (float(pair[1][p_key]), pair[0])
    )
    adjusted_by_index: dict[int, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(row[p_key]))
        running = max(running, adjusted)
        adjusted_by_index[original_index] = running
    output = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["holm_adjusted_p"] = adjusted_by_index[index]
        item["holm_reject_0_05"] = adjusted_by_index[index] < 0.05
        output.append(item)
    return output
