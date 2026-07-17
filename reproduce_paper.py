#!/usr/bin/env python3
"""Reproduce the primary publication table, figure, or exact-DP check."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from scripts.mask_pattern_kl_control import (
    enumerate_gibbs_paths,
    policy_path_distribution,
    solve_exact_kl_control,
    uniform_reference,
)


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "runs" / "mask_pattern_sb_publication_confirmation_v1" / "final_publication_package_v1"


def check_dp() -> dict[str, object]:
    maximum_error = 0.0
    checks = []
    for n in range(2, 7):
        costs = {
            (mask, index): 0.1 + 0.07 * index + 0.03 * ((mask + index) % 5)
            for mask in range((1 << n) - 1)
            for index in range(n)
            if not mask & (1 << index)
        }
        reference = uniform_reference(n)
        solution = solve_exact_kl_control(costs, n, beta=1.25, reference=reference)
        brute = enumerate_gibbs_paths(costs, n, beta=1.25, reference=reference)
        recovered = policy_path_distribution(solution.policy, n)
        error = max(abs(brute[path] - recovered[path]) for path in itertools.permutations(range(n)))
        maximum_error = max(maximum_error, error)
        checks.append({"target_length": n, "maximum_probability_error": error})
    report = {
        "acceptance_pass": maximum_error < 1e-10,
        "maximum_probability_error": maximum_error,
        "checks": checks,
        "llada_loaded": False,
    }
    if not report["acceptance_pass"]:
        raise RuntimeError(report)
    return report


def reproduce_table() -> Path:
    source = PACKAGE / "main_results_table.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = list(csv.DictReader(source.open(newline="")))
    required = {"family", "bucket", "target_length", "full_target_exact"}
    if not rows or not required <= set(rows[0]):
        raise RuntimeError("Main results table has an invalid schema")
    output = PACKAGE / "reproduced_main_results_table.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def reproduce_figure() -> Path:
    import matplotlib.pyplot as plt

    source = PACKAGE / "target_length_table.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    rows = list(csv.DictReader(source.open(newline="")))
    readiness = json.loads((PACKAGE / "top_tier_readiness.json").read_text())
    finite = str(readiness["finite_controller"])
    baseline = str(readiness["compute_matched_baseline"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for family, marker in ((finite, "o"), (baseline, "s")):
        selected = sorted(
            (
                (int(row["target_length"]), float(row["full_target_exact"]))
                for row in rows
                if row["family"] == family and row["bucket"] == "rewrite"
            ),
            key=lambda pair: pair[0],
        )
        if selected:
            ax.plot(
                [row[0] for row in selected],
                [row[1] for row in selected],
                marker=marker,
                linewidth=2,
                label=family,
            )
    ax.set_xlabel("Target length")
    ax.set_ylabel("Full-target rewrite exact")
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = PACKAGE / "main_figure.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=("main",))
    parser.add_argument("--figure", choices=("main",))
    parser.add_argument("--check-dp", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.table, args.figure, args.check_dp)) != 1:
        parser.error("Choose exactly one of --table, --figure, or --check-dp")
    if args.check_dp:
        print(json.dumps(check_dp(), sort_keys=True))
    elif args.table:
        print(reproduce_table())
    else:
        print(reproduce_figure())


if __name__ == "__main__":
    main()
