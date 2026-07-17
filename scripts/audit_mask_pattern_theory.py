#!/usr/bin/env python3
"""Run P2 finite-DAG exactness, beta-limit, and mathematical naming audits."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mask_pattern_kl_control import (
    enumerate_gibbs_paths,
    normalized_reference,
    path_cost,
    policy_path_distribution,
    solve_deterministic_global,
    solve_exact_kl_control,
    uniform_reference,
)
from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    git_commit,
    now_utc,
    record_stage,
    write_csv,
    write_json,
)


NAMING_DECISION = "doob_transformed_mask_pattern_control"


def _fixture(n: int, seed: int) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    rng = random.Random(seed)
    terminal = (1 << n) - 1
    costs = {}
    confidence_weights = {}
    for mask in range(terminal):
        for index in range(n):
            if mask & (1 << index):
                continue
            # The state-action interaction ensures trajectory costs depend on reveal order.
            cost = (
                0.15
                + 0.31 * (index + 1)
                + 0.07 * mask.bit_count()
                + 0.11 * ((mask + 3 * index) % (n + 2))
                + rng.random() * 0.01
            )
            costs[(mask, index)] = cost
            confidence_weights[(mask, index)] = math.exp(-0.5 * cost) + 0.01
    return costs, normalized_reference(n, confidence_weights)


def _numerical_tests() -> tuple[list[dict[str, object]], bool]:
    rows = []
    for n in range(2, 7):
        for reference_name in ("uniform", "confidence"):
            costs, confidence = _fixture(n, 1700 + n)
            reference = uniform_reference(n) if reference_name == "uniform" else confidence
            for beta in (0.0, 0.25, 1.0, 4.0):
                solution = solve_exact_kl_control(costs, n, beta=beta, reference=reference)
                brute = enumerate_gibbs_paths(costs, n, beta=beta, reference=reference)
                from_policy = policy_path_distribution(solution.policy, n)
                maximum_path_error = max(abs(brute[path] - from_policy[path]) for path in brute)
                brute_expected_cost = sum(path_cost(path, costs) * probability for path, probability in brute.items())
                transition_sum_error = max(
                    abs(sum(probabilities.values()) - 1.0)
                    for probabilities in solution.policy.values()
                )
                partition_from_paths = sum(
                    math.exp(-beta * path_cost(path, costs))
                    * math.prod(
                        reference[(sum(1 << prior for prior in path[:step]), index)]
                        for step, index in enumerate(path)
                    )
                    for path in itertools.permutations(range(n))
                )
                relative_partition_error = abs(
                    math.exp(solution.log_partition[0]) - partition_from_paths
                ) / max(partition_from_paths, 1e-300)
                row = {
                    "n": n,
                    "reference": reference_name,
                    "beta": beta,
                    "maximum_path_probability_error": maximum_path_error,
                    "transition_sum_error": transition_sum_error,
                    "expected_cost_error": abs(solution.expected_cost - brute_expected_cost),
                    "partition_relative_error": relative_partition_error,
                    "pass": (
                        maximum_path_error <= 1e-8
                        and transition_sum_error <= 1e-8
                        and abs(solution.expected_cost - brute_expected_cost) <= 1e-8
                        and relative_partition_error <= 1e-8
                    ),
                }
                rows.append(row)
    return rows, all(bool(row["pass"]) for row in rows)


def _beta_limits() -> tuple[list[dict[str, object]], bool]:
    rows = []
    for n in range(2, 7):
        costs, reference = _fixture(n, 2600 + n)
        low = solve_exact_kl_control(costs, n, beta=1e-9, reference=reference)
        # Use a genuinely asymptotic scale: the fixture can contain globally
        # distinct paths whose total costs differ by less than 1e-3.
        high = solve_exact_kl_control(costs, n, beta=1_000_000.0, reference=reference)
        best_order, best_cost = solve_deterministic_global(costs, n)
        low_error = max(
            abs(low.policy[mask][index] - reference[(mask, index)])
            for mask, probabilities in low.policy.items()
            for index in probabilities
        )
        greedy_high_order = []
        mask = 0
        while len(greedy_high_order) < n:
            index = max(high.policy[mask], key=lambda value: (high.policy[mask][value], -value))
            greedy_high_order.append(index)
            mask |= 1 << index
        high_cost = path_cost(greedy_high_order, costs)
        row = {
            "n": n,
            "beta_near_zero_max_reference_error": low_error,
            "beta_large_greedy_order": json.dumps(greedy_high_order),
            "beta_large": 1_000_000.0,
            "deterministic_optimal_order": json.dumps(best_order),
            "beta_large_path_cost": high_cost,
            "deterministic_minimum_cost": best_cost,
            "pass": low_error <= 1e-8 and abs(high_cost - best_cost) <= 1e-8,
        }
        rows.append(row)
    return rows, all(bool(row["pass"]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "theory_and_naming_v1"
    )
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    started = now_utc()
    if args.output_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exactness_rows, exactness_pass = _numerical_tests()
    beta_rows, beta_pass = _beta_limits()
    write_json(
        args.output_dir / "numerical_exactness_tests.json",
        {"rows": exactness_rows, "acceptance_pass": exactness_pass, "tolerance": 1e-8},
    )
    write_csv(args.output_dir / "beta_limit_tests.csv", beta_rows)
    write_csv(
        args.output_dir / "notation_table.csv",
        [
            {"symbol": "S", "meaning": "set of already revealed answer positions"},
            {"symbol": "Q(j|S)", "meaning": "positive-support reference reveal transition"},
            {"symbol": "c(S,j)", "meaning": "edited-model target-token negative log probability"},
            {"symbol": "beta", "meaning": "inverse path-control temperature"},
            {"symbol": "Z(S)", "meaning": "backward desirability/partition function"},
            {"symbol": "P_beta(j|S)", "meaning": "optimal Doob-transformed reveal transition"},
        ],
    )
    (args.output_dir / "formal_objective.md").write_text(
        """# Formal Objective

For monotone reveal paths from the empty mask pattern to the fully revealed
pattern, minimize

```text
E_P[sum_t c(S_t,j_t)] + (1/beta) KL(P || Q).
```

`Q` has positive support on every valid reveal transition. The terminal state
is deterministic for every monotone policy; there is no nontrivial learned or
prescribed terminal marginal.
""",
        encoding="utf-8",
    )
    (args.output_dir / "exact_recurrence.md").write_text(
        """# Exact Backward Recurrence

```text
Z(full) = 1
Z(S) = sum_{j not in S} Q(j|S) exp(-beta c(S,j)) Z(S union {j})
P_beta(j|S) = Q(j|S) exp(-beta c(S,j)) Z(S union {j}) / Z(S)
```

The implementation performs this recursion in log space on the finite subset
DAG and requires positive support after epsilon smoothing.
""",
        encoding="utf-8",
    )
    (args.output_dir / "proposition_and_proof.md").write_text(
        """# Proposition and Proof Sketch

On the finite monotone subset DAG, the displayed policy is the unique optimum
whenever the reference process has positive support on every valid transition.

Starting at the last decision layer, the Gibbs variational identity minimizes
the conditional expected immediate cost, downstream value, and conditional KL.
Its normalizer is the backward desirability `Z(S)`. Backward induction applies
the same identity at every predecessor state. Chain-rule decomposition of path
KL then gives the global objective, and strict convexity of conditional KL on
the supported simplex gives uniqueness. Brute-force path enumeration through
N=6 verifies the recurrence numerically.
""",
        encoding="utf-8",
    )
    (args.output_dir / "complexity_analysis.md").write_text(
        """# Complexity

There are `2^N` mask states and `N 2^(N-1)` valid directed transitions, so
exact backward planning is `O(N 2^N)` time and `O(2^N)` dynamic-program memory,
excluding edited-model state-cost queries. Building the full model cost table
can dominate GPU time and must be reported separately from planner CPU time.
""",
        encoding="utf-8",
    )
    naming = f"""# Naming Decision

Decision: `{NAMING_DECISION}`.

The controller is a finite-horizon linearly-solvable KL path-control problem
and its transition is a Doob/Feynman-Kac transform of the reference reveal
process. Every monotone path ends in the same fully revealed mask pattern, so
the endpoint constraint is trivial rather than a nontrivial pair of marginals.
It is therefore not described as a classical endpoint-constrained
Schrödinger bridge. Historical artifact names remain unchanged as provenance.
"""
    (args.output_dir / "naming_decision.md").write_text(naming, encoding="utf-8")
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P2",
        "stage": "P2_theory_and_naming",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "naming_decision": NAMING_DECISION,
        "classical_endpoint_constrained_sb": False,
        "finite_dag_optimality_derived": True,
        "positive_support_assumption_stated": True,
        "complexity": "O(N * 2^N) transitions excluding model queries",
        "beta_zero_limit_pass": beta_pass,
        "beta_infinity_limit_pass": beta_pass,
        "numerical_exactness_pass": exactness_pass,
        "maximum_n_exhaustively_validated": 6,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "acceptance_pass": exactness_pass and beta_pass,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="P2_theory_and_naming",
        track="P2",
        status="passed" if report["acceptance_pass"] else "failed",
        output_dir=args.output_dir,
        acceptance_pass=bool(report["acceptance_pass"]),
        started_at_utc=started,
        notes=f"naming={NAMING_DECISION}; exactness={exactness_pass}; beta_limits={beta_pass}",
        next_stage="P3_planner_baselines_dev",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
