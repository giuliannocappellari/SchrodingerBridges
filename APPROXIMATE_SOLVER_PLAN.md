# P7 — Approximate Solver and Target-Length Scaling

## Objective

Characterize exact-solver scaling and test whether most of the gain can be
retained with fewer state queries.

## Exact solver

Evaluate exact dynamic programming for:

```text
N = 2,3,4,5,6
```

Report:

```text
number of mask states
number of transitions
cost-table forward evaluations
planner CPU time
peak memory
total wall-clock
```

## Approximate solver

Implement one bounded approximation family:

```text
beam-truncated backward recursion
or
Monte Carlo partition-function/backward-value estimation
```

The implementation must preserve the same reference process and cost function.

One approximation-configuration repair is allowed.

## Development grid

Bounded:

```text
beam/state budget in {2N,4N,8N}
or
rollouts in {8,16,32}
```

Do not expand the grid.

## Evaluation

For N=5 and N=6, compare approximate to exact.

If source data exists, evaluate N=7 through N=10 on at least 100 edits per
length, using the approximate solver and compute-matched baselines.

## Acceptance

Strong approximation:

```text
retains >=80% of exact rewrite gain
uses <=50% of exact unique state queries
same-subject and malformed constraints pass
```

Minimum useful approximation:

```text
retains >=70% of exact gain
uses <=50% of exact queries
```

For N=7 through N=10, a positive result is:

```text
rewrite delta >= +0.03 over compute-matched myopic/beam
or
rewrite within 0.02 with trajectory target cost >=20% lower
```

A failed approximation does not invalidate exact N<=6 results, but it must be
reported as a scalability limitation.

## Outputs

```text
runs/.../approximate_solver_v1/
  report_summary.json
  exact_scaling.csv
  approximate_vs_exact.csv
  long_target_results.csv
  query_budget_table.csv
  runtime_memory_table.csv
  approximation_decision.md
```
