# P3 — Reveal-Order and Compute-Matched Baseline Suite

## Objective

Determine whether the exact finite-beta controller adds value beyond fixed
orders, local heuristics, deterministic global planning, beam search, random
search, and extra model evaluations.

## Mandatory planners

```text
default edited-LLaDA confidence reveal
left-to-right
right-to-left
uniform random reveal, averaged over >=5 seeds
best fixed permutation selected on dev only
maximum-confidence reveal
minimum-entropy reveal
one-step myopic target-cost reveal
deterministic global minimum-cost DP (beta -> infinity)
beam search widths 2,4,8
random search with matched state-query budget
beta=0 reference process
exact finite-beta controller
```

If official code for a relevant modern adaptive scheduler is available and
compatible, include one official implementation. Its absence must not block
the campaign.

## Cost-table regime

For N <= 6, precompute the cost of every unique mask state once.

Every global planner receives the same table.

Report:

```text
planner quality
planner CPU time
path entropy
KL from reference
expected trajectory target cost
```

This regime isolates the planner/control mechanism.

## Online compute-matched regime

A state cost is hidden until queried.

Give all methods matched budgets in:

```text
unique state queries
LLaDA/Dream forward evaluations
candidate-path expansions
wall-clock cap
```

Required budgets include at least:

```text
N
2N
4N
2^N, where feasible
```

For exact DP, report the natural full-table cost and an approximate version
under smaller budgets.

## Development selection

On dev data only, select:

```text
reference process
finite beta from {0.25,0.5,1,2,4}
best beam width/query budget
best deterministic/non-SB baseline
online primary query budget
```

Also evaluate `beta=0` and deterministic `beta=infinity`.

Freeze the selection in:

```text
runs/.../dev_method_lock.json
```

## Mechanism criteria

For an SB/KL-control claim, finite beta must satisfy at least one:

```text
rewrite delta >= +0.03 over deterministic global planning
with paired lower bound > 0

or

rewrite within 0.02 of deterministic planning
with trajectory target cost >=20% lower

or

rewrite within 0.02
with a meaningfully lower KL/entropy-cost trade-off
```

Finite beta must also outperform `beta=0` and one-step myopic.

If `beta=infinity` is always best, classify the result as global planning rather
than an SB-specific contribution.

## Compute-matched publication criterion

On the locked primary comparison:

```text
rewrite delta >= +0.03 over best compute-matched non-SB baseline

or

rewrite within 0.02 while trajectory target cost is >=20% lower

or

an approximate controller retains >=80% of the exact-controller gain
with <=50% of state queries
```

## Outputs

```text
runs/.../planner_baselines_dev_v1/
  report_summary.json
  planner_results.csv
  compute_matched_results.csv
  beta_sweep.csv
  reference_process_ablation.csv
  beam_random_search_results.csv
  planner_query_accounting.csv
  path_entropy_kl.csv
  dev_method_lock.json
```
