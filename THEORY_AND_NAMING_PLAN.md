# P2 — Theory, Exact Solver, and Naming Audit

## Objective

Formalize the mask-pattern controller precisely and determine the strongest
mathematically defensible name.

## State space

For a target of length `N`, let:

```text
S subset of {1,...,N}
```

denote positions already revealed.

The initial state is the empty set and the terminal mask pattern is the full
set.

A transition reveals one unrevealed position `j`.

## Reference process

Define:

```text
Q(j | S)
```

for at least these references:

```text
uniform reveal
edited-model confidence reveal
paper/default LLaDA reveal reference
```

All valid transitions must have positive support after epsilon smoothing.

## Cost

Primary cost:

```text
c(S,j) = -log p_edited(target_token_j | prompt, revealed target tokens S)
```

Record optional span/global variants as ablations only.

## KL-control objective

Formalize:

```text
min_P  E_P[sum_t c(S_t,j_t)] + (1/beta) KL(P || Q)
```

over monotone mask-pattern paths.

Derive the backward recursion:

```text
Z(full) = 1

Z(S) =
  sum_{j not in S}
    Q(j|S) * exp(-beta*c(S,j)) * Z(S union {j})
```

and transition:

```text
P_beta(j|S) =
  Q(j|S) * exp(-beta*c(S,j)) * Z(S union {j}) / Z(S)
```

## Mandatory theory tasks

1. Prove or derive optimality on the finite DAG.
2. State assumptions and support conditions.
3. Derive complexity:
   `O(N * 2^N)` states/transitions, excluding model-cost queries.
4. Show:
   - `beta -> 0` approaches the reference process;
   - `beta -> infinity` approaches deterministic global minimum-cost planning.
5. Distinguish path-control cost from actual GPU compute.
6. Explain whether the terminal constraint is nontrivial.

## Numerical validation

For N <= 6:

```text
enumerate every reveal permutation
compute its reference probability and cumulative cost
compute exact Gibbs path distribution by brute force
compare path marginals/transitions with dynamic programming
```

Tolerance:

```text
partition function relative error <= 1e-8 in float64 synthetic tests
transition probabilities sum to 1 within 1e-8
DP and brute-force expected cost agree within 1e-8
```

## Naming decision

Write one of:

```text
classical_schrodinger_bridge
generalized_schrodinger_bridge
schrodinger_style_kl_path_control
doob_transformed_mask_pattern_control
entropy_regularized_global_planning
```

If all monotone paths share a deterministic terminal mask pattern and there is
no nontrivial endpoint marginal constraint, the default safe classification is
`schrodinger_style_kl_path_control` or
`doob_transformed_mask_pattern_control`.

Do not preserve an inflated name for marketing reasons.

## Mechanism ablations

Compare:

```text
beta = 0
finite beta grid
beta -> infinity deterministic global path
one-step myopic
DP without entropy regularization
```

A bridge/KL-control-specific claim requires a finite beta to outperform both
the uncontrolled reference and deterministic limit on a meaningful
quality/cost trade-off.

## Outputs

```text
runs/.../theory_and_naming_v1/
  report_summary.json
  formal_objective.md
  proposition_and_proof.md
  exact_recurrence.md
  complexity_analysis.md
  numerical_exactness_tests.json
  beta_limit_tests.csv
  naming_decision.md
  notation_table.csv
```
