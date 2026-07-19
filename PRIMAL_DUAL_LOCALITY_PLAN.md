# N3 — Primal-Dual Constrained Locality Plan

## Hypothesis

Fixed locality penalties do not guarantee safety. An augmented-Lagrangian/primal-dual editor should satisfy explicit locality constraints more reliably than a fixed weighted loss.

## Optimization

Minimize edit loss plus trust-region/Fisher cost subject to differentiable locality surrogates:

```text
same-subject target-margin surrogate <= eps_subject
near KL <= eps_near
far KL <= eps_far
update energy <= eps_energy
```

Use:

```text
L_AL = L_edit + sum_j lambda_j c_j + (mu/2) sum_j relu(c_j)^2
lambda_j <- relu(lambda_j + eta * c_j)
```

Exact TFPR is evaluation-only; it is never differentiated through.

## Required comparisons

```text
fixed penalty partial-state editor
primal-dual partial-state editor
primal-dual + Fisher trust region if N2 mechanism gate passed
```

## Mechanism gate

```text
constraint violation decreases over optimization
no multiplier divergence/NaNs
>= 80% of calibration edits satisfy all differentiable constraints
primal-dual constraint satisfaction exceeds fixed penalty by >= 15 percentage points
```

## Pilot success

Full success:

```text
rewrite >= 0.85
paraphrase >= 0.45
same-subject TFPR <= 0.03
near/far budgets pass
```

Constraint/Pareto success:

```text
rewrite >= 0.80
paraphrase >= 0.40
exact held-out locality constraint satisfaction improves >= 20 percentage points
same-subject TFPR reduced >= 25% versus fixed penalty
paired CI below 0
```

## Rescue

One rescue only:

```text
multiplier step in {0.01,0.05,0.1}
penalty growth in {1.5,2.0}
```

Architecture and constraints remain fixed.
