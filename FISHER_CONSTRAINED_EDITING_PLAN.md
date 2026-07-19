# N2 — Fisher-Constrained Editing Plan

## Hypothesis

Euclidean-small updates can be behaviorally large. An empirical Fisher metric built from protected prompts should identify edit directions with high target gain and low collateral distribution change.

## Method

Estimate protected Fisher sketches from training-only same-subject, near, far, and unrelated prompts across partial states:

```text
F = E[g_protect g_protect^T]
```

Compare:

```text
Euclidean update
diagonal-Fisher natural gradient
low-rank Fisher + damping
Fisher trust-region update
```

Solve approximately:

```text
maximize g_edit^T d
subject to d^T F d <= rho
```

or use `(F + lambda I)^-1 g_edit` via conjugate gradients/low-rank Woodbury updates.

## Mechanism gate

```text
edit-signal / protected-sensitivity ratio improves >= 20%
protected Fisher quadratic form falls >= 20% at matched linearized edit gain
all Fisher sketches are finite and positive semidefinite within tolerance
```

## Pilot success

Full success:

```text
rewrite >= 0.85
paraphrase >= 0.45
same-subject TFPR <= 0.03
```

Pareto success:

```text
rewrite/paraphrase each within 0.02 of Euclidean baseline
same-subject TFPR reduced >= 20%
protected KL reduced >= 20%
paired protected-KL CI below 0
```

## Rescue

One bounded rescue:

```text
select damping from {1e-4,1e-3,1e-2}
and low-rank dimension from {32,64,128}
```

No full Hessian and no unbounded grid.
