# D2 — State-Conditioned Locality Preservation

## Objective

Improve same-subject and ordinary locality without forcing one static protected subspace to cover every denoising state.

## Main protected residual formulation

For state bucket b, fit a residual memory under an explicit preservation penalty:

\[
\min_{M_b}
\|M_b K_{edit,b} - D_b\|_F^2
+ \lambda_{ridge}\|M_b\|_F^2
+ \lambda_{pres}\|M_b K_{protect,b}\|_F^2.
\]

Equivalent closed-form or stable numerical solvers are allowed if verified against a small synthetic solution.

State buckets:

```text
early masked
partially revealed middle
nearly complete late
```

## Protection anchors

Training-only anchors must include:

```text
same subject, different relation
near locality
far locality
attribute prompts
generation prompts
unrelated prompts
partial states for all anchor families
```

Held-out official/stress evaluation prompts are forbidden as protection anchors.

## Variants

```text
no protection
static global null-space projector
shared soft preservation penalty
state-conditioned preservation penalty
state-conditioned sparsification
state + relation-cluster conditioned protection (rescue only)
```

## Relation rescue trigger

The relation-conditioned rescue is legal only if:

```text
state-conditioned method keeps rewrite within 0.02 of the unprotected method;
same-subject TFPR improves by at least 10%;
but same-subject TFPR remains above the frozen full-editor threshold.
```

Use relation clusters derived from training metadata only.

## Main success definitions

### Full editor

```text
rewrite >= 0.85
paraphrase >= 0.40
same-subject TFPR <= base + 0.03
near/far TFPR <= base + 0.03
malformed <= 0.05
```

### Pareto locality result

At matched efficacy against the strongest baseline:

```text
rewrite loss <= 0.02
paraphrase loss <= 0.02
same-subject TFPR reduction >= 25%
paired CI for TFPR delta below 0
near/far no material worsening
```

### State-conditioning result

```text
state-conditioned protection reduces same-subject TFPR by >= 20%
relative to shared protection at matched efficacy
or improves stress-aware aggregate by >= 0.05.
```

## Required diagnostics

```text
residual activation by state bucket
same-subject target margin by bucket
preservation-key drift
retain distribution KL
residual norm and sparsity
relation breakdown
target-length breakdown
```
