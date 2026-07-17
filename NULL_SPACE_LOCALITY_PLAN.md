# Null-Space and Locality Preservation Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Objective

Reduce unintended changes—especially same-subject different-relation leakage—by constraining the parametric update to directions that minimally affect protected factual keys.

## Preservation data

Use training-only prompts, disjoint from all evaluation prompts.

Categories:

```text
same_subject_different_relation
different_subject_same_relation
near_locality
far_locality
attribute
generation
unrelated
```

The strongest protection weight is assigned to same-subject different-relation keys.

## Protected key matrix

For selected layer `l`:

```text
K_minus[l] = concatenated MLP input keys from preservation states
```

Compute:

```text
SVD or eigendecomposition of K_minus K_minus^T
U_r = protected basis covering configured explained variance
N = I - U_r U_r^T
```

Alternative covariance/statistic forms may be used only if derived from AlphaEdit/MEMIT and dimensionally validated.

## Constrained update

Primary formulation:

```text
Delta_W = D N
```

Solve:

```text
min_D ||(W + D N)K_plus - V_star||_F^2
    + lambda_update ||D N||_F^2
    + lambda_identity ||D N K_minus||_F^2
```

Report:

```text
protected dimension
remaining editable dimension
projection energy ratio
positive-key projection loss
condition number
update rank and norm
```

## Bounded grid

```text
protected_variance in {0.90, 0.95, 0.99}
lambda_update in {1e-4, 1e-3, 1e-2}
lambda_identity in {0.1, 1.0, 2.0}
```

Use staged selection:

```text
1. choose protected variance by editability/locality curve;
2. choose ridge/update regularization;
3. choose identity weight.
```

## Baselines

```text
ordinary MDM-MEMIT
partial-state MDM-MEMIT
MDM-MEMIT + target-value KL anchors
AlphaEdit-style projected MDM-MEMIT
causal partial-state editor without projection
causal partial-state null-space editor
```

## Metrics

```text
same-subject TFPR
near/far TFPR
generation/attribute target injection
pre/post output agreement
sparse answer-position KL/JS
protected-key output drift
rewrite/paraphrase efficacy
update norm/rank
model utility diagnostics
```

## Primary acceptance

The full method must satisfy:

```text
same-subject TFPR <= base + 0.03
near TFPR <= base + 0.03
far TFPR <= base + 0.03
malformed <=0.05
```

Relative to the strongest efficacy-matched baseline, at least one:

```text
same-subject TFPR reduction >=50%
distributional locality KL reduction >=25%
protected-key output drift reduction >=25%
```

while:

```text
rewrite drop <=0.05
paraphrase drop <=0.05
```

## Safety checks

Fail the configuration if:

```text
remaining editable dimension is effectively zero
condition number is unstable after permitted regularization
update contains NaN/Inf
projection is computed from evaluation prompts
locality improvement comes only from eliminating edit efficacy
```

## Bounded rescue

Only the predeclared protected-variance/ridge/identity grid is allowed. Do not add evaluation-derived preservation keys or lower the efficacy floor.
