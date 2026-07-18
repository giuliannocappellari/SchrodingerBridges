# D1 — Partial-State Target-Delta Optimization

## Hypothesis

A target delta optimized only at a fully masked answer state does not support the intermediate partially revealed states visited by masked diffusion decoding. Optimizing the residual target across these states should improve multi-token editing.

## State families

For a target of length N, materialize:

```text
fully masked state
all mask-count levels 1..N-1
random revealed-position patterns
actual confidence-decoding states
early revealed, middle, and late state buckets
```

Do not use evaluation outcomes to choose states.

## Target-delta variants

```text
fullmask_delta
uniform_partial_state_delta
mask_count_cycling_delta
trajectory_sampled_delta
state_bucketed_delta (early/middle/late)
```

For positive states, optimize target_new likelihood and target_true suppression. For identity/preservation states, the desired residual is zero or base-preserving.

## Required comparison

```text
ordinary MDM-MEMIT
partial-state MDM-MEMIT
fullmask temporal residual
partial-state shared temporal residual
partial-state state-bucketed temporal residual
```

## Multi-token datasets

Use fresh KAMEL splits:

```text
kamel_trm_dev_50_per_length
kamel_trm_pilot_100_per_length
kamel_trm_locked_200_per_length
lengths = 2,3,4
```

Where possible add lengths 5 and 6 as secondary scaling diagnostics.

## Primary metrics

```text
full-target rewrite exact
paraphrase exact
target-token F1
old-target suppression
malformed rate
state coverage
residual norm
model evaluations
wall-clock time
```

## Diffusion-specific pass

The partial-state temporal residual supports a positive diffusion-specific claim if:

```text
rewrite gain >= 0.10 over fullmask temporal residual
on at least two target-length bins;
pooled paired-bootstrap lower bound > 0;
malformed rate <= 0.05;
no material same-subject/locality regression.
```

## Stronger mechanism evidence

At least one must hold:

```text
state-bucketed residual > shared residual by >= 0.05 on the stress-aware aggregate;
actual-trajectory states > ordinary independent masking by >= 0.05;
state shuffling materially reduces performance.
```

## Bounded rescue

One ridge/sparsity rescue may adjust only the predeclared `alpha`, `lambda`, and `q` grid. No architecture expansion is allowed here.
