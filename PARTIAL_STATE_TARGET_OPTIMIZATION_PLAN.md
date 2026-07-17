# Partial-State Target Optimization Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Objective

Make a parametric edit robust to the intermediate partially unmasked states visited by masked-diffusion decoding, especially for multi-token targets.

## State construction

For target tokens `y_1 ... y_N`, create states with exactly `k` revealed target positions for every:

```text
k = 0, 1, ..., N-1
```

At each optimization step:

```text
1. choose mask count k by the configured schedule;
2. sample the revealed-position subset uniformly or from the configured policy;
3. insert true target tokens at revealed positions;
4. leave remaining positions masked;
5. compute target loss only on masked target positions;
6. collect the factual key/value at the selected causal site.
```

## Required policies

```text
fully_masked_only
all_mask_counts_random_positions
uniform_mask_count_states
confidence_trajectory_states
three_bucket_states
```

`three_bucket_states` means:

```text
full: k = 0
intermediate: 1 <= k <= N-2
late: k = N-1
```

## Target-value objective

Primary objective:

```text
L_target:
  masked target-token negative log likelihood across states

L_old_suppress:
  margin against target_true on positive edit states

L_state_consistency:
  penalize variance in target support across mask states

L_value_norm:
  cap target residual magnitude
```

Starting loss:

```text
L = L_target
  + 0.25 * L_old_suppress
  + 0.10 * L_state_consistency
  + lambda_value_norm * L_value_norm
```

## Bounded grid

```text
value_lr in {0.05, 0.10}
value_steps in {25, 50}
state_consistency_weight in {0.0, 0.1}
old_target_suppression_weight in {0.0, 0.25}
revealed_position_seed in {0, 1}
```

Use staged narrowing rather than a full Cartesian sweep.

## Datasets

```text
CounterFact:
  single-token reproduction and standard editing

KAMEL-compatible data:
  exact target lengths 2, 3, 4; optional 5
```

All target lengths are computed with the active model tokenizer and context-aware suffix tokenization.

## Required comparisons

```text
fullmask-only MDM-MEMIT
partial-state MDM-MEMIT
causal-site fullmask editor
causal-site partial-state editor
causal partial-state null-space editor
```

## Metrics

```text
full-span rewrite exact
full-span paraphrase exact
target-token F1
partial-target rate
old-target retention/suppression
malformed rate
success by mask count
success by revealed-position pattern
variance of target probability across states
```

## Acceptance

A diffusion-state-specific result requires:

```text
full-span exact gain >=0.10 on at least two of target lengths 2,3,4
```

or:

```text
pooled multi-token paired delta >0 with positive confidence interval
and no same-subject/locality regression
```

A strong partial-state result additionally requires:

```text
paraphrase gain >=0.08 on at least two lengths
malformed <=0.05
```

## Failure analysis

If partial-state optimization fails, classify failures as:

```text
target not in candidate support
key drift across mask states
target value conflict across states
wrong target position tokenization
old target persists
partial completion/malformed span
state schedule mismatch
```

## Bounded rescue

If one shared target value fails because state-specific keys are incompatible, one rescue may use three state-bucket target values or residuals. No per-state unconstrained model family is allowed.
