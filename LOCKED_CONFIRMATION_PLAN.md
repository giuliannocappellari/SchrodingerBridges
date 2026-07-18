# F2 — Fresh Locked Confirmation

## Purpose

Confirm the selected positive claim on new untouched data after all method and threshold choices are frozen.

## Fresh locked sets

```text
cf_trm_locked_500
kamel_trm_locked_200_per_length for lengths 2,3,4
same-subject locked stress set
near/far locked locality set
```

All facts and prompt fingerprints must be disjoint from localization, smoke, pilot, and dev sets.

## Lock file

Before opening locked data, write:

```text
runs/partial_state_temporal_residual_editor_v1/dev_method_lock.json
```

It must include:

```text
method ID
site policy
coordinate/layer selection
residual-memory formula
state buckets
alpha/lambda/q
protection policy
relation clusters if used
model/tokenizer hashes
split hashes
metrics
claim class
bootstrap procedure
code commit
```

No changes after locked inspection.

## Confirmation by claim class

### Full editor

```text
rewrite >= 0.85
paraphrase >= 0.40
same-subject TFPR <= base + 0.03
near/far TFPR <= base + 0.03
malformed <= 0.05
```

### Pareto locality

```text
rewrite and paraphrase within 0.02 of baseline
same-subject TFPR reduction >= 25%
paired 95% CI for delta below 0
near/far no material worsening
```

### Diffusion-specific

```text
pooled multi-token rewrite gain >= 0.10 on at least two lengths
paired pooled lower bound > 0
malformed <= 0.05
```

### State-conditioning

```text
same-subject TFPR reduction >= 20% vs shared residual
at matched efficacy
paired evidence positive
```

## Failure

A locked failure is terminal for v1. Do not tune on locked data or create v2 automatically.
