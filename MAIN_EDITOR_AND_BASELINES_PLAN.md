# E/F — Main Editor, Baselines, Pilot, and Selection

## Required method registry

```text
base
prompt_memory
target_logit_bias
ordinary_mdm_memit
partial_state_mdm_memit
static_nullspace_partial_state_memit
timerome_source_style_fullmask
timerome_counterfact_fullmask
timerome_counterfact_partial_state
timerome_partial_state_state_bucketed
timerome_partial_state_state_protected
random_site_partial_state_residual
fixed_site_partial_state_residual
```

If triggered:

```text
timerome_partial_state_state_relation_protected
```

## Pilot ladder

### Smoke20

Purpose: integration and bounded calibration only.

Allowed calibration:

```text
alpha in a small predeclared grid
lambda ridge in a small predeclared grid
sparsity q in a small predeclared grid
state-bucket threshold sanity
```

No architecture changes.

### Pilot100

Fixed architecture comparison on fresh edits. Select at most three Pareto candidates:

```text
best full-editor candidate
best locality candidate
best multi-token/diffusion-specific candidate
```

### Dev200

Run the selected architectures and bounded shared hyperparameters. Freeze one primary and up to two secondary candidates.

## Core metrics

```text
rewrite exact
paraphrase exact
target-token F1
old-target suppression
same-subject TFPR
near/far TFPR
generation/attribute TFPR
malformed rate
distributional locality KL
residual norm and sparsity
model evaluations/edit
GPU minutes/edit
memory bytes/edit or per batch
```

## Primary comparison rules

1. Compare against the strongest efficacy-matched baseline, not only base.
2. Report paired bootstrap by edit ID.
3. Match or stratify compute where feasible.
4. Distinguish train-seen rewrite from held-out paraphrase/locality.
5. Report ordinary and state-conditioned TimeROME variants separately.

## Pilot eligibility

A candidate may advance if it satisfies at least one predeclared positive class:

```text
full editor claim
Pareto locality claim
diffusion-specific partial-state claim
state-conditioning claim
```

and:

```text
malformed <= 0.05
no data leakage
no catastrophic general-utility regression
all runtime inputs deployable
```

## Failure

If no candidate satisfies any positive class after permitted rescues, write a formal bounded negative package. Do not open locked confirmation.
