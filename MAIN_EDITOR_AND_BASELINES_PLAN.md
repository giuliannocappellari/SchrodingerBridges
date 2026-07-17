# Main Editor and Baselines Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Method registry

### B0 — Base

Frozen unedited model.

### B1 — Prompt memory

Runtime statement of the edit; required non-parametric baseline.

### B2 — Target logit bias

Simple target pressure; diagnostic baseline.

### B3 — MDM-MEMIT

Source-aligned locate-then-edit baseline using masked answer inputs.

### B4 — Partial-state MDM-MEMIT

MDM-MEMIT target value optimized over partial mask states.

### B5 — MDM-MEMIT + target-value KL anchors

Preservation enforced only during target-value optimization; distinguishes objective regularization from null-space geometry.

### B6 — AlphaEdit-style MDM-MEMIT

Standard update projected into a preservation null space.

### B7 — TimeROME-DLM-style residual memory

Temporal causal site plus ridge-regularized sparse low-rank residual applied during diffusion forwards. Label as `-style` unless official algorithm/code is reproduced exactly.

### B8 — Random-site partial-state editor

Same target/state/locality objective at random sites.

### B9 — Fixed-site partial-state editor

Best global source-aligned layer/position.

### B10 — Causal-site fullmask editor

Causal site with only fully masked target optimization.

### B11 — Causal-site partial-state editor

Causal site and partial-state target value, no null-space constraint.

### M0 — Causal partial-state null-space MEMIT

Primary method.

### R0 — State-conditioned low-rank residual rescue

Three mask-state buckets; only if rescue trigger is met.

## Main evaluation table

| Method | Rewrite | Paraphrase | Target F1 | Same-subj TFPR | Near TFPR | Far TFPR | Dist. KL | Update norm | Rank | Edit time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Stress-aware aggregate

Use a harmonic/geometric aggregate only after hard constraints:

```text
rewrite
paraphrase
clipped self-normalized locality
```

Methods violating TFPR/malformed constraints remain infeasible regardless of aggregate.

## Pilot sequence

### Smoke20

Purpose: integration and catastrophic-failure detection.

One bounded calibration may adjust:

```text
update scale
projector rank within the frozen grid
```

No architecture changes.

### Pilot100

Purpose: choose site/partial-state/null-space configuration.

Select at most three candidates:

```text
best aggregate
best locality
best multi-token robustness
```

### Dev200

Purpose: final method selection. No new method family after inspection.

## Mechanism ablations

Required:

```text
causal site vs random site
partial state vs fullmask only
null space vs no null space
state-conditioned vs step-agnostic, only if rescue triggered
same preservation anchors with KL-only vs geometric projection
```

## Acceptance to proceed to locked analysis

At least one primary candidate must:

```text
rewrite >=0.75
paraphrase >=0.40
same-subject/near/far TFPR <= base +0.03
malformed <=0.05
improve locality over the strongest efficacy-matched baseline
show causal or partial-state mechanism value
```

## Failure taxonomy

```text
edit construction failure
causal-site instability
target-value conflict across states
null-space removes edit direction
same-subject leakage
near/far leakage
old target persists
multi-token partial completion
malformed span
batch/sequential interference
```
