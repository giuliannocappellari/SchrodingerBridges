# N4 — Selective Conformal Safe Editing Plan

## Hypothesis

Not every fact is safely editable. A pre-edit risk model plus calibrated selective policy may certify a substantial subset with strong efficacy and acceptable same-subject locality.

## Candidate underlying editors

Evaluate the selective wrapper on:

```text
partial-state MDM-MEMIT
best confirmed N1/N2/N3 candidate, if any
```

## Allowed risk features

Pre-edit or edit-training diagnostics only:

```text
base target rank/margin
subject-relation residual strength
rewrite/paraphrase gradient agreement estimated from training augmentations
rewrite/same-subject gradient conflict estimated from training anchors
Fisher condition/sensitivity summaries
causal-site stability
update norm prediction
target length and tokenizer structure
```

Forbidden:

```text
confirmation outcomes
evaluation prompt labels
post-edit test metrics
case IDs as predictive features
```

## Splits

```text
statistics_train: risk-model fitting
calibration: threshold/risk calibration
pilot100: first held-out selective evaluation
confirmation200: final direction confirmation
```

## Calibration

Use split-conformal/selective risk control or an explicitly documented one-sided risk bound. Report exchangeability assumptions and exact finite-sample calculation.

## Pilot success

Selective safe success:

```text
coverage >= 0.50
accepted rewrite >= 0.85
accepted paraphrase >= 0.45
accepted same-subject TFPR <= 0.03
95% upper risk bound <= 0.05
malformed <= 0.05
```

Strong success:

```text
coverage >= 0.60
95% upper risk bound <= 0.03
```

The rejected/abstained subset must be reported, not discarded silently.

## Rescue

One calibration rescue only:

```text
switch between monotone isotonic and logistic risk calibration
```

No new features after pilot inspection.
