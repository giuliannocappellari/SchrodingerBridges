# N1 — Relation-Residualized Editing Plan

## Hypothesis

A factual key is confounded by broad subject and relation main effects. Removing those nuisance effects should isolate a subject×relation interaction direction that edits the requested fact with less same-subject leakage.

## Method

For state bucket `b`, estimate with cross-fitting:

```text
h(s,r,b) = mu_b + a(s,b) + b(r,b) + c(s,r,b) + error
```

Use a difference-in-differences residual:

```text
h_resid(s,r,b)
  = h(s,r,b)
  - mean_{r'!=r} h(s,r',b)
  - mean_{s'!=s} h(s',r,b)
  + mean_{s',r'} h(s',r',b)
```

Construct edit keys/gradients from `h_resid`, not the raw subject representation.

## Required variants

```text
raw partial-state MDM-MEMIT
subject-centered only
relation-centered only
full subject+relation residualization
hierarchical relation-cluster shrinkage rescue
```

## Offline mechanism gate

On calibration data:

```text
positive-vs-same-subject Fisher discriminant ratio improves >= 25%
rewrite-gradient cosine with same-subject gradients decreases >= 20%
relation classification from residualized representation remains above chance
cross-fitted estimates use no evaluation rows
```

At least two must pass to proceed to actual editing.

## Pilot success

Full success:

```text
rewrite >= 0.85
paraphrase >= 0.45
same-subject TFPR <= 0.03
```

Pareto success:

```text
rewrite/paraphrase each within 0.02 of baseline
same-subject TFPR reduced >= 25%
protected KL reduced >= 20%
paired TFPR CI below 0
```

## Rescue

One rescue only:

```text
replace raw relation means with hierarchical relation-cluster shrinkage
```

No new feature family or evaluation data may be added.
