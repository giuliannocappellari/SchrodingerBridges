# C — CounterFact Temporal Residual Adaptation

## Objective

Adapt temporal residual edit memory from forgetting/insertion to factual replacement:

```text
(subject, relation, target_true -> target_new)
```

## Fresh data

Create disjoint fresh manifests from allowed CounterFact rows:

```text
cf_trm_localize_50
cf_trm_smoke_20
cf_trm_pilot_100
cf_trm_dev_200
cf_trm_locked_500
cf_trm_scaling_100
```

Exclude historical source rows, fact tuples, rendered prompts, and fingerprints used in earlier campaigns. Keep historical analysis/final splits closed.

## Temporal localization

For each localization edit, evaluate:

```text
layers: all or a coarse-to-fine sweep
module families: MLP, attention, residual stream
positions: first subject, last subject, answer mask
states: fully masked, early, middle, late, actual confidence trajectory
restoration target: target_new probability/margin and decoded support
```

Compute:

```text
standard indirect effect
temporal indirect effect
site stability across paraphrases
site stability across mask patterns
site stability across seeds
```

## Site policies

```text
source-paper fixed site
per-edit highest-TIE site
stable temporal site set
last-subject early/mid-MLP site
random site
late answer-mask site
```

## CounterFact residual memory

For a chosen coordinate, extract keys and optimize a target delta that increases target_new and suppresses target_true. Fit:

\[
M = D K^\top (K K^\top + \lambda I)^{-1}.
\]

At runtime:

\[
h'_t = h_t + \alpha\,\mathrm{Sparse}_q(M k_t).
\]

## Acceptance for temporal localization

At least one causal/temporal site policy must either:

```text
improve the pilot stress-aware aggregate by >= 0.05 over random-site residual editing;
```

or:

```text
match rewrite within 0.02 while reducing update energy or the number of active coordinates/layers by >= 25%.
```

## Outputs

```text
causal_trace_summary.csv
temporal_trace_summary.csv
site_stability.csv
site_policy_comparison.csv
residual_memory_schema.json
report_summary.json
```

## Bounded rescue

One site-policy rescue may replace unstable per-edit coordinates with the best stable layer/position set selected on `cf_trm_localize_50`. No rescue may use pilot or locked evaluation outcomes.
