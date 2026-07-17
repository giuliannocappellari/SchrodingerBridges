# P4 — Fresh Locked LLaDA Confirmation

## Objective

Confirm the mask-pattern controller on fresh, previously unseen KAMEL facts
under a completely frozen LLaDA protocol.

## Pre-lock requirements

Before opening the locked split, freeze:

```text
MDM-MEMIT implementation and layer window
partial-state baseline implementation
edited model/update artifacts
target-value optimization
mask-pattern state definition
reference process
cost function
finite beta
online query budget
best non-SB planner
beam width
generation steps and schedule
span policy
random seeds
metrics
bootstrap code
```

Validate:

```text
runs/.../dev_method_lock.json
```

## Locked data

Primary:

```text
500 fresh N=3 edits
500 fresh N=4 edits
```

Secondary:

```text
300 fresh N=2 edits
300 fresh N=5 edits
300 fresh N=6 edits
```

If primary data falls below 500 because of source limitations, record the exact
power loss. Fewer than 300 primary edits caps the final classification at
`narrow_method_ready`.

## Locked methods

```text
ordinary MDM-MEMIT + default confidence
paper-matched partial-state MDM-MEMIT + default confidence
left-to-right
right-to-left
uniform random
best fixed permutation
one-step myopic
deterministic global minimum-cost DP
best compute-matched beam/random planner
beta=0 reference
exact finite-beta controller
approximate controller if already frozen
```

No new method may be added after locked results are inspected.

## Seeds

```text
>=3 generation seeds
>=5 random-policy seeds
```

Average random policies over seeds. Do not select a favorable seed.

## Metrics

```text
full-target rewrite exact
paraphrase exact
target-token F1
old-target suppression
malformed rate
same-subject TFPR
near/far locality
trajectory target cost
path entropy
KL from reference
unique state queries
model evaluations
planner time
GPU time
wall-clock time
```

## Primary statistical test

Primary comparison:

```text
finite-beta exact controller
vs best dev-selected compute-matched non-SB planner
pooled over N=3 and N=4
```

Required:

```text
10,000 paired bootstrap samples by edit_id
95% CI
Holm correction over the separate N=3 and N=4 tests
```

## Pass criteria

Minimum credible locked pass:

```text
pooled N=3/N=4 rewrite delta >= +0.05
pooled paired-bootstrap lower bound > 0
Holm-corrected primary result positive
each primary length mean delta >= 0
at least one primary length delta >= +0.05 with lower bound > 0
trajectory target cost reduction >= 15%
malformed rate <= 0.05
same-subject TFPR increase <= 0.03
target-token F1 does not fall by more than 0.02
```

Strong pass:

```text
significant positive rewrite gain at both N=3 and N=4
positive or nonnegative paraphrase trend
compute-matched criterion passes
```

## Failure

If the locked primary result fails, do not tune on it.

Write:

```text
fresh_confirmation_failed
```

and continue only with analyses that were preplanned and do not alter the
primary claim.

## Outputs

```text
runs/.../llada_locked_confirmation_v1/
  report_summary.json
  main_results.csv
  target_length_results.csv
  compute_matched_results.csv
  paired_bootstrap.csv
  holm_corrected_tests.csv
  random_seed_summary.csv
  same_subject_stress.csv
  locality_malformed.csv
  trajectory_cost_table.csv
  failure_cases.csv
  locked_result_interpretation.md
```
