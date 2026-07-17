# P8 — Statistics, Reproducibility, and Publication Package

## Objective

Produce a self-contained package that supports a precise readiness decision and
can reproduce every paper table and figure.

## Statistical analysis

Required:

```text
paired bootstrap by edit_id
10,000 resamples
95% confidence intervals
Holm correction for primary N=3/N=4 tests
micro average
macro-by-relation average
macro-by-target-length average
seed-level variability
effect sizes
```

Do not report only aggregate point estimates.

## Required analyses

```text
beta sweep and beta limits
finite beta vs deterministic planning
full-table vs online compute-matched
target length 2 through 6
base target rank/probability buckets
relation buckets
token-frequency buckets
shared-prefix/subword structure
number of plausible reveal orders
trajectory cost vs exact-success correlation
same-subject and locality safety
malformed and partial-target failures
```

## Trajectory examples

For successful and failed edits, show:

```text
default confidence trajectory
one-step myopic trajectory
deterministic global trajectory
finite-beta trajectory
```

At every state report:

```text
revealed positions
target-token probabilities
immediate cost
backward value/partition
transition probabilities
remaining expected cost
```

## Reproducibility

Freeze and record:

```text
Git commit
model revisions
tokenizer hashes
dataset/source fingerprints
split manifests
MEMIT configuration
edited layer window
covariance source/hash
cost definition
reference process
beta
planner/query budget
generation schedule
seeds
package versions
RunPod image/CUDA/GPU
```

Required commands:

```text
python reproduce_paper.py --table main
python reproduce_paper.py --figure main
python reproduce_paper.py --check-dp
```

`--check-dp` must run cheaply without loading LLaDA.

## Final package

Create:

```text
runs/mask_pattern_sb_publication_confirmation_v1/
  final_publication_package_v1/
```

Required:

```text
report_summary.json
top_tier_readiness.json
main_results_table.csv
compute_matched_table.csv
second_backbone_table.csv
editor_generality_table.csv
target_length_table.csv
beta_ablation.csv
planner_ablation.csv
same_subject_stress_table.csv
malformed_and_locality_table.csv
paired_bootstrap.csv
holm_corrected_tests.csv
power_analysis.json
theory_statement.md
naming_decision.md
complexity_analysis.md
trajectory_examples.md
failure_cases.csv
artifact_availability.json
reproducibility_manifest.json
final_research_report.md
paper_outline.md
paper_claim_recommendation.md
```

## Readiness decision

### `top_tier_ready`

All:

```text
partial-state discrepancy resolved or concretely explained
fresh locked LLaDA primary test passes
best compute-matched non-SB planner is beaten
finite beta adds value beyond beta=0 and beta=infinity
second backbone has consistent positive evidence
positive result under at least two editor conditions
formal naming is defensible
locality and malformed constraints pass
complete reproducibility package validates
```

### `narrow_method_ready`

Examples:

```text
fresh LLaDA result passes but second backbone is unavailable/inconclusive
method beats fixed schedules but not best compute-matched global planning
formal result is entropy-regularized planning rather than classical SB
approximate solver fails but exact short-span method is positive
```

### `diagnostic_only`

Examples:

```text
fresh gain disappears under compute matching
deterministic global planning explains all gain
finite beta adds no value
benefit occurs at only one target length
second backbone reverses the result
```

### `fresh_confirmation_failed`

The fresh locked LLaDA primary result fails.

After package validation, update campaign state and stop the Pod.
