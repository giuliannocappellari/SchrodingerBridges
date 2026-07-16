# F2 — Fixed-Template Categorical Text CSBM Sanity Experiment

Protocol: `toy_text_csbm_v1`

## Trigger

Run only if both M3 and M4 fail to establish an SB-specific positive result.

## Purpose

Test a true categorical Schrödinger bridge on a controlled relational-text distribution with fixed length and finite vocabulary. This is an algorithmic sanity result, not LLaDA model editing.

## Dataset

Construct fixed-template examples such as:

```text
Alice lives in Paris .
Bob lives in Rome .
Carlos lives in Madrid .
```

Define source and target distributions by systematic factual transformations with:

```text
fixed sequence length
fixed relation vocabulary
many source and target samples
train/validation/test split by entity/fact
```

Create at least:

```text
5,000 train pairs or unpaired samples per endpoint domain
1,000 validation
1,000 test
```

## Methods

```text
ordinary categorical noising + endpoint predictor
forward-only bridge matching
bidirectional D-IMF / CSBM
simple conditional classifier/denoiser
```

Use a finite categorical reference with full support.

## Mechanism criteria

Positive CSBM result requires:

```text
endpoint exact >= 0.90
identity/unaffected-token preservation >= 0.95
bidirectional D-IMF improves endpoint exact or path KL over forward-only
bridge-state training improves over ordinary noising
held-out entity/fact generalization reported
```

Do not claim LLaDA editing.

## Outputs

```text
runs/masked_diffusion_memit_sb_positive_result_v1/F2_toy_text_csbm_v1/
  report_summary.json
  dataset_spec.json
  train_curves.csv
  endpoint_results.csv
  path_metrics.csv
  ablation_results.csv
  generated_examples.jsonl
  final_track_report.md
```
