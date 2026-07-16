# F1 — Adaptive External Edit-Memory Guidance Fallback

Protocol: `llada_adaptive_edit_memory_v1`

## Trigger

Run only if M1 fails to establish a positive MDM-MEMIT result after its one bounded rescue.

## Purpose

Obtain a strong corrected-answer result in LLaDA and determine whether the failure is specific to parametric editing rather than to the model's ability to use corrected factual evidence.

This is an engineering-positive fallback. It is not a permanent parameter edit and must not be presented as a strong Schrödinger-bridge result.

## Method

For each edit, store a provenance-bearing memory statement:

```text
subject
relation
new object
canonical declarative statement
source/edit ID
```

At inference:

1. retrieve the edit by exact subject/relation or a frozen learned relation gate;
2. form conditional and base/unconditioned LLaDA logits;
3. compute the distributional shift induced by the memory;
4. adapt the guidance scale over denoising steps based on a predeclared reliability/SNR statistic;
5. suppress guidance when the edit evidence is weak or the prompt asks another relation.

## Baselines

```text
base LLaDA
static prompt memory
fixed-scale retrieval guidance
adaptive retrieval guidance
hard constrained fill oracle (unranked)
```

## Data

Use fresh campaign CounterFact splits only. Do not use old locked analysis/final prompts.

## Acceptance

Positive engineering result:

```text
rewrite >= 0.70
paraphrase >= 0.40
same-subject TFPR <= base + 0.05
malformed <= 0.05
adaptive guidance beats fixed guidance on stress-aware aggregate
```

If this passes, report:

```text
LLaDA can express corrected facts with adaptive external evidence,
but the result is external-memory runtime control rather than parametric editing.
```

## Outputs

```text
runs/masked_diffusion_memit_sb_positive_result_v1/F1_adaptive_edit_memory_v1/
  report_summary.json
  retrieval_metrics.csv
  guidance_schedule.csv
  main_results.csv
  stress_results.csv
  paired_bootstrap.csv
  final_track_report.md
```
