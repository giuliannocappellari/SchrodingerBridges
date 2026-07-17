# P5 — Second-Backbone Confirmation on Dream

## Objective

Test whether the path-control effect generalizes beyond LLaDA.

Primary second backbone:

```text
Dream-v0-Instruct-7B
```

## Step 1 — Model and editor audit

Map:

```text
tokenizer and mask token
denoising API
hidden/module names
MLP target matrices
subject-token location
editable floating-point weights
generation schedule
```

Reproduce a small paper-style MDM-MEMIT edit smoke before evaluating the path
controller.

One bounded model-adapter/module-mapping repair is allowed.

## Step 2 — Dream dev

Use matched source facts for target lengths 3, 4, and 5 under the Dream
tokenizer.

Target:

```text
100 dev edits per length
```

Tune only model-specific integration details that cannot be shared with LLaDA:

```text
module map
Dream generation API
Dream-specific default reference reveal process
```

Do not retune the core cost function or scientific acceptance threshold.

## Step 3 — Dream locked confirmation

Target:

```text
300 locked edits per length N=3,4,5
```

Methods:

```text
default Dream reveal
one-step myopic
deterministic global planner
best compute-matched non-SB planner
finite-beta controller
```

Use the LLaDA-selected beta whenever the reference/cost scale is compatible.
If calibration is mathematically necessary because score scale differs, permit
temperature normalization on Dream dev only and freeze it.

## Acceptance

Cross-backbone consistency:

```text
same direction of pooled N=3/N=4 improvement as LLaDA
```

At least one Dream length must achieve:

```text
rewrite delta >= +0.03
paired-bootstrap lower bound > 0
same-subject TFPR increase <= 0.03
malformed rate <= 0.05
```

A strong pass has positive pooled N=3/N=4 evidence.

## Fallback

If Dream integration is impossible after the one repair, run the predeclared
weaker fallback on:

```text
GSAI-ML/LLaDA-8B-Base
```

The fallback cannot support `top_tier_ready` by itself.

## Outputs

```text
runs/.../dream_confirmation_v1/
  report_summary.json
  model_module_map.json
  dream_memit_smoke.csv
  dev_results.csv
  locked_results.csv
  paired_bootstrap.csv
  tokenizer_alignment.csv
  compute_table.csv
  cross_backbone_interpretation.md
```
