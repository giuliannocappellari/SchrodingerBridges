# M1 — MDM-MEMIT Reproduction Plan

Protocol: `llada_mdm_memit_reproduction_v1`

## Objective

Reproduce a strong locate-then-edit result on `GSAI-ML/LLaDA-8B-Instruct` by adapting MEMIT to masked bidirectional prediction.

## Paper-matched algorithm

Keep the original MEMIT key/value framework, target-value optimization, covariance term, closed-form update, and residual distribution across layers.

Mask-augment every forward-pass input distribution:

```text
1. Rewriting prompts:
   prompt + one [MASK] per target token; supervise target at masks.

2. KL anchor:
   anchor prompt + [MASK]; read anchor logits at the mask.

3. Key extraction:
   last-subject MLP input/up-projection activation under the same mask-augmented prompt.

4. Residual baseline:
   last-subject hidden/value under the same mask-augmented prompt.
```

Primary paper-matched hyperparameters:

```text
learning_rate = 0.1
target_optimization_steps = 25
clamp_norm_factor = 0.75
kl_factor = 0.0625
edit_window_width = 4
paper LLaDA window = layers 4-7
```

Use the official MEMIT/EasyEdit covariance and closed-form update equations. Pin source commit.

## Stage M1.1 — Implementation scaffold

Tasks:

- implement LLaDA module adapter;
- implement contextual target tokenization;
- implement last-subject token locator;
- implement mask-block renderer;
- implement hooks for MLP key/value replacement;
- implement covariance cache interface;
- implement closed-form update application and rollback;
- add fake-model tests.

Tests:

```text
mask count equals target length
all four MEMIT distributions share identical masked context
last-subject index correct under tokenization
edit matrices restore exactly after rollback
closed-form shapes valid
no quantized weight editing
```

Acceptance:

```text
all tests pass
one fake edit increases target probability
rollback restores pre-edit checksum
```

## Stage M1.2 — One-edit GPU smoke

Use one fresh single-token CounterFact edit.

Required diagnostics:

```text
base target_new probability
target-value optimization curve
clamp activity
KL anchor drift
matrix update norm
post-edit target probability
greedy completion
rollback checksum
```

Acceptance:

```text
target loss decreases
post-edit target probability increases materially
no NaN/Inf
model can generate after update
rollback exact within tolerance
```

## Stage M1.3 — Smoke20

Run the paper layer window `L4-7` on `cf_memit_smoke_20`.

Acceptance:

```text
rewrite exact >= 0.50
paraphrase exact >= 0.20
no catastrophic model corruption
malformed <= 0.05
all 20 edits accounted
```

### Bounded M1 rescue

If smoke20 fails:

1. audit module orientation/transposition and key/value location;
2. compare paper L4-7 with two adjacent 4-layer windows;
3. verify covariance scaling and target-value injection;
4. keep paper hyperparameters otherwise fixed.

No second rescue.

## Stage M1.4 — Layer-window selection

On `cf_layer_select_500`, sweep every feasible contiguous 4-layer MLP window, or a compute-efficient coarse-to-fine equivalent that still includes every window in the final comparison.

Select by paper-matched rank-sum:

```text
rank efficacy + rank generalization
tie-break by efficacy
specificity reported but not used for layer selection
```

Also report same-subject stress as a diagnostic.

Acceptance:

```text
all windows evaluated or justified coarse-to-fine coverage
best window frozen
paper L4-7 result reported
selection manifest and hashes written
```

## Stage M1.5 — Locked reproduction on 500 facts

Run a single batch of 500 fresh CounterFact edits on `cf_repro_main_500` using the frozen window and paper hyperparameters.

Required evaluation:

```text
efficacy
generalization
classic specificity
same-subject TFPR
near/far TFPR
old-target suppression
update norm
editing time
inference time
```

Reproduction pass:

```text
efficacy >= 0.75
generalization >= 0.40
pre-edit target_new efficacy <= 0.10
all metrics complete
```

Strong reproduction:

```text
efficacy >= 0.85
generalization >= 0.50
```

Paper comparison target, not a hard requirement:

```text
efficacy about 0.91
generalization about 0.56
```

Locality is reported honestly; the campaign does not redefine reproduction failure solely from stricter historical same-subject budgets.

## Stage M1.6 — Generation robustness

Evaluate the frozen edited model under:

```text
target-length / target-length steps
8 / 8
16 / 16
32 / 32
```

Acceptance:

```text
efficacy at each fixed length >= reproduction efficacy - 0.15
generalization trend reported
specificity and malformed reported
```

## Outputs

```text
runs/masked_diffusion_memit_sb_positive_result_v1/M1_mdm_memit_reproduction_v1/
  report_summary.json
  implementation_config.json
  covariance_manifest.json
  layer_sweep.csv
  counterfact_reproduction.csv
  generation_robustness.csv
  paired_bootstrap.csv
  failure_cases.csv
  final_track_report.md
```

## Terminal interpretation

```text
passed_reproduction:
  hard reproduction thresholds pass

partial_reproduction:
  clear positive edit result but below thresholds

after_rescue_formal_negative:
  smoke or reproduction does not establish effective MDM-MEMIT
```
