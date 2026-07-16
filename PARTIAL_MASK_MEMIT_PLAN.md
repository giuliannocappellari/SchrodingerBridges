# M2 — Partial-Mask MDM-MEMIT Plan

Protocol: `llada_partial_mask_memit_v1`

## Objective

Reproduce the reported correction for multi-token MDM editing by optimizing the MEMIT target residual over the partially unmasked states encountered during iterative denoising.

## Method

For a target of `N` tokens at target-value optimization step `t`:

```text
k = t mod N
```

Then:

1. randomly choose `k` target positions;
2. fill those positions with the correct target tokens;
3. leave the remaining `N-k` positions masked;
4. compute the target loss only on still-masked positions;
5. cycle through `k=0,1,...,N-1` over optimization;
6. resample revealed positions each step.

Everything else in MEMIT remains unchanged.

## Stage M2.1 — KAMEL baseline smoke

On `kamel_smoke_20_per_length`, run fully-masked-only MDM-MEMIT for N=1,2,3,4.

Acceptance:

```text
all lengths execute
full-target exact degrades with target length or is otherwise measured
partial-token coverage and full-target exact both reported
no malformed pipeline behavior
```

## Stage M2.2 — Partial-mask implementation

Tests:

```text
k cycles deterministically from 0 to N-1
revealed positions are resampled
loss excludes revealed positions
all positions are supervised over a cycle
N=1 reduces to fully masked behavior
random seed reproducibility
```

## Stage M2.3 — Dev ablations

Use `kamel_dev_50_per_length`.

State-count schedules:

```text
fully masked only
bias toward fewer revealed
uniform random count
bias toward more revealed
cycle count
```

Reveal policies for the selected count schedule:

```text
left-to-right
base-confidence
random resampling
```

Freeze the best policy by average efficacy/generalization rank over N=2,3,4, with N=4 efficacy as tie-break.

Acceptance:

```text
all predeclared policies evaluated
one frozen policy selected
no policy added after inspection
```

Expected paper default:

```text
cycle counts + random revealed positions
```

## Stage M2.4 — Locked KAMEL 200-per-length evaluation

Compare:

```text
pre-edit
fully-masked MDM-MEMIT
partial-mask MDM-MEMIT
```

Metrics by N:

```text
full-target efficacy
generalization
per-target-token appearance
full-target assembly gap
malformed
classic and same-subject locality
edit time
```

Minimum positive pass: at least two of N=2,3,4 satisfy both:

```text
efficacy improvement >= 0.15 absolute
generalization improvement >= 0.08 absolute
```

Strong pass:

```text
N=2 efficacy >= 0.75
N=3 efficacy >= 0.60
N=4 efficacy >= 0.55
```

Paper comparison targets:

```text
N=2: 0.60 -> 0.87 efficacy, 0.36 -> 0.52 gen
N=3: 0.33 -> 0.76 efficacy, 0.20 -> 0.36 gen
N=4: 0.27 -> 0.73 efficacy, 0.14 -> 0.36 gen
```

## Stage M2.5 — Two-seed confirmation

Repeat target-value optimization and evaluation with a second seed for revealed-position sampling.

Acceptance:

```text
positive direction persists in both seeds
mean improvement passes minimum criterion
no single seed contributes all gain
```

### Bounded M2 rescue

If partial-mask gains fail:

- verify revealed-token conditioning and loss masking;
- increase target-value optimization steps only enough to complete at least five full cycles for N=4;
- keep learning rate, clamp, KL, layer window, and update method fixed;
- rerun one seed.

No further rescue.

## Outputs

```text
runs/masked_diffusion_memit_sb_positive_result_v1/M2_partial_mask_memit_v1/
  report_summary.json
  kamel_manifest_summary.json
  state_schedule_ablation.csv
  reveal_policy_ablation.csv
  main_results_by_length.csv
  token_assembly_gap.csv
  seed_confirmation.csv
  paired_bootstrap.csv
  final_track_report.md
```
