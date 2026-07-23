# C2 — Partial-State Continual Replay

## Hypothesis

Continual factual forgetting in a diffusion LM is partly caused by replaying only clean or fully masked prompts. Old edits should be rehearsed over the partial denoising states actually visited at inference.

## Replay variants

```text
R0 no replay
R1 clean rewrite replay
R2 fully masked answer replay
R3 uniformly sampled mask-ratio replay
R4 early/middle/late balanced replay
R5 actual-trajectory state replay
R6 dark replay: stored top-k logits
R7 state-balanced dark replay
R8 interference-prioritized dark replay
```

Use identical replay-item budgets when comparing variants.

## Memory budgets

```text
0 items/edit
1 clean item/edit
4 state items/edit
8 state items/edit
```

Dark replay stores compressed top-k logits and schema fingerprints, not final outcomes.

## Prioritization

Estimate interference using gradient cosine or observed retention loss on training-only old-edit probes.

## Pass

At matched new-block efficacy:

```text
forgetting reduction >= 30%
past retention +0.10
or equal retention with >=25% less storage than clean replay
```

For a diffusion-specific claim:

```text
state replay beats clean/full-mask replay by >=0.05 past retention
with paired lower bound > 0.
```

## Rescue

One rescue may change only the replay allocation across early/middle/late buckets, not the total replay budget.
