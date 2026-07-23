# C4 — Gated Continual Adapter Expansion

## Hypothesis

Adding a small adapter per edit block and learning gates that suppress new adapters on old tasks can improve stability without forcing all tasks through one adapted computation.

## Variants

```text
C-LoRA-style continually self-regularized adapters
GainLoRA-style new branch per block
uniform branch averaging
learned prompt gate
timestep-conditioned prompt gate
relation-aware gate
```

## Gate objective

For old-task training anchors, penalize contribution from the newest branch. For the current block, allow plasticity.

## Fairness

Report cumulative parameters and storage. Compare at fixed total rank and fixed per-block rank.

## Pass

Class A, B, or D.

## Rescue

One rescue may use a shared low-rank basis with block-specific coefficients to reduce linear parameter growth.
