# C7 — Schrödinger-Bridge Generative Trajectory Replay

## Hypothesis

Ordinary replay samples old edit states from the forward masking/noising distribution. A reference bridge conditioned on the old edit endpoint may generate more informative rehearsal states and reduce forgetting at a fixed replay budget.

## State construction

For an old edit with clean target span \(x_1\) and a chosen start distribution \(x_0\), sample intermediate states from:

\[
q^{ref}(x_t \mid x_0, x_1)
\]

or a bounded CSBM/Doob approximation.

Use the frozen LLaDA-compatible mask process as the reference.

## Variants

```text
ordinary random-mask replay
actual stored trajectory replay
reference-bridge replay
CSBM-lite endpoint-conditioned replay
unbalanced bridge replay prioritized by forgetting risk
```

## Distillation

On generated states, match the previous accepted model's top-k distribution and old edited target support.

## Fair comparison

Match:

```text
number of replay states
stored bytes
model evaluations
old edit IDs
timestep histogram
```

## SB-specific pass

Compared with ordinary random-mask replay:

```text
past retention +0.05
or forgetting reduction >=25%
or equal retention with >=25% less replay storage
```

with paired lower bound > 0.

## Rescue

One rescue may change only the bridge stochasticity/reference mixture within a predeclared 3-point grid.
