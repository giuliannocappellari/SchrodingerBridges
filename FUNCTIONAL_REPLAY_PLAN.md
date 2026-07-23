# C6 — Functional Distillation and Gradient-Constrained Replay

## Hypothesis

Preserving the previous model's function over diffusion states may be more effective than constraining parameters.

## Variants

```text
Learning without Forgetting
Dark Experience Replay
experience replay with labels
DER + replay labels
GEM
A-GEM
LwF + partial-state replay
DER + partial-state replay
```

## Distillation targets

Store or recompute previous-model distributions over:

```text
old edit answer positions
same-subject training anchors
base retention prompts
early/middle/late mask states
```

## GEM constraint

Require the new gradient not to increase old replay loss beyond the bounded slack.

## Pass

Class B or D. A mechanism result requires partial-state functional replay to beat clean-prompt functional replay.

## Rescue

One rescue may adjust the old/new loss balance in a fixed grid `{0.25,0.5,1.0,2.0}`.
