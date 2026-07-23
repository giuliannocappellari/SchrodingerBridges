# C8 — Multi-Marginal and Function-Space SB Consolidation

## Hypothesis

Sequential adaptation can be viewed as transporting the denoiser through a sequence of functional marginals. A KL-regularized consolidation may preserve old edit functions better than direct branch overwrite or linear merging.

## Tractable setting

Do not solve SB over all model parameters. Build a cache of deployable states and top-k distributions for:

```text
base denoiser
previous accepted continual model
current-block adapted model
old edit states
current edit states
base retention states
```

## Variants

```text
linear logit interpolation
EMA teacher
task-vector/adapter averaging
TIES-style sign-aware merge
two-marginal entropic barycenter
multi-marginal KL barycenter
iterative proportional/Markovian fitting on cached distributions
```

## Objective

Find consolidated distributions close to the reference/pretrained process while satisfying old- and new-edit marginals. Distill the selected consolidated function into the growth branch.

## Pass

SB-specific Class C relative to the strongest non-SB merge at matched state cache and compute.

## Rescue

One rescue may adjust the entropic regularization and old/new marginal weights within a bounded grid.
