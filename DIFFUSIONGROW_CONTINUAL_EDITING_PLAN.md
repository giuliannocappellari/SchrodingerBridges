# C1 — DiffusionGrow Continual Factual Editing

## Hypothesis

A function-preserving growth branch can acquire sequential factual edits without corrupting the frozen pretrained denoiser because the original path remains explicitly available and the new branch is zero-initialized and timestep-gated.

## Source-compatible reproduction

First reproduce the source-style domain-adaptation behavior using available code/checkpoints or an equation-level reimplementation. Record exact status.

## Factual adaptation

At selected early-middle MLP layers, use:

\[
h'_{\ell,t} = h_{\ell,t} + g_{\ell}(h_{\ell,t}, t)\,B_{\ell}A_{\ell}h_{\ell,t}.
\]

Initialize the residual path to zero so the expanded model equals the base denoiser before training.

## Variants

```text
C1-A shared branch updated across all blocks
C1-B one branch per edit block
C1-C shared branch + block gate
C1-D block branches + prompt/edit gate
C1-E partial-state branch training
```

Base weights remain frozen.

## Training

Use current-block rewrite data, training-only paraphrase augmentations, and training-only locality anchors. Do not replay held-out evaluation prompts.

## Mechanism metrics

```text
exact function equality at initialization
gate activation by prompt family and timestep
branch norm
old/new branch contribution
base-path availability
```

## Pass

Class A, B, or D from the master plan.

## Rescue

One rescue may adjust only:

```text
branch rank in {4,8,16}
growth layers within the source-compatible layer family
gate initialization/temperature
```

No new branch architecture.
