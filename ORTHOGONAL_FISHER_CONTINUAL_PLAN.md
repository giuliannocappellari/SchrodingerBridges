# C5 — Orthogonal and Fisher-Protected Growth

## Hypothesis

New factual updates interfere because their gradient/update directions overlap with base-denoiser and previous-edit directions. Orthogonalization and Fisher-guided masking may improve stability.

## Variants

```text
O-Edit update orthogonalization
online O-Edit basis with truncation
FGGM diagonal-Fisher gradient mask
NuSA-style null-space low-rank update
online EWC baseline
O-Edit + FGGM, only if both pass independently
```

## Protected data

Use training-only base retention prompts, old edits, same-subject anchors, and partial-state variants.

## Metrics

```text
gradient cosine with old edits
Fisher-weighted update norm
subspace rank growth
plasticity loss
retention gain
```

## Pass

At current efficacy within 0.03:

```text
forgetting reduction >=30%
protected KL reduction >=20%
or past retention +0.10
```

## Rescue

One bounded threshold/rank sweep only:

```text
orthogonal rank {16,32,64}
Fisher mask keep ratio {0.1,0.25,0.5}
```
