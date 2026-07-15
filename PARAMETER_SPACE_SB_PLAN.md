# Track T5 — Parameter-Space Schrödinger Bridge over Low-Rank Adapter Latents

Protocol: `counterfact_parameter_space_sb_v1`

## 1. Hypothesis

A Schrödinger bridge in a compact low-rank adapter latent space can transport the zero-update model toward edit-specific parameter updates while regularizing the path to preserve locality.

The base LLaDA remains frozen. The state space is not the full model weights; it is a compact adapter latent.

## 2. Adapter family

Pilot adapter:

```text
rank-2 low-rank residual at one predeclared answer-relevant module/layer
or equivalent small answer-position residual adapter
```

Bounded rescue:

```text
rank 2 -> rank 4
or latent dimension 64 -> 128
not both.
```

Target storage goal:

```text
<=1 MB per edit
```

## 3. Endpoint adapters

For each training edit, optimize a direct endpoint adapter using only allowed edit-time training prompts and anchors.

Loss:

```text
rewrite target loss
training-only relation augmentation loss
same-subject negative preservation
locality anchor KL
L2/low-rank regularization
```

Evaluation paraphrases/locality/stress prompts remain held out.

Required direct endpoint baseline metrics:

```text
rewrite
paraphrase
same-subject TFPR
near/far locality
training time
storage
```

If direct endpoint adapters cannot achieve useful editing, stop the track: a parameter-space bridge cannot rescue an unusable endpoint family.

Direct endpoint viability:

```text
rewrite >=0.30
paraphrase >=0.20
same-subject TFPR <= base +0.03
malformed <=0.05
storage <=1 MB/edit
training <=5 GPU min/edit
```

Pilot sizes:

```text
train endpoint adapters: 50 edits
val endpoint adapters: 20 edits
smoke actual: 10 then 20 held-out edits
```

## 4. Adapter latent representation

Flatten adapter parameters and fit:

```text
PCA or small autoencoder
latent_dim in {64,128}
```

The zero adapter maps to `z0`.
Each optimized endpoint adapter maps to `zT(edit)`.

Acceptance:

```text
validation reconstruction cosine >=0.95
reconstructed adapter retains >=90% of direct adapter rewrite/paraphrase performance
identity zero adapter reconstructs near zero
```

## 5. Parameter-space reference process

Use a Brownian or OU reference in adapter latent space:

```text
z_t = interpolation(z0,zT,t) + scheduled noise
```

Train a conditional endpoint predictor or drift:

```text
p(zT | z_t, edit, t)
or
u_psi(z_t,edit,t)
```

Condition on deployable edit representations:

```text
subject
relation
old target
new target
rewrite template
```

Do not use held-out evaluation prompt outcomes.

## 6. Required baselines

```text
direct per-edit optimized adapter
mean adapter
linear regression adapter generator
conditional MLP adapter generator
linear interpolation in latent
parameter-space SB generator
```

SB-specific evidence requires beating the conditional MLP or linear latent generator at matched storage/compute.

## 7. Offline training and metrics

Metrics:

```text
latent endpoint MSE/cosine
adapter reconstruction quality
predicted adapter parameter cosine
predicted-vs-direct logit agreement on training-only probes
identity/zero-update drift
edit/relation shuffle sensitivity
```

Offline pass:

```text
predicted adapter cosine >=0.70
rewrite-probe logit agreement >=0.70 correlation
identity drift norm <=0.10 * endpoint norm
edit/relation shuffle degrades endpoint cosine >=0.05
parameter-space SB beats conditional MLP by >=0.05 on endpoint cosine or held-out probe metric
```

## 8. Bounded rescue

One rescue only:

```text
rank 2 ->4
or latent 64 ->128
```

Do not increase both or add more edited layers in v1.

## 9. Actual smoke10

For 10 held-out edits, generate adapter from edit request without optimizing on evaluation prompts.

Methods:

```text
base
target_logit_bias
direct per-edit adapter oracle/reference
conditional MLP adapter generator
parameter-space SB adapter generator
```

Green smoke10:

```text
SB-generated adapter rewrite >= base +0.15
paraphrase >= base +0.10
same-subject TFPR <= base +0.03
malformed <=0.05
storage <=1 MB/edit
inference compute <=60% of MC bridge
SB beats conditional MLP on feasible score
```

If smoke10 passes, run fixed smoke20/confirmation subset.

## 10. Smoke20/confirmation

Freeze generator/checkpoint/rank/latent.

Acceptance:

```text
same qualitative gains
same-subject budget passes
SB-generated adapter remains better than conditional MLP or linear baseline
training/generation/storage costs stay within limits
```

## 11. Scale/dev

If pilot passes:

```text
generate direct endpoint adapters for scaled train/val
train best latent bridge config
no new architecture
run common dev_tune_200 with generated adapters
compare direct per-edit optimization on a predeclared subset if full coverage is too expensive
```

Nominate at most one T5 candidate.

## 12. Track claim

Parameter-space SB claim requires:

```text
usable direct endpoint adapter family;
SB generator generalizes to unseen edits;
SB beats conditional MLP/linear parameter generator at matched size/compute;
actual factual editing and locality pass.
```

If direct adapters work but SB generation does not, report a per-edit adapter result, not a parameter-space SB result.
