# Track T2 — Activation-Space Schrödinger Bridge

Protocol: `counterfact_activation_space_sb_v1`

## 1. Hypothesis

Token space has poor categorical geometry, but LLaDA hidden activations live in a continuous space where a Schrödinger bridge or entropic transport can move old-fact states toward new-fact states with smaller, more local interventions.

The track learns a conditional transport in a low-dimensional activation latent:

```text
h_old -> h_new
```

for edit prompts, and identity transport:

```text
h_base -> h_base
```

for same-subject and locality negatives.

## 2. Intervention location

Start with:

```text
selected answer-position hidden state
middle layer and final layer diagnostics
one chosen intervention layer for actual decoding
```

Do not modify base LLaDA weights.

At runtime:

```text
h_edited = transport(h_base, edit, timestep, gate)
```

then feed the transported state through the remaining frozen model / LM head path.

## 3. Endpoint dataset

For positive prompts:

```text
h0 = frozen base activation under old/base factual behavior
h1 = target-conditioned activation supporting target_new
```

Target activation construction options, in order:

```text
1. teacher-forced new-target answer-span state
2. successful raw-bridge trajectory state
3. optimized activation target with small locality regularization
```

For negatives:

```text
h0 = h1 = base activation
```

Required prompt types:

```text
rewrite
paraphrase
same-subject different-relation
near/far locality
generation
attribute/unrelated when available
```

## 4. Stage T2.1 — Activation endpoint collection

Use frozen LLaDA and common train/val/smoke splits.

Collect:

```text
edit_id
prompt_id
prompt provenance
step/timestep
selected layer
selected position
h0
h1
edit representation
relation representation
positive/identity label
base logits
```

Outputs:

```text
activation_endpoint_cache_v1/
  train.safetensors
  val.safetensors
  index.jsonl
  schema.json
  endpoint_quality.csv
  leakage_audit.json
  report_summary.json
```

Acceptance:

```text
all vectors finite
train/val edit overlap =0
positive and identity pairs present
real-prompt provenance passes
h0/h1 not identical for positive rows
h0/h1 identical within tolerance for identity rows
zero teacher/outcome leakage in runtime features
```

One collection repair is allowed.

## 5. Stage T2.2 — Latent geometry

Fit PCA/whitening on training activations only:

```text
latent_dim in {128,256}
```

Report retained variance and reconstruction error.

Acceptance:

```text
retained variance >=90% or exact retained variance documented
validation reconstruction cosine >=0.95
no validation fit leakage
```

If 256 dimensions retain <80%, use 512 only as the one bounded representation rescue.

## 6. Stage T2.3 — Gaussian/linear SB pilot

Primary pilot:

```text
conditional Gaussian Schrödinger bridge / bridge matching in activation latent
```

Reference process:

```text
OU or Brownian reference with fixed variance schedule
```

Conditioning:

```text
current activation latent
edit/relation embedding
timestep
```

Identity negatives remain stationary.

Required baselines:

```text
direct mean difference
conditional linear regression map
entropic OT barycentric map without dynamic bridge
straight interpolation
```

Offline metrics:

```text
endpoint cosine/error
identity drift norm
transport energy
relation/edit shuffle sensitivity
logit change toward target_new
logit change on negatives
```

Offline pass:

```text
endpoint cosine >=0.70
identity drift norm <=0.10 * positive transport norm
SB endpoint error improves >=10% over direct mean shift
SB transport energy <= direct shift at matched endpoint quality
relation shuffle degrades endpoint quality >=0.05
negative target-logit increase <=0 on average
```

## 7. Bounded rescue — neural activation bridge

If Gaussian/linear SB fails one of endpoint-quality criteria but identity drift remains acceptable, allow one rescue:

```text
small conditional neural drift / bridge-matching MLP
latent_dim fixed to best prior value
2-4 bridge timesteps
no new layer family
```

No second rescue.

## 8. Stage T2.4 — Runtime integration

Implement activation intervention for one chosen layer and schedule:

```text
final-step-only diagnostic
late-step intervention
all-step diagnostic
```

Fake tests:

```text
state alignment
transport feature parity
frozen base weights
identity transport no-op
compute accounting
```

## 9. Stage T2.5 — Smoke20 actual decode

Methods:

```text
base
target_logit_bias
direct mean shift
linear activation map
activation SB
activation SB + learned gate if gate available from T1 or retrained under this protocol
```

Green pass:

```text
rewrite >= base +0.15
paraphrase >= base +0.10
same-subject TFPR <= base +0.03
near/far budgets pass
malformed <=0.05
activation SB beats direct mean shift or linear map on feasible selection score
```

Diffusion-specific diagnostic:

```text
late/all-step SB should beat final-only by >=0.03 on rewrite or paraphrase at matched locality,
or diffusion-trajectory claim is rejected.
```

Yellow pass:

```text
useful efficacy gain
same-subject TFPR <=0.10
identity drift remains low
```

One intervention-strength calibration is allowed:

```text
transport scale in {0.25,0.5,1.0}
```

## 10. Confirmation30

Freeze layer, schedule, and scale. Run once.

Acceptance:

```text
same qualitative efficacy trend
same-subject budget passes
activation SB remains better than direct shift baseline
malformed <=0.05
```

## 11. Scale/dev

If pilot passes:

```text
train on scaled common train/val
layers limited to middle vs final
latent_dim limited to best pilot plus one adjacent value
schedule limited to late vs all
transport scale limited to top two pilot values
```

Nominate at most one T2 candidate.

## 12. Track claim

Activation-space SB claim requires:

```text
SB transport beats direct shift/linear/OT-static baseline at matched intervention energy;
identity negatives remain near identity;
actual factual editing passes stress constraints.
```

If only direct activation shift works, report an activation-editing result, not an SB result.
