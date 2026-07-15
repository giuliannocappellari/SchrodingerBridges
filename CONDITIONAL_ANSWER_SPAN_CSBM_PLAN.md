# Track T3 — Conditional Answer-Span Categorical Schrödinger Bridge Matching

Protocol: `counterfact_conditional_answer_span_csbm_v1`

## 1. Hypothesis

A faithful categorical bridge-matching process over the factual answer span can learn edit-conditioned forward and backward transitions that outperform ordinary target-conditioned denoising while preserving identity behavior on negative prompts.

This is not score distillation. The track trains endpoint-conditioned Markov transitions through reciprocal bridge sampling and bidirectional D-IMF-style fitting.

## 2. State space

Start with answer-span states only.

Pilot scope:

```text
single-token targets primary
multi-token >=2 diagnostic after single-token pass
```

State representation:

```text
answer-span token sequence with absorbing [MASK]
context prompt fixed
current span state x_t
edit tuple e
```

Do not train full-sequence CSBM in v1.

## 3. Endpoints

Positive prompt:

```text
x0 = old/base answer span
xT = target_new answer span
```

Identity negatives:

```text
x0 = base answer span
xT = base answer span
```

Use smoothed endpoint distributions:

```text
nu_edit = (1-epsilon) * target_delta + epsilon * base_support
nu_identity = base_support
```

Pilot epsilon grid:

```text
{0.01,0.05}
```

## 4. Reference process

Use an LLaDA-compatible absorbing-mask reference with small full-support mixing.

Implement and test:

```text
forward corruption q(x_t|x0)
reciprocal bridge sampler q_ref(x_t|x0,xT)
next-step conditional q_ref(x_{t+1}|x_t,xT)
identity bridge behavior
multi-token span masking
```

Reference tests must verify normalization, finite probabilities, endpoint consistency, and deterministic seeded sampling.

## 5. Stage T3.1 — Pilot data

Use common train/val rows with real prompts.

Initial pilot sizes:

```text
train 200 edits
val 50 edits
single-token primary
identity negatives for every edit where available
```

If the required legal data exists, add 50 multi-token training edits and 20 multi-token validation edits as a diagnostic; monetary cost must not be used as a reason to skip this predeclared diagnostic.

Outputs:

```text
csbm_pilot_data_v1/
  train.jsonl
  val.jsonl
  identity_negatives.jsonl
  summary.json
  overlap_audit.csv
```

## 6. Stage T3.2 — Ordinary-noising baseline

Train an endpoint predictor under ordinary masked noising:

```text
p_endpoint(xT | x_t,p,e,t)
```

This is the non-CSBM baseline.

Report:

```text
endpoint top1/top3
span exact
identity-negative drift
```

## 7. Stage T3.3 — Forward-only CSBM

Train the forward endpoint-conditioned transition model using reciprocal bridge states but no backward fitting.

Purpose:

```text
separate bridge-state sampling benefit from bidirectional D-IMF benefit.
```

## 8. Stage T3.4 — Bidirectional D-IMF CSBM

Train:

```text
forward endpoint predictor p_theta(xT|x_t,p,e,t)
backward endpoint predictor p_phi(x0|x_t,p,e,t)
```

Outer iterations:

```text
2 in initial pilot
4 only as bounded rescue
```

For each outer iteration:

```text
1. sample reciprocal bridge trajectories;
2. fit forward Markov projection;
3. sample trajectories from updated forward process;
4. fit backward Markov projection;
5. refresh cache and log endpoint/identity metrics.
```

Candidate factorization must be explicit. For multi-token spans, report dependency limitations.

## 9. Losses

```text
endpoint cross-entropy
bridge-transition negative log likelihood
identity KL/locality loss
target-support ranking loss with small weight
path/intervention regularization
```

Do not use a dominant target CE term.

## 10. Offline acceptance

Required comparisons:

```text
bidirectional CSBM vs ordinary noising
bidirectional CSBM vs forward-only CSBM
identity negatives vs no identity training
```

Hard offline pass:

```text
endpoint top1 improvement over base >=0.15
bridge-state model endpoint accuracy >= ordinary-noising +0.05
bidirectional endpoint/span metric >= forward-only +0.03
identity-negative average KL <=0.05 on sparse support
same-subject target advantage <=0
all transition probabilities finite/normalized
zero locked split leakage
```

Representation-use/shortcut checks:

```text
edit/relation shuffle reduces endpoint accuracy >=0.05
target-indicator-only baseline weaker than full CSBM by >=0.05 on endpoint or identity-aware metric
```

## 11. Bounded rescue

One rescue only:

```text
outer iterations 2 -> 4
or endpoint epsilon/temperature calibration within declared grid
```

Do not change state space or add full-sequence modeling in v1.

## 12. Runtime integration

Add methods:

```text
csbm_forward_only
csbm_bidirectional
csbm_bidirectional_identity
```

Schedules:

```text
final-only diagnostic
late
all
```

Runtime must use the trained transition process, not collapse to target logit bias.

## 13. Smoke20 actual decode

Methods:

```text
base
target_logit_bias
ordinary-noising endpoint model
forward-only CSBM
bidirectional CSBM
raw MC bridge
```

Green pass:

```text
rewrite >= base +0.20
paraphrase >= base +0.10
same-subject TFPR <= base +0.03
near/far budgets pass
malformed <=0.05
bidirectional CSBM beats ordinary-noising and forward-only on feasible score
multi-step late/all beats final-only by >=0.03 or diffusion-specific claim rejected
```

Yellow pass:

```text
useful efficacy gain
same-subject TFPR <=0.10
bidirectional advantage visible offline but not yet significant actual
```

## 14. Confirmation30

Freeze epsilon, outer iterations, schedule, and guidance strength. Run once.

Acceptance:

```text
same qualitative efficacy/locality trend
bidirectional remains >= forward-only
same-subject budget passes
```

## 15. Scale/dev

If pilot passes:

```text
train 1000/200; use the predeclared 500/100 fallback only for documented data feasibility or Pod hardware-capacity constraints, never for monetary budget reasons
single-token and multi-token >=2 represented
outer iterations fixed from pilot
epsilon fixed from pilot
bounded schedule/top_k sweep only
```

Nominate at most one T3 candidate.

## 16. Track claim

Categorical CSBM claim requires all of:

```text
bridge-state sampling > ordinary noising
bidirectional > forward-only
identity-negative locality control works
actual decode passes stress constraints
```

Otherwise report a target-conditioned denoising result, not a CSBM result.
