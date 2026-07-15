# Track T4 — Unbalanced / Partial Categorical Schrödinger Bridge

Protocol: `counterfact_unbalanced_partial_csbm_v1`

## 1. Hypothesis

Factual editing should transport only a small prompt-conditioned fraction of probability mass while leaving most behavior under the identity process. A balanced bridge forces too much transport; a partial/unbalanced bridge can learn how much of the reference process should be edited.

The primary transition mixture is:

```text
P_partial(next | state,p,e)
  = (1-rho(p,e,state)) * Q_identity(next | state)
  + rho(p,e,state) * P_edit_bridge(next | state,p,e)
```

`rho` is learned transport mass, not merely a post-hoc binary gate.

## 2. Dependencies

Reuse validated categorical bridge infrastructure from T3 when available.

If T3 is scientifically negative but implementation-valid, use its best balanced checkpoint/reference implementation as the balanced baseline.

If T3 implementation itself is invalid, repair the shared implementation before T4; do not skip T4 silently.

## 3. Training data

Use the same legal prompt/edit splits as T3.

Targets:

```text
positive edit prompts: desired transport mass high but not forced to 1
identity negatives: desired transport mass near 0
```

Do not use prompt_type/negative_type as runtime inputs.

## 4. Objectives

Total objective:

```text
balanced bridge transition loss
+ mass calibration loss
+ unbalanced KL / mass regularization
+ identity/locality loss
+ target-support loss with small weight
```

Mass priors:

```text
positive prior rho_pos in {0.7,0.9}
negative prior rho_neg in {0.01,0.05}
```

Unbalanced penalty grid:

```text
lambda_mass in {0.1,1.0,5.0}
```

## 5. Required variants

```text
balanced CSBM (rho=1)
fixed partial mixture
external learned gate + balanced CSBM
learned partial/unbalanced CSBM
rho-only target-bias diagnostic
```

The track's SB claim requires the learned internal partial bridge to beat both balanced CSBM and external-gate balanced CSBM on a locality/efficacy trade-off.

## 6. Stage T4.1 — Offline mass model

Inputs:

```text
current span state
prompt/edit representations
relation features
timestep
balanced bridge confidence/entropy available at runtime
```

Forbidden:

```text
prompt_type
negative_type
future success
teacher final outcome
```

Offline metrics:

```text
rho calibration
positive mean rho
same-subject mean rho
near/far mean rho
mass ROC/PR diagnostics
identity KL
target advantage
```

Offline pass:

```text
positive mean rho >=0.70
same-subject mean rho <=0.05
near/far mean rho <=0.02
mass ROC-AUC >=0.90
identity-negative KL <=0.05
negative target advantage <=0
```

## 7. Stage T4.2 — Partial bridge training

Train the bridge transition and mass model separately first, then one short joint calibration.

Hard comparison:

```text
at matched rewrite/endpoint quality,
partial CSBM reduces same-subject target advantage or TFPR proxy by >=50% versus balanced CSBM.
```

Efficacy preservation:

```text
positive endpoint/span metric >=90% of balanced CSBM when balanced CSBM is useful.
```

## 8. Bounded rescue

One rescue:

```text
select best existing architecture;
change lambda_mass within declared grid;
calibrate rho temperature;
no new feature family.
```

## 9. Runtime integration

At every transition:

```text
compute balanced edit transition
compute identity transition
compute rho
mix transitions
sample/choose next state
```

Log:

```text
rho per step
cumulative transported mass
identity/edit contribution
sparse guidance KL
```

## 10. Smoke20 actual decode

Methods:

```text
base
balanced CSBM
external learned gate + balanced CSBM
fixed partial mixture
learned partial CSBM
learned gate + raw bridge reference
```

Green pass:

```text
rewrite >= base +0.15
paraphrase >= base +0.10
same-subject TFPR <= base +0.03
near/far budgets pass
malformed <=0.05
same-subject TFPR improves >=50% versus balanced CSBM
rewrite/paraphrase retain >=90% of balanced CSBM when balanced is effective
learned partial method beats external-gate balanced method on feasible score or transport-mass calibration
```

Yellow pass:

```text
partial method is safer but loses >10% efficacy;
report diagnostic, allow one mass calibration only.
```

## 11. Confirmation30

Freeze mass penalty, calibration, balanced checkpoint, and schedule. Run once.

Acceptance:

```text
same-subject budget passes
partial remains safer than balanced
positive efficacy remains useful
```

## 12. Scale/dev

If pilot passes:

```text
reuse T3 scaled data/checkpoint where legal
train mass model on scaled data
lambda_mass limited to top two pilot values
rho calibration fixed on validation
bounded gamma/top_k/schedule sweep only
```

Nominate at most one T4 candidate.

## 13. Track claim

Unbalanced/partial SB claim requires:

```text
internal learned transport mass improves the balanced bridge trade-off;
not explainable solely by an external gate;
identity mass remains near base;
actual stress constraints pass.
```
