# Track T1 — Learned Edit-Intent Gate + Raw Runtime Bridge

Protocol: `counterfact_learned_gate_raw_bridge_v1`

## 1. Hypothesis

A learned relation-aware gate can solve the localization failure of Direction 1 while preserving the already demonstrated raw runtime steering signal.

The system is:

```text
edited_logits(v)
  = base_logits(v)
  + gamma * gate(prompt, edit) * raw_controller_score(v)
```

Raw controller families:

```text
myopic_score
no_rollout_bridge
mc_bridge
```

The gate answers only whether the prompt requests the edited relation. It does not approximate the bridge value function.

## 2. Scientific comparisons

Required:

```text
no gate
subject gate
best rule-based relation/hybrid gate
learned gate + myopic
learned gate + no-rollout
learned gate + MC bridge
```

Bridge-specific evidence requires:

```text
learned-gated MC > learned-gated no-rollout at matched compute/KL
and
learned-gated MC competitive with learned-gated myopic.
```

If learned-gated myopic wins, the track may support an edit-intent localization claim but not an MC-bridge superiority claim.

## 3. Data

Use common campaign splits.

Gate training positives:

```text
real rewrite prompts
real declarative paraphrases
```

Gate training negatives:

```text
same-subject different-relation
near locality
far locality
generation
attribute
unrelated
```

Do not use Direction 1 Step 3D/3D.1 stress evaluation prompts as training rows.

Gate inputs:

```text
real prompt representation
subject representation
relation/rewrite-template representation
prompt-relation interactions
relation_id embedding
lexical similarity diagnostics
```

Forbidden inputs:

```text
prompt_type
negative_type
teacher controller scores
future output metrics
evaluation-bucket identity
```

## 4. Stage T1.1 — Gate dataset and provenance

Outputs:

```text
runs/counterfact_learned_gate_raw_bridge_v1/gate_data_v1/
  gate_train.jsonl
  gate_val.jsonl
  gate_smoke.jsonl
  gate_data_summary.json
  prompt_provenance_audit.csv
  split_overlap_audit.csv
```

Acceptance:

```text
real rewrite coverage >=95%
real paraphrase coverage >=95%
real near/far coverage >=95%
same-subject negatives for >=80% of edits
train/val/smoke edit overlap = 0
synthetic fallback explicitly tagged
analysis/final use = false
```

One repair is allowed for prompt-materialization bugs.

## 5. Stage T1.2 — Train learned gate

Models:

```text
linear lexical/scalar baseline
small frozen-representation MLP
optional contrastive pair encoder within the same frozen feature family
```

Bounded grid:

```text
loss in {weighted_bce,focal}
projection_dim in {128,256}
same_subject_weight in {3,5}
seeds in {0,1}
```

Threshold calibration uses gate validation only.

Outputs:

```text
gate_train_v1/
  checkpoints
  configs
  train_metrics.csv
  val_metrics.csv
  threshold_sweep.csv
  feature_leakage_audit.json
  report_summary.json
```

Hard gate acceptance:

```text
ROC-AUC >=0.90
PR-AUC reported
rewrite activation >=0.90
paraphrase activation >=0.85
same-subject activation <=0.05
near activation <=0.02
far activation <=0.02
zero leaked runtime features
relation-shuffle AUC drop >=0.05
```

Bounded rescue:

```text
one calibration/hard-negative reweighting rescue;
no new feature family;
no threshold lowering.
```

If gate still fails, mark track pilot failed without actual decoding.

## 6. Stage T1.3 — Runtime integration tests

Extend runtime evaluator with:

```text
learned_gate_myopic
learned_gate_no_rollout
learned_gate_mc_bridge
soft gate
hard gate diagnostic
```

Fake-model tests:

```text
feature-schema fingerprint
same gate output offline/runtime
no teacher-field access
compute accounting
soft/hard gate behavior
candidate support alignment
```

Acceptance:

```text
all tests pass
runtime gate parity drift <=0.01 per prompt type
no analysis/final access
```

## 7. Stage T1.4 — Smoke20 actual decode

Methods:

```text
base
myopic_score
no_rollout_bridge
mc_bridge
rule_gated_myopic
rule_gated_mc
learned_gate_myopic
learned_gate_no_rollout
learned_gate_mc
```

Bounded calibration:

```text
gamma in {0.5,1.0,2.0}
gate temperature in {0.5,1.0,2.0}
```

Green pass:

```text
rewrite >= base +0.15
paraphrase >= base +0.10
same-subject TFPR <= base +0.03
malformed <=0.05
learned gate reduces same-subject TFPR by >=50% versus corresponding ungated or subject-gated controller
```

Yellow pass:

```text
rewrite/paraphrase improve over base
same-subject TFPR <=0.10
malformed <=0.05
```

One bounded calibration is allowed on smoke20. No gate retraining after smoke inspection.

Red fail:

```text
no rewrite gain
same-subject TFPR >0.20
malformed >0.05
runtime/offline gate mismatch
```

## 8. Stage T1.5 — Confirmation30

Run the fixed smoke-selected configuration once.

Acceptance:

```text
rewrite/paraphrase direction matches smoke
same-subject TFPR <= base +0.03
malformed <=0.05
paired trend favors learned-gated method over base
learned gate remains safer than rule-gated equivalent
```

No tuning on confirmation30.

## 9. Pilot status

Pilot passes only if gate acceptance, smoke, and confirmation all pass.

Write:

```text
pilot_summary_v1/
  report_summary.json
  pilot_results.csv
  same_subject_table.csv
  compute_table.csv
  paired_bootstrap.csv
  pilot_decision.md
```

## 10. Scale and dev_tune_200

If pilot passes:

1. Retrain gate on common scaled train/val data.
2. Freeze gate and controller settings.
3. Run staged dev sweep:

```text
controller in {myopic,no_rollout,mc_bridge}
gamma in {0.5,1.0,2.0}
gate in {soft,hard diagnostic}
top_k in {4,8}
steps in {4,8}
schedule in {late,all}
```

Do not run a full Cartesian grid; narrow after each stage.

Dev eligibility:

```text
common hard locality/stress/malformed constraints
clear rewrite/paraphrase gain over base
learned gate safer than rule gate
compute reported
```

Nominate at most one T1 candidate.

## 11. Track claim classification

Strong bridge claim:

```text
learned-gated MC beats learned-gated no-rollout and is competitive with learned-gated myopic.
```

Edit-intent claim:

```text
learned gate fixes rule-based gate leakage, even if learned-gated myopic wins.
```

Negative result:

```text
learned gate fails actual locality or suppresses paraphrase activation excessively.
```
