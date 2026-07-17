# Locked Confirmation Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Purpose

Prevent development-set overfitting and establish whether a selected diffusion-native parametric editor retains efficacy/locality on untouched data.

## Stage L0 — Dev selection lock

Before any locked evaluation, write:

```text
runs/diffusion_native_causal_partial_state_editor_v1/dev_method_lock.json
```

Required fields:

```text
protocol version
Git commit
model and tokenizer revisions
edited layer window
edited token position/site policy
causal tracing configuration
partial-state state-bank policy
positive/preservation manifest hashes
target-value objective and hyperparameters
null-space basis construction
protected variance and ridge
update rank and layer allocation
optional state-conditioned residual policy
sampling schedule and steps
random seeds
normalization and metrics
selected baselines
report-script hashes
```

The lock must explicitly state:

```text
no further tuning planned
analysis_500_allowed = true
final_test_500_allowed = false
```

Only then set:

```bash
export DEV_METHOD_LOCKED=1
```

## Stage L1 — KAMEL locked multi-token confirmation

Run the frozen candidate on untouched target-length-specific KAMEL sets:

```text
dnpe_kamel_locked_2
dnpe_kamel_locked_3
dnpe_kamel_locked_4
```

Methods:

```text
fullmask MDM-MEMIT
partial-state MDM-MEMIT
AlphaEdit-style MDM-MEMIT
causal partial-state editor
causal partial-state null-space editor
state-conditioned rescue if it was preselected
```

Primary criteria:

```text
full-span exact gain over fullmask baseline >=0.10 on at least two lengths
or positive pooled paired delta with lower CI >0
same-subject/locality budgets pass
malformed <=0.05
```

No tuning on locked KAMEL results.

## Stage L2 — analysis_500

Run the frozen selected candidate and required baselines once.

### Pass criteria

```text
rewrite >=80% of dev rewrite
paraphrase >=80% of dev paraphrase
same-subject TFPR <= base +0.03
near/far TFPR <= base +0.03
malformed <=0.05
locality advantage over strongest baseline remains in the same direction
paired bootstrap preserves the central qualitative claim
```

If analysis fails:

```text
mark protocol failed
write formal negative package
do not tune on analysis
do not open final_test_500
```

If analysis passes, write:

```text
runs/diffusion_native_causal_partial_state_editor_v1/analysis_confirmation_lock.json
```

and set:

```bash
export FINAL_METHOD_LOCKED=1
```

## Stage L3 — final_test_500

Run exactly once.

Required methods:

```text
base
MDM-MEMIT
partial-state MDM-MEMIT
AlphaEdit-style MDM-MEMIT
TimeROME-DLM-style residual memory
selected main editor
one precommitted causal/null-space ablation
```

Required metrics:

```text
rewrite
paraphrase
target F1
old-target suppression
same-subject TFPR
near/far locality
locality exact/self-normalized locality
distributional KL/JS
malformed
update rank/norm
edit time
inference time
sequential/batch diagnostics where precommitted
paired bootstrap by edit_id
```

A rerun is allowed only for a documented infrastructure failure before result inspection.

## Statistical protocol

```text
paired bootstrap by edit_id
10,000 resamples for primary final comparisons
95% confidence intervals
report micro and macro-by-relation metrics
Holm correction if multiple primary comparisons are declared
```

## Strong confirmation

A strong result requires:

```text
main editor meets efficacy floors
same-subject and near/far budgets pass
locality improvement over strongest baseline has positive paired evidence
partial-state/causal mechanism evidence survives
```

A narrower locality result may be claimed if efficacy is within 0.05 of baseline and locality improvement is robust.
