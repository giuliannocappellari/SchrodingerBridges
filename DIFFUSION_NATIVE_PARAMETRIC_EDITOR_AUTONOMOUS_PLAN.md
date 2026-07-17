# Diffusion-Native Causal Partial-State Parametric Editor: Autonomous Research Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Research question

> **Can a diffusion-native parametric editor preserve locality by optimizing updates over causal internal representations and partial denoising states?**

The campaign starts from a positive factual-editing baseline: locate-then-edit can work in masked diffusion language models, and multi-token failures are associated with the partially unmasked states traversed during denoising. The new contribution to test is whether a parametric update can be made both diffusion-native and local by combining:

```text
causal/temporal site selection
+ partial-state target-value optimization
+ preservation/null-space constrained low-rank updates
```

The campaign is bounded, autonomous, and must finish with either a validated positive package or a formal negative package. There is no monetary budget stop. The existing RunPod Pod remains running until terminal completion or unrecoverable infrastructure failure.

---

# 0. Main hypotheses

## H1 — Causal-site hypothesis

The edit site should be selected from internal coordinates that causally mediate factual recall across multiple denoising states, not only from a fixed AR-derived layer/position.

Success means at least one of:

```text
causal-site editor beats random-site editor by >= 0.05 on stress-aware aggregate

or

causal-site editor matches efficacy within 0.02 with >=25% lower update norm
or fewer edited layers
```

## H2 — Partial-state hypothesis

A parametric edit optimized over fully masked and partially revealed target states should be more robust for multi-token targets than a full-mask-only edit.

Success means:

```text
full-span exact improves by >=0.10 absolute on at least two of target lengths 2, 3, 4
```

or a paired positive improvement on the pooled multi-token set with no locality regression.

## H3 — Preservation-subspace hypothesis

Projecting or solving the update in the null space of training-only preservation keys should reduce same-subject and locality interference while retaining most edit efficacy.

Success means:

```text
same-subject TFPR reduced >=50% relative to the strongest efficacy-matched baseline
or distributional locality KL reduced >=25%

while rewrite and paraphrase each decline <=0.05 absolute
```

## H4 — Diffusion-native parametric editor hypothesis

The full causal + partial-state + preservation method should outperform ordinary MDM-MEMIT, partial-state MDM-MEMIT, and null-space-only editing on the joint efficacy/locality trade-off.

## H5 — Generality hypothesis

The method should show the same qualitative benefit on Dream or, if Dream integration is infeasible after one repair, on a second LLaDA checkpoint with a clearly limited claim.

---

# 1. Main method

## 1.1 Positive state bank

For edit `e = (subject, relation, target_true -> target_new)`, construct positive states from:

```text
rewrite prompt
training-only relation-template augmentations
all target mask counts
multiple revealed-position subsets per mask count
optional states visited by the default confidence trajectory
```

For each selected layer and token position, collect factual keys:

```text
K_plus[l, state] = MLP up-projection input at the selected causal site
```

## 1.2 Target-value optimization

Optimize target residual/value across the same state bank:

```text
L_target = average masked-token negative log likelihood of target_new
L_old_suppress = old-target margin penalty
L_state_consistency = variance penalty across mask states
L_target_value_norm = target residual norm penalty
```

The primary target-value objective is:

```text
L_value = L_target
        + 0.25 * L_old_suppress
        + 0.10 * L_state_consistency
        + lambda_value_norm * L_target_value_norm
```

The exact starting coefficients may be changed only inside the bounded development grid.

## 1.3 Preservation key bank

Collect training-only preservation keys from:

```text
same subject, different relation
different subject, same relation
near locality
far locality
attribute
generation
random unrelated
```

Evaluation prompts are forbidden from this bank.

## 1.4 Null-space/constrained update

Estimate the protected subspace from the preservation key covariance or SVD:

```text
K_minus = preservation keys
U_r = top singular vectors covering protected_variance
N = I - U_r U_r^T
```

Solve the edit update in the remaining subspace:

```text
min_D ||(W + D N) K_plus - V_star||_F^2
    + lambda_update ||D N||_F^2
    + lambda_identity ||D N K_minus||_F^2
```

Record the remaining editable dimension. The method must fail safely if the projector removes essentially all useful edit directions.

## 1.5 Optional state-conditioned rescue

Only if the shared permanent update passes efficacy but fails partial-state robustness, one bounded rescue may learn or solve three low-rank residuals:

```text
full-mask bucket
intermediate-mask bucket
late-mask bucket
```

At inference, select or interpolate the residual using active-mask count. This variant is an inference-conditioned parametric editor and must be reported separately from permanent weight editing.

---

# 2. Campaign outputs and state

Create:

```text
runs/diffusion_native_causal_partial_state_editor_v1/
  autonomous_campaign_v1/
    campaign_state.json
    stage_history.csv
    autonomous_log.md
    cost_state.json
```

`cost_state.json` is informational only and cannot block work.

Every stage writes versioned artifacts and an explicit acceptance report.

---

# Phase A — Bootstrap, sources, and fresh protocol

## A0 — Campaign bootstrap

### Tasks

1. Read all root campaign files.
2. Inspect historical terminal packages without modifying them.
3. Verify the latest Git commit and clean working tree or record intentional changes.
4. Initialize campaign state.
5. Start the existing RunPod Pod and keep it running.
6. Verify GPU, SSH, `/workspace/SB`, model cache, and tests.

### Acceptance

```text
autonomous mode enabled
active protocol matches root registry
Pod running with GPU
SSH and /workspace valid
remote tests pass
historical protocols marked read-only
analysis_500_used = false
final_test_used = false
```

### Infrastructure rescue

Up to `DNPE_MAX_INFRA_RETRIES` retries for Pod start, SSH refresh, package repair, or model-cache repair. No scientific work begins if the source tree or persistent storage is unsafe.

---

## A1 — Source and implementation audit

### Required primary sources

```text
Knowledge Editing in Masked Diffusion Language Models
TimeROME-DLM
ROME
MEMIT
AlphaEdit
LLaDA
Dream
CounterFact
KAMEL
```

### Tasks

1. Fetch/inspect official paper and official code when available.
2. Record exact algorithmic assumptions, model versions, layer conventions, tokenization, and evaluation definitions.
3. Map existing repository code to the required components.
4. Identify missing implementations.
5. Freeze a source audit before experiments.

### Outputs

```text
source_audit/report_summary.json
source_audit/source_matrix.csv
source_audit/implementation_gap.md
source_audit/algorithm_equation_map.md
source_audit/model_version_lock.json
```

### Acceptance

```text
all required sources identified
paper/code differences documented
model/tokenizer revisions recorded
no algorithm claimed reproduced without source-aligned settings
```

---

## A2 — Fresh manifests and exclusion audit

### CounterFact splits

```text
dnpe_smoke_20
dnpe_pilot_100
dnpe_dev_200
analysis_500  # historical locked manifest, untouched until lock
final_test_500  # historical locked manifest, untouched until lock
```

### KAMEL multi-token splits

For lengths 2, 3, and 4:

```text
dnpe_kamel_smoke_20_<N>
dnpe_kamel_dev_100_<N>
dnpe_kamel_locked_200_<N>
```

Optional length 5 if enough valid rows exist.

### Rules

```text
fresh dev/smoke rows must exclude historical used fingerprints
train/dev/locked overlap = 0
context-aware tokenization is authoritative
source split/index namespace verified
real prompts preferred; synthetic prompts explicitly tagged
```

### Acceptance

```text
all required manifests exist
zero overlap
required target lengths present
relation and target-length histograms written
locked manifests inaccessible to tuning scripts
```

One split-builder repair is allowed for implementation errors, not policy weakening.

---

# Phase B — Reproduce strong baselines

## B1 — MDM-MEMIT reproduction

### Primary configuration

```text
model = GSAI-ML/LLaDA-8B-Instruct
editable weights = floating point
edit site = last subject token
early-to-middle MLP layer window selected on development only
masked input = one mask per target token
key and target value computed from the same masked-input format
```

### Stages

```text
smoke20
pilot100
dev200
```

### Acceptance

```text
rewrite exact >= 0.75
paraphrase exact >= 0.40
pre-edit target_new rewrite <= 0.10
no NaN or invalid updates
```

Strong reproduction:

```text
rewrite >= 0.85
paraphrase >= 0.50
```

One bounded reproduction repair is allowed to align layer indexing, tokenizer, target-value optimization, or update convention with the source implementation.

If MDM-MEMIT does not meet the minimum after repair, the main campaign may continue only as a diagnostic causal/locality study; write this limitation explicitly.

---

## B2 — Partial-state MDM-MEMIT reproduction

Use target lengths 2, 3, and 4.

### Required policies

```text
fully_masked_only
all_mask_counts_random_positions
confidence_trajectory_states
uniform_mask_count_states
```

### Acceptance

At least two lengths must show:

```text
full-span exact gain >= 0.15
paraphrase gain >= 0.08
```

or reproduce the source paper's qualitative trend with statistically positive pooled gains.

If the correction fails, identify exact implementation/protocol differences. Continue the main method, but do not claim a reproduced partial-state baseline.

---

## B3 — AlphaEdit-style null-space MDM-MEMIT baseline

### Tasks

1. Build a preservation-key covariance from train-only anchors.
2. Project the standard MDM-MEMIT update into the protected null space.
3. Sweep only the predeclared protected-variance/ridge grid.
4. Measure editability remaining after projection.

### Bounded grid

```text
protected_variance in {0.90, 0.95, 0.99}
ridge in {1e-4, 1e-3, 1e-2}
```

### Acceptance as a valid baseline

```text
rewrite >= MDM-MEMIT - 0.10
no numerical collapse
projector dimension reported
same-subject or locality metric improves
```

This baseline need not pass the main method criteria; it must be implemented faithfully enough to be a serious comparator.

---

## B4 — TimeROME-DLM-style temporal residual memory baseline

Implement a paper-faithful or clearly labelled inspired baseline:

```text
temporal indirect effect localization
closed-form low-rank residual memory
ridge regularization
sparsification
application at the selected coordinate during each diffusion forward
```

### Acceptance as a valid baseline

```text
temporal localization runs
residual memory is finite
rewrite/paraphrase and retain/locality metrics are complete
runtime overhead and storage are reported
```

If official code is unavailable, record every deviation and use the label `timerome_dlm_style`, not `TimeROME-DLM reproduction`.

---

# Phase C — Temporal causal localization

## C1 — Standard causal tracing

Run clean, corrupted, and restored passes on old factual recall.

### Coordinates

```text
all layers
first subject token
last subject token
relation-cue token
first answer mask
MLP contribution
attention contribution
full hidden state
```

### Metrics

```text
old-target probability recovery
normalized AIE
causal peak layer/position
random-restoration baseline
```

### Acceptance

```text
last-subject early/mid MLP peak is detectable or a different stable peak is documented
causal effect exceeds random restoration by >=0.15 normalized effect
```

---

## C2 — Temporal/partial-state causal tracing

For target lengths 1–4, trace across:

```text
fully masked state
all mask counts
multiple reveal subsets
actual confidence trajectory states
```

Define Temporal Indirect Effect (TIE) as future target-log-probability recovery caused by restoring a coordinate at an earlier state and continuing the frozen denoising policy.

### Required outputs

```text
tie_by_layer_position_state.csv
site_stability_by_edit.csv
site_overlap_by_mask_count.csv
causal_heatmaps.png
causal_vs_editability.csv
```

### Acceptance

```text
TIE finite and reproducible
site stability reported
selected temporal site policy beats random on causal effect
no evaluation target leakage into site selection
```

---

## C3 — Select and freeze site policies for pilot

Select at most three:

```text
fixed_global_site
per_edit_top_tie_site
stable_temporal_site_set
```

No more site policies may be introduced after pilot inspection.

---

# Phase D — Build the main editor

## D1 — Positive and preservation state banks

### Positive bank

```text
rewrite states
training-only augmentations
all mask counts
random reveal subsets
optional actual trajectory states
```

### Preservation bank

```text
training-only same-subject different-relation
near/far locality
attribute
generation
unrelated
```

### Acceptance

```text
train/eval prompt overlap = 0
all state categories present or explicitly unavailable
keys aligned to selected sites
all activations finite
```

---

## D2 — Multi-state target-value optimization

Train/optimize target values under:

```text
fullmask_only
partial_state_shared_value
partial_state_state_weighted_value
```

### Bounded grid

```text
value_lr in {0.05, 0.10}
value_steps in {25, 50}
state_consistency_weight in {0.0, 0.1}
old_target_suppression_weight in {0.0, 0.25}
```

No larger grid.

### Acceptance

```text
loss finite and decreases
full target probability improves on held-out partial states
target value norm within configured cap
```

---

## D3 — Causal multi-state update

Solve updates using each frozen site policy and compare:

```text
fixed-site fullmask
fixed-site partial-state
causal-site fullmask
causal-site partial-state
```

### Acceptance

```text
closed-form residual equations validate on toy tests
update rank/norm finite
edited model loads and decodes
all metrics complete
```

---

## D4 — Null-space constrained main method

Apply the preservation projection/constrained solve to the best causal partial-state update.

Primary method:

```text
causal_partial_state_nullspace_memit
```

### Bounded locality grid

```text
protected_variance in {0.90, 0.95, 0.99}
lambda_update in {1e-4, 1e-3, 1e-2}
lambda_identity in {0.1, 1.0, 2.0}
```

Use staged narrowing; do not run the full Cartesian product unless cheap from cached statistics.

### Acceptance to advance

On pilot100:

```text
rewrite >= 0.75
paraphrase >= 0.40
same-subject TFPR <= base + 0.03
near/far TFPR <= base + 0.03
malformed <= 0.05
```

and relative to the strongest efficacy-matched baseline:

```text
same-subject TFPR reduced >=50%
or distributional KL reduced >=25%

with rewrite/paraphrase loss <=0.05 each
```

---

## D5 — Optional state-conditioned residual rescue

Trigger only if:

```text
permanent main update passes single-token efficacy and locality
but fails multi-token partial-state acceptance
```

Build three low-rank residuals for full/intermediate/late mask buckets. Keep the same causal sites and preservation subspace.

No new feature family or architecture is allowed.

---

# Phase E — Pilot and bounded selection

## E1 — Smoke20

Methods:

```text
base
prompt_memory
target_logit_bias
mdm_memit
partial_state_mdm_memit
alphaedit_style_mdm_memit
timerome_dlm_style_residual_memory
random_site_partial_state_editor
fixed_site_partial_state_editor
causal_site_partial_state_editor
causal_partial_state_nullspace_memit
```

Smoke catches integration failures. One bounded calibration is allowed for update scale and projector rank only.

Red stop examples:

```text
NaN/corrupt updates
rewrite no better than base
same-subject TFPR >0.50 for every parametric method
train/eval leakage
```

---

## E2 — Pilot100 and KAMEL dev

Evaluate:

```text
100 CounterFact edits
100 KAMEL edits per length 2,3,4
```

Primary pilot criteria are the hard criteria in AGENTS.md.

### Pilot pass

At least one main candidate must:

```text
pass efficacy floor
pass locality budgets
improve locality over partial-state MDM-MEMIT or AlphaEdit-style baseline
show causal-site or partial-state mechanism value
```

If no candidate passes after permitted rescues, finish formally negative without opening analysis/final.

---

# Phase F — Dev selection, scaling, and second backbone

## F1 — dnpe_dev_200 selection

Use staged selection:

```text
1. site policy
2. partial-state policy
3. projector rank/ridge
4. update rank/layer window
5. optional state-conditioned rescue if triggered
```

Required metrics and baselines remain fixed.

Select at most three candidates:

```text
A = best stress-aware aggregate
B = best locality/safety
C = best multi-token robustness
```

At least one must satisfy all hard constraints to proceed.

---

## F2 — Multi-edit and sequential scaling

Test edit counts:

```text
1
10
50
100
```

Compare:

```text
ordinary MDM-MEMIT
AlphaEdit-style projection
TimeROME-DLM-style residual memory
main editor
```

Report:

```text
rewrite/generalization retention
same-subject/locality drift
inter-edit interference
update rank/norm growth
wall clock
storage
```

This stage is required for a strong parametric editing claim but may remain secondary if the single-edit method is the primary contribution.

---

## F3 — Dream second backbone

Run one bounded integration repair if needed.

Minimum confirmation:

```text
same direction of locality improvement
rewrite >= baseline -0.10
same-subject/locality improvement on at least one primary metric
```

If Dream remains infeasible, run LLaDA-8B-Base as a fallback and limit the claim to cross-checkpoint, not cross-architecture generality.

---

# Phase G — Lock, analysis, and final

## G1 — Dev lock

Write:

```text
runs/diffusion_native_causal_partial_state_editor_v1/dev_method_lock.json
```

Freeze every method, metric, seed, layer, projector, and report choice.

Only after validation set `DEV_METHOD_LOCKED=1`.

---

## G2 — analysis_500

Run the frozen primary candidate and required baselines once.

### Analysis pass

```text
retain >=80% of dev rewrite
retain >=80% of dev paraphrase
same-subject/near/far budgets pass
malformed <=0.05
locality advantage over strongest baseline remains
paired confidence intervals preserve the qualitative claim
```

If analysis fails, do not tune. Finish negatively.

---

## G3 — final_test_500

After analysis lock, run exactly once.

Required final methods:

```text
base
mdm_memit
partial_state_mdm_memit
alphaedit_style_mdm_memit
timerome_dlm_style_residual_memory
selected main editor
key causal/null-space ablation
```

---

# Phase H — Final package and Pod shutdown

Create:

```text
runs/diffusion_native_causal_partial_state_editor_v1/final_research_package_v1/
```

Required artifacts are listed in AGENTS.md and PAPER_REPRODUCIBILITY_PLAN.md.

Classify the strongest claim honestly:

```text
strong_diffusion_native_parametric_editor
locality_preservation_improvement
partial_state_editing_improvement
causal_localization_result
reproduction_only
bounded_negative_result
infrastructure_blocked
```

Validate hashes and campaign state, then stop the Pod.

---

# Terminal decision rules

## Positive completion

A strong result requires:

```text
successful MDM-MEMIT baseline
main editor passes rewrite/paraphrase floors
same-subject and near/far locality budgets pass
main editor improves the stress-aware trade-off over strong baselines
causal or partial-state mechanism value is established
locked analysis and final results survive
```

## Locality-only positive completion

A narrower result is valid if:

```text
main editor preserves baseline efficacy within 0.05
and reduces same-subject TFPR or distributional drift substantially
and survives locked confirmation
```

## Formal negative completion

Finish negatively if, after bounded rescues:

```text
no method simultaneously meets efficacy and locality floors
causal-site selection adds no editing value
partial-state optimization adds no multi-token robustness
null-space constraints destroy editability or do not improve locality
analysis fails
```

The negative report must distinguish implementation failure, protocol infeasibility, and scientific rejection.
