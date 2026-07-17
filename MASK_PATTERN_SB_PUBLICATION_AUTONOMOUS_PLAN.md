# Autonomous Publication-Confirmation Plan

Protocol root:

```text
mask_pattern_sb_publication_confirmation_v1
```

## 1. Goal

Turn the validated pilot result into a rigorous ICLR/NeurIPS-level evaluation
of answer-span mask-pattern path control for multi-token factual editing in
masked diffusion language models.

The campaign must determine whether the observed improvement is caused by:

```text
a genuine finite-beta entropy/KL-control mechanism
rather than
a weak baseline, more model evaluations, deterministic global planning,
seed selection, or one-model implementation quirks.
```

The primary hypothesis is:

> After an MDM-MEMIT factual update, an exact finite-state
> entropy-regularized controller over answer-span mask configurations improves
> full-target realization over the strongest compute-matched non-SB reveal
> planner.

## 2. Starting state

Validated historical seed results:

```text
MDM-MEMIT on LLaDA-8B-Instruct:
  rewrite exact = 0.864
  paraphrase exact = 0.491

Exact mask-pattern controller pilot:
  length 3 rewrite delta = +0.090
  length 4 rewrite delta = +0.065 to +0.075
  target trajectory-cost reduction = approximately 24% to 26%
```

Historical limitations:

```text
partial-state MEMIT correction did not improve in the prior campaign
one length-3 paraphrase comparison was inconclusive
the exact controller used more model evaluations
only one primary backbone was tested
the precise classical-SB naming remains to be audited
```

No historical `analysis_500`, `final_test_500`, or `final_test_full` may be
opened for this campaign.

## 3. Campaign state

Create:

```text
runs/mask_pattern_sb_publication_confirmation_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  cost_state.json
  artifact_availability.json
```

The cost state is informational only.

## 4. Phase A — Bootstrap

### A1. Validate campaign files and environment

Required:

```text
MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_MODE = 1
existing RunPod Pod starts or is already running
GPU available
SSH available
persistent /workspace available
Git worktree synchronized
all tests pass
```

### A2. Validate historical seed artifacts

Audit:

```text
masked_diffusion_memit_sb_positive_result_v1 final package
M1 MDM-MEMIT result artifacts
M2 negative result artifacts
M4 exact mask-pattern controller artifacts
all source split/fingerprint manifests
```

Write:

```text
runs/mask_pattern_sb_publication_confirmation_v1/source_audit/
  report_summary.json
  historical_artifact_registry.json
  historical_result_reproduction_table.csv
  missing_artifacts.md
```

Historical pilot values may motivate the design but may not be copied into the
fresh locked result table.

## 5. Phase B — Source and implementation audit

### B1. Primary-source audit

Pin and record:

```text
knowledge-editing-in-MDM paper/version
LLaDA model/tokenizer revision
Dream model/tokenizer revision
MEMIT reference implementation
KAMEL source/version
mask-pattern solver implementation commit
```

Check once for official released code and record exact commits when available.

### B2. Implementation-difference register

Compare the current MDM-MEMIT/partial-state implementation with the source
paper in:

```text
model variant
edited layer window
last-subject position mapping
target-value optimizer
learning rate
optimization steps
clamp norm
KL anchor
target tokenization
mask count
partial-mask schedule
revealed-position resampling
loss positions
batch-edit convention
generation steps and schedule
KAMEL construction
```

Write:

```text
source_audit/implementation_difference_register.md
source_audit/model_module_maps.json
source_audit/paper_matched_config.json
```

## 6. Phase C — Fresh publication data protocol

### C1. Historical exclusion manifest

Exclude every source fact, source row, rendered prompt, target pair, and
fingerprint used in the prior positive campaign.

Read historical locked manifests only for identifiers and fingerprints.

### C2. LLaDA KAMEL splits

Build fresh, disjoint splits under LLaDA contextual tokenization.

Development target:

```text
200 edits for each N in {2,3,4,5,6}
```

Locked-confirmation targets:

```text
N=3: 500 edits
N=4: 500 edits
N=2: 300 edits
N=5: 300 edits
N=6: 300 edits
```

Primary locked lengths are 3 and 4.

Requirements:

```text
zero overlap with historical or new dev sets
same-relation counterfactual target swaps
exact contextual target length
real rewrite prompt
held-out paraphrase
same-subject negative or documented construction
near/far locality prompts where available
at least 20 relations for primary lengths if source permits
no one relation above 20% where feasible
```

If the source cannot supply 500 fresh primary examples after all exclusions,
the campaign must write a data-feasibility report. The primary sample size may
not be reduced below 300 without classifying the final result as
`narrow_method_ready` at best.

### C3. Dream matched splits

Construct source-fact-matched Dream splits where the contextual target length
under the Dream tokenizer is in `{3,4,5}`.

Target:

```text
Dream dev: 100 per length
Dream locked: 300 per length
```

Record all facts that cannot be matched across tokenizers.

### C4. Power analysis

Use historical paired deltas only to estimate the locked sample size. Write the
power analysis before the locked set is opened.

## 7. Phase D — P1 partial-state MEMIT discrepancy

Execute `PARTIAL_STATE_MEMIT_AUDIT_PLAN.md`.

This is a hard prerequisite for a top-tier readiness classification.

Outcomes:

```text
reproduced_paper_trend
concrete_protocol_difference_explained
unresolved_baseline_discrepancy
```

If unresolved after the single permitted repair, continue the remaining
experiments but final readiness cannot exceed `narrow_method_ready`.

## 8. Phase E — P2 theory and naming

Execute `THEORY_AND_NAMING_PLAN.md`.

The output must include:

```text
formal objective
exact dynamic-programming recurrence
controlled transition
proof/derivation
complexity
beta limits
exhaustive numerical validation
naming decision
```

If the audit concludes that the method is KL-control rather than a classical
endpoint-constrained SB, all code and paper artifacts must use the safer name.

## 9. Phase F — P3 serious baseline suite

Execute `COMPUTE_MATCHED_BASELINES_PLAN.md`.

Two regimes are mandatory:

```text
full cost-table regime
online compute-matched regime
```

Development selects:

```text
reference process
finite beta
best non-SB planner
beam/query budget
approximation parameters used later
```

All decisions are frozen before the locked LLaDA set is opened.

Write:

```text
runs/.../dev_method_lock.json
```

## 10. Phase G — P4 fresh locked LLaDA confirmation

Execute `LOCKED_LLADA_CONFIRMATION_PLAN.md`.

Primary comparisons:

```text
finite-beta exact controller
vs best compute-matched non-SB planner
at N=3 and N=4
```

Primary success requires:

```text
pooled N=3/N=4 rewrite delta >= +0.05
paired-bootstrap lower bound > 0
Holm-corrected primary test remains positive
each primary length has nonnegative mean delta
at least one primary length has delta >= +0.05 and lower bound > 0
trajectory target cost reduction >= 15%
malformed rate <= 0.05
same-subject TFPR increase <= 0.03
target-token F1 does not materially decline
```

A strong result has significant positive gains at both N=3 and N=4.

The locked set may be executed once only.

## 11. Phase H — P5 second backbone

Execute `SECOND_BACKBONE_DREAM_PLAN.md`.

Dream is required for `top_tier_ready`.

One bounded model-integration repair is allowed.

If Dream is technically impossible after that repair, run the same predeclared
test on `GSAI-ML/LLaDA-8B-Base` and classify cross-backbone evidence as weaker.
The final readiness cannot be `top_tier_ready` without a second architecture.

## 12. Phase I — P6 editor generality

Execute `EDITOR_GENERALITY_PLAN.md`.

The same frozen reveal controller must be evaluated with at least:

```text
ordinary MDM-MEMIT
paper-matched partial-state MDM-MEMIT
```

A third edit-conditioning mechanism is secondary:

```text
prompt-memory edit statement
or target-logit edit guidance
```

The controller should show a positive effect under at least two editor
conditions.

## 13. Phase J — P7 approximate solver and scaling

Execute `APPROXIMATE_SOLVER_PLAN.md`.

The exact solver is evaluated through target length 6.

The approximate solver is evaluated on lengths 5 and 6 against exact DP and, if
data exists, on lengths 7 through 10.

A failed approximation does not invalidate the exact short-span result, but it
limits significance and must be reported.

## 14. Phase K — P8 statistics and final package

Execute `PAPER_REPRODUCIBILITY_PLAN.md`.

No result may be called positive based only on an uncorrected point estimate.

Final readiness rules:

### `top_tier_ready`

All must hold:

```text
partial-state baseline discrepancy resolved or concretely explained
fresh locked LLaDA primary comparison passes
best compute-matched non-SB planner is beaten
finite-beta mechanism adds value beyond beta=0 and beta=infinity
second backbone has consistent positive evidence
positive effect appears under at least two editor conditions
formal naming is defensible
locality and malformed constraints pass
complete reproducibility package validates
```

### `narrow_method_ready`

Examples:

```text
fresh LLaDA confirmation passes
but Dream is inconclusive or unavailable

or

method beats fixed policies but not the best compute-matched global planner

or

formal audit classifies the method as entropy-regularized planning rather than SB
```

### `diagnostic_only`

Examples:

```text
fresh gain does not survive compute matching
finite beta adds no benefit over deterministic planning
positive effect occurs only for one target length
second backbone reverses the result
```

### `fresh_confirmation_failed`

The fresh locked LLaDA primary test fails.

## 15. Campaign shutdown

After the final publication package validates:

```text
mark campaign_state.json terminal
mirror compact artifacts locally or to durable storage
verify no Python/GPU process is active
stop the configured Pod
```

Do not stop earlier because of a scientific failure in one track.
