# Schrödinger-Bridge Alternatives Autonomous Research Plan

Complete autonomous execution plan for `counterfact_sb_alternatives_campaign_v1`.

The campaign tests five scientifically distinct ways to use Schrödinger bridges for factual editing in LLaDA:

```text
T1 — learned edit-intent gate + raw runtime bridge
T2 — activation-space Schrödinger bridge
T3 — conditional answer-span categorical Schrödinger bridge matching
T4 — unbalanced / partial categorical Schrödinger bridge
T5 — parameter-space Schrödinger bridge over low-rank adapter latents
```

The campaign is a bounded, breadth-first state machine. Every track receives a mandatory pilot before any track receives expensive scale-up. A track failure does not end the campaign. The Pod remains running until all tracks are terminal and the final cross-track package is complete, or until the budget/infrastructure makes continuation impossible.

---

# 0. Scientific objective

The common research question is:

> Can a Schrödinger-bridge formulation produce factual edits in a masked diffusion language model that improve rewrite and paraphrase success while controlling same-subject leakage, preserving locality, and remaining computationally defensible?

The five tracks test different possible locations for the bridge:

```text
T1: bridge remains an inference-time token controller; only edit intent is learned.
T2: bridge transports continuous hidden activations.
T3: bridge is learned directly over discrete answer-span states.
T4: bridge transports only a learned fraction of prompt mass and leaves the rest as identity.
T5: bridge transports low-rank adapter parameters from zero update to an edit-specific update.
```

Historical evidence is not a new result:

```text
Direction 1: raw runtime bridge can edit; rule-based gates failed same-subject stress.
Direction 2 v1: protocol-infeasible before adapter science.
Direction 3 v1: learned gate was promising; learned value controller failed representation-use tests.
```

---

# 1. Campaign ordering and breadth-first policy

Mandatory pilot order:

```text
1. T1 learned gate + raw bridge
2. T2 activation-space SB
3. T3 conditional answer-span CSBM
4. T4 unbalanced/partial CSBM
5. T5 parameter-space SB
```

This order reflects expected cost and reuse:

```text
T1 is cheapest and uses components already supported by evidence.
T2 tests a continuous representation where SB geometry is more natural.
T3 implements the most faithful categorical bridge-matching idea.
T4 reuses T3 infrastructure and tests sparse/partial transport.
T5 is the highest-risk and most expensive formulation.
```

Breadth-first rule:

```text
No track enters scale-up or common dev_tune_200 until all five mandatory pilots are terminal:
  pilot_passed,
  pilot_failed,
  formal_negative,
  or budget_not_run.
```

A track may use one explicitly listed rescue during its pilot. After rescue failure, write its stop package and continue.

---

# 2. Common protocol and data construction

## 2.1 Campaign directories

Create:

```text
runs/counterfact_sb_alternatives_campaign_v1/
  autonomous_campaign_v1/
  common_protocol_v1/
  cross_track_dev_v1/
  final_research_package_v1/
```

Track directories:

```text
runs/counterfact_learned_gate_raw_bridge_v1/
runs/counterfact_activation_space_sb_v1/
runs/counterfact_conditional_answer_span_csbm_v1/
runs/counterfact_unbalanced_partial_csbm_v1/
runs/counterfact_parameter_space_sb_v1/
```

## 2.2 Legal source pool

Use official CounterFact train rows.

Exclude from every track's training pool:

```text
dev_tune_200
ablation_500
analysis_500
final_test_500
final_test_full
official evaluation-only same-subject stress prompts
```

Previously used historical training rows may be reused as new-campaign training rows if:

```text
they are not locked evaluation rows;
historical teacher outputs/checkpoints are not silently reused;
the overlap is recorded;
all methods use an explicitly declared source policy.
```

Locked manifests may be read before unlock only for:

```text
case_id
source split
source index
fingerprint
exclusion auditing
```

Do not read locked prompt text, labels, generated outputs, or metrics for training/tuning.

## 2.3 Common split ladder

Create deterministic common splits:

```text
sb_alt_train_2000
sb_alt_val_300
sb_alt_smoke_50
sb_alt_confirmation_50
```

If the legal pool cannot satisfy 2000/300, use the largest deterministic predeclared fallback:

```text
sb_alt_train_1000
sb_alt_val_200
```

Do not invent smaller common split sizes without a formal budget/protocol stop.

`sb_alt_smoke_50` is split before inspection into:

```text
smoke_20 = first stratified 20
confirmation_30 = remaining 30
```

`sb_alt_confirmation_50` is reserved for tracks that require an independent post-pilot confirmation beyond the 20/30 split; use must be predeclared by the track plan.

## 2.4 Stratification

Stratify by:

```text
relation_id
target length exact bin: 1,2,3,>=4
primary stratum: single-token vs multi-token >=2
subject ambiguity
base target-new success when available
availability of real rewrite/paraphrase/locality prompts
availability of same-subject hard negatives
```

Do not require exact bin 2 in every split if the legal pool cannot support it. Always report exact target-length composition.

## 2.5 Prompt materialization

Use real allowed prompt fields whenever available:

```text
rewrite prompts
declarative paraphrases
near/far locality prompts
generation prompts
attribute prompts
same-subject different-relation prompts
unrelated prompts
```

Synthetic prompts are permitted only as tagged fallback:

```text
synthetic_from_metadata = true
synthetic_reason = <reason>
```

No official evaluation prompt for an edit may be used to train that edit's method.

## 2.6 Common protocol outputs

Write:

```text
common_protocol_v1/report_summary.json
common_protocol_v1/source_policy.md
common_protocol_v1/split_summary.json
common_protocol_v1/split_overlap_audit.csv
common_protocol_v1/target_length_histograms.csv
common_protocol_v1/relation_histograms.csv
common_protocol_v1/prompt_provenance_summary.csv
common_protocol_v1/common_baseline_registry.json
```

Acceptance:

```text
zero overlap between train/val/smoke/confirmation and locked evaluation splits
all fingerprints recorded
real prompt coverage reported
single-token and multi-token examples present where legally available
same-subject hard negatives available for >=80% of training edits, or exact shortfall documented
analysis_500_used_for_tuning = false
final_test_used_for_tuning = false
```

---

# 3. Common baseline registry

Required common runtime baselines:

```text
base
target_logit_bias
prompt_memory
myopic_score
no_rollout_bridge
mc_bridge
best historical raw bridge configuration
```

Track-specific baselines:

```text
T1: rule-based subject gate, rule-based relation gate, no gate
T2: direct mean activation shift, linear regression map, entropic OT map without bridge dynamics
T3: direct target CE, ordinary-noising endpoint model, forward-only CSBM
T4: balanced CSBM, external learned gate + balanced CSBM, fixed transport-mass mixture
T5: direct per-edit adapter, conditional MLP adapter generator, linear interpolation in adapter latent
```

All baseline configs must be frozen before their comparison stage.

---

# 4. Common metrics and constraints

## 4.1 Efficacy/generalization

```text
rewrite exact
declarative paraphrase exact
target-token F1
old-target suppression
probability margin when available
```

## 4.2 Locality/safety

```text
near locality exact
far locality exact
self-normalized locality
same-subject target false-positive rate
generation target false-positive rate
attribute target false-positive rate
malformed-span rate
```

## 4.3 Bridge/intervention metrics

```text
sparse-support guidance KL
transport/intervention energy
transport mass for partial methods
endpoint error
teacher/student candidate agreement where applicable
number of denoising steps affected
```

## 4.4 Efficiency

```text
model evaluations per edit
GPU minutes per edit
training GPU minutes
checkpoint/adapter bytes per edit
feature-cache bytes
peak VRAM
```

## 4.5 Statistical reporting

Use paired bootstrap resampling by `edit_id`, not by prompt row.

Report:

```text
mean
95% confidence interval
number of edits
number of prompt cases
micro and edit-macro metrics
relation-macro diagnostics
exact target-length breakdown
```

## 4.6 Common hard constraints

```text
same_subject_tfpr <= base_same_subject_tfpr + 0.03
near_tfpr <= base_near_tfpr + 0.03
far_tfpr <= base_far_tfpr + 0.03
malformed_rate <= 0.05
gpu_minutes_per_edit <= 2.0 unless a track declares a stricter bound
```

A pilot may receive a yellow diagnostic status under a looser track-specific threshold, but it cannot become a dev candidate without passing the common hard constraints.

---

# 5. Phase A — Campaign bootstrap

## A.1 State initialization

Create:

```text
runs/counterfact_sb_alternatives_campaign_v1/autonomous_campaign_v1/
  campaign_state.json
  budget_state.json
  track_registry.csv
  stage_history.csv
  autonomous_log.md
```

Record:

```text
current Git commit
Pod ID/GPU/hourly rate
authorized budget
historical terminal states
all five pending tracks
analysis/final lock state
```

Acceptance:

```text
SB_ALT_AUTONOMOUS_MODE = 1
budget variables parse
all required plans exist
ACTIVE_RESEARCH_CAMPAIGN.json matches this campaign
analysis_500_used = false
final_test_used = false
```

## A.2 Start and retain Pod

Start configured existing Pod, refresh SSH details if needed, then verify:

```text
GPU allocated
SSH works
/workspace/SB available
Git synchronized
remote tests pass
persistent storage mounted
```

After A.2, keep the Pod running until campaign terminal completion.

## A.3 Budget feasibility

Create a conservative pilot estimate for every track before launching the first scientific job.

The budget must reserve at least:

```text
next mandatory pilot estimate
+ minimum reserve for every still-untested track
+ final reporting reserve
```

If the total budget cannot support all mandatory pilots, write configuration-level budget completion before consuming significant compute.

---

# 6. Phase B — Mandatory minimum pilots for every track

Execute in order. Each detailed procedure is defined in its own plan.

## B.1 T1 — Learned gate + raw bridge

Plan: `LEARNED_GATE_RAW_BRIDGE_PLAN.md`

Mandatory pilot completion requires:

```text
real-prompt learned gate trained and validated
feature leakage audit passed
smoke20 actual decode completed
confirmation30 completed if smoke is green/yellow after bounded calibration
track classified pilot_passed or pilot_failed
```

Minimum pilot pass:

```text
rewrite >= base + 0.15
paraphrase >= base + 0.10
same-subject TFPR <= base + 0.03
malformed <= 0.05
learned-gated method improves safety over corresponding ungated/rule-gated controller
```

## B.2 T2 — Activation-space SB

Plan: `ACTIVATION_SPACE_SB_PLAN.md`

Mandatory pilot completion requires:

```text
activation endpoint dataset built
identity-negative transport included
Gaussian/linear activation SB pilot trained
offline transport audit completed
smoke20 actual activation intervention completed if offline passes
one neural-drift rescue at most
track classified
```

Minimum pilot pass:

```text
endpoint transport clearly beats direct mean shift on held-out edit states
identity-negative drift remains small
actual rewrite/paraphrase improve over base
same-subject TFPR passes
transport is not equivalent to a constant target direction
```

## B.3 T3 — Conditional answer-span CSBM

Plan: `CONDITIONAL_ANSWER_SPAN_CSBM_PLAN.md`

Mandatory pilot completion requires:

```text
reference reciprocal bridge sampler validated
single-token answer-span pilot trained
bidirectional D-IMF compared with forward-only and ordinary noising
identity negatives evaluated
smoke20 actual decode completed if offline passes
one bounded outer-iteration/temperature rescue at most
track classified
```

Minimum pilot pass:

```text
bridge-state training beats ordinary-noising endpoint training
bidirectional training beats forward-only
identity-negative transport stays near base
actual rewrite/paraphrase improve over base
same-subject TFPR passes
```

## B.4 T4 — Unbalanced/partial CSBM

Plan: `UNBALANCED_PARTIAL_CSBM_PLAN.md`

Mandatory pilot completion requires:

```text
learned transport-mass objective implemented
balanced CSBM and external-gate baselines available
partial/unbalanced pilot trained
offline mass calibration audited
smoke20 actual decode completed if offline passes
one mass-regularization rescue at most
track classified
```

Minimum pilot pass:

```text
negative prompts receive low transport mass
positive prompts retain useful transport mass
same-subject leakage improves over balanced CSBM
rewrite/paraphrase loss relative to balanced method <=10% when balanced method is effective
actual hard constraints pass
```

## B.5 T5 — Parameter-space SB

Plan: `PARAMETER_SPACE_SB_PLAN.md`

Mandatory pilot completion requires:

```text
direct low-rank endpoint adapters generated
direct adapter baseline evaluated
adapter latent representation built
conditional parameter-space bridge trained
conditional MLP and linear latent baselines compared
smoke10/20 actual adapter evaluation completed if offline passes
one bounded rank/latent rescue at most
track classified
```

Minimum pilot pass:

```text
direct endpoint adapter is itself viable
SB-generated adapters generalize to held-out edits
SB beats conditional MLP or linear interpolation on stress-aware score at matched storage/compute
same-subject TFPR passes
storage/training costs are reported
```

## B.6 Pilot registry freeze

After all five pilots are terminal, write:

```text
runs/counterfact_sb_alternatives_campaign_v1/pilot_registry_lock.json
```

It must record:

```text
track status
pilot evidence paths
rescue used
scientific failure type
whether scale-up is allowed
projected scale-up cost
```

Only `pilot_passed` tracks may enter Phase C.

---

# 7. Phase C — Scale every pilot-passed track

Do not scale tracks that failed their pilot.

For each passed track:

1. Build the track's declared scaled training/validation data.
2. Run only the bounded scale grid in its plan.
3. Evaluate on a track-specific held-out validation set.
4. Run the common `dev_tune_200` method selection stage.
5. Nominate at most one frozen candidate.

Budget rule:

```text
If multiple tracks pass, scale each passed track only after verifying the remaining budget can still execute a bounded scale stage for every other passed track and preserve analysis/final/reporting reserve.
```

If a track cannot be scaled for budget reasons after passing its pilot, mark:

```text
pilot_passed_scale_not_run_budget
```

Do not label it scientifically failed.

---

# 8. Phase D — Common dev_tune_200 cross-track comparison

## D.1 Preflight

All candidate checkpoints/configs must be frozen before reading common dev results beyond their planned evaluation.

Required comparison:

```text
base and common baselines
one frozen candidate per passed/scaled track
key track-specific ablations predeclared in each plan
```

## D.2 Common constraints

Every dev candidate must pass:

```text
same-subject TFPR budget
near/far TFPR budgets
malformed <=0.05
compute/storage constraints declared by track
no feature leakage
complete target-length reporting
```

## D.3 Primary selection

Choose the primary candidate on dev by:

```text
1. feasibility under all hard constraints;
2. highest stress-aware feasible selection score;
3. lower compute at statistically indistinguishable quality;
4. stronger evidence that the SB-specific component matters over its non-SB ablation.
```

A method cannot support an SB-specific claim unless it beats its direct/non-SB baseline on a predeclared metric or Pareto comparison.

## D.4 Cross-track dev lock

Write:

```text
runs/counterfact_sb_alternatives_campaign_v1/cross_track_dev_lock.json
```

Freeze primary, secondary candidates, baselines, all hyperparameters, metrics, and fallback policy before analysis.

---

# 9. Phase E — Locked analysis and final evaluation

## E.1 analysis_500

After lock validation, set `DEV_METHOD_LOCKED=1`.

Run once:

```text
primary dev-selected candidate
all precommitted frozen secondary passed-track candidates
required common baselines
```

Primary analysis pass:

```text
retains >=80% of dev rewrite
retains >=80% of dev paraphrase
passes same-subject, near/far, and malformed budgets
retains the main compute/storage advantage
paired CIs preserve the qualitative claim
```

Analysis cannot change the primary.

If primary fails, campaign finishes negatively. A secondary cannot be promoted unless the exact fallback was precommitted before analysis.

## E.2 final_test_500

After analysis pass, write `analysis_confirmation_lock.json`, set `FINAL_METHOD_LOCKED=1`, and run exactly once.

Final package methods:

```text
base
common runtime baselines
primary method
all precommitted analysis-passed secondary track candidates if budget permits
key ablations precommitted before analysis
```

No rerun for tuning.

`final_test_full` is optional secondary replication only if precommitted and budget permits.

---

# 10. Phase F — Final cross-track reporting and Pod shutdown

Create:

```text
runs/counterfact_sb_alternatives_campaign_v1/final_research_package_v1/
```

Required artifacts:

```text
report_summary.json
cross_track_status_table.csv
cross_track_main_results.csv
cross_track_pilot_results.csv
same_subject_stress_table.csv
target_length_table.csv
relation_table.csv
compute_storage_table.csv
paired_bootstrap.csv
rewrite_locality_pareto.png
aggregate_compute_pareto.png
same_subject_plot.png
failure_cases.csv
track_failure_taxonomy.csv
artifact_availability_manifest.json
reproducibility_manifest.json
final_research_report.md
paper_claim_matrix.md
next_research_recommendation.md
```

The final report must answer:

```text
Which tracks were technically implemented?
Which completed a scientific pilot?
Which passed scale/dev?
Which were protocol-infeasible, scientifically negative, or budget-not-run?
Did any SB-specific component beat its direct/non-SB ablation?
What is the strongest defensible paper claim?
What evidence remains limited by missing raw artifacts or small sample size?
```

After validation:

```text
mark campaign terminal
update ACTIVE_RESEARCH_CAMPAIGN.json or terminal status artifact
record total spend
verify no GPU processes remain
stop the Pod
provide one final user-facing summary
```

---

# 11. Track-specific claim requirements

## T1 learned gate + raw bridge

Strong edit-intent claim:

```text
learned gate fixes rule-based-gate same-subject leakage while preserving useful raw-controller efficacy.
```

Bridge-specific claim additionally requires learned-gated MC bridge to beat learned-gated no-rollout and be competitive with learned-gated myopic at matched compute/KL.

## T2 activation-space SB

Activation-SB claim requires the SB transport to beat direct mean shift/linear mapping at matched intervention energy and to preserve identity negatives.

## T3 conditional answer-span CSBM

Categorical-CSBM claim requires bridge-state sampling to beat ordinary noising and bidirectional D-IMF to beat forward-only training.

## T4 unbalanced/partial CSBM

Partial-transport claim requires better locality/same-subject behavior than balanced CSBM at comparable efficacy, not merely an external gate effect.

## T5 parameter-space SB

Parameter-space claim requires the SB-generated adapter to beat direct conditional adapter generation or linear latent interpolation at matched storage/compute.

If these SB-specific comparisons fail, the method may still be reported as a useful editor, but not as evidence that Schrödinger bridges contributed.

---

# 12. Global failure taxonomy

Use exactly one primary failure category per failed stage:

```text
protocol_infeasible
implementation_failed
infrastructure_blocked
data_integrity_failed
offline_scientific_failed
actual_decode_failed
generalization_failed
compute_or_storage_failed
budget_not_run
```

Do not describe an unrun track as scientifically failed.

---

# 13. Initial autonomous command

Use the command in `START_SB_ALTERNATIVES_GOAL.md` under Codex Goal mode.
