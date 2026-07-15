# Direction 3 Autonomous Research Plan

Complete autonomous execution plan for `counterfact_direction3_controller_v1`.

The goal is to determine whether a deployable learned controller and edit-intent gate can amortize raw bridge behavior for factual editing in LLaDA while controlling same-subject side effects and reducing inference compute.

The plan is a bounded state machine. Codex must execute the stages in order, automatically advance on pass, use only the specified rescue on failure, and finish with either a positive final package or a formal negative stop checkpoint.

---

## 0. Current starting state

Completed before autonomous execution resumes:

```text
Direction 1 formally closed as blocked under tested rule-based runtime gates
Direction 2 bridge-adapter v1 formally closed as protocol-infeasible before adapter science
Direction 2 bin-2 exclusion audit accounted for all 284 pre-exclusion bin-2 rows
Direction 3 protocol/scaffold
controller_train_100 / controller_val_50 / dev_smoke_50 splits
real teacher cache train100/val50
teacher-cache audit
clean deployable scalar v2 controller
metric audit showing strong groupwise candidate ranking
frozen LLaDA deployable feature extraction
feature integrity/alignment/leakage checks
```

Direction 2 v1 status:

```text
closed_protocol_infeasible
adapter_training_run = false
actual_decode_run = false
analysis_500_used = false
final_test_used = false
do_not_resume_without_new_protocol_version = true
```

Active protocol:

```text
counterfact_direction3_controller_v1
```

Current next stage:

```text
Stage 1B.4A — feature-cache readiness and prompt-provenance audit
then Stage 1B.4B/C/D — representation-aware deployable v3 offline training
```

No `analysis_500` or final-test split has been used for Direction 3 tuning.

The autonomous campaign must not create Direction 2 v2, switch directions, or resume Direction 2 v1.

---

# Campaign Phase A — Bootstrap and state management

## A.1 Create autonomous campaign state

### Tasks

Validate the repository-root campaign marker first:

```text
ACTIVE_RESEARCH_CAMPAIGN.json
```

Required values:

```text
active_protocol = counterfact_direction3_controller_v1
active_direction = direction3
active_plan = DIRECTION3_AUTONOMOUS_RESEARCH_PLAN.md
campaign_status = active
analysis_500_locked = true
final_test_locked = true
```

Then create or resume:

```text
runs/counterfact_direction3_controller_v1/autonomous_campaign_v1/
  campaign_state.json
  budget_state.json
  stage_history.csv
  autonomous_log.md
```

Record:

```text
current git commit
RunPod pod ID
GPU type
hourly rate
budget
completed historical stages
Direction 2 v1 closed status and audit fingerprint
current stage = stage_1b4a_feature_cache_readiness_audit
```

### Acceptance

```text
autonomous_mode = true
budget variables parse as positive numbers
active campaign marker validates
campaign state files exist
no Direction 2 state is reused
analysis_500_used = false
final_test_used = false
```

### Failure

If budget variables or required RunPod configuration are missing, write `campaign_configuration_failed` and do not start paid work.

---

## A.2 Start and retain the RunPod Pod

### Tasks

```bash
runpodctl pod start "$RUNPOD_POD_ID"
runpodctl pod list
```

Connect, then verify:

```bash
cd /workspace/SB
git pull
python -m pytest tests -q
nvidia-smi
```

### Acceptance

```text
Pod running with GPU
SSH working
/workspace/SB accessible
tests pass
feature cache train100/val50 exists
```

### Lifecycle rule

After A.2 passes, do not stop the Pod until campaign completion, budget exhaustion, or formal failure completion.

---

# Campaign Phase B — Representation-aware v3 offline system

## Stage 1B.4A — Local/remote feature-cache readiness audit

### Objective

Confirm the extracted frozen representations are scientifically usable for controller/gate training.

### Tasks

Audit:

```text
state_features.safetensors
candidate_features.safetensors
edit_features.safetensors
gate_features.safetensors
feature_index.jsonl
feature_schema.json
```

Write:

```text
runs/counterfact_direction3_controller_v1/
  deployable_feature_cache_train100_val50_v1_local_audit/
    report_summary.json
    tensor_schema_audit.csv
    group_alignment_audit.csv
    prompt_provenance_audit.csv
    feature_distribution_summary.csv
    relation_distribution_summary.csv
    target_length_summary.csv
```

### Required checks

```text
100 train edits
50 validation edits
2,994 candidate groups
candidate width 8
train/val overlap 0
all tensors finite
all index-to-tensor mappings valid
all teacher-cache groups aligned
zero leaked runtime features
model/tokenizer fingerprints recorded
```

Prompt provenance:

```text
real rewrite coverage >= 95%
real paraphrase coverage >= 95%
real near/far coverage >= 95%
same-subject negative provenance reported
synthetic fallback explicitly tagged
```

### Acceptance

```text
feature_integrity_pass = true
feature_alignment_pass = true
prompt_provenance_pass = true
runtime_feature_leakage_pass = true
```

### Failure/rescue

One repair attempt is allowed for schema/index/provenance bugs. No model training until pass.

---

## Stage 1B.4B — Train representation-aware value controller

### Objective

Learn candidate-level bridge advantage from deployable frozen state/candidate/edit representations.

### Required variants

```text
d3_value_repr
d3_value_repr_no_target_indicator
d3_target_indicator_only
scalar_feature_v2 baseline
```

### Architecture

```text
state projection: source_dim -> 128 or 256
candidate projection: source_dim -> 128 or 256
edit projection: source_dim -> 128 or 256
MLP hidden dim: 256
layers: 2-3
dropout: 0.1
output: one score per candidate
```

Interaction features:

```text
state
candidate
edit
state*candidate
state*edit
candidate*edit
abs(state-candidate)
base logprob
candidate rank
step embedding
active-mask embedding
target position/length
target indicators where allowed
```

### Teacher target

Normalize within each candidate group:

```text
z-score or centered teacher scores
teacher_distribution = softmax(normalized_raw_bridge_score / temperature)
student_distribution = softmax(controller_score / temperature)
```

### Loss

```text
1.0 * groupwise KL distillation
0.5 * ranking loss
1.0 * negative identity/locality loss
0.01 * guidance magnitude penalty
0.1 * positive target-support loss
```

### Training grid

Bounded grid only:

```text
projection_dim in {128, 256}
hidden_dim = 256
teacher_temperature in {0.5, 1.0}
target_loss_weight in {0.0, 0.1}
seeds in {0, 1}
```

Do not expand this grid.

### Outputs

```text
offline_train_repr_value_gate_train100_val50_v3/
  value checkpoints
  configs
  train/val curves
  report_summary.json
```

### Acceptance for value component

On controller_val_50:

```text
macro groupwise Spearman >= 0.40
NDCG@8 >= 0.70
pairwise ranking accuracy >= 0.70
teacher top1 agreement >= 0.40
teacher top3 overlap >= 0.65
target top3 improvement over base >= 0.15
all losses finite
```

Representation-use evidence:

```text
full Spearman or NDCG >= target-indicator-only + 0.05
state shuffle reduces a primary value metric by >= 0.05
candidate shuffle reduces a primary value metric by >= 0.05
```

### Failure/rescue

One bounded value rescue is allowed:

```text
- choose the best existing projection/grid configuration,
- increase projection dimension to 256 if 128 was best attempted,
- increase negative identity weight from 1.0 to 2.0,
- do not add new feature families or teacher-derived inputs.
```

If value criteria still fail, finish Direction 3 as a negative result.

---

## Stage 1B.4C — Train representation-aware edit-intent gate

### Objective

Recognize whether a prompt asks for the edited relation, especially under same-subject different-relation negatives.

### Inputs

```text
prompt representation
rewrite/relation representation
subject representation
prompt*relation
abs(prompt-relation)
cosine similarity
prompt*subject
relation_id embedding
```

Forbidden:

```text
prompt_type
negative_type
teacher scores
outcome labels
split labels
```

### Architecture

```text
projection dim: 128
MLP: 256 -> 128 -> 1
dropout: 0.1
weighted BCE or focal loss
```

### Negative weights

```text
same_subject_different_relation = 3.0
near_locality = 2.0
far_locality = 2.0
generation = 1.5
attribute = 1.5
unrelated = 1.0
```

### Bounded grid

```text
loss in {weighted_bce, focal}
focal_gamma in {1.0, 2.0} when applicable
same_subject_weight in {3.0, 5.0}
projection_dim in {128, 256}
seeds in {0, 1}
```

Threshold selection uses controller_val_50 only.

### Gate acceptance

```text
ROC-AUC >= 0.85
PR-AUC reported
rewrite activation >= 0.90
paraphrase activation >= 0.85
same-subject activation <= 0.05
near-locality activation <= 0.02
far-locality activation <= 0.02
```

Representation-use evidence:

```text
relation representation shuffle reduces gate AUC by >= 0.05
full gate AUC >= lexical/scalar gate AUC + 0.05, when baseline available
```

### Failure/rescue

One gate-only rescue is allowed:

```text
- select best architecture,
- calibrate temperature/isotonic mapping on controller_val_50,
- reweight hard negatives within the predefined grid,
- do not alter value-controller architecture,
- do not lower acceptance thresholds.
```

If gate criteria still fail, finish Direction 3 as a negative result.

---

## Stage 1B.4D — Combine value and gate

### Objective

Evaluate deployable soft-gated guidance without LLaDA decoding.

```text
combined_advantage = gate_probability * value_advantage
```

Evaluate offline guidance scales:

```text
gamma in {0.5, 1.0, 2.0}
```

Train separately first; optional joint calibration is limited to one short epoch at low learning rate after both components individually pass.

### Required variants

```text
d3_value_repr
d3_gate_repr
d3_value_gate_repr
d3_value_repr_no_target_indicator
d3_target_indicator_only
```

### Negative guidance metric

```text
negative_guidance_ratio =
  mean(abs(gate * advantage) on negatives)
  / mean(abs(gate * advantage) on positives)
```

### Combined acceptance

```text
negative_guidance_ratio <= 0.15
same_subject_target_advantage_vs_base <= 0
all value criteria pass
all gate criteria pass
```

---

## Stage 1B.5 — Offline replay, leakage, and shortcut audits

### Outputs

```text
offline_replay_repr_train100_val50_v3/
stage1b_feature_leakage_audit_v3/
representation_shortcut_audit_v3/
```

Required reports:

```text
groupwise_ranking_metrics.csv
gate_threshold_sweep.csv
negative_guidance_diagnostics.csv
per_prompt_type_metrics.csv
per_step_metrics.csv
per_target_length_metrics.csv
per_relation_metrics.csv
representation_ablation.csv
paired_bootstrap.csv
scientific_status.json
```

### Hard pass

All must hold:

```text
zero leaked runtime features
train/val overlap 0
value metrics pass
gate metrics pass
negative_guidance_ratio <= 0.15
same_subject_target_advantage_vs_base <= 0
full model meaningfully beats target-indicator-only
state/relation shuffle tests show representation use
analysis_500_used = false
final_test_used = false
llada_loaded = false during training/replay
```

### Decision

- Pass: automatically advance to Stage 2A.
- Fail after bounded rescue: write Direction 3 stop checkpoint and finish negatively.

---

# Campaign Phase C — Actual decoding smoke and confirmation

## Stage 2A.0 — Runtime integration implementation

### Tasks

Extend runtime evaluation to support:

```text
d3_value_repr
d3_value_gate_soft
d3_value_gate_hard_diagnostic
```

At runtime, construct exactly the same deployable feature schema used offline.

Add fake-model integration tests for:

```text
feature alignment
candidate group width
controller inference
gate inference
soft guidance application
no teacher-field access
compute accounting
```

### Acceptance

```text
all tests pass
runtime feature schema fingerprint matches training schema
leakage audit passes
no analysis/final access
```

---

## Stage 2A — Actual decode smoke20

### Split

Create deterministic:

```text
dev_smoke_20 = first stratified 20 of dev_smoke_50
confirmation_30 = remaining 30
```

Smoke20 may be used for bounded integration calibration. Confirmation30 must remain untouched until configuration is fixed.

### Methods

```text
base
target_logit_bias diagnostic
myopic_score
no_rollout_bridge
mc_bridge
d3_value_repr
d3_value_gate_soft_gamma0.5
d3_value_gate_soft_gamma1.0
d3_value_gate_soft_gamma2.0
```

### Metrics

```text
rewrite exact
declarative paraphrase exact
target-token F1
old-target suppression
near/far locality
same-subject TFPR
generation TFPR
malformed rate
sparse guidance KL
model evals/edit
GPU minutes/edit
```

### Green pass

```text
D3 rewrite >= base + 0.15
D3 paraphrase >= base + 0.10
same-subject TFPR <= base + 0.03
malformed <= 0.05
D3 model evals <= 0.60 * MC bridge
D3 GPU time <= 0.60 * MC bridge
```

### Yellow pass

```text
rewrite/paraphrase improve over base
same-subject TFPR <= 0.10
malformed <= 0.05
D3 cheaper than MC bridge
```

One bounded calibration is allowed on smoke20:

```text
gamma in {0.25, 0.5, 1.0, 1.5, 2.0}
gate temperature in {0.5, 1.0, 2.0}
```

No architecture retraining.

### Red fail

```text
no rewrite gain
same-subject TFPR > 0.20
malformed > 0.05
not cheaper than MC bridge
runtime/offline feature mismatch
```

Red fail ends Direction 3 negatively.

---

## Stage 2B — Confirmation30

Run the fixed smoke20-selected configuration once on the untouched confirmation30 subset.

### Acceptance

```text
rewrite/paraphrase direction matches smoke20
same-subject TFPR <= base + 0.03
malformed <= 0.05
compute advantage remains >= 30%
paired bootstrap trends favor D3 over base
```

If confirmation30 fails, finish Direction 3 negatively. Do not tune on confirmation30.

---

# Campaign Phase D — Scale controller training data

## Stage 3A — Expand to controller_train_1000 / controller_val_200

### Objective

Scale Direction 3 while preserving the legally scarce multi-token examples already assigned to its seed train/validation splits.

### Tasks

Build deterministic expansions from allowed CounterFact train rows:

```text
controller_train_1000 must be a superset of controller_train_100
controller_val_200 must be a superset of controller_val_50
dev_smoke_50 remains held out
```

Exclusion policy:

```text
exclude Direction 1 dev/analysis/ablation/final case IDs
exclude Direction 3 dev_smoke_50 and all locked analysis/final cases
do not exclude Direction 3 controller_train_100 or controller_val_50 from their expanded parent splits
use source_split + source_index/fingerprint namespaces
```

Stratify new additions by:

```text
relation_id
target length
subject ambiguity
availability of real prompt categories
```

Target-length scarcity policy:

```text
preserve every existing legal bin-2 example in train100/val50
report bin-2 counts and uncertainty explicitly
do not fail merely because no additional unseen bin-2 rows remain
do not move dev_smoke_50 bin-2 cases into training
do not silently substitute bin 3 for bin 2
```

### Acceptance

```text
1000 train edits
200 val edits
controller_train_100 subset preserved exactly
controller_val_50 subset preserved exactly
dev_smoke_50 overlap = 0
zero overlap with locked Direction 1/analysis/final splits
bins 1,2,3 represented by preserving legal seed records
relation histograms written
same-subject negatives available for >= 80% edits
bin-2 scarcity and per-bin effective sample size reported
```

Budget-adaptive fallback:

If 1000/200 teacher generation is projected to exceed remaining budget, use the largest predeclared fallback:

```text
controller_train_500 / controller_val_100
```

The fallback must still preserve the existing train100/val50 seed records. Record that this limits claim scale. Do not invent smaller sizes.

---

## Stage 3B — Sharded teacher-cache generation

Generate in resumable shards:

```text
100 train edits per shard
50 val edits per shard
```

Settings:

```text
top_k = 8
steps = 4
mc_rollouts = 2
methods = base,myopic_score,no_rollout_bridge,mc_bridge
```

Merge only after every shard validates.

### Acceptance

```text
all expected edits present
>=3 observed steps
active-mask count >1 present
target bins 1 and 2 present
all top-k arrays valid
all scores finite and nonconstant
same-subject/locality negatives present
no locked split use
```

One retry per failed shard is allowed.

---

## Stage 3C — Scaled deployable feature extraction

Extract the same frozen feature schema and verify exact schema fingerprint compatibility with v3.

### Acceptance

```text
all scaled candidate groups represented
all tensors finite
zero runtime feature leakage
model/tokenizer/schema fingerprints recorded
```

---

## Stage 3D — Train scaled D3 v4

Train the best v3 architecture on the scaled cache.

Bounded hyperparameter grid:

```text
best v3 config
one lower-regularization config
one higher-locality config
seeds {0,1,2}
```

No architecture expansion.

### Acceptance

Stronger offline criteria:

```text
groupwise Spearman >= 0.45
NDCG@8 >= 0.72
gate AUC >= 0.90
paraphrase activation >= 0.85
same-subject activation <= 0.05
negative guidance ratio <= 0.12
representation-use ablations still pass
```

If scaled training fails to match the 100/50 pilot qualitatively, finish negatively.

---

# Campaign Phase E — dev_tune_200 method selection

## Stage 4A — Runtime preflight on 20 dev-tune cases

Run the scaled controller on 20 deterministic dev_tune cases only to catch integration/schema errors. No selection decisions from this subset.

Acceptance:

```text
runtime works
no schema mismatch
no leakage
outputs complete
```

---

## Stage 4B — Staged dev_tune_200 sweep

Avoid a full combinatorial grid.

### Sweep 1: gate/gamma

```text
gate mode = soft learned, hard diagnostic
gamma in {0.5,1.0,2.0}
```

### Sweep 2: candidate support

For top two configurations:

```text
top_k in {4,8,16}
```

### Sweep 3: diffusion schedule

For top two remaining:

```text
steps in {4,8}
schedule in {final_only,late,all}
```

### Sweep 4: span policy

For top two remaining:

```text
exact
slack_1
```

### Required baselines

```text
base
target_logit_bias
prompt_memory
myopic_score
no_rollout_bridge
mc_bridge
best historical raw bridge configuration
D3 candidates
```

### Constraints

```text
same-subject TFPR <= base + 0.03
near/far TFPR budgets pass
malformed <= 0.05
GPU minutes/edit <= 2.0
compute advantage over MC bridge reported
```

### Selection

Select at most three frozen Pareto candidates:

```text
A = best stress-aware aggregate
B = best safety/locality
C = best compute efficiency
```

### Pass

At least one D3 candidate must:

```text
clearly beat base on rewrite/paraphrase
pass same-subject stress
be cheaper than MC bridge
be competitive with myopic/no-rollout on stress-aware aggregate
not collapse on target length 2
```

If none passes, finish Direction 3 negatively.

---

## Stage 4C — Preplanned frozen ablations

Use `ablation_500` only after candidates are selected and frozen. Results cannot change selection.

Ablations:

```text
value without gate
gate without value
no target indicator
target-indicator-only
final-only vs all-step
state-shuffle/relation-shuffle offline evidence
```

If budget is insufficient, run ablations on the predeclared 200-case subset and label coverage clearly.

---

# Campaign Phase F — Lock, analysis, and final test

## Stage 5 — Dev method lock

Write:

```text
runs/counterfact_direction3_controller_v1/dev_method_lock.json
```

Include:

```text
controller/gate checkpoint hashes
feature schema fingerprint
teacher cache version
training/validation split hashes
method/gate type
gate threshold or calibration
gamma
top_k
steps
schedule
span policy
normalization
metrics
budgets
random seeds
selected Pareto candidates
report-script commit
```

Acceptance:

```text
all dev criteria passed
no further tuning planned
lock validates
```

Then automatically set `DEV_METHOD_LOCKED=1`.

---

## Stage 6 — analysis_500 locked confirmation

Run the frozen candidate(s) and required baselines once.

No parameter or threshold change after inspection.

### Analysis pass

```text
D3 retains >=80% of dev rewrite
D3 retains >=80% of dev paraphrase
same-subject TFPR remains within budget
near/far and malformed budgets pass
compute advantage over MC bridge remains
paired CIs preserve the main qualitative claim
```

If analysis fails:

```text
mark counterfact_direction3_controller_v1 failed
write stop checkpoint
do not tune on analysis
finish campaign negatively
```

If analysis passes, write:

```text
analysis_confirmation_lock.json
```

and set `FINAL_METHOD_LOCKED=1`.

---

## Stage 7 — final_test_500

Run one locked final configuration plus required baselines.

Required final methods:

```text
base
target_logit_bias
prompt_memory
myopic_score
no_rollout_bridge
mc_bridge
selected D3 method
key D3 ablation if precommitted
```

Required metrics:

```text
rewrite
paraphrase
locality/self-normalized locality
same-subject TFPR
generation TFPR
malformed
sparse guidance KL
model evals/edit
GPU minutes/edit
target-length breakdown
paired bootstrap by edit_id
```

No rerun for tuning. A rerun is allowed only for a documented infrastructure failure before results were inspected.

---

# Campaign Phase G — Final reporting and shutdown

## Stage 8 — Final research package

Create:

```text
runs/counterfact_direction3_controller_v1/final_research_package_v1/
```

Artifacts:

```text
main_results_table.csv
same_subject_stress_table.csv
target_length_table.csv
compute_table.csv
paired_bootstrap.csv
pareto_plot.png
same_subject_plot.png
target_length_plot.png
failure_cases.csv
reproducibility_manifest.json
final_research_report.md
paper_claim_recommendation.md
report_summary.json
```

### Claim decision

#### Strong method claim

```text
D3 beats strong runtime baselines on stress-aware aggregate,
controls same-subject leakage,
is cheaper than MC bridge,
and survives analysis/final.
```

#### Efficiency claim

```text
D3 matches MC bridge within confidence intervals
with substantially lower compute.
```

#### Safety/evaluation claim

```text
D3 is not the best overall,
but learned edit-intent control improves safety and exposes a key failure mode.
```

#### Negative result

```text
D3 fails the bounded offline/actual/generalization pipeline.
```

After the final package validates, mark campaign complete and stop the Pod.

---

# Budget and failure completion

At any stage, if remaining budget cannot cover the projected stage plus reserve:

1. do not start the stage;
2. write `direction3_stop_checkpoint_v1`;
3. summarize completed evidence and missing stages;
4. classify the strongest current claim;
5. stop the Pod.

At any bounded scientific failure with no rescue remaining, follow the same negative completion process.

The campaign must not automatically switch to Direction 2 v2, a CSBM-lite branch, or another protocol after a Direction 3 stop. It must finish and report the active Direction 3 campaign first.

---

# Initial autonomous Codex Goal-mode command

Use Goal mode because the outcome, constraints, state machine, and verification criteria are already fully specified.

```text
/goal

Read AGENTS.md, ACTIVE_RESEARCH_CAMPAIGN.json, and
DIRECTION3_AUTONOMOUS_RESEARCH_PLAN.md.

Resume the active Direction 3 autonomous campaign from the current repository
state. Execute all remaining planned stages without asking for per-stage,
per-command, or per-GPU-job approval.

Start the configured existing RunPod Pod if it is stopped. Keep it running
between all planned CPU and GPU stages until one terminal campaign state is
reached: positive completion, formal scientific negative completion, budget
completion, unrecoverable infrastructure block, or unsafe data-integrity stop.

Maintain campaign_state.json, budget_state.json, stage_history.csv, and
autonomous_log.md. Obey all split locks, feature-leakage rules, bounded rescues,
budget guards, and acceptance criteria. Do not lower thresholds, invent
experiments, resume Direction 2 v1, create Direction 2 v2, tune on analysis_500,
or rerun final_test_500 for tuning.

The goal is complete only after the corresponding final or stop package is
validated, durable summaries are preserved, campaign state is terminal, and
the Pod is stopped.
```
