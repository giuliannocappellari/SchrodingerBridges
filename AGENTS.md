# AGENTS.md

Operational and scientific rules for Codex in the LLaDA / masked-diffusion model-editing repository.

The active research campaign asks:

> **Can a diffusion-native parametric editor preserve locality by optimizing updates over causal internal representations and partial denoising states?**

The campaign is autonomous. Once Goal mode is started with the required environment variables, Codex must execute the complete bounded plan without requesting per-stage, per-command, or per-GPU-job approval.

---

## 0. Active project identity

```text
active_protocol_version = diffusion_native_causal_partial_state_editor_v1
active_plan = DIFFUSION_NATIVE_PARAMETRIC_EDITOR_AUTONOMOUS_PLAN.md
base_model_primary = GSAI-ML/LLaDA-8B-Instruct
base_model_secondary = Dream-v0-Instruct-7B
base_model_weights = frozen except explicit low-rank/closed-form edited modules
trained_or_solved_parameters = edit-specific low-rank updates, target values, optional state-conditioned residual basis
edit_access = edit request available at edit time
training_access = fresh development and train-only preservation anchors
analysis_access = locked confirmation only
final_access = one locked final evaluation only
```

Historical campaigns and their artifacts are immutable evidence:

```text
counterfact_direction1_v1
counterfact_direction2_bridge_adapter_v1
counterfact_direction3_controller_v1
counterfact_sb_alternatives_campaign_v1
masked_diffusion_memit_sb_positive_result_v1
mask_pattern_sb_publication_confirmation_v1
```

Codex may read historical code and compact summaries, but must not overwrite, delete, silently reinterpret, or resume a closed historical protocol.

---

## 1. Research question and primary method

The primary hypothesis is that locality can be improved by combining three diffusion-native ideas:

```text
1. Temporal/causal localization:
   identify internal coordinates that causally mediate the factual object across denoising states.

2. Partial-state edit optimization:
   optimize the target value/update over fully masked and partially revealed answer states.

3. Preservation-subspace constraints:
   project or solve the update in directions that minimally affect same-subject, locality, and unrelated keys.
```

The main candidate is called:

```text
causal_partial_state_nullspace_memit
```

It is a locate-then-edit parametric editor. The primary form writes a permanent low-rank update into selected MLP down-projection matrices. A state-conditioned low-rank residual variant is allowed only as a predeclared bounded rescue and must be reported separately as inference-conditioned editing.

The intended update for layer `l` is conceptually:

```text
positive key bank K_plus:
  causal subject-site keys collected across rewrite prompts and partial denoising states

target value bank V_star:
  values optimized to support target_new across the same states

preservation key bank K_minus:
  train-only same-subject-different-relation, near/far locality, attribute,
  generation, and unrelated anchors

null-space projector N:
  N = I - U U^T, where U spans the protected key subspace

constrained update:
  Delta_W = argmin_D ||(W + D N) K_plus - V_star||^2
                    + lambda_update ||D N||_F^2
                    + lambda_identity ||D N K_minus||^2
```

Exact implementation details must follow the master plan and be validated dimensionally and numerically.

---

## 2. Authoritative files and read order

Codex must read these files in order before acting:

```text
1. AGENTS.md
2. ACTIVE_RESEARCH_CAMPAIGN.json
3. EXPERIMENT_PROTOCOL_REGISTRY.json
4. DIFFUSION_NATIVE_PARAMETRIC_EDITOR_AUTONOMOUS_PLAN.md
5. the relevant detailed stage plan
6. persisted campaign state under runs/diffusion_native_causal_partial_state_editor_v1/
```

Detailed plans:

```text
CAUSAL_LOCALIZATION_PLAN.md
PARTIAL_STATE_TARGET_OPTIMIZATION_PLAN.md
NULL_SPACE_LOCALITY_PLAN.md
MAIN_EDITOR_AND_BASELINES_PLAN.md
LOCKED_CONFIRMATION_PLAN.md
SECOND_BACKBONE_AND_SCALING_PLAN.md
PAPER_REPRODUCIBILITY_PLAN.md
```

The persisted campaign state is authoritative for completed stages. Root control files determine which campaign is active but must not cause already validated work to be rerun unnecessarily.

---

## 3. Autonomous mode

Autonomous execution is enabled only when:

```bash
export DNPE_AUTONOMOUS_MODE=1
export DNPE_MAX_INFRA_RETRIES="3"
export DNPE_MAX_SCIENTIFIC_RESCUES_PER_STAGE="1"
```

Required RunPod variables:

```bash
export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

`runpodctl` must already be configured with a RunPod API key.

When `DNPE_AUTONOMOUS_MODE=1`, Codex has one-time approval to execute every task explicitly listed in the plan without asking the user between stages.

Codex must not:

```text
- expand the scientific scope beyond the plan;
- lower acceptance thresholds after seeing results;
- invent unplanned rescues;
- add evaluation prompts to training;
- use teacher-only or outcome-only features as runtime inputs;
- open locked analysis/final data before the required lock;
- switch to a different research direction;
- create or delete a Pod;
- ask for monetary budget approval or stop because of estimated cost.
```

---

## 4. Monetary policy

There is no monetary budget guard in this campaign.

Codex may record:

```text
Pod running time
estimated GPU spend
per-stage runtime
model evaluations
energy/compute proxies
```

These are informational only. Monetary cost must never:

```text
block a planned stage
trigger a scientific stop
cause a track to be skipped
cause the Pod to be stopped
cause Codex to ask for a budget increase
```

Scientific compute-efficiency thresholds remain valid experimental criteria. They are not monetary guards.

---

## 5. Pod lifecycle

### Start

At campaign start, Codex must start the configured existing Pod if stopped:

```bash
runpodctl pod start "$RUNPOD_POD_ID"
runpodctl pod list
```

Verify:

```text
Pod status is RUNNING
at least one GPU is allocated
SSH works
nvidia-smi works
/workspace is mounted
/workspace/SB exists or can be cloned
```

If the host or port changes, Codex must refresh connection details through configured RunPod tooling/API. It must not guess.

### Keep running

After successful startup, keep the Pod running through:

```text
source and artifact audits
CPU-only implementation and tests
GPU causal tracing
feature/statistics collection
edit generation
controller/update solving
actual decoding
bootstrap analysis
second-backbone work
locked analysis/final evaluation
final reporting
```

Do not stop the Pod because:

```text
one stage finished
one method failed
one method passed
no GPU process is active temporarily
the next stage is CPU-only
estimated spend is high
```

### Stop

Stop the Pod only after one of these terminal conditions:

```text
1. Positive completion:
   the full final research package validates.

2. Formal bounded scientific negative completion:
   every required stage and permitted rescue is complete,
   the terminal negative package validates, and no further planned stage remains.

3. Unrecoverable infrastructure failure:
   Pod/GPU/SSH/storage remains unusable after the allowed retries and a complete
   infrastructure checkpoint has been written.

4. Unsafe data-integrity failure:
   split leakage, irrecoverable corruption, or missing authoritative artifacts makes
   continuation scientifically invalid, and a terminal checkpoint has been validated.
```

Stop command:

```bash
runpodctl pod stop "$RUNPOD_POD_ID"
```

Never terminate/delete the Pod unless the user explicitly requests deletion.

---

## 6. Local and RunPod Python environments

### Local MacBook

Use `uv`:

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script_name>.py
```

Do not use `pip install` directly in the local project environment.

### RunPod

Use the Python environment available in the GPU image:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script_name>.py
```

Do not require `uv` on RunPod. Record Python, CUDA, PyTorch, Transformers, PEFT, and bitsandbytes versions in every GPU run summary.

The primary closed-form parametric edit must operate on floating-point editable MLP weights. Do not apply the update directly to 4-bit quantized matrices. Quantization may be used only for diagnostics or explicitly labelled inference-only comparisons.

---

## 7. Git, storage, and long-job rules

Code moves through Git. Large artifacts remain under `/workspace/SB/runs` or durable storage.

Before every stage:

```bash
cd /workspace/SB
git status
git pull
python -m pytest tests -q
```

After code changes:

```bash
git diff
git status
python -m pytest tests -q
```

Commit only after tests pass. Do not commit secrets, model weights, large tensors, or large run directories.

Long jobs must use `tmux`, `set -o pipefail`, logs, and explicit exit-code files:

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage_name>.log; \
   code=${PIPESTATUS[0]}; echo "$code" > logs/<stage_name>.exitcode; exit "$code"'
```

Every stage must write:

```text
report_summary.json
run_config.json or equivalent
validation report
log path
exit-code path for long jobs
artifact hashes or manifest
```

Never overwrite a completed run by default. Use versioned directories.

---

## 8. Campaign state machine

Campaign directory:

```text
runs/diffusion_native_causal_partial_state_editor_v1/autonomous_campaign_v1/
```

Required files:

```text
campaign_state.json
stage_history.csv
autonomous_log.md
cost_state.json  # informational only
```

For each stage:

```text
1. Read active state and authoritative plan.
2. Run preflight checks and tests.
3. Execute the stage.
4. Validate every hard acceptance criterion.
5. Write versioned artifacts, logs, and exit code.
6. Update campaign state and track status.
7. Advance automatically on pass.
8. Apply only the bounded rescue specified for that stage on failure.
9. If no rescue remains, write the formal terminal package.
```

No unplanned hyperparameter expansion is allowed.

---

## 9. Data and split discipline

Primary datasets:

```text
CounterFact:
  single-token and standard factual-editing evaluation

KAMEL-compatible multi-token set:
  controlled target lengths 2, 3, and 4; optional 5 if available
```

Fresh protocol manifests must exclude every case/prompt fingerprint used for method development in historical campaigns when practical. Historical locked manifests may be read only for ID/fingerprint exclusion until the new method lock permits evaluation.

Canonical new split roles:

```text
dnpe_smoke_20:
  integration only; bounded calibration allowed

dnpe_pilot_100:
  architecture pilot and bounded rescue

dnpe_dev_200:
  final shared-hyperparameter and method selection

dnpe_kamel_dev_<N>:
  multi-token method selection for target length N

dnpe_kamel_locked_<N>:
  locked confirmation for target length N

analysis_500:
  locked proceed/stop confirmation only

final_test_500:
  one locked final evaluation only
```

The old `analysis_500`, `final_test_500`, and `final_test_full` remain locked until the new protocol writes and validates the required lock files.

No prompt may be both adapter/update training data and evaluation-only data for the same edit.

Allowed edit-time optimization data:

```text
rewrite prompt and target_new
edit tuple
training-only prompt augmentations from unused rows
training-only same-subject different-relation anchors
training-only locality/unrelated anchors
partial-mask states derived from the allowed rewrite/augmentation prompts
base activations and logits from those allowed states
```

Evaluation-only data:

```text
official paraphrases
QA generalization
held-out same-subject stress
near/far locality evaluation prompts
generation/attribute evaluation prompts
locked analysis/final prompts
```

Hard rule:

```text
train_prompt_ids intersect eval_prompt_ids = empty
```

The rewrite prompt may be reported as `train_seen=true`; all paraphrase/locality/stress metrics must be `train_seen=false`.

---

## 10. Causal localization rules

Causal tracing must use only information available before applying the new edit.

Primary measures:

```text
normalized indirect effect (AIE)
temporal indirect effect (TIE)
old-target probability recovery
site stability across partial mask states
site stability across paraphrases
edit-effect per update norm
```

Required coordinates:

```text
layer
module type: MLP, attention, full hidden state
token position: first subject, last subject, relation cue, first answer mask
partial-mask count / denoising state
```

The campaign must compare:

```text
fixed early-mid last-subject site
per-edit top causal site
stable temporal site aggregated across states
random layer/position controls
late answer-position control
```

Causal tracing may guide site selection, but the final method must validate that the selected site improves editing/locality over random or fixed-site baselines. Localization alone is not evidence of editing value.

---

## 11. Partial-state optimization rules

For a target of length `N`, include all mask counts:

```text
k = 0, 1, ..., N-1 revealed target positions
```

At each optimization step, sample or cycle which `k` target positions are revealed. Compute target loss only on still-masked positions unless a predeclared joint-span objective is used.

Required state policies:

```text
fully_masked_only
all_mask_counts_random_positions
confidence_trajectory_states
uniform_mask_count_states
three_bucket_states: full, intermediate, late
```

The primary partial-state comparison is:

```text
all_mask_counts_random_positions
vs
fully_masked_only
```

No method may claim diffusion-specific value unless partial-state or state-conditioned optimization beats a matched full-mask-only/step-agnostic baseline.

---

## 12. Locality and preservation rules

The main locality mechanism is a protected-key/null-space constraint.

Protection data must be training-only and disjoint from evaluation prompts.

Protection categories:

```text
same subject, different relation
different subject, same relation
near locality
far locality
attribute
generation
random unrelated
```

Required comparisons:

```text
ordinary MDM-MEMIT
partial-state MDM-MEMIT
MDM-MEMIT + target-value KL anchors
AlphaEdit-style null-space projection
causal multi-state editor without null space
causal partial-state null-space editor
TimeROME-DLM-style temporal residual memory baseline
```

Null-space rank/energy must be reported. A projection that removes every useful edit direction is not acceptable.

Main locality metrics:

```text
same-subject target false-positive rate
near/far target false-positive rate
pre/post output agreement
distributional KL/JS to base
identity-key output drift
update norm and rank
general utility diagnostics
```

---

## 13. Runtime deployability and leakage

The primary permanent editor may use base activations, keys, values, and covariance statistics during edit construction. Evaluation must use only the resulting edited weights and the frozen decoding configuration.

The optional state-conditioned residual editor may use at inference:

```text
current hidden state
mask ratio / active mask count
step index
edit-specific low-rank residual parameters
```

Forbidden runtime shortcuts:

```text
evaluation bucket identity
prompt_type
negative_type
case ID as learned feature
final success labels
future decoded outcome
analysis/final membership
```

Teacher-only scores may be optimization labels but not runtime inputs.

Every checkpoint/update must include:

```text
runtime feature schema
edited layer/position policy
update hashes
parameter count/rank
feature-leakage audit
```

---

## 14. Required baselines

At minimum:

```text
base
prompt_memory
target_logit_bias
ordinary MDM-MEMIT
partial-state MDM-MEMIT
AlphaEdit-style null-space MDM-MEMIT
TimeROME-DLM-style temporal residual memory
random-site partial-state editor
fixed-site partial-state editor
causal-site full-mask editor
causal-site partial-state editor
causal partial-state null-space editor
state-conditioned residual rescue, only if triggered
```

Where feasible, include:

```text
per-edit LoRA rank 4
ROME-style single-layer update
MEMIT batch edit
```

The main claim requires the proposed editor to beat the strongest direct/null-space baseline, not merely the unedited model.

---

## 15. Hard scientific acceptance criteria

### Baseline reproduction

```text
MDM-MEMIT rewrite exact >= 0.75
MDM-MEMIT paraphrase exact >= 0.40
pre-edit target_new rewrite exact <= 0.10
```

### Causal localization

```text
causal-site normalized effect exceeds random-site mean by >= 0.15
site stability across partial states is reported
selected site improves edit/locality aggregate or reduces update norm at matched efficacy
```

### Main editor efficacy

On dev/confirmation:

```text
rewrite exact >= 0.75
paraphrase exact >= 0.40
malformed rate <= 0.05
old-target suppression improves over base
```

### Locality

At least one main candidate must satisfy:

```text
same-subject TFPR <= base + 0.03
near-locality TFPR <= base + 0.03
far-locality TFPR <= base + 0.03
```

and relative to the strongest efficacy-matched baseline:

```text
same-subject TFPR reduced >= 50%
or distributional locality KL reduced >= 25%
```

with:

```text
rewrite loss <= 0.05 absolute
paraphrase loss <= 0.05 absolute
```

### Diffusion-native value

At least one:

```text
partial-state editor improves full-span exact by >= 0.10 on at least two
multi-token lengths

or

state-conditioned editor beats step-agnostic editor by >= 0.05 on the
stress-aware aggregate with positive paired CI
```

### Causal-mechanism value

At least one:

```text
causal-site editor beats random-site editor by >= 0.05 aggregate

or

matches efficacy within 0.02 using >= 25% lower update norm / fewer edited layers
```

### Strong-method claim

Requires all:

```text
passes efficacy floors
passes same-subject and near/far locality constraints
beats ordinary partial-state MDM-MEMIT on stress-aware aggregate
beats AlphaEdit-style projection or TimeROME-DLM-style baseline on at least one
primary locality/efficacy axis without losing the others
shows diffusion-state-specific value
survives locked analysis and final evaluation
```

---

## 16. Bounded rescues

Only these rescues are allowed:

```text
1. Causal-site rescue:
   switch between global fixed layer window, per-edit top TIE site, and stable
   temporal site set. No new localization family.

2. Locality rescue:
   bounded projector-rank / ridge grid specified in the plan.

3. Partial-state rescue:
   replace one shared update with a three-bucket state-conditioned low-rank
   residual using the same sites/features.

4. Dream integration repair:
   one model-specific compatibility repair.
```

No rescue may:

```text
lower hard thresholds
use evaluation prompts in optimization
add outcome labels as features
change metrics after analysis
expand grids beyond the plan
```

If all permitted rescues fail, produce a formal negative result.

---

## 17. Analysis and final locks

Before `analysis_500`, write and validate:

```text
runs/diffusion_native_causal_partial_state_editor_v1/dev_method_lock.json
```

It must freeze:

```text
model/checkpoint
edited layers and positions
causal-site policy
partial-state policy
null-space construction
projector rank/ridge
update rank
all hyperparameters
sampling configuration
metrics
random seeds
report scripts
selected candidate
```

Only then set:

```bash
export DEV_METHOD_LOCKED=1
```

`analysis_500` is proceed/stop only. After seeing it, do not alter the method.

If analysis passes, write:

```text
analysis_confirmation_lock.json
```

and set:

```bash
export FINAL_METHOD_LOCKED=1
```

Run `final_test_500` exactly once. Rerun only for a documented infrastructure failure before results were inspected.

---

## 18. Final package and claim classes

Final directory:

```text
runs/diffusion_native_causal_partial_state_editor_v1/final_research_package_v1/
```

Required files:

```text
report_summary.json
main_results_table.csv
same_subject_stress_table.csv
multi_token_table.csv
causal_localization_table.csv
locality_distribution_table.csv
compute_storage_table.csv
sequential_edit_table.csv
paired_bootstrap.csv
rewrite_locality_pareto.png
causal_heatmap.png
partial_state_plot.png
failure_cases.csv
artifact_availability_manifest.json
reproducibility_manifest.json
final_research_report.md
paper_claim_recommendation.md
```

Claim classifications:

```text
strong_diffusion_native_parametric_editor
locality_preservation_improvement
partial_state_editing_improvement
causal_localization_result
reproduction_only
bounded_negative_result
infrastructure_blocked
```

The claim must follow the evidence.

---

## 19. Codex completion behavior

In Goal mode, Codex must execute the complete campaign and return only after:

```text
the final positive or negative package validates
campaign_state.json is terminal
all required artifact hashes are recorded
historical analysis/final use is correctly reported
all tests pass or failures are documented
the Pod is stopped
```

Codex must not ask the user for intermediate approval.
