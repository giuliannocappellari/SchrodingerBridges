# AGENTS.md

Operational and scientific rules for Codex in the LLaDA masked-diffusion model-editing repository.

The active autonomous research program is:

```text
protocol_version = partial_state_temporal_residual_editor_v1
research_question = Can a temporally localized residual editor, optimized across partial denoising states and protected by state-conditioned locality constraints, improve factual-editing locality at matched efficacy?
primary_model = GSAI-ML/LLaDA-8B-Instruct
source_reproduction_model = GSAI-ML/LLaDA-8B-Base or the exact TimeROME-DLM source checkpoint when available
base_weights = frozen
trained_or_fitted_objects = temporal residual memories, calibration parameters, optional state-conditioned gates/projectors
```

All earlier protocols and campaigns are immutable historical evidence. Do not resume, overwrite, delete, or silently reinterpret them.

---

## 1. Authoritative files and reading order

Before acting, Codex must read:

1. `AGENTS.md`
2. `ACTIVE_RESEARCH_CAMPAIGN.json`
3. `PARTIAL_STATE_TEMPORAL_RESIDUAL_EDITOR_AUTONOMOUS_PLAN.md`
4. `EXPERIMENT_PROTOCOL_REGISTRY.json`
5. every detailed plan referenced by the active stage
6. persisted campaign state under `runs/partial_state_temporal_residual_editor_v1/autonomous_campaign_v1/`

The persisted campaign state is authoritative for completed stages. Root control files determine the active protocol and must not reset validated work.

---

## 2. Autonomous Goal-mode authorization

Autonomous mode is enabled only when:

```bash
export PS_TRM_AUTONOMOUS_MODE=1
```

When enabled, Codex has one-time authorization to execute every task explicitly listed in the active plan without asking for per-stage, per-command, or per-GPU-job approval.

Codex must not:

- invent unplanned experiments;
- lower frozen thresholds after observing results;
- use evaluation prompts as training or protection anchors;
- use outcome labels, evaluation-bucket labels, or teacher-only fields as runtime inputs;
- reopen historical analysis/final splits;
- switch to another research direction;
- create a new protocol version automatically;
- delete existing run directories;
- terminate or delete a RunPod Pod.

---

## 3. Campaign completion and Pod lifecycle

The Pod must be started once and retained across all planned CPU and GPU stages.

Keep the Pod running through:

```text
source reproduction
implementation and testing
causal tracing
partial-state target construction
residual-memory fitting
state-conditioned protection
pilot and locked evaluations
multi-token evaluation
scaling and second-backbone work
statistics, plots, and final reporting
```

Do not stop the Pod because:

- one job finished;
- the next task is CPU-only;
- one method failed;
- one method passed;
- the GPU is temporarily idle between planned stages;
- the estimated monetary cost is high.

Stop the Pod only after one of these terminal conditions:

1. **Validated positive completion:** the final package and claim classification validate.
2. **Validated formal bounded negative completion:** all permitted stages/rescues are exhausted and the negative package validates.
3. **Unrecoverable infrastructure failure:** the configured Pod/GPU/SSH/persistent volume remains unusable after all permitted retries.
4. **Unsafe data-integrity failure:** split leakage or artifact corruption makes continued science invalid and cannot be repaired within the plan.

There is no monetary budget guard. Cost tracking is informational only.

Never terminate/delete the Pod unless the user explicitly requests deletion.

---

## 4. Python environments

### Local MacBook

Use `uv`:

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script>.py
```

Do not use `pip install` directly in the local project environment.

### RunPod

Use the Python available in the image unless a compatible environment already exists:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script>.py
```

Do not require `uv` on RunPod.

---

## 5. Required RunPod configuration

```bash
export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
export PS_TRM_MAX_INFRA_RETRIES="3"
export PS_TRM_MAX_SCIENTIFIC_RESCUES_PER_STAGE="1"
```

`runpodctl` must already be configured with a RunPod API key.

If host/port changes after restart, refresh it using configured RunPod tooling. Do not guess.

---

## 6. Campaign state machine

Required directory:

```text
runs/partial_state_temporal_residual_editor_v1/autonomous_campaign_v1/
```

Required state files:

```text
campaign_state.json
stage_history.csv
autonomous_log.md
cost_state.json
artifact_registry.json
```

For every stage:

1. read the active plan and campaign state;
2. run preflight and tests;
3. execute the exact planned task;
4. validate every acceptance criterion;
5. write versioned artifacts, logs, and explicit exit-code files;
6. update state and registry;
7. advance automatically on pass;
8. apply only the listed bounded rescue on failure;
9. finish formally if no rescue remains.

---

## 7. Git, storage, and long-job rules

Code moves through Git. Large artifacts remain under `/workspace` and `runs/`.

Before each stage:

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

Commit only after tests pass. Do not commit secrets, weights, raw large runs, or private keys.

Long-job template:

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
log file
exit-code file for long jobs
```

Never overwrite a completed run by default. Use `_v1`, `_v2`, and explicit rescue directories.

---

## 8. Split and leakage discipline

Create fresh manifests for this protocol. Exclude all historically used development facts, prompt fingerprints, and source-row IDs where required by the active plan.

Historical Direction 1 `analysis_500`, `final_test_500`, and `final_test_full` remain untouched and are forbidden for this campaign.

Training/protection data may include only predeclared training prompts and anchors. Evaluation-only prompts include:

```text
official held-out paraphrases
held-out same-subject different-relation stress prompts
held-out near/far locality prompts
held-out generation/attribute prompts
locked confirmation facts and prompts
```

Hard rule:

```text
train_prompt_ids ∩ evaluation_prompt_ids = empty
```

The rewrite prompt may be reported as `train_seen=true`. Generalization and locality metrics must be `train_seen=false`.

---

## 9. Scientific method families

Required methods:

```text
base
prompt_memory
target_logit_bias
ordinary_mdm_memit
partial_state_mdm_memit
static_nullspace_partial_state_memit
timerome_source_reproduction
timerome_counterfact_fullmask
timerome_counterfact_partial_state
timerome_partial_state_state_bucketed
timerome_partial_state_state_protected
random_site_partial_state_residual
fixed_site_partial_state_residual
```

One relation-conditioned protection rescue is allowed only if the state-conditioned method reaches the trigger defined in the plan.

The main method must not be described as a permanent weight editor. It is a frozen-backbone, low-rank inference-time residual memory unless a stage explicitly modifies weights.

---

## 10. Runtime deployability and forbidden shortcuts

Allowed runtime inputs:

```text
current hidden state at the selected temporal coordinate
subject key
prompt/edit tuple
relation identifier or precomputed relation cluster
step/timestep or early-middle-late bucket
active mask count and selected answer-span position
base logits/log-probabilities
fitted residual-memory parameters
sparsification/calibration parameters
```

Forbidden runtime inputs:

```text
evaluation prompt type
evaluation negative type
split label
case ID as a learned feature
final decoded outcome
final rewrite/locality success
teacher-only future reward
analysis/final metrics
```

Every checkpoint or fitted memory must serialize its runtime feature schema and pass a leakage audit.

---

## 11. Frozen positive-result definitions

This campaign can finish positively under one or more predeclared claim classes.

### Full editor claim

```text
rewrite_exact >= 0.85
paraphrase_exact >= 0.40
same_subject_tfpr <= base + 0.03
near_tfpr <= base + 0.03
far_tfpr <= base + 0.03
malformed_rate <= 0.05
```

### Pareto locality claim

Relative to the strongest efficacy-matched baseline:

```text
rewrite_loss <= 0.02
paraphrase_loss <= 0.02
same_subject_tfpr reduction >= 25%
paired 95% CI for TFPR delta is below 0
near/far locality do not materially worsen
```

### Diffusion-specific partial-state claim

```text
partial-state temporal residual improves multi-token rewrite by >= 0.10
on at least two target-length bins
with positive paired evidence on the pooled comparison
```

### State-conditioning claim

```text
state-conditioned protection reduces same-subject TFPR by >= 20%
versus shared residual at matched efficacy
or improves the frozen stress-aware aggregate by >= 0.05
with positive paired evidence
```

Do not retroactively redefine a failed claim. A method may support one claim class without supporting all classes.

---

## 12. Bounded rescues

Allowed total rescues:

```text
one TimeROME source-integration repair
one temporal-site policy rescue
one residual-memory ridge/sparsity rescue
one state-conditioned protection rescue
one relation-conditioned protection rescue, only if triggered
one Dream integration repair
```

No rescue after locked confirmation inspection.

Never lower hard thresholds or expand grids beyond the plan.

---

## 13. Final package

Required final directory:

```text
runs/partial_state_temporal_residual_editor_v1/final_research_package_v1/
```

Required artifacts:

```text
report_summary.json
main_results_table.csv
multi_token_table.csv
same_subject_stress_table.csv
locality_table.csv
causal_localization_table.csv
compute_storage_table.csv
paired_bootstrap.csv
rewrite_locality_pareto.png
state_bucket_plot.png
multi_token_plot.png
failure_cases.csv
artifact_availability_manifest.json
reproducibility_manifest.json
final_research_report.md
paper_claim_recommendation.md
terminal_package_validation.json
```

The final claim must be classified as:

```text
full_editor_positive
pareto_locality_positive
diffusion_specific_positive
state_conditioning_positive
reproduction_only
diagnostic_negative
formal_bounded_negative
infrastructure_blocked
```

After the final package validates, mark campaign terminal and stop the Pod.
