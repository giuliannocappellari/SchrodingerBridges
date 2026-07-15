# AGENTS.md

Operational and scientific rules for Codex in the LLaDA / CounterFact model-editing repository.

The latest research program is **Direction 3: learned edit-conditioned bridge controller** under `counterfact_direction3_controller_v1`. Its autonomous v1 campaign is now closed with a bounded negative result.

Historical status:

```text
Direction 1 = closed; blocked under tested rule-based runtime gates
Direction 2 v1 = closed; protocol-infeasible before adapter science
Direction 2 v2 = not created and must not be created automatically
Direction 3 v1 = closed; offline hard criteria failed after the one allowed bounded rescue
```

Direction 1 and Direction 2 v1 artifacts are immutable historical evidence. They may be read for provenance, baselines, code reuse, or reporting, but must not be overwritten, deleted, resumed, or used to change locked split policy.

---

## 0. Active project identity

```text
active_protocol_version = counterfact_direction3_controller_v1
active_direction = none; Direction 3 v1 completed negative
active_campaign_file = ACTIVE_RESEARCH_CAMPAIGN.json
authoritative_plan = DIRECTION3_AUTONOMOUS_RESEARCH_PLAN.md
historical_protocols = counterfact_direction1_v1,counterfact_direction2_bridge_adapter_v1
base_model = GSAI-ML/LLaDA-8B-Base
theta0 = frozen
trained_parameters = controller_and_gate_only
base_model_weight_update = none
edit_access = given_at_edit_time
training_access = controller_train_only
current_stage = complete_negative
```

Direction 3 trains a small deployable controller and edit-intent gate over frozen LLaDA representations. The intended runtime form is:

```text
edited_logits(v)
  = base_logits(v)
  + guidance_scale
    * gate(prompt, edit)
    * controller_advantage(v, state, step, edit)
```

Canonical split roles:

```text
controller_train_* = controller/gate training only
controller_val_*   = controller/gate validation, early stopping, calibration, threshold selection
dev_smoke_50       = bounded integration smoke; split once into smoke20 + confirmation30
dev_tune_200       = only split for final D3 method/hyperparameter selection
ablation_500       = frozen, pre-planned ablations only; cannot affect selection
analysis_500       = locked proceed/stop confirmation only
final_test_500     = primary locked final result
final_test_full    = optional secondary replication only if pre-committed and budget allows
```

The scarce target-length-bin-2 examples already assigned to Direction 3 train/validation must be preserved when scaled splits are built. Scaled Direction 3 splits extend the existing D3 train/validation splits; they do not exclude their own seed records.

---

## 1. Authoritative autonomous plan and instruction precedence

Codex must read these files before acting, in this order:

```text
1. AGENTS.md
2. ACTIVE_RESEARCH_CAMPAIGN.json
3. DIRECTION3_AUTONOMOUS_RESEARCH_PLAN.md
4. existing autonomous campaign state under runs/
```

If the files conflict, the earlier item in the list takes precedence. Codex must stop with `campaign_configuration_conflict` rather than guessing.

The authoritative execution plan is:

```text
DIRECTION3_AUTONOMOUS_RESEARCH_PLAN.md
```

The active campaign marker is:

```text
ACTIVE_RESEARCH_CAMPAIGN.json
```

When autonomous mode is enabled, the user grants one-time approval for Codex to execute every task explicitly listed in the Direction 3 plan without requesting per-stage, per-command, or per-GPU-job approval.

Autonomous mode is enabled only when:

```bash
export D3_AUTONOMOUS_MODE=1
export D3_AUTONOMOUS_BUDGET_USD="<total authorized budget>"
export RUNPOD_HOURLY_RATE_USD="<current hourly rate>"
```

Recommended additional variables:

```bash
export D3_AUTONOMOUS_BUDGET_RESERVE_USD="5"
export D3_AUTONOMOUS_MAX_INFRA_RETRIES="3"
export D3_AUTONOMOUS_MAX_SCIENTIFIC_RESCUES_PER_STAGE="1"
```

If `D3_AUTONOMOUS_MODE != 1`, use normal approval rules and do not start paid GPU work without explicit approval.

If autonomous mode is enabled, Codex must not pause for routine approval between planned stages. It must follow the state machine, acceptance criteria, budget guard, bounded rescues, split locks, and stop conditions in the plan.

Codex must not:

```text
switch to Direction 2 v2 automatically
resume Direction 2 v1
lower hard acceptance thresholds
invent extra rescue attempts
expand the hyperparameter grids
use teacher-derived runtime inputs
use analysis/final data for tuning
```

### 1.1 Codex interaction mode

For the autonomous campaign, use **Goal mode** (`/goal`), not Plan mode. The research plan, constraints, state machine, and completion criteria already exist in repository files.

Use `/plan` only when the user explicitly wants to redesign or amend the research protocol before execution. A plan-mode session must not start RunPod or execute science unless it is later converted into a separately approved goal.

Goal mode must treat the repository plan and active-campaign file as the completion contract.

---

## 2. Autonomous campaign completion definition

The Direction 3 campaign is complete when either:

### Positive completion

```text
- a locked D3 configuration passes analysis_500,
- final_test_500 is executed once,
- final tables/plots/reproducibility artifacts are generated,
- the strongest defensible claim is classified and documented.
```

### Negative completion

```text
- a bounded scientific stop criterion in the plan is reached,
- no further planned rescue remains,
- no unjustified analysis/final run is performed,
- a formal Direction 3 stop checkpoint and negative-result report are generated.
```

### Budget completion

```text
- remaining authorized budget cannot cover the next required stage plus reserve,
- a budget-exhaustion checkpoint records completed evidence and unrun stages,
- the Pod is stopped.
```

All three are valid research completions.

---

## 3. Local MacBook Python environment

Outside an active RunPod campaign, the MacBook/local environment must use `uv`.

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script_name>.py
```

Rules:

- Prefer `uv run ...` for every local Python command.
- Use `uv sync` for dependency installation.
- Do not use `pip install` directly in the local project environment.
- Do not manually activate `.venv` unless explicitly required.
- If `pyproject.toml` or `uv.lock` changes, inspect and report the diff.

---

## 4. RunPod Python environment

On RunPod, use the Python environment available in the selected PyTorch/CUDA image unless a custom environment already exists in the repo.

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script_name>.py
```

Rules:

- Do not require `uv` on RunPod.
- Do not assume `.venv` exists.
- Prefer `python -m pip` if a dependency is missing.
- Record the Python executable and package versions in every GPU run summary.
- Never install packages from untrusted sources.

---

## 5. Normal task routing

When autonomous mode is disabled:

| Task | Default environment |
|---|---|
| Code editing/refactoring | MacBook |
| Unit tests/fake-model tests | MacBook |
| CSV/JSON/report aggregation | MacBook |
| Cached-state training/replay | MacBook unless too slow |
| Frozen-LLaDA feature extraction | RunPod |
| Teacher-cache generation | RunPod |
| Actual LLaDA decoding | RunPod |
| analysis_500/final_test_500 | RunPod after locks |

---

## 6. Autonomous task routing

When `D3_AUTONOMOUS_MODE=1`:

1. Start the existing configured RunPod Pod at campaign start if it is stopped.
2. Use `/workspace/SB` as the authoritative worktree for the remainder of the campaign.
3. Run both CPU and GPU campaign tasks on the Pod so the campaign can progress without waiting for local synchronization.
4. Use the remote system Python for tests, training, reports, and decoding.
5. Commit/push code checkpoints after tests pass; do not commit large artifacts.
6. Keep the Pod running between stages, including CPU-only stages.
7. Do not stop the Pod merely because a GPU job ended.

The Pod must remain running until one of the campaign stop conditions in Section 9 occurs.

---

## 7. Required RunPod variables

Before autonomous campaign start, these must be configured:

```bash
export RUNPOD_POD_ID="<existing pod id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current host>"
export RUNPOD_SSH_PORT="<current port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

Also configure `runpodctl` with a RunPod API key.

Codex must never print private keys, API keys, Hugging Face tokens, or OpenAI credentials.

If SSH host/port changes after a restart, Codex should refresh connection details using the configured RunPod tooling/API available in the environment. Do not guess values.

If connection details cannot be refreshed after the configured infrastructure retry limit, mark the campaign `infrastructure_blocked`, write a checkpoint, and stop the Pod to protect the budget.

---

## 8. RunPod lifecycle in autonomous mode

### Start

```bash
runpodctl pod start "$RUNPOD_POD_ID"
runpodctl pod list
```

After start, verify:

```text
- desired status is RUNNING,
- at least one GPU is allocated,
- SSH works,
- nvidia-smi works,
- /workspace/SB exists or can be cloned.
```

### Keep running

During the autonomous campaign:

- Do not stop the Pod between plan stages.
- Do not stop after teacher-cache generation, feature extraction, training, replay, smoke decoding, dev tuning, analysis, or final reporting if another planned task remains.
- If no GPU process is active, immediately continue to the next CPU/report/training task on the same Pod.
- Use `tmux` for long jobs.

### Stop

Stop the Pod only when:

1. the complete Direction 3 research campaign finishes positively;
2. the Direction 3 campaign finishes with a formal negative stop checkpoint;
3. the autonomous budget is exhausted or insufficient for the next stage;
4. an unrecoverable infrastructure failure remains after all allowed retries;
5. a data-integrity or split-leakage failure makes further execution unsafe.

Stop command:

```bash
runpodctl pod stop "$RUNPOD_POD_ID"
```

Never terminate/delete the Pod unless the user explicitly requests deletion.

---

## 9. Budget guard

The campaign must track cost continuously.

Required state file:

```text
runs/counterfact_direction3_controller_v1/autonomous_campaign_v1/budget_state.json
```

It must contain:

```json
{
  "budget_usd": 0.0,
  "hourly_rate_usd": 0.0,
  "reserve_usd": 0.0,
  "estimated_spend_usd": 0.0,
  "remaining_budget_usd": 0.0,
  "campaign_start_utc": "",
  "last_updated_utc": "",
  "stage_costs": []
}
```

Cost source priority:

1. RunPod API/account usage if available;
2. actual Pod running duration multiplied by `RUNPOD_HOURLY_RATE_USD`;
3. conservative stage-runtime estimate.

Before every expensive stage, Codex must estimate stage cost.

Proceed only if:

```text
estimated_stage_cost <= remaining_budget - reserve
```

If not, Codex must:

```text
- not start the stage,
- write a budget-exhaustion checkpoint,
- summarize completed science and missing work,
- stop the Pod.
```

Codex must not request a budget top-up during autonomous execution.

---

## 10. Campaign state machine

Required campaign directory:

```text
runs/counterfact_direction3_controller_v1/autonomous_campaign_v1/
```

Before creating or resuming that directory, Codex must validate `ACTIVE_RESEARCH_CAMPAIGN.json`:

```text
active_protocol = counterfact_direction3_controller_v1
active_direction = direction3
campaign_status = active
analysis_500_locked = true
final_test_locked = true
```

If an old Direction 2 campaign state exists, preserve it as historical and do not reuse its state, budget, locks, or current-stage fields.

Required files:

```text
campaign_state.json
budget_state.json
stage_history.csv
autonomous_log.md
```

`campaign_state.json` must contain:

```json
{
  "protocol_version": "counterfact_direction3_controller_v1",
  "autonomous_mode": true,
  "current_stage": "",
  "current_stage_status": "pending",
  "next_stage": "",
  "campaign_status": "running",
  "scientific_claim_status": "undetermined",
  "analysis_500_used": false,
  "final_test_used": false,
  "last_git_commit": "",
  "completed_stages": [],
  "failed_stages": [],
  "rescues_used": {}
}
```

For each stage:

1. Read the plan and current campaign state.
2. Run preflight checks.
3. Run tests.
4. Estimate budget.
5. Execute the stage.
6. Validate all acceptance criteria.
7. Write versioned artifacts and report summary.
8. Update campaign and budget state.
9. Advance automatically if passed.
10. Apply at most the bounded rescue specified in the plan if failed.
11. If no rescue remains, finish as a negative result.

Codex must not invent unplanned experiments or expand hyperparameter grids beyond the plan.

---

## 11. Git and code rules

Code moves through Git. Large artifacts do not.

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

Commit only after tests pass.

Do not commit:

```text
.env
private keys
tokens
RunPod API keys
model weights
large run artifacts
*.safetensors
*.pt
*.ckpt
```

The `.gitignore` should include:

```text
.env
runs/
.cache/
*.pt
*.pth
*.safetensors
*.ckpt
__pycache__/
.ipynb_checkpoints/
```

---

## 12. Storage and artifact rules

Authoritative remote paths:

```text
/workspace/SB
/workspace/SB/runs
/workspace/.cache/huggingface
/workspace/SB/logs
```

Every stage must use a new versioned output directory. Never overwrite a completed run by default.

Every stage must write:

```text
report_summary.json
run_config.json or equivalent
validation report
log file
exit-code file for long jobs
```

Long job pattern:

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage_name>.log; \
   code=${PIPESTATUS[0]}; echo "$code" > logs/<stage_name>.exitcode; exit "$code"'
```

Large artifacts remain on `/workspace`. At major checkpoints, copy summaries and small reports to a durable backup. Do not stop the campaign solely for backup synchronization.

---

## 13. Split locks and scientific safety

Direction 1 and Direction 2 v1 protocol directories are read-only historical evidence.

Direction 2 v1 ended before adapter science because all 284 legal target-length-bin-2 training examples were consumed by the predeclared Direction 1 and Direction 3 reservations. It must not be resumed. Direction 2 v2 requires an explicit new protocol and is outside this autonomous campaign.

For Direction 3 scaled split construction:

```text
controller_train_1000 must be a deterministic superset of controller_train_100
controller_val_200 must be a deterministic superset of controller_val_50
dev_smoke_50 remains held out and is never absorbed into training
existing D3 target-length-bin-2 train/val records must be preserved
no new requirement may demand additional bin-2 cases beyond those legally available
bin-2 scarcity must be reported in every scaled summary
```

No script may use `analysis_500` or final-test splits unless the relevant lock exists.

### Analysis lock

Required file:

```text
runs/counterfact_direction3_controller_v1/dev_method_lock.json
```

Only after that file validates may Codex set:

```bash
export DEV_METHOD_LOCKED=1
```

### Final lock

Required file:

```text
runs/counterfact_direction3_controller_v1/analysis_confirmation_lock.json
```

Only after analysis passes may Codex set:

```bash
export FINAL_METHOD_LOCKED=1
```

Every split-capable script must enforce:

```python
import os

if split_role == "analysis_500":
    assert os.environ.get("DEV_METHOD_LOCKED") == "1"

if split_role.startswith("final_test"):
    assert os.environ.get("FINAL_METHOD_LOCKED") == "1"
```

After inspecting `analysis_500`, do not change:

```text
controller architecture
controller checkpoint
gate checkpoint
gate threshold
guidance scale
top_k
steps
schedule
span policy
normalization
metrics
budgets
filtering
random seeds
```

If analysis fails, mark `counterfact_direction3_controller_v1` failed. Do not tune on analysis.

---

## 14. Feature leakage and deployability rules

Runtime controller/gate inputs must be available during actual inference.

Forbidden runtime inputs include:

```text
raw_bridge_scores_top_k
mc_rollout_rewards_top_k
myopic/no_rollout scores or margins
teacher chosen token
final decoded output
final edit/locality success
malformed outcome
completed-trajectory KL
prompt_type
negative_type
split label
case/edit identifiers as semantic features
```

Teacher-derived fields may be labels/targets only.

Every controller-training stage must write a feature schema and pass an actual-tensor leakage audit with:

```text
num_leaked_runtime_features = 0
feature_leakage_audit_pass = true
```

No actual decoding is allowed if leakage audit fails.

---

## 15. Direction 3 scientific integrity rules

The controller must be evaluated groupwise over each top-k candidate group. Do not use flattened global correlation as the primary ranking metric.

Primary offline value metrics:

```text
macro groupwise Spearman
Kendall tau
NDCG@8
pairwise ranking accuracy
teacher top-1 agreement
teacher top-3 overlap
target top-3 improvement over base
```

Primary gate metrics:

```text
ROC-AUC
PR-AUC
rewrite activation
paraphrase activation
same-subject activation
near/far locality activation
```

Primary safety metric:

```text
negative_guidance_ratio
same_subject_target_advantage_vs_base
```

Target-indicator-only and representation-shuffle ablations are mandatory before actual decoding.

---

## 16. Retry policy

### Infrastructure retries

Use up to `D3_AUTONOMOUS_MAX_INFRA_RETRIES` for:

```text
SSH disconnect
transient package/download error
RunPod restart/capacity issue
single-process crash without data corruption
```

Resume from checkpoints when possible.

### Scientific rescues

Use at most the rescue explicitly defined for the stage in the plan, normally one.

Do not:

```text
run unbounded sweeps
add teacher-derived runtime features
change locked splits
reuse analysis/final for tuning
silently lower acceptance criteria
```

If the bounded rescue fails, finish the Direction 3 campaign as a negative result.

---

## 17. Reporting rules

Every `report_summary.json` must include at least:

```json
{
  "protocol_version": "counterfact_direction3_controller_v1",
  "stage": "",
  "git_commit": "",
  "split_role": "",
  "analysis_500_used": false,
  "final_test_used": false,
  "llada_loaded": false,
  "actual_decode_performed": false,
  "acceptance_pass": false,
  "acceptance_failures": [],
  "artifacts": {}
}
```

Every GPU report must additionally record:

```text
model_id
dtype
use_4bit
device_map
Python executable/version
GPU name
CUDA version
torch version
transformers version
bitsandbytes version
wall time
model evaluation count
estimated cost
```

---

## 18. Direction 3 final claim classification

At campaign completion, classify the strongest defensible result as exactly one of:

### Strong method claim

```text
D3-gated beats myopic/no-rollout/raw bridge baselines on the stress-aware aggregate,
controls same-subject TFPR,
uses less compute than MC bridge,
and survives analysis and final test.
```

### Efficiency/amortization claim

```text
D3-gated matches MC bridge within confidence intervals
while using substantially fewer model evaluations and GPU time.
```

### Safety/evaluation claim

```text
D3 does not win overall,
but learned edit-intent control materially improves same-subject safety
and exposes a central failure mode of diffusion-LM runtime editing.
```

### Negative result

```text
Deployable controller/gate behavior fails offline or actual decoding
under the bounded plan.
```

Never overstate the claim.

---

## 19. Autonomous stop checkpoint

When the campaign stops for scientific failure or budget exhaustion, write:

```text
runs/counterfact_direction3_controller_v1/direction3_stop_checkpoint_v1/
```

with:

```text
report_summary.json
direction3_stop_checkpoint.md
direction3_evidence_table.csv
remaining_unrun_stages.csv
budget_summary.json
paper_claim_recommendation.md
```

Then stop the Pod.

---

## 20. Final completion package

When final_test_500 completes, write:

```text
runs/counterfact_direction3_controller_v1/final_research_package_v1/
```

with:

```text
report_summary.json
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
```

Only after this package validates is positive campaign completion reached. Then stop the Pod.

---

## 21. Codex behavior expectations

Codex must:

- Read `AGENTS.md`, `ACTIVE_RESEARCH_CAMPAIGN.json`, and the autonomous Direction 3 plan.
- Follow the stage state machine without asking for approval when autonomous mode is enabled.
- Keep the existing RunPod Pod running between stages.
- Maintain budget and campaign state.
- Run tests before every stage.
- Preserve old artifacts.
- Use versioned outputs.
- Stop automatically only at campaign completion, budget exhaustion, or unrecoverable safety/infrastructure failure.
- Report uncertainty and scientific failures honestly.

Codex must not:

- Ask for routine stage approval in autonomous mode.
- Stop the Pod between planned stages.
- Create or delete Pods.
- Tune on `analysis_500` or final-test splits.
- Reintroduce teacher-derived runtime features.
- Lower acceptance thresholds to manufacture a pass.
- Run experiments outside the autonomous plan.
- Create Direction 2 v2 or switch research directions without an explicit user-authored protocol change.
- Reuse Direction 2 autonomous state as Direction 3 state.


---

## 22. Goal-mode completion contract

When the campaign is launched with `/goal`, Codex must continue until exactly one terminal state is written and validated:

```text
positive_complete
negative_complete
budget_complete
infrastructure_blocked
unsafe_data_integrity_stop
```

A goal is not complete merely because one command or one stage finishes. It is complete only after the appropriate final or stop package exists, campaign/budget state is updated, durable summaries are preserved, and the Pod is stopped.
