# AGENTS.md

Operational and scientific rules for Codex in the LLaDA / CounterFact Schrödinger-bridge model-editing repository.

The repository contains three completed historical campaigns and one new active autonomous campaign.

```text
Direction 1 = closed; runtime bridge signal was useful, but tested rule-based gates failed same-subject locality.
Direction 2 v1 = closed; protocol-infeasible before adapter science.
Direction 3 v1 = closed; bounded negative result because the learned value controller used a target-token shortcut.
Active campaign = counterfact_sb_alternatives_campaign_v1.
```

The active campaign tests five new alternatives:

```text
T1 = learned edit-intent gate + raw runtime bridge
T2 = activation-space Schrödinger bridge
T3 = conditional answer-span categorical Schrödinger bridge matching
T4 = unbalanced / partial categorical Schrödinger bridge
T5 = parameter-space Schrödinger bridge over low-rank adapter latents
```

All historical Direction 1, Direction 2 v1, and Direction 3 v1 artifacts are immutable evidence. They may be read, but must not be overwritten, deleted, or silently resumed.

---

## 0. Active project identity

```text
campaign_protocol = counterfact_sb_alternatives_campaign_v1
base_model = GSAI-ML/LLaDA-8B-Base
theta0 = frozen
edit_access = given_at_edit_time
historical_protocols = counterfact_direction1_v1,counterfact_direction2_bridge_adapter_v1,counterfact_direction3_controller_v1
analysis_500 = locked confirmation only
final_test_500 = locked final evaluation only
final_test_full = optional secondary replication only if precommitted and budget allows
```

Track protocol names:

```text
counterfact_learned_gate_raw_bridge_v1
counterfact_activation_space_sb_v1
counterfact_conditional_answer_span_csbm_v1
counterfact_unbalanced_partial_csbm_v1
counterfact_parameter_space_sb_v1
```

---

## 1. Authoritative files and read order

Before acting, Codex must read these files in order:

```text
1. AGENTS.md
2. ACTIVE_RESEARCH_CAMPAIGN.json
3. ALTERNATIVE_PROTOCOL_REGISTRY.json
4. SB_ALTERNATIVES_AUTONOMOUS_RESEARCH_PLAN.md
5. the relevant per-track plan file
6. existing campaign state under runs/counterfact_sb_alternatives_campaign_v1/
```

Per-track plans:

```text
LEARNED_GATE_RAW_BRIDGE_PLAN.md
ACTIVATION_SPACE_SB_PLAN.md
CONDITIONAL_ANSWER_SPAN_CSBM_PLAN.md
UNBALANCED_PARTIAL_CSBM_PLAN.md
PARAMETER_SPACE_SB_PLAN.md
```

`ACTIVE_RESEARCH_CAMPAIGN.json` is the authority on the active campaign. Codex must refuse to resume a historical direction or create a new protocol unless the file and master plan explicitly permit it.

---

## 2. Autonomous mode

Autonomous execution is enabled only when:

```bash
export SB_ALT_AUTONOMOUS_MODE=1
export SB_ALT_AUTONOMOUS_BUDGET_USD="<total authorized budget>"
export RUNPOD_HOURLY_RATE_USD="<current hourly rate>"
```

Recommended variables:

```bash
export SB_ALT_AUTONOMOUS_BUDGET_RESERVE_USD="5"
export SB_ALT_AUTONOMOUS_MAX_INFRA_RETRIES="3"
export SB_ALT_AUTONOMOUS_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"
export SB_ALT_MIN_UNTESTED_TRACK_RESERVE_USD="8"
```

When `SB_ALT_AUTONOMOUS_MODE=1`, the user grants one-time approval for Codex to execute every task explicitly listed in the master and track plans without requesting per-stage, per-command, or per-GPU-job approval.

Codex must not:

```text
ask for approval between planned stages
change protocols during execution
lower hard acceptance criteria after seeing results
invent unplanned rescues
add teacher-only fields as runtime inputs
tune on analysis_500 or final_test_500
stop the Pod merely because one track fails
```

When autonomous mode is disabled, normal approval rules apply for paid GPU work.

---

## 3. Campaign completion definition

The campaign reaches a terminal state only after every track has a terminal status or the budget/infrastructure makes further execution impossible.

### Positive completion

```text
- all five tracks have completed their mandatory bounded pilot;
- every passed track has completed its planned dev evaluation;
- one primary method is selected on dev before analysis;
- the primary method passes analysis_500;
- final_test_500 is executed exactly once for the frozen final package;
- cross-track reports, tables, plots, failure cases, and reproducibility artifacts validate;
- the Pod is stopped.
```

### Scientific negative completion

```text
- all five tracks have completed their mandatory bounded pilot;
- no track satisfies the predeclared dev eligibility criteria, or the dev-selected primary fails locked analysis;
- each failed track has a formal stop package;
- a cross-track negative-result package is generated;
- no unjustified final-test run occurs;
- the Pod is stopped.
```

### Budget completion

```text
- the remaining budget cannot cover the next required stage plus the reserve for all untested tracks;
- completed and untested work is documented;
- no partially hidden scientific claim is made;
- the Pod is stopped.
```

### Infrastructure/data-integrity completion

```text
- an unrecoverable infrastructure, split-leakage, artifact-corruption, or credential failure remains after allowed retries;
- a formal infrastructure/data-integrity stop package is written;
- the Pod is stopped when possible.
```

A single track failure is not campaign completion. Codex must continue to the next untested track.

---

## 4. Local and RunPod Python environments

Outside the autonomous campaign, local MacBook Python work must use `uv`:

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script_name>.py
```

Do not use `pip install` directly in the local project environment.

During the autonomous campaign, `/workspace/SB` on RunPod is the authoritative worktree. Use the Python available in the RunPod image:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script_name>.py
```

Do not require `uv` on RunPod. Record Python, PyTorch, Transformers, bitsandbytes, CUDA, GPU, model, and tokenizer versions in GPU-stage summaries.

---

## 5. RunPod task routing and retention

When autonomous mode is enabled:

1. Start the existing configured Pod at campaign start if it is stopped.
2. Use `/workspace/SB` as the authoritative campaign worktree.
3. Run CPU and GPU campaign tasks on the Pod to avoid synchronization pauses.
4. Use `tmux` and explicit exit-code files for long jobs.
5. Commit and push code checkpoints after tests pass; never commit large artifacts.
6. Keep the Pod running between all tracks and between CPU/GPU stages.
7. Do not stop the Pod after an individual track fails or finishes.
8. Immediately continue to the next planned track after validating the current track.

The Pod may be stopped only when:

```text
all five tracks and the final cross-track package are complete;
the campaign reaches a formal scientific negative completion;
the budget is exhausted or insufficient for the next required stage;
an unrecoverable infrastructure/data-integrity failure remains after retries.
```

Never terminate/delete a Pod unless the user explicitly authorizes deletion.

---

## 6. Required RunPod variables

```bash
export RUNPOD_POD_ID="<existing pod id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current host>"
export RUNPOD_SSH_PORT="<current port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

`runpodctl` must already be configured with the RunPod API key.

Codex must never print or commit:

```text
private SSH keys
RunPod API keys
Hugging Face tokens
OpenAI credentials
other secrets
```

If SSH host/port changes after Pod restart, Codex should refresh it through configured tooling/API. Do not guess. Retry only up to the configured infrastructure limit.

---

## 7. Budget guard and breadth-first testing rule

Required budget state:

```text
runs/counterfact_sb_alternatives_campaign_v1/autonomous_campaign_v1/budget_state.json
```

Track the following:

```json
{
  "budget_usd": 0.0,
  "hourly_rate_usd": 0.0,
  "reserve_usd": 0.0,
  "estimated_spend_usd": 0.0,
  "remaining_budget_usd": 0.0,
  "untested_tracks": [],
  "minimum_reserve_for_untested_tracks_usd": 0.0,
  "stage_costs": []
}
```

Cost-source priority:

1. RunPod API/account usage when available.
2. Actual Pod running duration multiplied by hourly rate.
3. Conservative stage-runtime estimate.

Before each expensive stage, Codex must verify:

```text
estimated_stage_cost
  <= remaining_budget
     - reserve
     - minimum_reserve_for_all_untested_tracks
```

Breadth-first rule:

```text
No track may enter its expensive scale-up phase until every other track has completed its mandatory minimum pilot or has a formal pilot-level negative result.
```

This prevents an early promising track from consuming the budget before all alternatives are tested.

If the budget cannot cover the next mandatory untested-track pilot, end as budget completion. Do not ask for a budget top-up.

---

## 8. Campaign state machine

Required directory:

```text
runs/counterfact_sb_alternatives_campaign_v1/autonomous_campaign_v1/
```

Required files:

```text
campaign_state.json
budget_state.json
track_registry.csv
stage_history.csv
autonomous_log.md
```

`campaign_state.json` must include:

```json
{
  "campaign_protocol": "counterfact_sb_alternatives_campaign_v1",
  "autonomous_mode": true,
  "campaign_status": "running",
  "current_track": "",
  "current_stage": "",
  "analysis_500_used": false,
  "final_test_used": false,
  "completed_tracks": [],
  "failed_tracks": [],
  "passed_tracks": [],
  "rescues_used": {},
  "last_git_commit": ""
}
```

Track statuses:

```text
pending
running
pilot_passed
pilot_failed
scaled_dev_passed
scaled_dev_failed
analysis_passed
analysis_failed
final_reported
formal_negative
budget_not_run
```

For every stage:

1. Read master plan, track plan, and current state.
2. Run preflight checks and tests.
3. Estimate budget and reserve untested tracks.
4. Execute the stage.
5. Validate every acceptance criterion.
6. Write versioned artifacts, validation report, log, and exit code.
7. Update campaign, track, and budget state.
8. Apply only the bounded rescue listed in the track plan.
9. Mark the track terminal at the appropriate level.
10. Continue to the next track without stopping the Pod.

---

## 9. Historical evidence and immutability

These directories are immutable:

```text
runs/counterfact_direction1_v1/
runs/counterfact_direction2_bridge_adapter_v1/
runs/counterfact_direction3_controller_v1/
```

They may be read for:

```text
historical baselines
frozen manifests and fingerprints
reusable code
teacher/checkpoint initialization only when explicitly permitted
scientific comparison
```

They must not be:

```text
overwritten
deleted
silently resumed
used as hidden evaluation training data
reported as new results
```

Direction 2 v1 remains protocol-infeasible, not adapter-failed. Direction 3 v1 remains a bounded negative value-controller result with a promising learned gate.

---

## 10. Common data and split policy

The common campaign source is the official CounterFact train split.

Locked manifests may be read for ID/source-index/fingerprint exclusion only until their evaluation stage is legally unlocked.

Training pool rules:

```text
Exclude from training:
- dev_tune_200
- ablation_500
- analysis_500
- final_test_500
- final_test_full
- all evaluation-only same-subject stress prompts

Allow as training rows:
- legal CounterFact train rows previously used for historical controller training,
  provided historical teacher outputs/checkpoints are not silently reused;
- explicit reusable training-only prompt augmentations;
- real prompt fields from allowed train rows.
```

Evaluation ladder:

```text
track_train / track_val = method training and early stopping
track_smoke_20 = integration and bounded calibration
track_confirmation_30 = untouched confirmation; no tuning
common dev_tune_200 = final method/hyperparameter selection
ablation_500 = frozen preplanned ablations only
analysis_500 = locked proceed/stop confirmation
final_test_500 = primary locked result
```

Target-length policy:

```text
Primary training strata = single-token (length 1) and multi-token (length >=2).
Always report exact bins 1,2,3,>=4.
Do not require bin 2 in every split when the legal pool cannot support it.
Never silently substitute bin 3 for bin 2; report exact composition and effective sample size.
```

Prompt provenance:

```text
Prefer real rewrite, paraphrase, locality, generation, and attribute prompts.
Synthetic fallback must be tagged and reported.
No official evaluation paraphrase/locality/stress prompt may be used for training the same edit.
```

---

## 11. Common baselines and metrics

Every actual-decode track must compare against applicable baselines:

```text
base
target_logit_bias
prompt_memory
myopic_score
no_rollout_bridge
mc_bridge
best historical raw bridge configuration
```

Additional track-specific baselines are defined in each track plan.

Common metrics:

```text
rewrite exact
declarative paraphrase exact
target-token F1
old-target suppression
near/far locality exact
self-normalized locality
same-subject target false-positive rate
generation target false-positive rate
malformed-span rate
sparse-support guidance KL or intervention cost
model evaluations per edit
GPU minutes per edit
training GPU minutes per edit
storage bytes per edit when applicable
paired bootstrap by edit_id
exact target-length breakdown
relation breakdown
```

Common hard constraints for a viable runtime candidate:

```text
same_subject_tfpr <= base_same_subject_tfpr + 0.03
near_tfpr <= base_near_tfpr + 0.03
far_tfpr <= base_far_tfpr + 0.03
malformed_rate <= 0.05
gpu_minutes_per_edit <= 2.0 unless the track plan declares a stricter bound
```

Primary feasible selection score:

```text
maximize harmonic_mean(rewrite_exact,
                       declarative_paraphrase_exact,
                       clipped_self_normalized_locality)
subject to all hard constraints.
```

Report compute-quality and guidance-cost Pareto curves. Do not call sparse-support KL full path KL.

---

## 12. Runtime deployability and leakage

Runtime inputs must exist during real inference.

Teacher-only fields may be labels but never runtime inputs:

```text
raw bridge scores
MC rollout rewards
myopic/no-rollout teacher scores
teacher chosen token
future final success/locality outcomes
completed-trajectory cost
```

Forbidden shortcuts:

```text
prompt_type
negative_type
evaluation bucket identity
split label
case ID as a learned feature
final outcome labels
```

Every learned checkpoint must serialize its runtime feature schema and pass a leakage audit.

Track-specific scientific shortcut audits are mandatory where defined.

---

## 13. Bounded rescue rules

Each track gets only the rescue explicitly declared in its plan.

Global limits:

```text
one scientific rescue per track by default
no architecture expansion after a track confirmation split
no rescue after common dev selection
no rescue after analysis_500
no final-test rerun for tuning
```

Codex must never:

```text
lower acceptance thresholds
add evaluation prompts to training
add teacher-only runtime inputs
expand grids beyond the plan
use analysis/final data for rescue
reinterpret a failed track as passed
```

A failed track must write a formal track stop package, then the campaign continues.

---

## 14. Cross-track selection and locked evaluation

All five mandatory pilots must finish before any track scale-up.

After scale-up, each passed track may nominate at most one frozen dev candidate.

Before `analysis_500`, write:

```text
runs/counterfact_sb_alternatives_campaign_v1/cross_track_dev_lock.json
```

It must freeze:

```text
primary candidate selected on dev
all frozen secondary track candidates
checkpoints and hashes
training split fingerprints
method-specific hyperparameters
gate/transport thresholds
top_k
steps and schedule
span policy
normalization
metrics and budgets
report-script commit
```

The primary candidate must be selected on dev before analysis. Analysis cannot change the primary.

After the lock validates, set:

```bash
export DEV_METHOD_LOCKED=1
```

`analysis_500` may run all frozen passed-track candidates and baselines once. It is confirmation only.

If the primary fails analysis, the campaign ends negatively. Do not promote a secondary after inspecting analysis unless that fallback rule was precommitted in the lock.

If analysis passes, write `analysis_confirmation_lock.json`, set `FINAL_METHOD_LOCKED=1`, and run final evaluation exactly once.

---

## 15. Git, storage, and long-job rules

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

Commit code only after tests pass. Do not commit secrets, model weights, checkpoints, safetensors, or large run artifacts.

Authoritative remote paths:

```text
/workspace/SB
/workspace/SB/runs
/workspace/SB/logs
/workspace/.cache/huggingface
```

Every stage must use a versioned output directory and write:

```text
report_summary.json
run_config.json or equivalent
validation report
log file
exit-code file for long jobs
```

Long job template:

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage_name>.log; \
   code=${PIPESTATUS[0]}; echo "$code" > logs/<stage_name>.exitcode; exit "$code"'
```

At major checkpoints, mirror compact summaries, manifests, hashes, and reports to durable storage. Keep the Pod running; do not stop merely for synchronization.

---

## 16. Required track stop package

Every failed or budget-not-run track must write:

```text
report_summary.json
track_stop_checkpoint.md
negative_result_report.md
track_evidence_table.csv
artifact_availability_manifest.json
next_recommendation.md
```

The report must distinguish:

```text
implementation failure
protocol infeasibility
offline scientific failure
actual-decode failure
generalization failure
budget-not-run
```

Do not call an untested hypothesis a failed method.

---

## 17. Final campaign package

Required directory:

```text
runs/counterfact_sb_alternatives_campaign_v1/final_research_package_v1/
```

Required files:

```text
report_summary.json
cross_track_status_table.csv
cross_track_main_results.csv
same_subject_stress_table.csv
target_length_table.csv
relation_table.csv
compute_storage_table.csv
paired_bootstrap.csv
rewrite_locality_pareto.png
aggregate_compute_pareto.png
same_subject_plot.png
failure_cases.csv
artifact_availability_manifest.json
reproducibility_manifest.json
final_research_report.md
paper_claim_matrix.md
next_research_recommendation.md
```

Claim categories:

```text
strong SB editing method claim
efficiency/amortization claim
edit-intent localization claim
activation-space transport claim
categorical CSBM claim
partial/unbalanced transport claim
parameter-space editing claim
diagnostic/negative result
```

The claim must follow the evidence. The final report must state which hypotheses were supported, rejected, protocol-infeasible, budget-not-run, or left untested.

---

## 18. Codex behavior expectations

Codex must:

- read all authoritative files before acting;
- execute every mandatory track pilot before scaling any track;
- continue automatically between tracks;
- keep RunPod running until campaign terminal completion or budget/infrastructure stop;
- preserve historical campaigns;
- use tests, versioned artifacts, explicit acceptance reports, and split locks;
- stop only at a campaign-level terminal state;
- provide the user with one final cross-track results report at the end.

Codex must not:

- ask for stage-by-stage approval in autonomous mode;
- create or delete a Pod;
- stop the Pod after an individual track;
- guess missing secrets;
- invent experiments or rescues;
- tune on analysis/final;
- overwrite historical artifacts;
- silently skip a track without recording `budget_not_run` or a formal reason.
