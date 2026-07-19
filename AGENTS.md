# AGENTS.md

Operational and scientific rules for Codex in the LLaDA factual-editing repository.

## 0. Active campaign

```text
active_protocol = diffusion_editor_next_direction_selection_v1
active_goal = select the single most promising next research direction
base_model = GSAI-ML/LLaDA-8B-Instruct
authoritative_plan = NEXT_DIRECTION_SELECTION_AUTONOMOUS_PLAN.md
campaign_output = runs/diffusion_editor_next_direction_selection_v1/
```

Historical protocols are immutable evidence. Do not modify, overwrite, resume, or reinterpret their pass/fail decisions:

```text
counterfact_direction1_v1
counterfact_direction2_bridge_adapter_v1
counterfact_direction3_controller_v1
counterfact_sb_alternatives_campaign_v1
masked_diffusion_memit_sb_positive_result_v1
mask_pattern_sb_publication_confirmation_v1
diffusion_native_causal_partial_state_editor_v1
partial_state_temporal_residual_editor_v1
```

The campaign is a **selection campaign**, not a full-scale paper campaign. It must test every mandatory candidate direction under bounded, comparable pilots, confirm only the candidates that pass, and return one ranked recommendation. It must not automatically continue into a full new research protocol after selection.

## 1. Files Codex must read before acting

Read in this order:

```text
1. AGENTS.md
2. ACTIVE_RESEARCH_CAMPAIGN.json
3. CANDIDATE_DIRECTION_REGISTRY.json
4. NEXT_DIRECTION_SELECTION_AUTONOMOUS_PLAN.md
5. the individual candidate plan files
6. existing campaign state under runs/diffusion_editor_next_direction_selection_v1/
```

## 2. Autonomous authorization

Autonomous mode is enabled only when:

```bash
export NEXT_DIRECTION_AUTONOMOUS_MODE=1
```

When enabled, the user grants one-time permission to execute every task explicitly listed in the authoritative plan without requesting per-stage, per-command, or per-GPU-job approval.

Codex must not:

```text
ask for intermediate approval
lower thresholds after seeing results
invent additional candidate directions
expand hyperparameter grids beyond the plan
open historical analysis_500 or final-test splits
reuse evaluation prompts as training anchors
add teacher/output labels as runtime features
silently change the active protocol
start the selected full campaign automatically
```

## 3. Mandatory candidate directions

Every mandatory direction must receive its minimum pilot before final selection:

```text
N1 relation-residualized editing
N2 Fisher-constrained / natural-gradient editing
N3 primal-dual constrained locality optimization
N4 selective conformal safe editing
N5 joint answer-span coupling for multi-token edits
```

A predeclared integrated candidate may run only after the individual pilots:

```text
N6 relation-residualized + Fisher/primal-dual + selective wrapper
```

One candidate failing does not stop the campaign. Write its formal track result and continue.

## 4. RunPod lifecycle

There is no monetary budget guard.

At campaign start:

```bash
runpodctl pod start "$RUNPOD_POD_ID"
runpodctl pod list
```

Use:

```text
/workspace/SB
```

as the authoritative remote worktree. Keep the Pod running through all CPU and GPU stages, including failed tracks, reporting, confirmation, and final selection.

Stop the Pod only when:

```text
1. all mandatory tracks are terminal and the final selection package validates; or
2. an unrecoverable Pod/infrastructure issue remains after the allowed retries; or
3. a data-integrity/split-leakage failure makes continuation unsafe and a terminal package has been written.
```

Do not stop the Pod because a job, stage, or track finishes.

Never create, terminate, or delete a Pod. Only start/stop the configured existing Pod.

## 5. RunPod environment

Required variables:

```bash
export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
export NEXT_DIRECTION_MAX_INFRA_RETRIES="3"
export NEXT_DIRECTION_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"
```

Use the Python available in the RunPod image:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script>.py
```

Do not require `uv` on RunPod. On the MacBook, use `uv`.

## 6. Local environment

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script>.py
```

Do not use `pip install` directly in the local project environment.

## 7. Campaign state

Maintain:

```text
runs/diffusion_editor_next_direction_selection_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  track_registry.json
  infrastructure_events.csv
```

The campaign state is authoritative for progress. Preserve completed validated work and resume from the first incomplete stage.

## 8. Git rules

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

Commit only after tests pass. Never commit secrets, model weights, or large run artifacts.

## 9. Long-job pattern

Use `tmux`, `set -o pipefail`, logs, and explicit exit-code files:

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage_name>.log; \
   code=${PIPESTATUS[0]}; echo "$code" > logs/<stage_name>.exitcode; exit "$code"'
```

## 10. Data and split safety

Create fresh manifests for this campaign. Historical `analysis_500`, `final_test_500`, and `final_test_full` remain untouched.

Locked historical manifests may be read only for ID/fingerprint exclusion. Do not read their prompt contents, outputs, labels, or metrics for training or selection.

For every edit:

```text
training anchors and evaluation prompts must be disjoint
same-subject evaluation prompts must not be used as training anchors
evaluation bucket labels must never become model features
```

## 11. Runtime leakage rules

Allowed runtime/edit-time inputs include:

```text
edit tuple
current hidden state
base logits/log probabilities
candidate embeddings/ranks
partial-mask state
training-only statistics fitted from allowed data
pre-edit risk features
```

Forbidden runtime inputs include:

```text
evaluation bucket identity
prompt_type or negative_type labels
case IDs as learned features
teacher scores unless they are explicit training labels only
final outcomes
future decoded outputs
analysis/final results
```

Every learned component must serialize its input feature schema and pass a leakage audit.

## 12. Scientific rescues

Each candidate receives at most one bounded rescue exactly as defined in its plan. No rescue may:

```text
lower hard thresholds
add evaluation prompts to training
add forbidden runtime features
open locked historical splits
expand the model family beyond the predeclared rescue
```

## 13. Final selection rule

The final recommendation must follow this priority order:

```text
1. confirmed full-editor success
2. confirmed selective-safe-editor success
3. confirmed efficacy-matched locality Pareto improvement
4. confirmed multi-token coupling success
5. confirmed mechanism-only result
6. no promising next direction
```

Within the same class, rank by paired evidence, same-subject safety, robustness, compute, and implementation risk.

Codex must not pick a preferred method based on intuition if the predeclared evidence ranks another method higher.

## 14. Campaign completion

The campaign ends only after writing and validating:

```text
runs/diffusion_editor_next_direction_selection_v1/final_direction_selection_package_v1/
  report_summary.json
  direction_selection_matrix.csv
  track_results.csv
  paired_bootstrap.csv
  efficacy_locality_pareto.png
  coverage_risk_plot.png
  multi_token_results.csv
  artifact_availability_manifest.json
  reproducibility_manifest.json
  final_research_report.md
  next_direction_recommendation.md
  SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md
  terminal_package_validation.json
```

`SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md` is a draft only. Do not execute it.

After the final package validates, stop the Pod and report the result.
