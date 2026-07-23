# AGENTS.md

Operational and scientific rules for the autonomous continual-learning campaign in the LLaDA factual-editing repository.

## 0. Active project identity

```text
active_protocol = continual_diffusion_editing_sb_selection_v1
campaign_type = bounded next-direction selection
primary_backbone = GSAI-ML/LLaDA-8B-Instruct
secondary_backbone = GSAI-ML/LLaDA-8B-Base only after a confirmed primary result
base_backbone_parameters = frozen unless a baseline explicitly requires otherwise
main_new_source = DiffusionGrow: Continual Learning for Diffusion Language Models
```

All previous Direction 1, Direction 2, Direction 3, MDM-MEMIT, mask-pattern-control, parametric-editor, temporal-residual, and next-direction-selection protocols are immutable historical evidence. They may be read for code reuse, exclusions, baseline summaries, and failure analysis, but must not be overwritten or resumed.

The goal is to determine whether continual-learning mechanisms can turn diffusion-LM factual editing into a sequentially stable editor that acquires new facts, retains earlier edits, preserves the original denoiser, and controls same-subject side effects.

## 1. Authoritative files

Codex must read these files in order:

```text
1. AGENTS.md
2. ACTIVE_RESEARCH_CAMPAIGN.json
3. CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md
4. CANDIDATE_DIRECTION_REGISTRY.json
5. the relevant per-track plan
6. persisted campaign state under runs/
```

The root campaign file selects the active protocol. Persisted state under `runs/continual_diffusion_editing_sb_selection_v1/autonomous_campaign_v1/` is authoritative for completed stages.

## 2. Autonomous approval

Autonomous execution is enabled only when:

```bash
export CL_DLLM_AUTONOMOUS_MODE=1
```

Recommended infrastructure variables:

```bash
export CL_DLLM_MAX_INFRA_RETRIES="3"
export CL_DLLM_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"

export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

When autonomous mode is enabled, Codex has one-time approval to execute every task explicitly listed in the master plan without asking for per-stage, per-command, or per-GPU-job approval.

Codex must not:

- lower scientific thresholds after seeing results;
- invent unplanned rescues;
- use evaluation prompts as replay/training examples;
- route teacher-only labels into runtime features;
- open historical `analysis_500`, `final_test_500`, or `final_test_full`;
- create a full follow-up campaign before the selection package is terminal;
- terminate or delete the RunPod resource.

## 3. Pod lifecycle

There is no monetary budget guard.

Codex must:

1. start the configured existing Pod if stopped;
2. use `/workspace/SB` as the authoritative campaign worktree;
3. keep the Pod running through all mandatory pilots, CPU analyses, GPU jobs, confirmation, and reporting;
4. continue after an individual track fails;
5. stop the Pod only after the final selection package validates, or after an unrecoverable Pod/infrastructure/data-integrity failure remains after the allowed retries.

Do not stop the Pod merely because:

```text
one track failed
one track passed
one GPU job finished
the next task is CPU-only
the Pod is temporarily idle between stages
```

## 4. Local and remote Python

Local MacBook:

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script>.py
```

RunPod:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script>.py
```

Do not require `uv` on RunPod. Do not use `pip install` directly in the local project environment.

## 5. Campaign state

Create and maintain:

```text
runs/continual_diffusion_editing_sb_selection_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  infrastructure_state.json
  optional_cost_state.json
```

The campaign state must include:

```json
{
  "protocol_version": "continual_diffusion_editing_sb_selection_v1",
  "campaign_status": "running",
  "current_stage": "",
  "next_stage": "",
  "completed_stages": [],
  "failed_stages": [],
  "rescues_used": {},
  "analysis_500_used": false,
  "final_test_used": false,
  "pod_status": ""
}
```

Cost is informational only and cannot block execution.

## 6. Long-job discipline

Use `tmux` and explicit exit-code files:

```bash
tmux new -d -s "<stage>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage>.log; \
   code=${PIPESTATUS[0]}; echo "$code" > logs/<stage>.exitcode; exit "$code"'
```

Every stage writes:

```text
report_summary.json
run_config.json or equivalent
validation report
log
exit-code file for long jobs
```

Never overwrite a completed output directory. Use versioned directories.

## 7. Fresh-data and split rules

Create fresh sequential-edit streams. Historical manifests may be read only for source-ID/fingerprint exclusion.

Forbidden for training, tuning, selection, or confirmation:

```text
historical analysis_500
historical final_test_500
historical final_test_full
historical held-out prompts
historical same-subject evaluation prompts
```

For each edit:

```text
train_prompt_ids ∩ eval_prompt_ids = ∅
```

Allowed training/replay data:

```text
rewrite prompt
edit tuple
training-only paraphrase augmentations
training-only same-subject different-relation anchors
training-only near/far/unrelated anchors
partial-mask states derived from allowed training prompts
stored old-model logits or generated replay from allowed training prompts
```

Evaluation-only:

```text
official held-out paraphrases
held-out same-subject prompts
held-out near/far prompts
held-out generation/attribute prompts
confirmation stream outcomes
```

## 8. Continual-learning definitions

The campaign must separately measure:

```text
plasticity:
  acquisition of the newest edit block

edit retention:
  performance on all earlier edited facts

base retention:
  preservation of the original denoiser and general capabilities

locality:
  same-subject, near, far, and distributional side effects

forgetting:
  historical best performance minus current performance for each old block

backward transfer:
  change in previous-block performance after learning later blocks

forward transfer:
  effect of previous learning on new-block acquisition
```

Evaluate after every edit block, not only at the end.

## 9. Runtime leakage rules

Runtime inputs may use:

```text
current hidden states
base logits/log-probabilities
timestep/mask ratio
candidate token embeddings
edit request
branch/adaptor parameters
routing keys derived from prompt/edit text
```

Teacher/replay labels may not be runtime features:

```text
future success
final decoded outcome
evaluation bucket
prompt_type
negative_type
teacher score arrays
post-edit locality labels
case ID
split role
```

Every checkpoint must serialize its runtime feature schema and pass a leakage audit.

## 10. Breadth-first rule

Mandatory tracks C1-C9 must all receive their bounded pilot before any track is scaled.

A failed track must:

```text
write a formal track stop package
update the registry
continue to the next mandatory track
```

An integrated candidate may combine only components that independently passed their mechanism criteria.

## 11. Scientific claim classes

### Class A — Full continual editor

```text
new-block rewrite >= 0.80
new-block paraphrase >= 0.45
past-edit retention >= 0.75 after 200 edits
average forgetting <= 0.10
same-subject TFPR <= 0.03
near/far budgets pass
base denoising retention-loss increase <= 5%
malformed <= 0.05
```

### Class B — Retention/locality Pareto result

At matched current-block efficacy within 0.03:

```text
average forgetting reduced >= 30%
past-edit retention improves >= 0.10
protected KL reduced >= 20%
same-subject TFPR does not worsen
paired evidence favors the method
```

### Class C — SB-specific continual-learning result

At matched replay memory and compute:

```text
bridge replay or SB consolidation improves past-edit retention >= 0.05
or reduces average forgetting >= 25%
or reaches equal retention with >= 25% less replay storage
```

and it must beat the closest non-SB replay/consolidation baseline.

### Class D — Efficiency/scaling result

```text
storage <= 1 MB per edit or sublinear growth
inference overhead <= 25% over base
retention remains competitive with the best pilot
```

## 12. Bounded rescues

Allowed:

```text
one source-integration repair for DiffusionGrow
one implementation repair per track
one scientific rescue per mandatory track
one confirmation-only infrastructure rerun before results are inspected
```

Forbidden:

```text
threshold relaxation
new feature families after pilot failure
larger hidden models as rescue
evaluation-data replay
unbounded grids
post-confirmation retuning
```

## 13. Final completion

The campaign is terminal when:

1. all mandatory tracks are terminal;
2. eligible tracks have been confirmed on a fresh stream;
3. the integrated candidate, if triggered, is terminal;
4. one final recommendation or `no_promising_continual_direction` is written;
5. the final package validates;
6. the Pod is stopped.

The campaign selects a next direction. It does not automatically launch the selected direction's full-scale research program.
