# Start the Autonomous Schrödinger-Bridge Alternatives Campaign

Use Codex **Goal mode**. The research scope, stage order, acceptance criteria, bounded rescues, and terminal conditions are already specified.

## Environment

```bash
export SB_ALT_AUTONOMOUS_MODE=1
export SB_ALT_AUTONOMOUS_BUDGET_USD="<total authorized budget>"
export RUNPOD_HOURLY_RATE_USD="<current hourly rate>"
export SB_ALT_AUTONOMOUS_BUDGET_RESERVE_USD="5"
export SB_ALT_AUTONOMOUS_MAX_INFRA_RETRIES="3"
export SB_ALT_AUTONOMOUS_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"
export SB_ALT_MIN_UNTESTED_TRACK_RESERVE_USD="8"

export RUNPOD_POD_ID="<existing pod id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current host>"
export RUNPOD_SSH_PORT="<current port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

Ensure `runpodctl` is already configured with the RunPod API key.

## Goal-mode prompt

Open Codex in the repository, enter `/goal`, and paste:

```text
Read AGENTS.md, ACTIVE_RESEARCH_CAMPAIGN.json,
ALTERNATIVE_PROTOCOL_REGISTRY.json,
SB_ALTERNATIVES_AUTONOMOUS_RESEARCH_PLAN.md, and every referenced per-track
plan in the repository root.

Run the complete counterfact_sb_alternatives_campaign_v1 autonomous campaign.
Test all five mandatory alternatives in the prescribed breadth-first order:

1. learned edit-intent gate + raw bridge
2. activation-space Schrödinger bridge
3. conditional answer-span CSBM
4. unbalanced/partial CSBM
5. parameter-space Schrödinger bridge

Do not ask for per-stage, per-command, per-track, or per-GPU-job approval.
Start the configured existing RunPod Pod if it is stopped and keep it running
between every planned CPU and GPU stage and between all five alternatives.

An individual track failure must produce its formal track stop package, after
which the campaign must continue to the next untested track. Do not stop the Pod
after an individual track succeeds or fails.

Use the mandatory breadth-first budget rule: reserve enough compute for the
minimum pilot of every untested track before scaling an early promising track.
Maintain campaign_state.json, budget_state.json, track_registry.csv,
stage_history.csv, and autonomous_log.md.

After every track has a terminal pilot status, scale every pilot-passed track
that the remaining budget can support, run the common dev_tune_200 comparison,
and freeze the primary candidate before analysis. Run analysis_500 only under
the validated dev lock. Run final_test_500 exactly once only if the preselected
primary passes analysis.

Obey all split locks, feature-leakage rules, real-prompt provenance rules,
acceptance criteria, bounded rescues, historical immutability rules, and budget
guards. Never lower thresholds, invent experiments, add teacher-derived runtime
features, reuse evaluation prompts for training, silently change target-length
scope, tune on analysis_500, or rerun final_test_500 for tuning.

Stop the Pod only when every track and the final cross-track package are
complete, or the authorized budget is insufficient, or an unrecoverable
infrastructure/data-integrity failure remains after allowed retries.

Before stopping, validate the final or partial campaign package, preserve
compact summaries and reproducibility manifests, commit/push code and terminal
summaries, verify that no tmux/Python/GPU job remains active, stop the Pod, and
return one complete report containing the outcome of every alternative,
cross-track comparisons, total spend, strongest defensible claim, limitations,
and next recommendation.
```
