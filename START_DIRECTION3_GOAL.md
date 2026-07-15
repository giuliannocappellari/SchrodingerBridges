# Start Direction 3 Autonomous Goal

## Required environment

```bash
export D3_AUTONOMOUS_MODE=1
export D3_AUTONOMOUS_BUDGET_USD="<total authorized budget>"
export RUNPOD_HOURLY_RATE_USD="<current hourly rate>"
export D3_AUTONOMOUS_BUDGET_RESERVE_USD="5"
export D3_AUTONOMOUS_MAX_INFRA_RETRIES="3"
export D3_AUTONOMOUS_MAX_SCIENTIFIC_RESCUES_PER_STAGE="1"

export RUNPOD_POD_ID="<existing pod id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/run_pod"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current host>"
export RUNPOD_SSH_PORT="<current port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

## Codex command

Use Goal mode:

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
