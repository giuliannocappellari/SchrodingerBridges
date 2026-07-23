# Continual Diffusion Editing Goal Bundle

This bundle launches an autonomous bounded selection campaign to test whether continual-learning methods can solve sequential factual editing in masked diffusion LMs.

## Central new source

The campaign is anchored in **DiffusionGrow**, a recent continual-learning proposal for diffusion language models that preserves an explicit frozen pretrained path and adds timestep-conditioned trainable branches.

## Why this campaign differs from earlier editing campaigns

Earlier work focused mostly on one edit or one fixed batch. This campaign evaluates:

```text
sequential acquisition
retention of earlier edits
forgetting after every block
preservation of the original denoiser
same-subject locality
storage and compute growth
```

## Files

```text
AGENTS.md
ACTIVE_RESEARCH_CAMPAIGN.json
CANDIDATE_DIRECTION_REGISTRY.json
CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md
CONTINUAL_LEARNING_ALTERNATIVES_CATALOG.md
DIFFUSIONGROW_CONTINUAL_EDITING_PLAN.md
PARTIAL_STATE_REPLAY_PLAN.md
SPARSE_MEMORY_ROUTING_PLAN.md
GATED_ADAPTER_EXPANSION_PLAN.md
ORTHOGONAL_FISHER_CONTINUAL_PLAN.md
FUNCTIONAL_REPLAY_PLAN.md
BRIDGE_GENERATIVE_REPLAY_PLAN.md
MULTIMARGINAL_SB_CONSOLIDATION_PLAN.md
DUAL_MEMORY_CONSOLIDATION_PLAN.md
FINAL_SELECTION_AND_REPORTING_PLAN.md
PRIMARY_SOURCES.md
START_CONTINUAL_DIFFUSION_EDITING_GOAL.md
```

## Environment

```bash
export CL_DLLM_AUTONOMOUS_MODE=1
export CL_DLLM_MAX_INFRA_RETRIES="3"
export CL_DLLM_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"

export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

## Launch

Use Codex Goal mode and paste `START_CONTINUAL_DIFFUSION_EDITING_GOAL.md`.

The Pod remains running until the final validated selection package is complete or an unrecoverable infrastructure issue remains.
