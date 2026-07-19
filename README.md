# Diffusion Editor Next-Direction Selection Goal

This bundle instructs Codex to run a bounded autonomous screening campaign over five remaining statistically and physically motivated directions:

```text
N1 relation residualization
N2 Fisher-constrained editing
N3 primal-dual locality constraints
N4 selective conformal editing
N5 joint answer-span coupling
```

A conditional N6 integration stage may combine only components that pass their own mechanism gates.

The campaign does not execute the final selected direction at full scale. It ends after fresh confirmation, ranked selection, and generation of a draft full-campaign plan.

## Installation

Copy every file in this bundle to the repository root, replacing the active campaign control files.

## Environment

```bash
export NEXT_DIRECTION_AUTONOMOUS_MODE=1
export NEXT_DIRECTION_MAX_INFRA_RETRIES=3
export NEXT_DIRECTION_MAX_SCIENTIFIC_RESCUES_PER_TRACK=1

export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

Ensure `runpodctl` is configured with the RunPod API key.

## Launch

Use Goal mode and paste `START_NEXT_DIRECTION_SELECTION_GOAL.md`.
