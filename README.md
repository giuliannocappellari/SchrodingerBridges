# Diffusion-Native Parametric Editor Autonomous Goal Bundle

This bundle starts a new autonomous research campaign:

```text
diffusion_native_causal_partial_state_editor_v1
```

Research question:

> Can a diffusion-native parametric editor preserve locality by optimizing updates over causal internal representations and partial denoising states?

## Files

```text
AGENTS.md
ACTIVE_RESEARCH_CAMPAIGN.json
EXPERIMENT_PROTOCOL_REGISTRY.json
DIFFUSION_NATIVE_PARAMETRIC_EDITOR_AUTONOMOUS_PLAN.md
START_DIFFUSION_NATIVE_PARAMETRIC_EDITOR_GOAL.md
PRIMARY_SOURCES.md
CAUSAL_LOCALIZATION_PLAN.md
PARTIAL_STATE_TARGET_OPTIMIZATION_PLAN.md
NULL_SPACE_LOCALITY_PLAN.md
MAIN_EDITOR_AND_BASELINES_PLAN.md
LOCKED_CONFIRMATION_PLAN.md
SECOND_BACKBONE_AND_SCALING_PLAN.md
PAPER_REPRODUCIBILITY_PLAN.md
BUNDLE_MANIFEST.json
```

Extract/copy the files directly into the repository root. They replace the previous root campaign-control files but do not modify historical run artifacts.

## Autonomous policy

- No per-stage approval requests.
- No monetary budget guard.
- Use the existing RunPod Pod.
- Keep the Pod running between all CPU/GPU stages.
- Stop only after terminal positive/negative completion or unrecoverable infrastructure/data-integrity failure.
- Preserve historical campaigns as read-only evidence.
- Do not open `analysis_500` or `final_test_500` before the new lock files exist.

## Environment

```bash
export DNPE_AUTONOMOUS_MODE=1
export DNPE_MAX_INFRA_RETRIES="3"
export DNPE_MAX_SCIENTIFIC_RESCUES_PER_STAGE="1"

export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

Ensure `runpodctl` has a configured RunPod API key.

## Start

Use Codex Goal mode:

```text
/goal
```

Then paste the contents of `START_DIFFUSION_NATIVE_PARAMETRIC_EDITOR_GOAL.md`.
