# Partial-State Temporal Residual Editor Autonomous Campaign

This bundle launches a fresh autonomous campaign for the question:

> Can a temporally localized residual editor preserve factual-editing locality when its target deltas and preservation constraints are optimized across partial denoising states?

## Files

```text
AGENTS.md
ACTIVE_RESEARCH_CAMPAIGN.json
EXPERIMENT_PROTOCOL_REGISTRY.json
PARTIAL_STATE_TEMPORAL_RESIDUAL_EDITOR_AUTONOMOUS_PLAN.md
START_PARTIAL_STATE_TEMPORAL_RESIDUAL_EDITOR_GOAL.md
PRIMARY_SOURCES.md
TIMEROME_REPRODUCTION_PLAN.md
COUNTERFACT_TEMPORAL_RESIDUAL_ADAPTATION_PLAN.md
PARTIAL_STATE_TARGET_DELTA_PLAN.md
STATE_CONDITIONED_LOCALITY_PLAN.md
MAIN_EDITOR_AND_BASELINES_PLAN.md
LOCKED_CONFIRMATION_PLAN.md
SECOND_BACKBONE_AND_SCALING_PLAN.md
PAPER_REPRODUCIBILITY_PLAN.md
BUNDLE_MANIFEST.json
```

## Core method

The base MDLM remains frozen. A temporally localized low-rank residual memory is applied during each diffusion forward pass. The campaign tests whether:

```text
partial-state target-delta construction
+ early/middle/late state conditioning
+ same-subject preservation anchors
+ sparse low-rank residual memory
```

moves the efficacy-locality Pareto frontier beyond MDM-MEMIT, static null-space projection, and ordinary temporal residual editing.

## Launch

Configure RunPod variables, set:

```bash
export PS_TRM_AUTONOMOUS_MODE=1
```

Use Codex Goal mode and paste `START_PARTIAL_STATE_TEMPORAL_RESIDUAL_EDITOR_GOAL.md`.

There is no monetary budget guard. The Pod remains running until the campaign and final package are terminal, or an unrecoverable infrastructure/data-integrity failure occurs.
