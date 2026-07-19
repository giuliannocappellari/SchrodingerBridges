# Codex Goal Prompt

Use Codex Goal mode from the repository root:

```text
/goal
```

Then paste:

```text
Read AGENTS.md, ACTIVE_RESEARCH_CAMPAIGN.json,
CANDIDATE_DIRECTION_REGISTRY.json, and
NEXT_DIRECTION_SELECTION_AUTONOMOUS_PLAN.md, followed by every candidate plan.

Execute the complete diffusion-editor next-direction selection campaign from the
current repository state without asking for per-stage, per-command, or
per-GPU-job approval.

Start the configured existing RunPod Pod if it is stopped. Keep it running
through every mandatory candidate pilot, bounded rescue, conditional integration,
fresh confirmation, statistics, and final reporting. Do not stop it when an
individual stage or track finishes.

Test every mandatory direction N1 through N5 before final selection. Run N6 only
when its predeclared trigger is satisfied. Preserve every historical protocol as
immutable evidence. Create fresh manifests and do not use historical analysis_500,
final_test_500, or final_test_full data for training, tuning, or selection.

Obey all train/evaluation separation rules, runtime-feature leakage rules,
bounded grids, rescue limits, and acceptance criteria. Do not lower thresholds,
invent new directions, or silently redefine a failed result.

The goal is to select the single strongest next research direction, not to execute
its full research campaign. After confirmation, produce and validate the complete
final_direction_selection_package_v1, including a ranked selection matrix,
scientific interpretation, next_direction_recommendation.md, and a draft-only
SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md.

Return only after the campaign is terminal, the final package validates, all
mandatory tracks have a terminal status, and the RunPod Pod is stopped. Stop early
only for an unrecoverable Pod/infrastructure issue or unsafe data-integrity failure
after the permitted retries.
```
