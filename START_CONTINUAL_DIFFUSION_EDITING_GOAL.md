# Codex Goal: Continual Learning for Diffusion-LM Factual Editing

Read, in order:

```text
AGENTS.md
ACTIVE_RESEARCH_CAMPAIGN.json
CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md
CANDIDATE_DIRECTION_REGISTRY.json
PRIMARY_SOURCES.md
all relevant per-track plans
```

Resume from the current repository state and execute the complete autonomous campaign `continual_diffusion_editing_sb_selection_v1`.

Requirements:

1. Do not ask for per-stage, per-command, or per-GPU-job approval.
2. Start the configured existing RunPod Pod if stopped.
3. Keep the Pod running through every mandatory pilot, confirmation, CPU analysis, GPU job, and final report.
4. Stop the Pod only after the final package validates, or after an unrecoverable Pod/infrastructure/data-integrity issue remains after the permitted retries.
5. Preserve all historical protocols as immutable evidence.
6. Use fresh sequential-edit streams and keep historical analysis/final splits closed.
7. Test every mandatory track C0-C9 breadth-first before scaling any track.
8. Use only the bounded rescues defined in the plans.
9. Do not lower thresholds, replay evaluation prompts, use teacher-only fields as runtime inputs, or invent new experiments.
10. Confirm every eligible track on a fresh untouched stream.
11. Trigger conditional tracks only under their predeclared conditions.
12. Select one next direction or `no_promising_continual_direction`.
13. Generate and validate the complete final package, preserve artifact hashes, update terminal campaign state, and stop the Pod.
14. Do not automatically execute the selected direction's full follow-up campaign; write only the draft protocol.

The final response must report:

```text
status of every track
sequential retention and forgetting results
same-subject and base-denoiser retention
SB-specific comparisons
compute/storage scaling
selected next direction or no-promising result
artifact locations
test counts
Pod stopped status
```
