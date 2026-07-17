# Goal-mode launch prompt

Paste the following into Codex Goal mode from the repository root:

```text
Read AGENTS.md, ACTIVE_RESEARCH_CAMPAIGN.json,
EXPERIMENT_PROTOCOL_REGISTRY.json, and
DIFFUSION_NATIVE_PARAMETRIC_EDITOR_AUTONOMOUS_PLAN.md.

Resume or start the active autonomous campaign:
  diffusion_native_causal_partial_state_editor_v1

Research goal:
Determine whether a diffusion-native parametric editor can preserve locality
by optimizing edits over causal internal representations and partial denoising
states.

Execute every remaining mandatory stage and every explicitly triggered bounded
rescue without requesting per-stage, per-command, or per-GPU-job approval.

Start the configured existing RunPod Pod if it is stopped. Keep the Pod running
between all implementation, CPU, GPU, causal tracing, editing, evaluation,
bootstrap, second-backbone, locked confirmation, and reporting stages.

Do not stop the Pod because a job finishes, a stage is CPU-only, a method fails,
or estimated monetary cost is high. Monetary budget guards are disabled.

Stop the Pod only after:
1. the complete positive or formal negative research package validates; or
2. an unrecoverable Pod/infrastructure or unsafe data-integrity failure remains
   after the allowed retries and a terminal checkpoint validates.

Preserve every historical protocol as read-only evidence. Do not resume or
modify closed historical campaigns.

Obey all fresh split rules, train/evaluation separation, causal-localization
rules, partial-state rules, null-space/locality constraints, feature-leakage
rules, bounded rescues, analysis/final locks, and hard acceptance criteria.

Do not lower thresholds after seeing results. Do not invent new method families
or expand grids beyond the plan. Do not apply closed-form edits directly to
quantized weights. Do not tune on analysis_500 or final_test_500.

Maintain:
  runs/diffusion_native_causal_partial_state_editor_v1/
    autonomous_campaign_v1/campaign_state.json
    autonomous_campaign_v1/stage_history.csv
    autonomous_campaign_v1/autonomous_log.md
    autonomous_campaign_v1/cost_state.json

Return only after the campaign is terminal, the final or stop package passes
validation, all required hashes and artifact-availability records exist, and
the Pod is stopped.
```
