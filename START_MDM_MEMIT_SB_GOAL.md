# Codex Goal Prompt

Use Goal mode and paste the text below.

```text
Read and follow, in order:
1. AGENTS.md
2. ACTIVE_RESEARCH_CAMPAIGN.json
3. EXPERIMENT_PROTOCOL_REGISTRY.json
4. MDM_MEMIT_SB_AUTONOMOUS_RESEARCH_PLAN.md
5. the active track plan for each stage
6. persisted campaign state under runs/

Resume or start the autonomous campaign:
masked_diffusion_memit_sb_positive_result_v1

Execute every remaining planned stage without asking for per-stage,
per-command, or per-GPU-job approval.

Start the configured existing RunPod Pod if it is stopped. Keep the Pod running
through all CPU work, GPU work, tests, dataset construction, editing, evaluation,
failed tracks, successful tracks, and reporting. Do not stop the Pod because a
job or individual track finished, the next task is CPU-only, the Pod is briefly
idle, or estimated monetary cost is high.

Stop the Pod only after:
1. the entire campaign reaches a validated terminal scientific outcome and the
   final cross-track package is durable; or
2. an unrecoverable Pod/infrastructure issue remains after the allowed retries.

Monetary budget is informational only and must not block, reorder, skip, or stop
scientific work.

Execute the mandatory tracks in order:
M1 MDM-MEMIT reproduction
M2 partial-mask MDM-MEMIT
M3 Schrodinger/path-KL-regularized MEMIT
M4 exact mask-pattern Schrodinger bridge

Run F1 only if M1 fails after its bounded rescue.
Run F2 only if M3 and M4 both fail to establish an SB-specific positive result.

Preserve all historical Direction 1, Direction 2, Direction 3, and SB-alternative
campaigns as immutable evidence. Build fresh campaign manifests and do not open
or tune on old analysis_500, final_test_500, or final_test_full splits.

Use GSAI-ML/LLaDA-8B-Instruct for the primary MDM-MEMIT reproduction. Do not
apply MEMIT updates directly to quantized 4-bit weights. Follow the paper-matched
masked-input adaptation, target-value optimization, layer selection, and
partial-mask correction defined in the plans.

Maintain:
- campaign_state.json
- stage_history.csv
- autonomous_log.md
- artifact_availability.json
- informational cost_state.json

For every stage:
- run preflight and tests;
- use versioned outputs;
- capture logs and explicit exit codes;
- validate every acceptance criterion;
- use only the single bounded rescue written for that stage;
- write a formal track stop package on bounded failure;
- continue to the next required track.

Do not lower thresholds, add evaluation prompts to optimization, silently change
datasets or target-length definitions, force target tokens in ranked methods,
invent unplanned experiments, or claim an SB-specific contribution when a direct
or non-SB ablation performs equally or better.

Return to the user only after all mandatory tracks and triggered fallbacks are
terminal, the final research package validates, compact results are durable,
campaign state is terminal, and the Pod has been stopped.
```
