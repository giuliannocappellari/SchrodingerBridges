# Direction 3 Autonomous Campaign Log

- 2026-07-14T00:42:03.643898+00:00: initialized autonomous campaign on pod tlss3w46xk3wzj.

- 2026-07-14T00:43:02.558819+00:00: Stage 1B.4A found missing feature_index.jsonl; using allowed schema/index repair attempt.

- 2026-07-14T00:43:52.985314+00:00: Stage 1B.4A passed after feature-index repair. Starting v3 representation training/replay.

## 2026-07-14T01:01:43.482785+00:00 - Stage 1B.4 attempt 1 failed hard offline criteria
- Process exit status: 0.
- scientific_acceptance_pass: false.
- Failed hard criteria: value_top3_pass, representation_beats_target_indicator_pass, state_shuffle_hurts_pass.
- Action: launch bounded value rescue with proj_dim=256, negative_identity_weight=2.0, target_loss_weight=0.0.

## 2026-07-14T01:33:06.294401+00:00 - Formal negative completion
- Bounded rescue failed hard offline criteria.
- Stage 2A actual decode was not run.
- Stop checkpoint: runs/counterfact_direction3_controller_v1/direction3_autonomous_stop_checkpoint_v1

## 2026-07-15T00:28:04.278670+00:00 - Terminal package revalidation and pod shutdown
- Durable stop summaries passed internal consistency validation.
- Raw v3 train/replay directories from the old pod are unavailable for metric rederivation.
- No rescue or Stage 2A work was rerun.
- Replacement pod `dbsjvi6dzv1ew3` was confirmed idle and stopped.
