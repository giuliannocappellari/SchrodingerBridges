# Direction 3 Autonomous Stop Checkpoint

Status: **formal_negative_completion**

Direction 3 stopped before Stage 2A because the representation-aware offline controller failed hard scientific criteria after the one allowed bounded rescue. No `analysis_500`, `final_test_500`, or `final_test_full` tuning/evaluation was run.

## Stop Reason

`stage_1b4_1b5_offline_hard_criteria_failed_after_allowed_rescue`

The deployable feature cache and learned gate are usable, but the value controller does not satisfy the representation-use requirements. The decisive failures after rescue were:

- `value_top3_pass = false`: teacher top-3 overlap was 0.5263, below 0.65.
- `representation_beats_target_indicator_pass = false`: target-indicator-only Spearman was 0.9589, while full value Spearman was 0.5224.
- `state_shuffle_hurts_pass = false`: shuffling state representations did not reduce the primary value metric by the required margin.

## What Passed

- Feature-cache audit passed.
- Runtime feature leakage audit passed with zero leaked runtime fields.
- Gate ROC-AUC remained strong: 0.9877.
- Negative guidance ratio remained safely low: 0.000711.
- Same-subject target advantage over base remained non-positive: -10.4719.

## What Was Not Run

- Stage 2A actual D3 decoding was not run.
- `analysis_500` was not used.
- `final_test_500` and `final_test_full` were not used.

## Decision

The current Direction 3 top-k/value controller pilot is blocked under the autonomous acceptance criteria. The next scientifically honest branch is not to decode this controller; it is to redesign the teacher objective or move to the planned fallback direction.
