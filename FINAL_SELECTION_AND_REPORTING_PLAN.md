# Final Selection and Reporting Plan

## Track result schema

Each track must be classified as one of:

```text
confirmed_full_editor
confirmed_selective_safe_editor
confirmed_pareto_improvement
confirmed_multi_token_result
confirmed_mechanism_only
pilot_failed
confirmation_failed
protocol_infeasible
infrastructure_blocked
```

## Selection matrix columns

```text
track_id
candidate_id
pilot_status
confirmation_status
success_class
rewrite
paraphrase
same_subject_tfpr
near_tfpr
far_tfpr
distributional_locality_kl
coverage
risk_upper_bound
multi_token_exact_delta
paired_ci_low
paired_ci_high
gpu_minutes_per_edit
implementation_risk
recommended
```

## Final recommendation

`next_direction_recommendation.md` must include:

```text
selected direction or no-promising-direction decision
why it outranked the alternatives
which claims are supported
which claims are not supported
exact fresh evidence
implementation and compute requirements
main scientific risks
predeclared next full-campaign stages
conditions that would falsify the direction
```

## Draft only

Generate `SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md` for the selected direction. It must not be executed within this campaign.
