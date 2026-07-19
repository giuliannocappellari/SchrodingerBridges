# Next-Direction Selection Autonomous Plan

Protocol: `diffusion_editor_next_direction_selection_v1`

## Goal

Run bounded, comparable pilots for five scientifically motivated directions and select the single strongest next research direction. The campaign is not allowed to continue automatically into the selected full research program.

The existing evidence shows three recurring facts:

```text
editing efficacy is achievable
partial denoising states matter for some permanent editors
same-subject relation locality remains the central failure mode
```

This campaign asks which remaining statistical/control formulation best addresses that gap.

---

# Phase S0 — Bootstrap, source audit, and fresh data

## S0.1 Campaign state

Create:

```text
runs/diffusion_editor_next_direction_selection_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  track_registry.json
  infrastructure_events.csv
```

Acceptance:

```text
autonomous_mode = true
all mandatory plan files exist
historical protocols marked immutable
analysis_500_used = false
final_test_used = false
```

## S0.2 Pod preflight

Start the configured Pod once. Verify Git, tests, GPU, persistent `/workspace`, and model access. Keep the Pod running until terminal completion.

## S0.3 Fresh manifests

Create fresh CounterFact manifests from allowed train rows after excluding every historical development/evaluation fingerprint:

```text
cf_nds_statistics_train_500
cf_nds_calibration_200
cf_nds_smoke_20
cf_nds_pilot_100
cf_nds_confirmation_200
```

Primary CounterFact screening may focus on one-token targets because the scientific target is locality, not multi-token realization.

Create fresh KAMEL manifests for N5:

```text
kamel_nds_train_200_per_length
kamel_nds_calibration_100_per_length
kamel_nds_pilot_100_per_length
kamel_nds_confirmation_200_per_length
```

for target lengths 2, 3, and 4.

Hard checks:

```text
all manifests deterministic
zero cross-split edit overlap
zero historical fingerprint overlap
all prompt IDs unique
training/evaluation prompt separation validated
historical analysis/final contents not read
```

Failure: one split-builder repair is allowed. If legal fresh manifests cannot be built, write a protocol-infeasibility package and stop.

---

# Phase S1 — Common baseline and diagnostics

## S1.1 Baseline editor

Reproduce on fresh CounterFact:

```text
base
ordinary MDM-MEMIT
partial-state MDM-MEMIT
best historical-style temporal residual, diagnostic only
static null-space partial-state editor, diagnostic only
```

Primary base editor for candidate modifications:

```text
partial-state MDM-MEMIT
```

Baseline floor on `cf_nds_pilot_100`:

```text
rewrite exact >= 0.75
paraphrase exact >= 0.40
pre-edit target-new rewrite <= 0.10
malformed <= 0.05
```

One source-compatible repair is allowed. If the baseline still fails, the direction-selection campaign is invalid and must terminate.

## S1.2 Shared measurements

Cache for every allowed training/calibration edit:

```text
subject/relation/state representations
edit gradients
same-subject gradients
near/far gradients
base target rank and margin
causal-site stability
partial-state statistics
protected-prompt empirical Fisher sketches
```

These are training/diagnostic assets only. Evaluation outcomes cannot become runtime features.

---

# Phase S2 — Mandatory breadth-first pilots

Every N1–N5 track receives its minimum pilot before N6 or final selection.

For each track:

```text
1. implement fake/unit tests
2. tune only on statistics_train/calibration
3. freeze one candidate
4. run smoke20 for integration
5. run pilot100 (or KAMEL pilot for N5)
6. apply at most one bounded rescue
7. write a track terminal report
8. continue to the next track
```

Common CounterFact metrics:

```text
rewrite exact
paraphrase exact
same-subject TFPR
near/far TFPR
self-normalized locality
distributional KL/JS
malformed rate
update norm / intervention energy
GPU minutes/edit
```

Common paired inference:

```text
bootstrap by edit_id
10,000 resamples for final confirmation
95% confidence intervals
```

---

# Phase S3 — Conditional integration

Run N6 only after N1–N5 pilots are terminal.

Trigger N6 if at least one holds:

```text
N1 improves subject-relation separation or locality
N2 improves Fisher/protected KL geometry
N3 improves exact constraint satisfaction
N4 provides a valid selective-safe operating point
```

N6 may combine only components that passed their own mechanism gate. It may not invent new architectures.

---

# Phase S4 — Fresh confirmation of nominated tracks

Every track that passes its pilot may nominate one frozen candidate.

Run nominated CounterFact candidates once on:

```text
cf_nds_confirmation_200
```

Run N5 once on:

```text
kamel_nds_confirmation_200_per_length
```

No tuning on confirmation.

Confirmation requires the pilot direction and paired evidence to persist. A candidate that fails confirmation cannot be selected.

---

# Phase S5 — Final selection

## Success classes

### Class A — Full editor

```text
rewrite >= 0.85
paraphrase >= 0.45
same-subject TFPR <= 0.03
near/far budgets pass
malformed <= 0.05
```

### Class B — Selective safe editor

```text
coverage >= 0.50
accepted rewrite >= 0.85
accepted paraphrase >= 0.45
accepted same-subject TFPR <= 0.03
95% upper risk bound <= 0.05
```

Strong selective result:

```text
coverage >= 0.60
95% upper risk bound <= 0.03
```

### Class C — Efficacy-matched locality Pareto improvement

Relative to the strongest efficacy-matched baseline:

```text
rewrite loss <= 0.02
paraphrase loss <= 0.02
same-subject TFPR reduction >= 25%
protected distributional KL reduction >= 20%
paired TFPR confidence interval below 0
```

### Class D — Multi-token coupling result

```text
full-span exact improvement >= 0.10 on at least two target lengths
pooled paired lower bound > 0
malformed <= 0.05
no material locality regression
```

### Class E — Mechanism-only result

A mechanism-only direction may be recommended only when no A–D result exists and the mechanism passes its own predeclared diagnostic threshold with confirmation.

## Ranking

Rank candidates in this order:

```text
A > B > C > D > E > no promising direction
```

Within class:

```text
paired evidence
same-subject safety
robustness across relations/lengths
compute/storage
implementation risk
```

## Final recommendation statuses

```text
pursue_relation_residualized_editor
pursue_fisher_constrained_editor
pursue_primal_dual_editor
pursue_selective_conformal_editor
pursue_joint_span_coupled_editor
pursue_integrated_statistical_editor
no_promising_next_direction
protocol_infeasible
infrastructure_blocked
```

---

# Phase S6 — Final package and shutdown

Create:

```text
runs/diffusion_editor_next_direction_selection_v1/final_direction_selection_package_v1/
```

Required artifacts:

```text
report_summary.json
direction_selection_matrix.csv
track_results.csv
paired_bootstrap.csv
efficacy_locality_pareto.png
coverage_risk_plot.png
multi_token_results.csv
failure_taxonomy.csv
artifact_availability_manifest.json
reproducibility_manifest.json
final_research_report.md
next_direction_recommendation.md
SELECTED_DIRECTION_FULL_CAMPAIGN_DRAFT.md
terminal_package_validation.json
```

The draft full campaign must contain the selected hypothesis, datasets, baselines, acceptance criteria, estimated stages, and stop rules. Do not execute it.

After validation, mark the campaign terminal and stop the Pod.
