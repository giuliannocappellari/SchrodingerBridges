# Continual Diffusion Editing Autonomous Selection Plan

Protocol: `continual_diffusion_editing_sb_selection_v1`

## Research question

Can continual-learning mechanisms—especially function-preserving growth for diffusion LMs—support sequential factual editing while preserving earlier edits, the original denoiser, and same-subject locality? Can Schrödinger-bridge replay or consolidation improve this stability-plasticity trade-off beyond ordinary continual-learning baselines?

The campaign is a bounded breadth-first selection study. It does not assume that one track will work.

---

# Phase A — Source, repository, and campaign bootstrap

## A0 — Source audit

Audit and record the exact available paper/code/checkpoint status for:

```text
DiffusionGrow
MDM-MEMIT / partial-state editing
O-Edit
MEMOIR
Sparse Memory Finetuning
GainLoRA
FGGM
NuSA-CL
C-LoRA
LwF
GEM/A-GEM
Dark Experience Replay
CSBM and existing SB code
```

Classify each implementation as:

```text
official code
author checkpoint only
equation-level reimplementation
conceptual adaptation
unavailable
```

Do not call an equation-level reimplementation an exact reproduction.

### Output

```text
runs/continual_diffusion_editing_sb_selection_v1/A0_source_audit_v1/
```

### Acceptance

```text
all mandatory sources classified
model/checkpoint licenses recorded
base revisions frozen
no historical evaluation data used
```

## A1 — Campaign state and Pod

Create campaign state, start the configured Pod, verify GPU, repository, tests, persistent storage, and source artifacts. Keep the Pod running.

---

# Phase B — Fresh sequential-edit protocol

## B0 — Build fresh streams

Create fresh disjoint manifests after excluding all historical development/evaluation fingerprints.

### CounterFact single-token streams

```text
cf_cl_smoke_20:
  4 sequential blocks of 5 edits

cf_cl_pilot_100:
  10 blocks of 10 edits

cf_cl_confirmation_200:
  20 blocks of 10 edits

cf_cl_scale_500:
  10 blocks of 50 edits
```

### KAMEL multi-token stream

```text
kamel_cl_pilot_90:
  30 edits each for target lengths 2, 3, 4

kamel_cl_confirmation_180:
  60 edits each for lengths 2, 3, 4
```

### Retention and locality sets

```text
base_denoising_retention_500
same_subject_eval_heldout
near_locality_eval_heldout
far_locality_eval_heldout
general_generation_retention
```

### Training-only anchors

Create separate same-subject, relation, near/far, and unrelated anchors. They must not overlap held-out evaluation prompts.

### Acceptance

```text
all source IDs and fingerprints disjoint
historical analysis/final untouched
all target lengths present where required
same-subject held-out prompts available
stream order and seeds frozen
```

## B1 — Sequential evaluation harness

After every block, compute:

```text
new-block rewrite/paraphrase
all-past-edit rewrite/paraphrase
average retention
average forgetting
backward transfer
forward transfer
same-subject TFPR
near/far locality
base denoising loss across mask-ratio buckets
partial-state consistency
general generation retention
storage and compute
```

Bootstrap by edit ID.

---

# Phase C — Common baselines

## C0 — Reproduce base sequential editors

Required baselines:

```text
sequential partial-state MDM-MEMIT
sequential LoRA
sequential full-mask MDM-MEMIT
O-Edit-style sequential partial-state MEMIT
ordinary replay
LwF functional distillation
DiffusionGrow source-style domain-adaptation reproduction
```

The factual-edit baseline must achieve on fresh smoke/pilot:

```text
current-block rewrite >= 0.75
current-block paraphrase >= 0.40
pre-edit target-new rewrite <= 0.10
malformed <= 0.05
```

If no factual acquisition baseline can be reproduced after one implementation repair, terminate as `baseline_infeasible`.

DiffusionGrow reproduction is evaluated on its source-compatible adaptation setting first, then adapted to sequential factual edits.

---

# Phase D — Mandatory breadth-first pilots

Every track receives smoke20 and pilot100 before any track reaches confirmation.

## C1 — Function-preserving growth branches

See `DIFFUSIONGROW_CONTINUAL_EDITING_PLAN.md`.

Test:

```text
shared growth branch
block-specific branches
timestep-conditioned gates
zero-init exact identity
partial-state factual training
```

## C2 — Partial-state real and dark replay

See `PARTIAL_STATE_REPLAY_PLAN.md`.

Test:

```text
clean-prompt replay
random-mask state replay
state-balanced replay
top-k dark replay
interference-prioritized replay
```

## C3 — Sparse routed residual memory

See `SPARSE_MEMORY_ROUTING_PLAN.md`.

Test MEMOIR/SMF-inspired sparse memory with timestep-conditioned routing.

## C4 — Gated continual adapter expansion

See `GATED_ADAPTER_EXPANSION_PLAN.md`.

Test C-LoRA/GainLoRA-style branch growth and routing.

## C5 — Orthogonal and Fisher-protected growth

See `ORTHOGONAL_FISHER_CONTINUAL_PLAN.md`.

Test O-Edit, FGGM, and null-space constraints on growth branches.

## C6 — Functional distillation and gradient-constrained replay

See `FUNCTIONAL_REPLAY_PLAN.md`.

Test LwF, DER, GEM/A-GEM, and combinations.

## C7 — SB generative trajectory replay

See `BRIDGE_GENERATIVE_REPLAY_PLAN.md`.

Compare bridge-conditioned old-edit state replay against ordinary random masking at exactly matched replay count and compute.

## C8 — Multi-marginal/function-space SB consolidation

See `MULTIMARGINAL_SB_CONSOLIDATION_PLAN.md`.

Consolidate old/new branch functions over cached states and compare against linear/EMA/task-vector baselines.

## C9 — Dual-memory fast/slow consolidation

See `DUAL_MEMORY_CONSOLIDATION_PLAN.md`.

Use a fast sparse episodic memory for immediate edits and a slow DiffusionGrow branch consolidated periodically.

---

# Phase E — Pilot eligibility

A track becomes confirmation-eligible if it satisfies at least one class on pilot100.

## Class A — Full continual editor

```text
new-block rewrite >= 0.80
new-block paraphrase >= 0.45
past-edit retention >= 0.75
average forgetting <= 0.10
same-subject TFPR <= 0.03
near/far budgets pass
base retention-loss increase <= 5%
malformed <= 0.05
```

## Class B — Retention/locality Pareto

At current efficacy within 0.03 of the strongest baseline:

```text
forgetting reduction >= 30%
past-edit retention improvement >= 0.10
protected KL reduction >= 20%
same-subject TFPR not worse
paired confidence interval favors method
```

## Class C — SB-specific

At matched replay storage and model evaluations:

```text
retention improvement >= 0.05
or forgetting reduction >= 25%
or replay storage reduction >= 25% at equal retention
```

over the closest non-SB baseline.

## Class D — Efficient scalable memory

```text
storage <= 1 MB/edit or sublinear branch growth
inference overhead <= 25%
retention competitive within 0.03
```

One bounded rescue per track may use only that track's predeclared rescue.

---

# Phase F — Fresh confirmation

Confirm every eligible track on `cf_cl_confirmation_200`.

No tuning on confirmation.

Required:

```text
same qualitative direction as pilot
paired retention/forgetting evidence remains positive
same-subject and base-retention constraints remain
storage/compute claim remains
```

For multi-token claims, confirm on `kamel_cl_confirmation_180`.

A track that fails confirmation becomes a bounded negative.

---

# Phase G — Conditional tracks

## C10 — Parameter-space SB consolidation

Trigger only if C1 or C4 produces a stable low-dimensional adapter state.

## C11 — Online Bayesian/Laplace adapter

Trigger only if C5 shows meaningful Fisher signal but misses end-to-end thresholds.

## C12 — Spectral post-hoc repair

Trigger only if a parametric method has strong acquisition and identifiable forgetting in accumulated deltas.

## C13 — Selective routing

Trigger only if pre-edit risk features predict unsafe edits with validation AUC >= 0.80.

## C14 — Integrated candidate

Combine only independently confirmed compatible components. Examples:

```text
DiffusionGrow + partial-state dark replay
DiffusionGrow + sparse memory
DiffusionGrow + FGGM
sparse memory + bridge replay
dual memory + bridge consolidation
```

No failed component may be revived through integration.

---

# Phase H — Final selection

Rank confirmed outcomes:

```text
1. full continual editor
2. SB-specific continual result
3. retention/locality Pareto result
4. efficiency/scaling result
5. mechanism-only result
6. no promising continual direction
```

Write one recommendation:

```text
pursue_diffusiongrow_continual_editor
pursue_partial_state_replay_editor
pursue_sparse_memory_editor
pursue_gated_adapter_editor
pursue_orthogonal_fisher_editor
pursue_functional_replay_editor
pursue_bridge_replay_editor
pursue_sb_consolidation_editor
pursue_dual_memory_editor
pursue_integrated_continual_editor
mechanism_only_result
no_promising_continual_direction
```

Do not launch the selected full program automatically. Generate `SELECTED_CONTINUAL_DIRECTION_FULL_CAMPAIGN_DRAFT.md`.

---

# Phase I — Final package

Create:

```text
runs/continual_diffusion_editing_sb_selection_v1/final_direction_selection_package_v1/
```

Required:

```text
report_summary.json
final_research_report.md
direction_selection_matrix.csv
sequential_retention_table.csv
forgetting_curves.csv
same_subject_table.csv
base_retention_table.csv
multi_token_table.csv
compute_storage_table.csv
paired_bootstrap.csv
track_status_registry.json
artifact_availability_manifest.json
reproducibility_manifest.json
next_direction_recommendation.md
SELECTED_CONTINUAL_DIRECTION_FULL_CAMPAIGN_DRAFT.md
terminal_package_validation.json
```

Validate hashes, stop the Pod, and return the consolidated result.
