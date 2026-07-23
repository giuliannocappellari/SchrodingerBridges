#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
# This writes the complete continual-learning goal bundle.

cat > 'AGENTS.md' <<'__FILE_0_AGENTS_md__'
# AGENTS.md

Operational and scientific rules for the autonomous continual-learning campaign in the LLaDA factual-editing repository.

## 0. Active project identity

```text
active_protocol = continual_diffusion_editing_sb_selection_v1
campaign_type = bounded next-direction selection
primary_backbone = GSAI-ML/LLaDA-8B-Instruct
secondary_backbone = GSAI-ML/LLaDA-8B-Base only after a confirmed primary result
base_backbone_parameters = frozen unless a baseline explicitly requires otherwise
main_new_source = DiffusionGrow: Continual Learning for Diffusion Language Models
```

All previous Direction 1, Direction 2, Direction 3, MDM-MEMIT, mask-pattern-control, parametric-editor, temporal-residual, and next-direction-selection protocols are immutable historical evidence. They may be read for code reuse, exclusions, baseline summaries, and failure analysis, but must not be overwritten or resumed.

The goal is to determine whether continual-learning mechanisms can turn diffusion-LM factual editing into a sequentially stable editor that acquires new facts, retains earlier edits, preserves the original denoiser, and controls same-subject side effects.

## 1. Authoritative files

Codex must read these files in order:

```text
1. AGENTS.md
2. ACTIVE_RESEARCH_CAMPAIGN.json
3. CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md
4. CANDIDATE_DIRECTION_REGISTRY.json
5. the relevant per-track plan
6. persisted campaign state under runs/
```

The root campaign file selects the active protocol. Persisted state under `runs/continual_diffusion_editing_sb_selection_v1/autonomous_campaign_v1/` is authoritative for completed stages.

## 2. Autonomous approval

Autonomous execution is enabled only when:

```bash
export CL_DLLM_AUTONOMOUS_MODE=1
```

Recommended infrastructure variables:

```bash
export CL_DLLM_MAX_INFRA_RETRIES="3"
export CL_DLLM_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"

export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

When autonomous mode is enabled, Codex has one-time approval to execute every task explicitly listed in the master plan without asking for per-stage, per-command, or per-GPU-job approval.

Codex must not:

- lower scientific thresholds after seeing results;
- invent unplanned rescues;
- use evaluation prompts as replay/training examples;
- route teacher-only labels into runtime features;
- open historical `analysis_500`, `final_test_500`, or `final_test_full`;
- create a full follow-up campaign before the selection package is terminal;
- terminate or delete the RunPod resource.

## 3. Pod lifecycle

There is no monetary budget guard.

Codex must:

1. start the configured existing Pod if stopped;
2. use `/workspace/SB` as the authoritative campaign worktree;
3. keep the Pod running through all mandatory pilots, CPU analyses, GPU jobs, confirmation, and reporting;
4. continue after an individual track fails;
5. stop the Pod only after the final selection package validates, or after an unrecoverable Pod/infrastructure/data-integrity failure remains after the allowed retries.

Do not stop the Pod merely because:

```text
one track failed
one track passed
one GPU job finished
the next task is CPU-only
the Pod is temporarily idle between stages
```

## 4. Local and remote Python

Local MacBook:

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script>.py
```

RunPod:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script>.py
```

Do not require `uv` on RunPod. Do not use `pip install` directly in the local project environment.

## 5. Campaign state

Create and maintain:

```text
runs/continual_diffusion_editing_sb_selection_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  infrastructure_state.json
  optional_cost_state.json
```

The campaign state must include:

```json
{
  "protocol_version": "continual_diffusion_editing_sb_selection_v1",
  "campaign_status": "running",
  "current_stage": "",
  "next_stage": "",
  "completed_stages": [],
  "failed_stages": [],
  "rescues_used": {},
  "analysis_500_used": false,
  "final_test_used": false,
  "pod_status": ""
}
```

Cost is informational only and cannot block execution.

## 6. Long-job discipline

Use `tmux` and explicit exit-code files:

```bash
tmux new -d -s "<stage>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage>.log; \
   code=${PIPESTATUS[0]}; echo "$code" > logs/<stage>.exitcode; exit "$code"'
```

Every stage writes:

```text
report_summary.json
run_config.json or equivalent
validation report
log
exit-code file for long jobs
```

Never overwrite a completed output directory. Use versioned directories.

## 7. Fresh-data and split rules

Create fresh sequential-edit streams. Historical manifests may be read only for source-ID/fingerprint exclusion.

Forbidden for training, tuning, selection, or confirmation:

```text
historical analysis_500
historical final_test_500
historical final_test_full
historical held-out prompts
historical same-subject evaluation prompts
```

For each edit:

```text
train_prompt_ids ∩ eval_prompt_ids = ∅
```

Allowed training/replay data:

```text
rewrite prompt
edit tuple
training-only paraphrase augmentations
training-only same-subject different-relation anchors
training-only near/far/unrelated anchors
partial-mask states derived from allowed training prompts
stored old-model logits or generated replay from allowed training prompts
```

Evaluation-only:

```text
official held-out paraphrases
held-out same-subject prompts
held-out near/far prompts
held-out generation/attribute prompts
confirmation stream outcomes
```

## 8. Continual-learning definitions

The campaign must separately measure:

```text
plasticity:
  acquisition of the newest edit block

edit retention:
  performance on all earlier edited facts

base retention:
  preservation of the original denoiser and general capabilities

locality:
  same-subject, near, far, and distributional side effects

forgetting:
  historical best performance minus current performance for each old block

backward transfer:
  change in previous-block performance after learning later blocks

forward transfer:
  effect of previous learning on new-block acquisition
```

Evaluate after every edit block, not only at the end.

## 9. Runtime leakage rules

Runtime inputs may use:

```text
current hidden states
base logits/log-probabilities
timestep/mask ratio
candidate token embeddings
edit request
branch/adaptor parameters
routing keys derived from prompt/edit text
```

Teacher/replay labels may not be runtime features:

```text
future success
final decoded outcome
evaluation bucket
prompt_type
negative_type
teacher score arrays
post-edit locality labels
case ID
split role
```

Every checkpoint must serialize its runtime feature schema and pass a leakage audit.

## 10. Breadth-first rule

Mandatory tracks C1-C9 must all receive their bounded pilot before any track is scaled.

A failed track must:

```text
write a formal track stop package
update the registry
continue to the next mandatory track
```

An integrated candidate may combine only components that independently passed their mechanism criteria.

## 11. Scientific claim classes

### Class A — Full continual editor

```text
new-block rewrite >= 0.80
new-block paraphrase >= 0.45
past-edit retention >= 0.75 after 200 edits
average forgetting <= 0.10
same-subject TFPR <= 0.03
near/far budgets pass
base denoising retention-loss increase <= 5%
malformed <= 0.05
```

### Class B — Retention/locality Pareto result

At matched current-block efficacy within 0.03:

```text
average forgetting reduced >= 30%
past-edit retention improves >= 0.10
protected KL reduced >= 20%
same-subject TFPR does not worsen
paired evidence favors the method
```

### Class C — SB-specific continual-learning result

At matched replay memory and compute:

```text
bridge replay or SB consolidation improves past-edit retention >= 0.05
or reduces average forgetting >= 25%
or reaches equal retention with >= 25% less replay storage
```

and it must beat the closest non-SB replay/consolidation baseline.

### Class D — Efficiency/scaling result

```text
storage <= 1 MB per edit or sublinear growth
inference overhead <= 25% over base
retention remains competitive with the best pilot
```

## 12. Bounded rescues

Allowed:

```text
one source-integration repair for DiffusionGrow
one implementation repair per track
one scientific rescue per mandatory track
one confirmation-only infrastructure rerun before results are inspected
```

Forbidden:

```text
threshold relaxation
new feature families after pilot failure
larger hidden models as rescue
evaluation-data replay
unbounded grids
post-confirmation retuning
```

## 13. Final completion

The campaign is terminal when:

1. all mandatory tracks are terminal;
2. eligible tracks have been confirmed on a fresh stream;
3. the integrated candidate, if triggered, is terminal;
4. one final recommendation or `no_promising_continual_direction` is written;
5. the final package validates;
6. the Pod is stopped.

The campaign selects a next direction. It does not automatically launch the selected direction's full-scale research program.
__FILE_0_AGENTS_md__

cat > 'ACTIVE_RESEARCH_CAMPAIGN.json' <<'__FILE_1_ACTIVE_RESEARCH_CAMPAIGN_json__'
{
  "schema_version": 1,
  "active_campaign": "continual_diffusion_editing_sb_selection_v1",
  "active_plan": "CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md",
  "execution_mode": "autonomous_goal",
  "campaign_status": "not_started",
  "current_stage": "A0_bootstrap_and_source_audit",
  "historical_protocols": {
    "direction1": "closed",
    "direction2_v1": "closed_protocol_infeasible",
    "direction3_v1": "closed_bounded_negative",
    "sb_alternatives": "closed_scientific_negative",
    "mdm_memit_sb": "closed_mixed_positive_negative",
    "mask_pattern_confirmation": "closed_fresh_confirmation_failed",
    "parametric_editor": "closed_bounded_negative",
    "temporal_residual_editor": "closed_bounded_negative",
    "next_direction_statistics": "closed_no_promising_direction"
  },
  "historical_analysis_500_locked": true,
  "historical_final_test_locked": true,
  "new_protocol_required": true,
  "pod_stop_policy": "stop_only_after_terminal_package_or_unrecoverable_infrastructure_issue"
}
__FILE_1_ACTIVE_RESEARCH_CAMPAIGN_json__

cat > 'CANDIDATE_DIRECTION_REGISTRY.json' <<'__FILE_2_CANDIDATE_DIRECTION_REGISTRY_json__'
{
  "schema_version": 1,
  "protocol_version": "continual_diffusion_editing_sb_selection_v1",
  "tracks": [
    {
      "track_id": "C0",
      "name": "DiffusionGrow source reproduction and continual factual-edit baseline",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C1",
      "name": "Function-preserving growth branches for sequential factual edits",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C2",
      "name": "Partial-state real and dark replay",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C3",
      "name": "Sparse routed residual memory",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C4",
      "name": "Gated continual adapter expansion",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C5",
      "name": "Orthogonal and Fisher-protected growth",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C6",
      "name": "Functional distillation, DER, and episodic gradient constraints",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C7",
      "name": "Schrodinger-bridge generative trajectory replay",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C8",
      "name": "Multi-marginal/function-space SB consolidation",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C9",
      "name": "Dual-memory fast/slow consolidation",
      "mandatory": true,
      "status": "pending"
    },
    {
      "track_id": "C10",
      "name": "Low-dimensional parameter-space SB consolidation",
      "mandatory": false,
      "trigger": "C1 or C4 passes and adapter latent is stable",
      "status": "not_triggered"
    },
    {
      "track_id": "C11",
      "name": "Online Bayesian/Laplace continual adapter",
      "mandatory": false,
      "trigger": "C5 shows Fisher signal but misses end-to-end criteria",
      "status": "not_triggered"
    },
    {
      "track_id": "C12",
      "name": "Spectral post-hoc unforgetting",
      "mandatory": false,
      "trigger": "a parametric track has strong acquisition but recoverable forgetting",
      "status": "not_triggered"
    },
    {
      "track_id": "C13",
      "name": "Selective safe routing/abstention",
      "mandatory": false,
      "trigger": "a track has a reliable pre-edit risk signal",
      "status": "not_triggered"
    },
    {
      "track_id": "C14",
      "name": "Integrated continual editor",
      "mandatory": false,
      "trigger": "two or more compatible components pass independently",
      "status": "not_triggered"
    }
  ],
  "final_decisions": [
    "pursue_diffusiongrow_continual_editor",
    "pursue_partial_state_replay_editor",
    "pursue_sparse_memory_editor",
    "pursue_gated_adapter_editor",
    "pursue_orthogonal_fisher_editor",
    "pursue_functional_replay_editor",
    "pursue_bridge_replay_editor",
    "pursue_sb_consolidation_editor",
    "pursue_dual_memory_editor",
    "pursue_integrated_continual_editor",
    "mechanism_only_result",
    "no_promising_continual_direction",
    "protocol_infeasible",
    "infrastructure_blocked"
  ]
}
__FILE_2_CANDIDATE_DIRECTION_REGISTRY_json__

cat > 'CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md' <<'__FILE_3_CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN_md__'
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
__FILE_3_CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN_md__

cat > 'CONTINUAL_LEARNING_ALTERNATIVES_CATALOG.md' <<'__FILE_4_CONTINUAL_LEARNING_ALTERNATIVES_CATALOG_md__'
# Continual-Learning Alternatives Catalog

This catalog lists the broad design space. The master campaign tests the highest-value families directly and keeps the remaining ideas as conditional extensions.

## Core diffusion-LM alternatives

1. **DiffusionGrow function-preserving growth**  
   Add timestep-conditioned trainable branches while retaining the frozen pretrained path. Zero initialization gives exact base behavior before adaptation.

2. **Shared growth branch with sequential updates**  
   One branch is updated across edit blocks; tests raw catastrophic forgetting.

3. **Block-specific growth branches**  
   Add one branch per edit block and route by prompt/edit relevance.

4. **Progressive branch compression**  
   Periodically distill multiple branches into a smaller shared branch.

5. **Small-block continual pretraining**  
   Use small diffusion blocks during consolidation to preserve informative contexts.

6. **Partial-state replay**  
   Replay old edits at fully masked, early, middle, and late denoising states.

7. **Trajectory-balanced replay**  
   Balance replay by timestep, active-mask count, relation, and target length.

8. **Dark experience replay**  
   Store top-k logits or branch outputs for old edit states instead of only target labels.

9. **Learning without forgetting across mask states**  
   Distill the previous denoiser on new-task inputs and training-only anchors.

10. **Gradient episodic memory / A-GEM**  
    Project new-edit gradients so they do not increase loss on stored old-edit states.

## Parameter-isolation and routing alternatives

11. **C-LoRA-style self-regularized continual LoRA**  
    Penalize interference between sequential low-rank branches.

12. **GainLoRA-style gated branch integration**  
    Add a branch per block and learn gates that suppress new branches on old tasks.

13. **O-Edit orthogonal subspace updates**  
    Orthogonalize each new update against earlier edit directions.

14. **FGGM Fisher-guided gradient masking**  
    Freeze or downweight parameters important to the base and previous edits.

15. **NuSA-style null-space adaptation**  
    Restrict new low-rank updates to an approximate protected null space.

16. **Progressive networks / expandable experts**  
    Freeze old branches and add lateral connections for new edit blocks.

17. **PackNet/dynamic sparse masks**  
    Allocate disjoint sparse parameter subsets to sequential edit groups.

18. **Relation- or subject-specific experts**  
    Route edits to specialized branches; requires careful same-subject gating.

## Memory alternatives

19. **MEMOIR sparse residual memory**  
    Use sparse activation masks to isolate edits in a dedicated memory module.

20. **Sparse Memory Finetuning**  
    Update only highly accessed memory rows.

21. **Fast episodic memory + slow semantic branch**  
    Apply edits immediately in external memory, then consolidate safely.

22. **Reservoir replay**  
    Maintain a bounded unbiased buffer of old edit states.

23. **Interference-prioritized replay**  
    Replay edits with high gradient conflict or high observed forgetting.

24. **Coreset selection**  
    Choose representative old states by relation, hidden-state geometry, or Fisher leverage.

25. **Free-text causal memory**  
    Store causal abstractions or edit descriptions instead of parameter changes.

## Schrödinger-bridge combinations

26. **Bridge generative replay**  
    Sample old-edit partial states from a reference bridge conditioned on old endpoints instead of ordinary random masking.

27. **Bridge replay with previous-model distillation**  
    Distill the previous branch on bridge-sampled states.

28. **Multi-marginal SB over edit blocks**  
    Treat successive old/new function distributions as multiple marginals and find a minimum-KL consolidation path.

29. **Function-space Schrödinger barycenter**  
    Consolidate old and new denoiser distributions at each state through an entropic KL barycenter.

30. **Unbalanced SB replay**  
    Allocate transport mass only to edits judged at risk of forgetting.

31. **Parameter-space SB over adapter latents**  
    Bridge from the previous adapter state to the new optimum in a low-dimensional adapter manifold.

32. **Fisher-metric parameter bridge**  
    Use a behavioral metric rather than Euclidean adapter distance.

33. **SB-guided branch merging**  
    Compare entropy-regularized consolidation with linear averaging, EMA, and task arithmetic.

34. **Bridge-based rehearsal scheduler**  
    Use bridge potential or path cost to prioritize which old edits need replay.

35. **Doob-transform retention control**  
    Reweight continual training trajectories toward states that retain old edits while acquiring the new block.

## Statistical and Bayesian alternatives

36. **EWC / online EWC**  
    Penalize changes to Fisher-important parameters.

37. **Synaptic Intelligence / MAS**  
    Accumulate online parameter importance without full replay.

38. **Online Laplace continual adapters**  
    Maintain a posterior precision over adapter parameters.

39. **Kalman-filter branch updates**  
    Treat the adapter state as a latent dynamical system with noisy edit observations.

40. **Hierarchical Bayesian relation adapters**  
    Share statistical strength across relations while retaining edit-specific corrections.

41. **CVaR forgetting optimization**  
    Optimize the worst-forgotten tail rather than only average retention.

42. **Conformal selective editing**  
    Abstain or route to external memory when safe continual adaptation cannot be certified.

## Post-hoc consolidation and repair

43. **Spectral unforgetting**  
    Remove low-signal/noise components from accumulated parameter deltas.

44. **Weight interpolation / WiSE-style merging**  
    Interpolate adapted and base branches to trade plasticity for retention.

45. **TIES/DARE-style task-vector merging**  
    Resolve sign conflicts before merging sequential branches.

46. **Knowledge-driven parameter fusion**  
    Weight branch contributions according to edit relevance and retention risk.

47. **Teacher-student periodic consolidation**  
    Train a clean student from the base plus all accepted edit memories.

## Recommended priority

Highest probability:

```text
DiffusionGrow + partial-state dark replay
DiffusionGrow + sparse routed residual memory
DiffusionGrow + O-Edit/FGGM protection
dual-memory fast/slow consolidation
```

Highest Schrödinger-specific value:

```text
bridge generative replay
multi-marginal/function-space SB consolidation
Fisher-metric parameter-space SB
```

Highest engineering risk:

```text
full multi-marginal SB
Bayesian parameter bridge
large progressive expert systems
```
__FILE_4_CONTINUAL_LEARNING_ALTERNATIVES_CATALOG_md__

cat > 'DIFFUSIONGROW_CONTINUAL_EDITING_PLAN.md' <<'__FILE_5_DIFFUSIONGROW_CONTINUAL_EDITING_PLAN_md__'
# C1 — DiffusionGrow Continual Factual Editing

## Hypothesis

A function-preserving growth branch can acquire sequential factual edits without corrupting the frozen pretrained denoiser because the original path remains explicitly available and the new branch is zero-initialized and timestep-gated.

## Source-compatible reproduction

First reproduce the source-style domain-adaptation behavior using available code/checkpoints or an equation-level reimplementation. Record exact status.

## Factual adaptation

At selected early-middle MLP layers, use:

\[
h'_{\ell,t} = h_{\ell,t} + g_{\ell}(h_{\ell,t}, t)\,B_{\ell}A_{\ell}h_{\ell,t}.
\]

Initialize the residual path to zero so the expanded model equals the base denoiser before training.

## Variants

```text
C1-A shared branch updated across all blocks
C1-B one branch per edit block
C1-C shared branch + block gate
C1-D block branches + prompt/edit gate
C1-E partial-state branch training
```

Base weights remain frozen.

## Training

Use current-block rewrite data, training-only paraphrase augmentations, and training-only locality anchors. Do not replay held-out evaluation prompts.

## Mechanism metrics

```text
exact function equality at initialization
gate activation by prompt family and timestep
branch norm
old/new branch contribution
base-path availability
```

## Pass

Class A, B, or D from the master plan.

## Rescue

One rescue may adjust only:

```text
branch rank in {4,8,16}
growth layers within the source-compatible layer family
gate initialization/temperature
```

No new branch architecture.
__FILE_5_DIFFUSIONGROW_CONTINUAL_EDITING_PLAN_md__

cat > 'PARTIAL_STATE_REPLAY_PLAN.md' <<'__FILE_6_PARTIAL_STATE_REPLAY_PLAN_md__'
# C2 — Partial-State Continual Replay

## Hypothesis

Continual factual forgetting in a diffusion LM is partly caused by replaying only clean or fully masked prompts. Old edits should be rehearsed over the partial denoising states actually visited at inference.

## Replay variants

```text
R0 no replay
R1 clean rewrite replay
R2 fully masked answer replay
R3 uniformly sampled mask-ratio replay
R4 early/middle/late balanced replay
R5 actual-trajectory state replay
R6 dark replay: stored top-k logits
R7 state-balanced dark replay
R8 interference-prioritized dark replay
```

Use identical replay-item budgets when comparing variants.

## Memory budgets

```text
0 items/edit
1 clean item/edit
4 state items/edit
8 state items/edit
```

Dark replay stores compressed top-k logits and schema fingerprints, not final outcomes.

## Prioritization

Estimate interference using gradient cosine or observed retention loss on training-only old-edit probes.

## Pass

At matched new-block efficacy:

```text
forgetting reduction >= 30%
past retention +0.10
or equal retention with >=25% less storage than clean replay
```

For a diffusion-specific claim:

```text
state replay beats clean/full-mask replay by >=0.05 past retention
with paired lower bound > 0.
```

## Rescue

One rescue may change only the replay allocation across early/middle/late buckets, not the total replay budget.
__FILE_6_PARTIAL_STATE_REPLAY_PLAN_md__

cat > 'SPARSE_MEMORY_ROUTING_PLAN.md' <<'__FILE_7_SPARSE_MEMORY_ROUTING_PLAN_md__'
# C3 — Sparse Routed Residual Memory

## Hypothesis

A dedicated sparse memory can isolate sequential edits better than modifying a shared branch. Query-dependent sparse activation should generalize to paraphrases while suppressing unrelated and same-subject activations.

## Variants

```text
MEMOIR-style sparse residual memory
Sparse Memory Finetuning row selection
timestep-conditioned memory keys
relation-conditioned memory keys
shared memory pool
block-partitioned memory pool
```

## Runtime

\[
h'_t = h_t + \sum_{i\in \mathcal A(x,t)} w_i(x,t) M_i,
\]

where only a small active set is retrieved.

## Required comparisons

```text
dense residual memory
sparse memory
random sparse routing
subject-only routing
subject+relation+timestep routing
```

## Metrics

```text
memory-row overlap between edits
activation sparsity
wrong-edit activation
same-subject activation
storage/edit
retrieval latency
retention across stream
```

## Pass

Class A, B, or D.

## Rescue

One rescue may alter the routing sparsity target or memory-row count within a bounded {64,128,256} row family.
__FILE_7_SPARSE_MEMORY_ROUTING_PLAN_md__

cat > 'GATED_ADAPTER_EXPANSION_PLAN.md' <<'__FILE_8_GATED_ADAPTER_EXPANSION_PLAN_md__'
# C4 — Gated Continual Adapter Expansion

## Hypothesis

Adding a small adapter per edit block and learning gates that suppress new adapters on old tasks can improve stability without forcing all tasks through one adapted computation.

## Variants

```text
C-LoRA-style continually self-regularized adapters
GainLoRA-style new branch per block
uniform branch averaging
learned prompt gate
timestep-conditioned prompt gate
relation-aware gate
```

## Gate objective

For old-task training anchors, penalize contribution from the newest branch. For the current block, allow plasticity.

## Fairness

Report cumulative parameters and storage. Compare at fixed total rank and fixed per-block rank.

## Pass

Class A, B, or D.

## Rescue

One rescue may use a shared low-rank basis with block-specific coefficients to reduce linear parameter growth.
__FILE_8_GATED_ADAPTER_EXPANSION_PLAN_md__

cat > 'ORTHOGONAL_FISHER_CONTINUAL_PLAN.md' <<'__FILE_9_ORTHOGONAL_FISHER_CONTINUAL_PLAN_md__'
# C5 — Orthogonal and Fisher-Protected Growth

## Hypothesis

New factual updates interfere because their gradient/update directions overlap with base-denoiser and previous-edit directions. Orthogonalization and Fisher-guided masking may improve stability.

## Variants

```text
O-Edit update orthogonalization
online O-Edit basis with truncation
FGGM diagonal-Fisher gradient mask
NuSA-style null-space low-rank update
online EWC baseline
O-Edit + FGGM, only if both pass independently
```

## Protected data

Use training-only base retention prompts, old edits, same-subject anchors, and partial-state variants.

## Metrics

```text
gradient cosine with old edits
Fisher-weighted update norm
subspace rank growth
plasticity loss
retention gain
```

## Pass

At current efficacy within 0.03:

```text
forgetting reduction >=30%
protected KL reduction >=20%
or past retention +0.10
```

## Rescue

One bounded threshold/rank sweep only:

```text
orthogonal rank {16,32,64}
Fisher mask keep ratio {0.1,0.25,0.5}
```
__FILE_9_ORTHOGONAL_FISHER_CONTINUAL_PLAN_md__

cat > 'FUNCTIONAL_REPLAY_PLAN.md' <<'__FILE_10_FUNCTIONAL_REPLAY_PLAN_md__'
# C6 — Functional Distillation and Gradient-Constrained Replay

## Hypothesis

Preserving the previous model's function over diffusion states may be more effective than constraining parameters.

## Variants

```text
Learning without Forgetting
Dark Experience Replay
experience replay with labels
DER + replay labels
GEM
A-GEM
LwF + partial-state replay
DER + partial-state replay
```

## Distillation targets

Store or recompute previous-model distributions over:

```text
old edit answer positions
same-subject training anchors
base retention prompts
early/middle/late mask states
```

## GEM constraint

Require the new gradient not to increase old replay loss beyond the bounded slack.

## Pass

Class B or D. A mechanism result requires partial-state functional replay to beat clean-prompt functional replay.

## Rescue

One rescue may adjust the old/new loss balance in a fixed grid `{0.25,0.5,1.0,2.0}`.
__FILE_10_FUNCTIONAL_REPLAY_PLAN_md__

cat > 'BRIDGE_GENERATIVE_REPLAY_PLAN.md' <<'__FILE_11_BRIDGE_GENERATIVE_REPLAY_PLAN_md__'
# C7 — Schrödinger-Bridge Generative Trajectory Replay

## Hypothesis

Ordinary replay samples old edit states from the forward masking/noising distribution. A reference bridge conditioned on the old edit endpoint may generate more informative rehearsal states and reduce forgetting at a fixed replay budget.

## State construction

For an old edit with clean target span \(x_1\) and a chosen start distribution \(x_0\), sample intermediate states from:

\[
q^{ref}(x_t \mid x_0, x_1)
\]

or a bounded CSBM/Doob approximation.

Use the frozen LLaDA-compatible mask process as the reference.

## Variants

```text
ordinary random-mask replay
actual stored trajectory replay
reference-bridge replay
CSBM-lite endpoint-conditioned replay
unbalanced bridge replay prioritized by forgetting risk
```

## Distillation

On generated states, match the previous accepted model's top-k distribution and old edited target support.

## Fair comparison

Match:

```text
number of replay states
stored bytes
model evaluations
old edit IDs
timestep histogram
```

## SB-specific pass

Compared with ordinary random-mask replay:

```text
past retention +0.05
or forgetting reduction >=25%
or equal retention with >=25% less replay storage
```

with paired lower bound > 0.

## Rescue

One rescue may change only the bridge stochasticity/reference mixture within a predeclared 3-point grid.
__FILE_11_BRIDGE_GENERATIVE_REPLAY_PLAN_md__

cat > 'MULTIMARGINAL_SB_CONSOLIDATION_PLAN.md' <<'__FILE_12_MULTIMARGINAL_SB_CONSOLIDATION_PLAN_md__'
# C8 — Multi-Marginal and Function-Space SB Consolidation

## Hypothesis

Sequential adaptation can be viewed as transporting the denoiser through a sequence of functional marginals. A KL-regularized consolidation may preserve old edit functions better than direct branch overwrite or linear merging.

## Tractable setting

Do not solve SB over all model parameters. Build a cache of deployable states and top-k distributions for:

```text
base denoiser
previous accepted continual model
current-block adapted model
old edit states
current edit states
base retention states
```

## Variants

```text
linear logit interpolation
EMA teacher
task-vector/adapter averaging
TIES-style sign-aware merge
two-marginal entropic barycenter
multi-marginal KL barycenter
iterative proportional/Markovian fitting on cached distributions
```

## Objective

Find consolidated distributions close to the reference/pretrained process while satisfying old- and new-edit marginals. Distill the selected consolidated function into the growth branch.

## Pass

SB-specific Class C relative to the strongest non-SB merge at matched state cache and compute.

## Rescue

One rescue may adjust the entropic regularization and old/new marginal weights within a bounded grid.
__FILE_12_MULTIMARGINAL_SB_CONSOLIDATION_PLAN_md__

cat > 'DUAL_MEMORY_CONSOLIDATION_PLAN.md' <<'__FILE_13_DUAL_MEMORY_CONSOLIDATION_PLAN_md__'
# C9 — Dual-Memory Fast/Slow Consolidation

## Hypothesis

Immediate factual acquisition and long-term stable consolidation should be separated.

## Architecture

```text
fast memory:
  sparse routed residual or external edit memory
  immediate low-cost updates

slow memory:
  DiffusionGrow function-preserving branch
  periodic consolidation using replay/distillation

base path:
  permanently frozen
```

## Consolidation schedule

Compare consolidation every:

```text
10 edits
25 edits
50 edits
```

## Variants

```text
fast only
slow only
fast + ordinary replay consolidation
fast + dark replay consolidation
fast + bridge replay consolidation
```

## Metrics

```text
immediate acquisition
post-consolidation retention
fast-memory eviction effects
slow-branch forgetting
storage scaling
latency
```

## Pass

Class A, B, C, or D.

## Rescue

One rescue may alter only the consolidation interval among the predeclared values.
__FILE_13_DUAL_MEMORY_CONSOLIDATION_PLAN_md__

cat > 'FINAL_SELECTION_AND_REPORTING_PLAN.md' <<'__FILE_14_FINAL_SELECTION_AND_REPORTING_PLAN_md__'
# Final Selection and Reporting

## Confirmation rule

Only pilot-eligible candidates run on the fresh 200-edit confirmation stream. No retuning.

## Selection hierarchy

```text
1. full continual editor
2. confirmed SB-specific continual result
3. confirmed retention/locality Pareto improvement
4. confirmed efficiency/scaling result
5. confirmed mechanism-only result
6. no promising continual direction
```

## Tie-breaking

1. lower average forgetting;
2. lower same-subject TFPR;
3. higher past-edit retention;
4. lower base retention loss;
5. lower storage growth;
6. lower compute.

## Required final outputs

```text
final_research_report.md
direction_selection_matrix.csv
plasticity_retention_curves.csv
forgetting_by_block.csv
same_subject_results.csv
base_denoising_retention.csv
multi_token_results.csv
compute_storage_results.csv
paired_bootstrap.csv
track_status_registry.json
next_direction_recommendation.md
SELECTED_CONTINUAL_DIRECTION_FULL_CAMPAIGN_DRAFT.md
terminal_package_validation.json
```

## Wording guard

A track that improves retention but fails factual acquisition is a mechanism result, not a successful editor. A bridge-based track must beat its matched non-SB baseline before receiving an SB-specific claim.
__FILE_14_FINAL_SELECTION_AND_REPORTING_PLAN_md__

cat > 'PRIMARY_SOURCES.md' <<'__FILE_15_PRIMARY_SOURCES_md__'
# Primary Sources and Source Status

The autonomous campaign must independently verify URLs, code availability, licenses, and revisions before execution.

## Diffusion-language-model continual learning

- **DiffusionGrow: Continual Learning for Diffusion Language Models**  
  OpenReview submission: https://openreview.net/forum?id=pvqJiLXmUn  
  Reported idea: function-preserving growth with timestep-conditioned trainable branches and learned gates, leaving the pretrained denoiser path frozen. Treat as a recent submission, not an established accepted result, until source audit confirms status.

- **Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model**  
  arXiv: https://arxiv.org/abs/2601.15892  
  Relevant idea: block-diffusion continual pretraining, small-block curriculum, and training/inference context alignment.

- **Multi-Mask Diffusion Language Models for Few-Step Generation**  
  arXiv: https://arxiv.org/abs/2607.19686  
  Relevant idea: continual training from a pretrained MDM using a modified mask process.

## Continual and lifelong editing

- **O-Edit**  
  https://arxiv.org/abs/2410.11469

- **MEMOIR**  
  https://arxiv.org/abs/2506.07899

- **EvoEdit**  
  https://arxiv.org/abs/2512.04545

- **Sparse Memory Finetuning**  
  https://arxiv.org/abs/2605.03229

## Continual PEFT and protected updates

- **Continual Diffusion / C-LoRA**  
  https://arxiv.org/abs/2304.06027

- **GainLoRA**  
  https://arxiv.org/abs/2505.15424

- **FGGM**  
  https://arxiv.org/abs/2601.18261

- **NuSA-CL**  
  https://arxiv.org/abs/2510.21175

## Replay and distillation

- **Learning without Forgetting**  
  https://arxiv.org/abs/1606.09282

- **Gradient Episodic Memory**  
  https://arxiv.org/abs/1706.08840

- **Dark Experience Replay**  
  https://arxiv.org/abs/2004.07211

## Schrödinger bridges

- **Categorical Schrödinger Bridge Matching**  
  https://arxiv.org/abs/2502.01416

- **Diffusion Schrödinger Bridge Matching**  
  https://arxiv.org/abs/2303.16852

- **Unbalanced Diffusion Schrödinger Bridge**  
  https://arxiv.org/abs/2306.09099

## Repository evidence

The user's dissertation proposal and prior campaign artifacts remain part of the scientific context. The proposal explicitly includes sequential editing/O-Edit and asks whether bridge adaptation can improve locality in diffusion LMs. Previous campaigns must remain immutable.
__FILE_15_PRIMARY_SOURCES_md__

cat > 'README.md' <<'__FILE_16_README_md__'
# Continual Diffusion Editing Goal Bundle

This bundle launches an autonomous bounded selection campaign to test whether continual-learning methods can solve sequential factual editing in masked diffusion LMs.

## Central new source

The campaign is anchored in **DiffusionGrow**, a recent continual-learning proposal for diffusion language models that preserves an explicit frozen pretrained path and adds timestep-conditioned trainable branches.

## Why this campaign differs from earlier editing campaigns

Earlier work focused mostly on one edit or one fixed batch. This campaign evaluates:

```text
sequential acquisition
retention of earlier edits
forgetting after every block
preservation of the original denoiser
same-subject locality
storage and compute growth
```

## Files

```text
AGENTS.md
ACTIVE_RESEARCH_CAMPAIGN.json
CANDIDATE_DIRECTION_REGISTRY.json
CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md
CONTINUAL_LEARNING_ALTERNATIVES_CATALOG.md
DIFFUSIONGROW_CONTINUAL_EDITING_PLAN.md
PARTIAL_STATE_REPLAY_PLAN.md
SPARSE_MEMORY_ROUTING_PLAN.md
GATED_ADAPTER_EXPANSION_PLAN.md
ORTHOGONAL_FISHER_CONTINUAL_PLAN.md
FUNCTIONAL_REPLAY_PLAN.md
BRIDGE_GENERATIVE_REPLAY_PLAN.md
MULTIMARGINAL_SB_CONSOLIDATION_PLAN.md
DUAL_MEMORY_CONSOLIDATION_PLAN.md
FINAL_SELECTION_AND_REPORTING_PLAN.md
PRIMARY_SOURCES.md
START_CONTINUAL_DIFFUSION_EDITING_GOAL.md
```

## Environment

```bash
export CL_DLLM_AUTONOMOUS_MODE=1
export CL_DLLM_MAX_INFRA_RETRIES="3"
export CL_DLLM_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"

export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

## Launch

Use Codex Goal mode and paste `START_CONTINUAL_DIFFUSION_EDITING_GOAL.md`.

The Pod remains running until the final validated selection package is complete or an unrecoverable infrastructure issue remains.
__FILE_16_README_md__

cat > 'START_CONTINUAL_DIFFUSION_EDITING_GOAL.md' <<'__FILE_17_START_CONTINUAL_DIFFUSION_EDITING_GOAL_md__'
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
__FILE_17_START_CONTINUAL_DIFFUSION_EDITING_GOAL_md__

cat > 'BUNDLE_MANIFEST.json' <<'__FILE_18_BUNDLE_MANIFEST_json__'
{
  "bundle_name": "continual_diffusion_editing_goal_bundle",
  "protocol_version": "continual_diffusion_editing_sb_selection_v1",
  "created_utc": "2026-07-23T19:13:00+00:00",
  "files": [
    {
      "path": "ACTIVE_RESEARCH_CAMPAIGN.json",
      "sha256": "37fb5344d5c6bba25b6bbec22516fb4485c9e81e0a605895421f09fa0e8cee0a",
      "bytes": 1010
    },
    {
      "path": "AGENTS.md",
      "sha256": "5d1857bdecae5e29d36dec490a284931f23b82a66705d8f258fcaadafb351413",
      "bytes": 9434
    },
    {
      "path": "BRIDGE_GENERATIVE_REPLAY_PLAN.md",
      "sha256": "d06aff700866200539b6644ca6dcdc4c7d2e94fc544a6050da6bce4325356baa",
      "bytes": 1377
    },
    {
      "path": "CANDIDATE_DIRECTION_REGISTRY.json",
      "sha256": "379a7c0389eb33697c570007030031b980f667c6a77dc8ff48cb513032882de5",
      "bytes": 3281
    },
    {
      "path": "CONTINUAL_DIFFUSION_EDITING_AUTONOMOUS_PLAN.md",
      "sha256": "265ead7bf50bf058e060b0be80f08a180a4766666136b2d80f8d448a7a1e90e9",
      "bytes": 9704
    },
    {
      "path": "CONTINUAL_LEARNING_ALTERNATIVES_CATALOG.md",
      "sha256": "75ac4559834cb4759693151995f019fb4ea86b56cee9f8c3ca386c049e51c56b",
      "bytes": 6756
    },
    {
      "path": "DIFFUSIONGROW_CONTINUAL_EDITING_PLAN.md",
      "sha256": "36df0a36bfef54a98db59cc63f2cf2819a302698c3284f7dee7286824543768d",
      "bytes": 1577
    },
    {
      "path": "DUAL_MEMORY_CONSOLIDATION_PLAN.md",
      "sha256": "dab9996518f9610a871144f75fc2cd12b618e2431a2c0f7252b5b433d93249b1",
      "bytes": 935
    },
    {
      "path": "FINAL_SELECTION_AND_REPORTING_PLAN.md",
      "sha256": "dd4d6b8a98832a855a53609151c97b0f6222884822862cabcedfa807348da905",
      "bytes": 1241
    },
    {
      "path": "FUNCTIONAL_REPLAY_PLAN.md",
      "sha256": "58fb2fd403cf68949b8b704d12f4e422d3cf560e48bb81e5ada14d655fa026fd",
      "bytes": 913
    },
    {
      "path": "GATED_ADAPTER_EXPANSION_PLAN.md",
      "sha256": "cccd0e474cf438f9d62086f923338fd5a1c4fd01b15f12eb2b146ae8eaf7bc3f",
      "bytes": 847
    },
    {
      "path": "MULTIMARGINAL_SB_CONSOLIDATION_PLAN.md",
      "sha256": "d5cb6c8138816d4a11a32928181bec0eb324d8f7792eb26d8b9778bd4fcd97f2",
      "bytes": 1251
    },
    {
      "path": "ORTHOGONAL_FISHER_CONTINUAL_PLAN.md",
      "sha256": "39994191db9d288a475f33a7775748c8897ab735d31232bc9c38355177355fa0",
      "bytes": 1017
    },
    {
      "path": "PARTIAL_STATE_REPLAY_PLAN.md",
      "sha256": "8d7c5b23dbb634c92ca82e49ac660fbeb39434bcfb00d39a20ef258c0c9cce1f",
      "bytes": 1385
    },
    {
      "path": "PRIMARY_SOURCES.md",
      "sha256": "d93274e0e1b3e903190eb7ca31eacfb7994709e7fc0e46bd66aaa1e54d681d18",
      "bytes": 2397
    },
    {
      "path": "README.md",
      "sha256": "391c2c3d2a76ae2ae35265dc0cf0caa801df44f3b473ec9cce693105fdfb0fbe",
      "bytes": 1973
    },
    {
      "path": "SPARSE_MEMORY_ROUTING_PLAN.md",
      "sha256": "fb83ff3504d08be9513162f4901a918176a5fcb7a0cfc3916a75d3ecc3746b3c",
      "bytes": 1090
    },
    {
      "path": "START_CONTINUAL_DIFFUSION_EDITING_GOAL.md",
      "sha256": "933e3f94e2c9905256f650682b83289e09fa83ff2aea69b66c0806b026c7c07a",
      "bytes": 1954
    }
  ]
}
__FILE_18_BUNDLE_MANIFEST_json__

printf '%s\n' 'Created continual-learning campaign files.'
