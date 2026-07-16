# Autonomous Research Plan: MDM-MEMIT, Partial-Mask Editing, and Schrödinger Extensions

Protocol root: `masked_diffusion_memit_sb_positive_result_v1`

## 1. Goal

The campaign has two nested goals:

1. obtain a reproducible positive factual-editing result in `GSAI-ML/LLaDA-8B-Instruct` using the recently demonstrated MDM adaptation of MEMIT;
2. test whether a principled Schrödinger-style component improves locality, intervention cost, or answer-span denoising trajectories beyond the successful editor.

The campaign is not complete after a positive MEMIT result. It must also execute the two mandatory SB extensions and produce a final cross-track interpretation.

## 2. Starting scientific state

Historical experiments established that:

```text
runtime bridge guidance can create edit pressure
rule-based gates failed same-subject locality
learned value controllers used target shortcuts
multiple standalone SB formulations failed bounded pilot criteria
```

A newer primary source, arXiv:2606.03924, demonstrates a different positive mechanism:

```text
locate factual recall at early/middle MLP layers and the last subject token
adapt MEMIT by mask-augmenting every relevant input distribution
optimize multi-token edits across partially unmasked intermediate states
```

This campaign tests that mechanism first and treats SB as a secondary minimum-intervention/trajectory component rather than requiring SB to create the edit from scratch.

## 3. Campaign state

Create:

```text
runs/masked_diffusion_memit_sb_positive_result_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  cost_state.json                # informational only
  artifact_availability.json
```

`campaign_state.json` must include:

```json
{
  "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
  "autonomous_mode": true,
  "campaign_status": "running",
  "current_stage": "",
  "next_stage": "",
  "completed_stages": [],
  "failed_stages": [],
  "rescues_used": {},
  "track_status": {},
  "old_analysis_500_used": false,
  "old_final_test_used": false,
  "last_git_commit": "",
  "pod_status": ""
}
```

## 4. Phase A — Bootstrap and source audit

### A1. Campaign configuration

Tasks:

- validate root campaign files;
- validate RunPod variables;
- start the existing Pod if stopped;
- verify GPU, persistent volume, Git, and Python;
- run tests;
- initialize campaign state;
- preserve historical campaign statuses as read-only.

Acceptance:

```text
MDM_MEMIT_SB_AUTONOMOUS_MODE = 1
Pod running with GPU
/workspace/SB accessible
all existing tests pass
campaign state files exist
historical protocols marked immutable
```

### A2. Primary-source/code availability audit

Tasks:

- record arXiv:2606.03924 version and paper hash;
- check once for official released code;
- pin official MEMIT or EasyEdit reference implementation commit;
- inspect `GSAI-ML/LLaDA-8B-Instruct` module names, MLP down-projection, layer count, tokenizer, mask token, forward outputs, and generation API;
- write an implementation-difference register.

Outputs:

```text
source_audit/report_summary.json
source_audit/source_registry.json
source_audit/implementation_difference_register.md
source_audit/model_module_map.json
```

Acceptance:

```text
paper and reference code sources recorded
model loads in editable floating-point mode
MLP target matrices identified
last-subject token mapping validated
mask-augmented forward pass validated
```

Failure/rescue:

- one architecture-adapter repair is allowed;
- if official paper code is unavailable, continue from the paper and official MEMIT/EasyEdit source;
- do not stop merely because paper code is unavailable.

## 5. Phase B — Fresh data protocol

### B1. Historical-exclusion manifest

Collect historical IDs/fingerprints only from prior campaigns. Verify namespacing by source split and source index.

Output:

```text
protocol/historical_exclusion_manifest.json
protocol/historical_exclusion_audit.csv
```

Acceptance:

```text
no cross-split ID collision
all historical inspected/tuned/analysis/final IDs accounted
prompt/label/output content not reused for new tuning
```

### B2. Fresh CounterFact manifests

Build disjoint:

```text
cf_memit_smoke_20
cf_layer_select_500
cf_repro_main_500
cf_sb_dev_200
cf_sb_analysis_200
```

Stratify by relation and contextual target length while favoring valid single-token targets for the paper-matched reproduction.

Acceptance:

```text
all counts exact
zero overlap
source and fingerprint metadata written
repro_main_500 not read by layer selection or hyperparameter tuning
sb_analysis_200 not read by SB tuning
```

### B3. KAMEL adaptation

Implement a deterministic KAMEL-to-CounterFact adapter.

For each source fact:

1. render the real relation cloze template;
2. contextual-tokenize the object;
3. choose another object from the same relation with the same contextual target length;
4. create the counterfactual target;
5. attach one held-out paraphrase per relation;
6. preserve provenance.

Build disjoint:

```text
kamel_smoke_20_per_length
kamel_dev_50_per_length
kamel_repro_200_per_length
```

for `N={1,2,3,4}`.

Acceptance:

```text
200 main examples per N if source permits
at least 15 relations per N
zero split overlap
same-relation target swaps
context-aware target length exact
paraphrase mapping valid and evaluation-only
```

If dual LLaDA/LLaMA tokenizer filtering is feasible, reproduce it. If not, record the deviation and use LLaDA contextual length as the primary definition.

## 6. Phase C — M1: MDM-MEMIT reproduction

Read `MEMIT_REPRODUCTION_PLAN.md` and execute it completely.

Stage sequence:

```text
C1 CPU/fake implementation and tests
C2 one-edit GPU smoke
C3 20-edit integration smoke
C4 layer-window selection on cf_layer_select_500
C5 locked cf_repro_main_500 batch reproduction
C6 generation-setting robustness
C7 formal M1 result package
```

M1 may use one bounded rescue described in its track plan.

M1 terminal statuses:

```text
passed_reproduction
partial_reproduction
after_rescue_formal_negative
infrastructure_blocked
```

If M1 fails after rescue, trigger F1 and determine whether M2 can still run on a scientifically usable editor implementation.

## 7. Phase D — M2: partial-mask MEMIT

Read `PARTIAL_MASK_MEMIT_PLAN.md` and execute it completely.

Stage sequence:

```text
D1 baseline fully-masked MEMIT on KAMEL smoke
D2 implement cycle/random partial-mask augmentation
D3 KAMEL dev schedule/reveal ablations
D4 freeze partial-mask policy
D5 locked KAMEL 200-per-length reproduction
D6 two-seed confirmation
D7 formal M2 result package
```

M2 is the primary diffusion-specific positive target.

## 8. Phase E — M3: Schrödinger/path-KL regularized MEMIT

Read `SB_REGULARIZED_MEMIT_PLAN.md`.

Stage sequence:

```text
E1 define sparse-support path-KL and identity-state objective
E2 implement loss and tests
E3 tune regularization on cf_sb_dev_200 and KAMEL dev
E4 freeze one or two Pareto candidates
E5 evaluate on cf_sb_analysis_200 and locked KAMEL main
E6 mechanism ablations
E7 formal M3 result package
```

M3 failure does not stop M4.

## 9. Phase F — M4: exact mask-pattern Schrodinger bridge

Read `MASK_PATTERN_SB_PLAN.md`.

Stage sequence:

```text
F1 exact finite-state DP implementation and unit tests
F2 toy analytical verification
F3 inference integration with edited LLaDA
F4 KAMEL smoke and fixed-schedule comparisons
F5 tune beta/reference policy on KAMEL dev
F6 locked KAMEL main evaluation
F7 mechanism ablations and formal M4 result package
```

If M3 and M4 both fail to establish SB-specific lift, trigger F2 toy text CSBM.

## 10. Conditional fallback F1 — adaptive edit memory

Trigger only if M1 fails after its one rescue.

Read `ADAPTIVE_EDIT_MEMORY_FALLBACK_PLAN.md`.

Purpose:

```text
obtain a strong corrected-answer result in an MDM
separate editor-implementation failure from model inability
provide a positive engineering fallback
```

This fallback is not a parametric-editing or strong SB claim.

## 11. Conditional fallback F2 — toy categorical text CSBM

Trigger only if neither M3 nor M4 yields an SB-specific positive result.

Read `TOY_TEXT_CSBM_FALLBACK_PLAN.md`.

Purpose:

```text
validate a true categorical bridge on a controlled relational text distribution
obtain a positive algorithmic SB demonstration if possible
bound the claim honestly
```

## 12. Cross-track decision

After required tracks are terminal, create:

```text
cross_track/track_evidence_matrix.csv
cross_track/claim_matrix.md
cross_track/mechanism_ablation_matrix.csv
```

Classify each claim:

```text
supported
partially_supported
rejected_under_protocol
not_triggered
protocol_infeasible
```

Priority of final claims:

1. successful MDM-MEMIT reproduction;
2. successful partial-mask multi-token improvement;
3. positive SB locality/path-cost improvement;
4. positive mask-pattern SB trajectory improvement;
5. fallback engineering or toy result;
6. bounded negative result.

## 13. Final package and Pod stop

Create the final package required by `AGENTS.md`.

Validate:

```text
all mandatory tracks terminal
conditional tracks correctly triggered/not triggered
all track summaries present
all scientific thresholds evaluated
no old locked analysis/final data opened
artifact availability documented
final report and claim recommendation complete
```

Then:

1. mark `campaign_state.json` terminal;
2. mirror compact summaries locally/durably;
3. verify no active job;
4. stop the Pod;
5. return one consolidated report to the user.

Do not return for intermediate approvals.
