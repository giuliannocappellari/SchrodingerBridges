# CounterFact Direction 1 Notebook Plan

This file maps the current Colab notebooks to the frozen
`counterfact_direction1_v1` protocol.

The notebooks live in:

```text
runtime bridge editing/
```

They do not map one-to-one to every numbered core experiment. Some notebooks are
engineering preflights, and Step 3A intentionally bundles the first dev-only
runtime comparison so all methods can share one edit-balanced report.

## Notebook Map

| Notebook | Protocol Coverage | Purpose | Split Access | Main Outputs |
| --- | --- | --- | --- | --- |
| `First_Step_Build_The_Protocol_Manifests.ipynb` | Core Experiment 1: Protocol manifests | Build official manifests and validate deterministic splits, overlap, fingerprints, target-length bins, and protocol metadata. | HF train/test only through manifest builder | `runs/counterfact_direction1_v1/protocol/*.jsonl`, `protocol_manifest.json` |
| `Step 2A - Tiny Runtime Smoke.ipynb` | First-sprint smoke tests; Core Experiment 3: G0 Candidate Support | Verify GPU runtime, LLaDA loading, protocol artifact loading, sprint-1 method registry, MC bridge execution, gated bridge execution, and full `dev_tune_200` candidate support. | `dev_tune_200` only | `dev_tune_200_smoke_*`, `dev_tune_200_base_coverage/candidate_coverage.jsonl`, G0 coverage summary |
| `Step 2B - Base Self-Consistency.ipynb` | Core Experiment 2: Base/self-consistency | Run frozen base LLaDA twice with different seeds and compute edit-balanced `SelfLoc_base`, base TFPR, malformed rates, and base success. | `dev_tune_200` only | `dev_tune_200_base_seed0`, `dev_tune_200_base_seed1`, `dev_tune_200_base_selfconsistency_report/base_selfconsistency_summary.json` |
| `Step 3A - Dev Runtime Baselines.ipynb` | Core Experiment 4: Raw MC bridge; Core Experiment 5: Runtime baselines; first pass of Core Experiment 6 and 8 | Run the first dev-only comparison of base, simple runtime baselines, bridge controls, MC bridge, and simple gated bridge. Build edit-balanced aggregate, target-length, coverage, and bootstrap tables. | `dev_tune_200` only | `dev_tune_200_runtime_baselines_*`, `dev_tune_200_runtime_baseline_report/*.csv` |
| `Step 3A.1 - Patch Runtime Baseline Report.ipynb` | Metrics and selection patch for Core Experiments 4-6 | Recompute the existing Step 3A report without rerunning decoding. Uses declarative paraphrases as the primary paraphrase metric and applies feasibility constraints. | Existing `dev_tune_200` outputs only | `dev_tune_200_runtime_baseline_report_v3/*.csv` |
| `Step 3B - Dev Gated Runtime Baselines.ipynb` | Core Experiment 9: Gating/locality, first fair-baseline pass | Run subject-gated and subject+relation-gated runtime baselines on `dev_tune_200`, then build a merged feasible report with gate activation summaries. | `dev_tune_200` only | `dev_tune_200_gated_*`, `dev_tune_200_runtime_baseline_report_gated_v1/*.csv` |
| `Step 3C - Dev Matched KL Sweep.ipynb` | Focused follow-up to Core Experiment 6, 8, and 10 | Reuse the Step 3B subject-gated contenders, sweep guidance scale for bridge controls and MC bridge, and build matched-KL, matched-rewrite, matched-bootstrap, feasible-ranking, constraint, and target-length comparison artifacts. | `dev_tune_200` only | `dev_tune_200_sweep_*`, `dev_tune_200_matched_kl_sweep_subject_v1/*` |
| `Step 3C.1 - Dev Dense Pareto Sweep.ipynb` | Dense follow-up to Core Experiment 6 and 10 | Add intermediate subject-gated guidance scales `1.25`, `1.5`, and `1.75` for myopic, no-rollout, and MC bridge; report best feasible methods under fixed KL budgets, including a guided-only KL-budget view and summarized target-length table. | `dev_tune_200` only | `dev_tune_200_dense_pareto_*`, `dev_tune_200_dense_pareto_subject_v1/*` |
| `Step 3D - Same Subject Locality Stress.ipynb` | Locality stress extension before dev lock | Build a dev-only same-subject stress input outside `protocol/`, then test high-guidance subject-gated contenders for target over-injection on same-subject non-edit prompts with stress-relative base budgets and paired bootstrap deltas. | `dev_tune_200` only | `same_subject_stress_inputs/*`, `dev_tune_200_same_subject_stress_*`, `dev_tune_200_same_subject_stress_report_v1/*` |
| `Step 3D.1 - Mid KL Same Subject Stress.ipynb` | Mid-KL locality stress before dev lock | Reuse the Step 3D stress input and high-guidance carry-forward outputs, run mid-guidance subject-gated contenders, validate carry-forward completeness, join stress leakage with dense Pareto edit metrics, and produce the stress-vs-edit decision table with explicit edit-usefulness thresholds. | `dev_tune_200` only | `dev_tune_200_same_subject_stress_*gs150`, `dev_tune_200_same_subject_stress_*gs175`, `dev_tune_200_same_subject_stress_midkl_v1/*` |
| `Step 3E.0 - Hybrid Gate Replay.ipynb` | Gate-only replay before any new decoding | Compose existing subject-gated outputs with base outputs under lexical, relation-bank, and hybrid relation-OR edit-intent gates. Reject gates that cannot preserve rewrite/paraphrase activation while passing same-subject stress. | Existing `dev_tune_200` outputs only | `dev_tune_200_hybrid_gate_replay_v1/*` |
| `Step 3E.1 - Hybrid Gate Actual Decode.ipynb` | Actual decode validation before dev lock | Run actual LLaDA decoding with the selected `hybrid_or_rel0.45_bank0.10` gate for top empirical, bridge, no-rollout, and prompt-memory candidates. Compare actual metrics against replay and same-subject stress budgets. | `dev_tune_200` only | `dev_tune_200_hybrid_decode_*`, `dev_tune_200_hybrid_gate_decode_v1/*` |

## Current Execution Status

| Notebook | Status | Notes |
| --- | --- | --- |
| `First_Step_Build_The_Protocol_Manifests.ipynb` | Complete | Official manifests exist and show no disallowed overlap. |
| `Step 2A - Tiny Runtime Smoke.ipynb` | Complete | Tiny smoke, method-registry smoke, MC bridge smoke, gated bridge smoke, and full G0 coverage passed. |
| `Step 2B - Base Self-Consistency.ipynb` | Complete after regenerated report | Seed runs completed. The official report must use `edit_id` as the balance unit and `sample_self_agreement_paired_edit_balanced` as the main self-locality denominator. |
| `Step 3A - Dev Runtime Baselines.ipynb` | Complete or ready, depending on local Colab state | Produces the first ungated dev baseline comparison. It should not touch `analysis_500`, `ablation_500`, or final-test data. |
| `Step 3A.1 - Patch Runtime Baseline Report.ipynb` | Complete or ready, depending on local Colab state | Does not rerun decoding. Recomputes Step 3A report with feasible selection logic into `report_v3`. |
| `Step 3B - Dev Gated Runtime Baselines.ipynb` | Complete or ready, depending on local Colab state | Runs fair gated variants before any move to `analysis_500`. |
| `Step 3C - Dev Matched KL Sweep.ipynb` | Ready after Step 3B | Focused subject-gated matched-KL / matched-guidance sweep among the serious contenders. |
| `Step 3C.1 - Dev Dense Pareto Sweep.ipynb` | Ready after Step 3C | Fills in the high-KL guidance grid before selecting a dev Pareto point. |
| `Step 3D - Same Subject Locality Stress.ipynb` | Ready after Step 3C | Stress-tests whether the subject gate over-activates on same-subject non-edit prompts. |
| `Step 3D.1 - Mid KL Same Subject Stress.ipynb` | Ready after Step 3C.1 and Step 3D | Tests whether mid-guidance subject-gated methods survive same-subject stress before any hybrid/edit-intent gate work. |
| `Step 3E.0 - Hybrid Gate Replay.ipynb` | Ready after Step 3D.1 | Cheap replay-only edit-intent gate search. Run this before spending GPU on hybrid-gated decoding. |
| `Step 3E.1 - Hybrid Gate Actual Decode.ipynb` | Ready after Step 3E.0 | Confirms whether the selected hybrid gate replay survives actual decoding before any move to `analysis_500`. |

## How Step 3A Relates To The Frozen Protocol

`Step 3A - Dev Runtime Baselines.ipynb` bundles several early dev-only checks:

```text
Core Experiment 4: Raw MC bridge
  Runs mc_bridge with the initial config:
  steps=4
  bridge_topk=4
  mc_rollouts=2
  guidance_scale=1.0
  reward_mode=soft_overlap
  reward_beta=6.0

Core Experiment 5: Runtime baselines
  Compares:
  base
  target_logit_bias
  prompt_memory
  target_candidate_insert
  myopic_score
  no_rollout_bridge
  mc_bridge
  raw_bridge_gated

Core Experiment 6: Bridge mechanism ablation, first pass
  Includes:
  myopic_score
  no_rollout_bridge
  mc_bridge

  The full matched-rewrite or matched-sparse-KL analysis is not complete until
  later ablation/sweep notebooks.

Core Experiment 8: Target-length analysis, first report pass
  Generates:
  aggregate_target_length.csv

  This is an initial breakdown, not the final target-length experiment.
```

## Not Yet Covered By Current Notebooks

The following frozen-protocol sections still need separate notebooks or scripts:

```text
Core Experiment 6:
  Full bridge mechanism ablation at matched rewrite or matched sparse guidance KL.
  Step 3C and Step 3C.1 cover the current dev-only matched-KL path.

Core Experiment 7:
  Diffusion schedule and active-step sweep.

Core Experiment 8:
  Full target-length/span analysis.

Core Experiment 9:
  Reward-threshold and hybrid gates, if needed after Step 3B.

Core Experiment 10:
  Compute-quality Pareto and final dev method selection.
  Step 3C covers the first focused matched-KL / matched-rewrite decision pass,
  and Step 3C.1 densifies the Pareto grid, but neither is the full final dev lock.

Core Experiment 11:
  Locked final evaluation on final_test_500.
```

`path_kl_bridge` is still staged as required before any valid `analysis_500`
run, but it is not required for the first dev smoke sprint.

## Split Discipline

Current notebooks obey this split discipline:

```text
Step 1:
  Builds manifests from the official HF train/test splits.

Step 2A:
  Uses dev_tune_200 only.

Step 2B:
  Uses dev_tune_200 only.

Step 3A:
  Uses dev_tune_200 only.
```

No current runtime notebook should inspect or evaluate:

```text
analysis_500
ablation_500
final_test_500
final_test_full
```

`analysis_500` remains proceed/stop only and cannot be used for tuning. Final
test remains locked.

## Artifact Discipline

The frozen protocol directory should be treated as read-only after Step 1:

```text
runs/counterfact_direction1_v1/protocol/
```

Derived smoke inputs and runtime outputs should live outside that directory,
for example:

```text
runs/counterfact_direction1_v1/smoke_inputs/
runs/counterfact_direction1_v1/dev_tune_200_*/
```

All runtime run configs must preserve:

```text
protocol_version = counterfact_direction1_v1
edit_access = given_at_edit_time
training_access = none
hyperparameter_access = dev_tune_only
```

Report aggregation and bootstrap should balance by `edit_id` when present, not
by prompt rows.
