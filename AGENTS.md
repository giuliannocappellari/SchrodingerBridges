# AGENTS.md

Operational and scientific rules for the autonomous masked-diffusion knowledge-editing campaign.

## 0. Active campaign

```text
active_campaign = masked_diffusion_memit_sb_positive_result_v1
active_plan = MDM_MEMIT_SB_AUTONOMOUS_RESEARCH_PLAN.md
primary_model = GSAI-ML/LLaDA-8B-Instruct
primary_positive_baseline = MDM-adapted MEMIT
primary_diffusion_extension = partial-mask MEMIT
primary_SB_extensions = Schrodinger-regularized MEMIT, exact mask-pattern SB
```

Historical protocols are immutable evidence:

```text
counterfact_direction1_v1 = closed_blocked
counterfact_direction2_bridge_adapter_v1 = closed_protocol_infeasible
counterfact_direction3_controller_v1 = closed_bounded_negative
counterfact_sb_alternatives_campaign_v1 = closed_scientific_negative
```

Codex must not modify, overwrite, resume, or reinterpret historical runs. They may be read as baselines and evidence only.

## 1. Authoritative files and read order

Before acting, read in this order:

1. `AGENTS.md`
2. `ACTIVE_RESEARCH_CAMPAIGN.json`
3. `EXPERIMENT_PROTOCOL_REGISTRY.json`
4. `MDM_MEMIT_SB_AUTONOMOUS_RESEARCH_PLAN.md`
5. the active track plan for the current stage
6. persisted campaign state under `runs/masked_diffusion_memit_sb_positive_result_v1/autonomous_campaign_v1/`

The persisted campaign state is authoritative for completed stages. Root files define policy and must not reset validated progress.

## 2. Autonomous Goal-mode authorization

Autonomous mode is enabled when:

```bash
export MDM_MEMIT_SB_AUTONOMOUS_MODE=1
```

When enabled, Codex has one-time authorization to execute every task explicitly listed in the authoritative plan without asking for per-stage, per-command, per-edit, or per-GPU-job approval.

Codex must not:

- expand the scientific scope beyond the plans;
- lower hard acceptance thresholds after seeing results;
- invent additional rescues beyond the bounded rescue written for a stage;
- switch to a different research direction;
- open locked historical analysis/final splits without the new campaign lock;
- silently replace datasets, target-length bins, models, or evaluation definitions;
- claim an SB contribution when the closest non-SB ablation performs equally or better.

## 3. Campaign completion

The campaign is complete only when all mandatory tracks are terminal and the final cross-track package validates.

Mandatory tracks:

```text
M1 = MDM-MEMIT reproduction on LLaDA-8B-Instruct / CounterFact
M2 = partial-mask MEMIT reproduction on multi-token KAMEL
M3 = Schrodinger/path-KL regularized partial-mask MEMIT
M4 = exact mask-pattern/reveal-order Schrodinger bridge
```

Conditional fallback tracks:

```text
F1 = adaptive external edit-memory guidance
F2 = fixed-template categorical text CSBM sanity experiment
```

F1 is mandatory only if M1 cannot achieve a positive editing result after its bounded rescue.
F2 is mandatory only if both M3 and M4 fail to establish any SB-specific positive result.

Valid terminal outcomes:

```text
positive_completion
mixed_completion_positive_editing_negative_SB
formal_negative_completion
infrastructure_blocked
unsafe_data_integrity_stop
```

A track failure does not stop the campaign. Codex must write the track stop package and continue to the next required track.

## 4. RunPod lifecycle: no monetary budget guard

Monetary budget must not block, reorder, skip, or stop scientific work.

Codex may record elapsed Pod hours and estimated spend for reporting, but cost is informational only.

### Start

Start the existing configured Pod if it is stopped:

```bash
runpodctl pod start "$RUNPOD_POD_ID"
runpodctl pod list
```

Verify:

```text
Pod status RUNNING
at least one GPU allocated
SSH works
nvidia-smi works
/workspace persistent storage is available
/workspace/SB exists or can be cloned
```

### Keep running

Once the autonomous campaign begins, keep the Pod running through:

- CPU and GPU stages;
- code implementation;
- tests;
- dataset construction;
- covariance/feature caching;
- editing;
- evaluation;
- reporting;
- failed individual tracks;
- successful individual tracks;
- final cross-track analysis.

Do not stop the Pod because:

- a job finished;
- the next task is CPU-only;
- one track passed or failed;
- the Pod is temporarily idle between planned stages;
- estimated cost is high;
- a previous campaign had a budget stop.

### Stop

Stop the Pod only when:

1. the entire campaign reaches a validated terminal outcome and the final package is durable; or
2. an unrecoverable Pod/infrastructure issue remains after the configured retries.

Examples of unrecoverable Pod issues:

```text
Pod cannot start after retries
no GPU allocation after retries
SSH cannot be restored
/workspace volume is unavailable or corrupted
repeated CUDA/runtime failures prevent the same validated command from running
```

Before stopping after scientific completion:

```text
all required terminal artifacts exist
campaign_state.json is terminal
stage_history.csv is complete
final package validation passes
compact summaries are mirrored locally or pushed to durable storage
no Python/GPU job is active
```

Never terminate/delete the Pod unless the user explicitly requests deletion.

## 5. RunPod connection and Python

Required environment variables:

```bash
export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

`runpodctl` must already be configured with a RunPod API key.

If host/port changes after restart, refresh it using the RunPod console/API/tooling. Never guess.

Use the Python environment available in the RunPod image:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script>.py
```

Do not require `uv` on RunPod.

## 6. Local MacBook rules

Outside the active autonomous Pod campaign, use `uv` locally:

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script>.py
```

Do not use `pip install` directly in the local project environment.

During autonomous Goal mode, `/workspace/SB` is the authoritative worktree. Codex may run CPU work on the Pod to avoid synchronization pauses.

## 7. Git and code rules

Before every stage:

```bash
cd /workspace/SB
git status
git pull
python -m pytest tests -q
```

After code changes:

```bash
git diff
git status
python -m pytest tests -q
```

Commit code only after tests pass. Push compact code/config/report checkpoints.

Do not commit:

```text
.env
private keys
tokens
RunPod API keys
model weights
full model checkpoints
large covariance caches
large run artifacts
*.safetensors
*.pt
*.pth
*.ckpt
```

Large artifacts stay under `/workspace/SB/runs` or `/workspace/checkpoints` and must have hashes and availability manifests.

## 8. Long-job execution

Use `tmux`, `set -o pipefail`, logs, and explicit exit-code files:

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage_name>.log; \
   code=${PIPESTATUS[0]}; echo "$code" > logs/<stage_name>.exitcode; exit "$code"'
```

Every long stage must record:

```text
tmux session
command
start/end UTC
Git commit
log path
exit-code path
output directory
GPU name
Python/CUDA/torch/transformers versions
```

## 9. Model and quantization policy

Primary reproduction model:

```text
GSAI-ML/LLaDA-8B-Instruct
```

MDM-MEMIT modifies MLP weights. The primary reproduction must therefore use editable floating-point weights:

```text
dtype = float16 or bfloat16
use_4bit = false
```

Do not apply the closed-form MEMIT update directly to 4-bit quantized weights.

If the A40 cannot fit the primary configuration:

1. use CPU offload or layer-wise loading;
2. keep edited MLP weights and required activations in fp32/fp16;
3. reduce batch/shard size, not the scientific method;
4. record the deviation.

A 4-bit run may be used only as a non-primary diagnostic and must never replace the full-precision reproduction silently.

## 10. Primary sources

Codex must inspect and cite these primary sources in implementation notes:

```text
arXiv:2606.03924 — Knowledge Editing in Masked Diffusion Language Models
arXiv:2210.07229 — MEMIT
Official MEMIT repository / official EasyEdit implementation
arXiv:2502.01416 — Categorical Schrodinger Bridge Matching
arXiv:2603.17677 — Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models (fallback only)
```

As of arXiv:2606.03924 v1, the authors state that code will be released upon publication. If no official code is available, implement from the paper and pin any reused MEMIT/EasyEdit source commit.

Do not copy untrusted third-party code without audit.

## 11. New protocol and historical split safety

Active campaign root:

```text
runs/masked_diffusion_memit_sb_positive_result_v1/
```

Do not tune on or overwrite historical:

```text
counterfact_direction1_v1/analysis_500
counterfact_direction1_v1/final_test_500
counterfact_direction1_v1/final_test_full
```

The new campaign must build fresh manifests from the official source datasets, excluding all historical case IDs/fingerprints used for tuning, analysis, final testing, or legacy test50 inspection.

Locked historical manifests may be read only for IDs, source split, source index, and fingerprints used for exclusion auditing.

## 12. New campaign split discipline

### CounterFact

Build fresh disjoint manifests:

```text
cf_memit_smoke_20
cf_layer_select_500
cf_repro_main_500
cf_sb_dev_200
cf_sb_analysis_200
```

Rules:

```text
smoke_20 may be used for integration
layer_select_500 selects the 4-layer window only
repro_main_500 is a locked reproduction set
sb_dev_200 tunes only SB regularization/reveal policy
sb_analysis_200 is proceed/stop for the SB extensions
```

No method or threshold change after inspecting `cf_repro_main_500` for the reproduction claim or `cf_sb_analysis_200` for the SB claim.

### KAMEL

Build fresh disjoint manifests adapted to the CounterFact schema:

```text
kamel_smoke_20_per_length
kamel_dev_50_per_length
kamel_repro_200_per_length
```

Target lengths:

```text
N in {1,2,3,4}
```

Follow the paper construction where feasible:

- start from single-answer KAMEL facts;
- compute contextual object token length;
- create a counterfactual target by swapping in another object from the same relation with the same target length;
- use real relation templates;
- create one held-out paraphrase per relation;
- record relation and tokenizer coverage.

The main KAMEL reproduction must use 200 examples per target length if available.

## 13. Training/evaluation separation

For MEMIT target-value optimization, allowed inputs are the rewriting prompt, target object, context templates, and explicitly predeclared KL anchors.

Evaluation-only prompts include:

```text
held-out paraphrases
classic specificity/neighborhood prompts
same-subject different-relation stress prompts
generation/attribute prompts
fresh unrelated prompts
```

Do not add evaluation prompts to target-value optimization.

All rows must record `train_seen` and prompt provenance.

## 14. Metrics

Report at minimum:

```text
efficacy/rewrite exact
generalization/paraphrase exact
classic specificity
same-subject target false-positive rate
near/far target false-positive rate
generation target false-positive rate
full-target exact
token-level target coverage
malformed rate
old-target suppression
base-vs-edited sparse KL at answer positions
partial-state path KL / intervention cost
weight-update norm
editing GPU time
inference GPU time
model evaluations
storage/update bytes
```

For stochastic decoding, report mean exact, greedy exact, and pass@k where applicable.

Use paired bootstrap by edit ID for comparisons.

## 15. Scientific wording

Use precise labels:

```text
MDM-MEMIT reproduction
partial-mask MDM-MEMIT
Schrodinger/path-KL-regularized MEMIT
exact mask-pattern Schrodinger bridge
adaptive edit-memory guidance
toy categorical text CSBM
```

Do not call path-KL regularization a solved/full Schrödinger bridge. Use `Schrodinger-regularized` or `minimum-intervention path regularization` unless the algorithm satisfies an actual bridge formulation.

Do not claim mask-pattern SB improves factual editing unless it beats the best fixed/random reveal schedule under the predeclared mechanism criteria.

## 16. Bounded rescues

Each mandatory track receives at most one scientific rescue explicitly listed in its plan.

Infrastructure retries are separate and may be repeated up to:

```bash
export MDM_MEMIT_SB_MAX_INFRA_RETRIES="3"
```

Never rescue by:

```text
lowering acceptance thresholds
adding evaluation data to optimization
changing the locked set after inspection
removing difficult target lengths silently
using forced target filling as a ranked method
relabeling a non-SB baseline as SB
```

## 17. Track continuation policy

- M1 failure after rescue: write formal M1 negative; execute F1 and continue M2 only if the editor implementation is scientifically usable.
- M2 failure after rescue: write formal M2 negative; continue M3/M4 if a working MEMIT baseline exists, otherwise execute F1/F2.
- M3 failure: write formal M3 negative and continue M4.
- M4 failure: write formal M4 negative; if M3 also failed, execute F2.
- A positive M1 or M2 result does not end the campaign; all mandatory SB extensions must still be tested.

## 18. Final package

Create:

```text
runs/masked_diffusion_memit_sb_positive_result_v1/final_research_package_v1/
```

Required:

```text
report_summary.json
track_registry_final.json
main_results_table.csv
counterfact_reproduction_table.csv
kamel_partial_mask_table.csv
sb_regularization_table.csv
mask_pattern_sb_table.csv
same_subject_stress_table.csv
target_length_table.csv
compute_storage_table.csv
paired_bootstrap.csv
rewrite_locality_pareto.png
multi_token_gain_plot.png
sb_mechanism_plot.png
failure_cases.csv
artifact_availability_manifest.json
reproducibility_manifest.json
final_research_report.md
paper_claim_recommendation.md
```

Claim categories:

```text
successful diffusion-MEMIT reproduction
successful multi-token partial-mask improvement
positive Schrodinger-regularization result
positive exact mask-pattern SB result
engineering-positive fallback only
bounded negative result
```

The claim must follow the evidence.

## 19. Codex behavior

Codex must:

- continue automatically between planned stages in autonomous mode;
- keep the Pod running until the entire goal is terminal or unrecoverable Pod failure occurs;
- preserve old campaigns;
- run tests before and after code changes;
- validate every acceptance criterion;
- write track stop packages on bounded failure;
- continue to the next required track;
- return to the user only after the complete final package validates and the Pod is stopped.

Codex must not:

- ask for intermediate approval;
- stop the Pod after an individual job or track;
- create/delete a Pod;
- open old locked analysis/final splits;
- expand the plan;
- hide negative results;
- overstate novelty or mechanism-specific evidence.
