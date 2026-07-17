# AGENTS.md

Operational and scientific rules for the autonomous publication-confirmation
campaign for mask-pattern path control in edited masked diffusion language
models.

## 0. Active campaign identity

```text
active_campaign = mask_pattern_sb_publication_confirmation_v1
active_plan = MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_PLAN.md
primary_backbone = GSAI-ML/LLaDA-8B-Instruct
secondary_backbone = Dream-v0-Instruct-7B
primary_editor = MDM-adapted MEMIT
primary_method = exact finite-state entropy-regularized mask-pattern control
primary_task = multi-token factual editing
```

The campaign starts from a validated positive seed result:

```text
historical M1 reproduction:
  rewrite exact = 0.864
  paraphrase exact = 0.491

historical M4 pilot:
  target length 3 rewrite gain = +0.090
  target length 4 rewrite gain = +0.065 to +0.075
  trajectory target-cost reduction = approximately 24% to 26%
```

These historical values motivate the new campaign but are not the new locked
confirmation result.

Historical protocol directories are immutable evidence:

```text
counterfact_direction1_v1
counterfact_direction2_bridge_adapter_v1
counterfact_direction3_controller_v1
counterfact_sb_alternatives_campaign_v1
masked_diffusion_memit_sb_positive_result_v1
```

Codex must not overwrite, modify, resume, or tune those protocols. They may be
read only for artifact validation, baseline implementation, exclusion
fingerprints, and historical context.

## 1. Authoritative files and read order

Before acting, read in this order:

1. `AGENTS.md`
2. `ACTIVE_RESEARCH_CAMPAIGN.json`
3. `PUBLICATION_PROTOCOL_REGISTRY.json`
4. `MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_PLAN.md`
5. the plan file for the current stage
6. persisted campaign state under:
   `runs/mask_pattern_sb_publication_confirmation_v1/autonomous_campaign_v1/`

Persisted campaign state is authoritative for completed work. Root files define
policy and must not reset validated progress.

## 2. Autonomous Goal-mode authorization

Autonomous mode is enabled only when:

```bash
export MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_MODE=1
```

When enabled, Codex has one-time authorization to execute every task explicitly
listed in the authoritative plans without requesting per-stage, per-command,
per-model, or per-GPU-job approval.

Codex must not:

- expand the scientific scope beyond the listed tracks;
- lower acceptance thresholds after seeing results;
- invent additional scientific rescues;
- use historical locked analysis/final data for new tuning;
- silently change target-length definitions, model variants, datasets, or
  metrics;
- call the method a classical Schrödinger bridge if the formal naming audit
  concludes that it is KL-control/Doob-transformed path control;
- select the best random or generation seed;
- rerun locked confirmation data for tuning;
- treat additional model evaluations as free;
- claim an SB-specific contribution when deterministic planning,
  compute-matched search, or `beta -> infinity` performs equally or better.

## 3. Campaign completion

The campaign is complete only after every mandatory publication track is
terminal and the final publication-readiness package validates.

Mandatory tracks:

```text
P0 = source, artifact, and implementation audit
P1 = paper-matched partial-state MDM-MEMIT discrepancy resolution
P2 = exact mathematical formulation and naming audit
P3 = serious reveal-order and compute-matched baseline suite
P4 = fresh locked LLaDA confirmation at target lengths 2 through 6
P5 = second-backbone confirmation on Dream-v0-Instruct-7B
P6 = editor-generality evaluation
P7 = approximate-solver and scaling evaluation
P8 = statistical analysis, reproducibility, and paper package
```

Valid terminal outcomes:

```text
top_tier_ready
narrow_method_ready
diagnostic_only
fresh_confirmation_failed
publication_blocked_baseline_discrepancy
publication_blocked_infrastructure
unsafe_data_integrity_stop
```

A failed track does not stop the campaign unless the master plan marks the
track as a hard prerequisite. Codex must write a formal track result package and
continue whenever the remaining experiments are still scientifically valid.

## 4. RunPod lifecycle: no monetary budget guard

Money must not stop, skip, reorder, or weaken this campaign.

Cost may be logged for reporting, but there is no monetary budget state, no
budget-based stage block, and no budget-based Pod shutdown.

### Start

Start the existing configured Pod if it is stopped:

```bash
runpodctl pod start "$RUNPOD_POD_ID"
runpodctl pod list
```

Verify:

```text
Pod status is RUNNING
at least one GPU is allocated
SSH works
nvidia-smi works
persistent /workspace is available
/workspace/SB exists or can be cloned
```

### Keep running

Once the Goal begins, keep the Pod running through:

- source and code audits;
- implementation and tests;
- CPU and GPU stages;
- data construction;
- MDM-MEMIT editing;
- LLaDA and Dream experiments;
- baseline searches;
- exact and approximate solvers;
- bootstrapping and plotting;
- failed individual tracks;
- successful individual tracks;
- final report generation.

Do not stop the Pod because:

- one job finished;
- the next stage is CPU-only;
- a track failed or passed;
- the Pod is temporarily idle between planned tasks;
- execution is expensive;
- a previous campaign had a monetary stop;
- a long test is running only on CPU.

### Stop

Stop the Pod only when:

1. the full campaign reaches a validated terminal outcome and all compact
   artifacts are durable; or
2. an unrecoverable Pod/infrastructure failure remains after the configured
   retries.

Examples of unrecoverable infrastructure failures:

```text
Pod cannot start after retries
no GPU can be allocated after retries
SSH cannot be restored
/workspace is unavailable or corrupted
repeated CUDA/runtime crashes block the same validated command
required model access cannot be restored
```

A scientific negative result is not a Pod issue. Complete the entire remaining
valid campaign and final package before stopping.

Never terminate/delete a Pod unless the user explicitly requests deletion.

## 5. Required environment variables

```bash
export MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_MODE=1
export MASK_PATTERN_SB_MAX_INFRA_RETRIES="3"
export MASK_PATTERN_SB_MAX_SCIENTIFIC_RESCUES_PER_TRACK="1"

export RUNPOD_POD_ID="<existing-pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/<private-key-file>"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-host>"
export RUNPOD_SSH_PORT="<current-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

`runpodctl` must already be configured with a RunPod API key.

If SSH host/port changes after restart, refresh it through the RunPod
console/API/tooling. Never guess it.

Never print or commit:

```text
private SSH keys
RunPod API keys
Hugging Face tokens
OpenAI credentials
```

## 6. Python environments

### RunPod

Use the Python environment available in the image unless a compatible existing
project environment is already present:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<script>.py
```

Do not require `uv` on RunPod.

### Local MacBook

Outside the autonomous campaign, use:

```bash
uv sync
uv run pytest tests -q
uv run python scripts/<script>.py
```

During Goal mode, `/workspace/SB` is the authoritative worktree and Codex may
execute CPU stages on the Pod to avoid synchronization delays.

## 7. Git and code rules

Before each stage:

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

Commit and push code only after tests pass.

Do not commit:

```text
.env
credentials
model weights
full edited model snapshots
large covariance files
large raw generations
*.safetensors
*.pt
*.pth
*.ckpt
```

Large artifacts remain under `/workspace/SB/runs` or `/workspace/checkpoints`.
Every large artifact must be represented in an availability manifest with path,
size, hash when feasible, and reconstructability.

## 8. Long jobs

Use `tmux`, `set -o pipefail`, logs, and explicit exit-code files:

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && set -o pipefail && \
   python scripts/<script>.py <args> 2>&1 | tee logs/<stage_name>.log; \
   code=${PIPESTATUS[0]}; \
   echo "$code" > logs/<stage_name>.exitcode; \
   exit "$code"'
```

After launch, verify:

```bash
tmux ls
nvidia-smi
tail -n 100 logs/<stage_name>.log
```

A stage does not pass merely because the process exits with code 0. Every
scientific and artifact acceptance check must pass.

## 9. Campaign state

Maintain:

```text
runs/mask_pattern_sb_publication_confirmation_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  cost_state.json              # informational only
  artifact_availability.json
```

`campaign_state.json` must include:

```json
{
  "campaign_id": "mask_pattern_sb_publication_confirmation_v1",
  "autonomous_mode": true,
  "campaign_status": "running",
  "current_stage": "",
  "next_stage": "",
  "completed_stages": [],
  "failed_stages": [],
  "rescues_used": {},
  "track_status": {},
  "historical_analysis_500_used": false,
  "historical_final_test_used": false,
  "locked_confirmation_opened": false,
  "last_git_commit": "",
  "pod_status": ""
}
```

For every stage:

1. read current state and the relevant plan;
2. run preflight tests;
3. execute the stage;
4. validate all acceptance criteria;
5. write versioned artifacts, logs, and exit code;
6. update state and track registry;
7. automatically advance on pass;
8. apply only the explicitly permitted rescue on failure;
9. write a formal result package if no rescue remains.

## 10. Fresh data and split safety

Do not use historical `analysis_500`, `final_test_500`, or
`final_test_full` for this campaign.

Historical artifacts may be read only for:

```text
implementation comparison
source-row/fingerprint exclusion
historical result context
reproducibility checks
```

Create a new KAMEL protocol with disjoint development and locked-confirmation
sets. Exclude every KAMEL source row/fact/target fingerprint used in the previous
positive campaign.

Primary new data roles:

```text
kamel_pub_dev_*:
  method, beta, scheduler, and compute-policy selection only

kamel_pub_locked_*:
  one locked confirmation after dev lock

dream_pub_dev_*:
  Dream integration and bounded method selection only

dream_pub_locked_*:
  one locked Dream confirmation
```

No locked set may be read before its corresponding lock file validates.

After opening a locked set, do not change:

```text
editor implementation
layer range
target-value optimization
reference process
cost function
beta
planner
beam width
query budget
generation schedule
span policy
metrics
normalization
seeds
filters
bootstrap procedure
```

A rerun is allowed only for a documented infrastructure failure before results
were inspected.

## 11. Full-precision editing

MEMIT updates must be applied to editable floating-point MLP weights.

Primary settings:

```text
dtype = bfloat16 or float16
use_4bit = false
base weights remain editable for the selected matrices
```

Do not apply the closed-form MEMIT update directly to 4-bit weights.

CPU offloading, sequential edit batches, or sharded covariance computation are
allowed if they preserve the algorithm.

## 12. Statistical rules

Primary resampling unit:

```text
edit_id
```

Primary comparisons:

```text
exact finite-beta mask-pattern controller
vs best compute-matched non-SB planner
for target lengths 3 and 4
```

Required:

```text
10,000 paired bootstrap resamples
95% confidence intervals
Holm correction for the two primary length comparisons
at least 3 generation seeds
at least 5 random reveal-policy seeds
macro-by-relation and micro averages
no best-seed selection
```

Random-policy results must average over seeds.

Power analysis must be written before locked confirmation. The locked sample
size cannot be reduced after results are seen.

## 13. Compute fairness

Report separately:

```text
LLaDA/Dream forward evaluations
unique mask states evaluated
candidate path evaluations
planner CPU time
GPU time
wall-clock time
peak memory
```

Two evaluation regimes are mandatory:

1. **full cost-table regime**  
   All global planners receive the same precomputed state-cost table. This
   isolates planning/control quality.

2. **online compute-matched regime**  
   Each method receives the same state-query or forward-pass budget.

The paper claim must survive the online compute-matched comparison. A gain only
under more model evaluations is not sufficient for a top-tier method claim.

## 14. Mathematical naming rules

The formal audit must determine whether the method is:

```text
a classical endpoint-constrained Schrödinger bridge
a generalized Schrödinger bridge
linearly-solvable KL control / path-space control
a Doob/Feynman-Kac transformed reference process
entropy-regularized global planning
```

The repository, tables, plots, and final paper title must use the conclusion of
that audit.

If the endpoint marginal is trivial because all monotone paths terminate in the
fully revealed mask pattern, do not call the method a classical Schrödinger
bridge without qualification.

## 15. Bounded rescue rules

Allowed scientific rescues:

```text
P1 partial-state MEMIT:
  one implementation/protocol correction after the source audit

P4 Dream integration:
  one model-adapter/module-mapping correction

P7 approximate solver:
  one approximation-configuration correction
```

No other scientific rescue is permitted.

Never:

```text
lower thresholds after seeing results
move locked examples into dev
add new baselines only after seeing a failure
change primary metrics after locked confirmation
select favorable seeds
replace Dream with another model without recording a weaker fallback status
```

## 16. Completion package

The final directory is:

```text
runs/mask_pattern_sb_publication_confirmation_v1/
  final_publication_package_v1/
```

It must include:

```text
report_summary.json
top_tier_readiness.json
main_results_table.csv
compute_matched_table.csv
second_backbone_table.csv
editor_generality_table.csv
target_length_table.csv
beta_ablation.csv
planner_ablation.csv
same_subject_stress_table.csv
malformed_and_locality_table.csv
paired_bootstrap.csv
power_analysis.json
theory_statement.md
naming_decision.md
complexity_analysis.md
trajectory_examples.md
failure_cases.csv
artifact_availability.json
reproducibility_manifest.json
final_research_report.md
paper_outline.md
paper_claim_recommendation.md
```

Readiness classifications:

```text
top_tier_ready
narrow_method_ready
diagnostic_only
fresh_confirmation_failed
```

After the final package validates, mark campaign state terminal and stop the
Pod.
