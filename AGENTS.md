# RUNPOD_CODEX_RULES.md

Rules for using RunPod SSH with Codex for `counterfact_direction1_v1`.

## 0. Project identity

This repository is for the LLaDA / CounterFact runtime-editing project:

- Protocol: `counterfact_direction1_v1`
- Main model: `GSAI-ML/LLaDA-8B-Base`
- Current research direction: runtime bridge editing / runtime guided editing
- Current split for tuning: `dev_tune_200`
- `analysis_500` and final-test splits are locked and must not be used for tuning.

Every run config must record:

```text
protocol_version = counterfact_direction1_v1
edit_access = given_at_edit_time
training_access = none
hyperparameter_access = dev_tune_only
```

## 1. What should run locally on the MacBook

Run these locally unless the dataset/files are too large:

- Code editing and refactoring
- Unit tests with fake tokenizer/model stubs
- CSV/report aggregation
- Plot generation
- Gate-only replay
- Gate parity audit
- Actual-gate activation grid
- Protocol validation
- Split/manifest checks
- Paired bootstrap over existing results
- Reading/writing markdown reports
- Creating notebooks that call scripts

Local-only or CPU-safe steps:

```text
Step 3E.2 — Hybrid Gate Parity Audit
Step 3E.3 — Actual-Gate Activation Grid
report regeneration
plotting
unit tests
small data inspections
```

Do not use the RunPod GPU for cheap CSV/report work unless it is already running for a GPU experiment.

## 2. What should run on RunPod

Use RunPod only for GPU-required work:

- Loading LLaDA-8B
- Any actual model decoding
- `mc_bridge` decoding
- `no_rollout_bridge` or `myopic_score` decoding over many edits
- GPU smoke tests
- Any future actual hybrid-gate decode
- `analysis_500` after the dev method is locked
- final-test evaluation after analysis confirmation
- any learned controller / adapter training that needs CUDA

GPU-required examples:

```text
Step 3E.4 — actual decode with stricter hybrid gate
future analysis_500 confirmation
future final_test_500 locked run
Direction 2 per-edit adapter pilot
Direction 3 learned controller training
```

## 3. RunPod cost discipline

Use RunPod as a temporary GPU workstation, not as always-on storage.

Rules:

1. Start the Pod only when a GPU job is ready.
2. Work inside `tmux` for long runs.
3. Stop the Pod when no GPU job is running.
4. Keep important project data in `/workspace` or a network volume.
5. Back up final artifacts to Git, local machine, Google Drive, or another durable store.
6. Do not rely on container disk for persistent data.
7. Do not leave the Pod running overnight unless an active run is executing.
8. Schedule an automatic stop for long runs when possible.

Suggested manual stop command:

```bash
runpodctl pod stop $RUNPOD_POD_ID
```

Suggested safety stop after N hours:

```bash
sleep 6h; runpodctl pod stop $RUNPOD_POD_ID &
```

## 4. Storage rules

Use:

```text
/workspace/SB
```

as the main repository directory on RunPod.

Recommended layout:

```text
/workspace/SB/                         # git repo
/workspace/SB/runs/                    # experiment artifacts
/workspace/.cache/huggingface/         # model cache if configured
/workspace/checkpoints/                # optional training checkpoints
```

Never assume data outside `/workspace` will survive a stop.

Before terminating a Pod, back up:

```text
runs/counterfact_direction1_v1/<new_report_or_run_dir>/
```

## 5. SSH and tmux rules

Always connect by SSH for long-running work.

After connecting:

```bash
cd /workspace/SB
tmux new -s sb
```

Inside tmux:

```bash
source .venv/bin/activate
nvidia-smi
git status
```

Detach from tmux:

```text
Ctrl-b then d
```

Reattach:

```bash
tmux attach -t sb
```

## 6. Git rules

Code moves through Git. Large artifacts do not.

Before running a GPU job:

```bash
git status
git pull
python -m pytest tests -q
```

After Codex changes code:

```bash
git diff
git status
```

Commit code changes only after tests pass.

Do not commit:

```text
.env
HF tokens
OpenAI tokens
RunPod API keys
model weights
large run artifacts
```

`.gitignore` should include at least:

```text
.env
runs/
.cache/
*.pt
*.pth
*.safetensors
*.ckpt
__pycache__/
.ipynb_checkpoints/
```

If run artifacts need to be versioned, export summarized CSV/JSON/markdown only.

## 7. Secret rules

Never print or commit secrets.

Allowed secret locations:

```text
.env
shell environment variables
RunPod secret/environment configuration
```

Required secrets may include:

```text
HF_TOKEN
OPENAI_API_KEY / Codex auth if needed
WANDB_API_KEY only if used
RUNPOD_API_KEY only if using runpodctl automation
```

Never paste private SSH keys into notebooks, source files, markdown, or prompts.

## 8. Split locks and safety guards

Codex must not run `analysis_500` or final-test commands unless explicitly unlocked.

Every script that can run a split must enforce:

```python
import os

if split_role == "analysis_500":
    assert os.environ.get("DEV_METHOD_LOCKED") == "1", (
        "analysis_500 is locked. Set DEV_METHOD_LOCKED=1 only after freezing the dev method."
    )

if split_role.startswith("final_test"):
    assert os.environ.get("FINAL_METHOD_LOCKED") == "1", (
        "final_test is locked. Set FINAL_METHOD_LOCKED=1 only after analysis confirmation."
    )
```

Default split for all tuning is:

```text
dev_tune_200
```

No threshold, gate, method, span policy, normalization, metric, or filtering choice may be changed after inspecting `analysis_500`.

## 9. Current next tasks

The next immediate tasks are CPU/report tasks first:

```text
Step 3E.2 — Hybrid Gate Parity Audit
Step 3E.3 — Actual-Gate Activation Grid
```

Do not decode LLaDA for these steps.

Only after a stricter actual gate passes activation criteria should Codex run GPU decoding:

```text
Step 3E.4 — Actual Decode With Stricter Hybrid Gate
```

## 10. Step 3E.2 requirements

Goal: explain why Step 3E.0 replay and Step 3E.1 actual decoding disagreed on stress activation.

Do not run LLaDA.
Do not use `analysis_500`.
Do not use final-test splits.

Inputs:

```text
runs/counterfact_direction1_v1/dev_tune_200_hybrid_gate_replay_v1/
runs/counterfact_direction1_v1/dev_tune_200_hybrid_gate_decode_v1/
runs/counterfact_direction1_v1/same_subject_stress_inputs/
runs/counterfact_direction1_v1/protocol/dev_tune_200.jsonl
```

Outputs:

```text
runs/counterfact_direction1_v1/dev_tune_200_hybrid_gate_parity_audit_v1/report_summary.json
runs/counterfact_direction1_v1/dev_tune_200_hybrid_gate_parity_audit_v1/gate_feature_parity.csv
runs/counterfact_direction1_v1/dev_tune_200_hybrid_gate_parity_audit_v1/gate_activation_parity_summary.csv
runs/counterfact_direction1_v1/dev_tune_200_hybrid_gate_parity_audit_v1/gate_mismatch_samples.csv
runs/counterfact_direction1_v1/dev_tune_200_hybrid_gate_parity_audit_v1/gate_threshold_diagnostics.csv
```

Required checks:

- Use the same normalization as actual decode.
- Use the same relation-bank construction as actual decode.
- Compare replay gate activation and actual gate activation by bucket.
- Report activation drift by bucket.
- Export mismatch samples.
- Add or update tests for gate normalization and relation similarity.

## 11. Step 3E.3 requirements

Goal: search stricter gates using the actual runtime gate implementation before spending GPU time.

Do not run LLaDA.
Do not use `analysis_500`.
Do not use final-test splits.

Sweep at least:

```text
hybrid_or_rel0.45_bank0.15
hybrid_or_rel0.45_bank0.20
hybrid_or_rel0.45_bank0.25
hybrid_or_rel0.45_bank0.30
hybrid_and_rel0.30_bank0.10
hybrid_and_rel0.30_bank0.15
```

Evaluate activation on:

```text
rewrite
declarative_paraphrases
qa_format_generalization
near_locality
far_locality
same_subject_template
generation
```

Acceptance criteria:

```text
rewrite activation >= 0.95
declarative paraphrase activation >= 0.85
same_subject_template activation <= 0.05
generation activation <= 0.10 ideally
near_locality activation <= 0.02
far_locality activation = 0
```

Outputs:

```text
runs/counterfact_direction1_v1/dev_tune_200_actual_gate_activation_grid_v1/report_summary.json
runs/counterfact_direction1_v1/dev_tune_200_actual_gate_activation_grid_v1/gate_activation_grid.csv
runs/counterfact_direction1_v1/dev_tune_200_actual_gate_activation_grid_v1/best_actual_gate_candidates.csv
runs/counterfact_direction1_v1/dev_tune_200_actual_gate_activation_grid_v1/gate_activation_plot.png
runs/counterfact_direction1_v1/dev_tune_200_actual_gate_activation_grid_v1/gate_activation_samples.csv
```

## 12. Step 3E.4 requirements

Only run this if Step 3E.3 finds a gate that passes activation criteria.

Run actual decoding only on RunPod/GPU.

Minimum method set:

```text
prompt_memory_gated_hybrid
myopic_score_gated_hybrid_gs2.0
mc_bridge_gated_hybrid_gs2.0
mc_bridge_gated_hybrid_gs1.75
```

Optional:

```text
myopic_score_gated_hybrid_gs1.75
no_rollout_bridge_gated_hybrid_gs2.0
```

Evaluate buckets:

```text
rewrite
declarative_paraphrases
qa_format_generalization
near_locality
far_locality
same_subject_template
generation
```

Required outputs:

```text
report_summary.json
hybrid_gate_method_bucket.csv
hybrid_gate_stress_summary.csv
hybrid_gate_selection.csv
hybrid_gate_paired_bootstrap.csv
hybrid_gate_target_length.csv
hybrid_gate_gate_activation_summary.csv
hybrid_gate_output_samples.csv
replay_vs_actual_comparison.csv
```

## 13. Run summaries

Every new report directory must include:

```text
report_summary.json
```

with at least:

```json
{
  "protocol_version": "counterfact_direction1_v1",
  "split_role": "dev_tune_200",
  "analysis_500_used": false,
  "final_test_used": false,
  "git_commit": "<current commit>",
  "stage": "<stage name>",
  "artifacts": {}
}
```

Every GPU run summary must also include:

```text
model_id
dtype
use_4bit
device_map
GPU name
CUDA version
torch version
transformers version
bitsandbytes version
GPU minutes per edit
model evals per edit
```

## 14. Overwrite rules

Default: never overwrite old reports.

Use new versioned directories:

```text
..._v1
..._v2
..._v3
```

A script may overwrite only if explicitly passed:

```text
--allow_overwrite 1
```

and the notebook/report clearly states that it was rerun.

## 15. Exit criteria before analysis_500

Do not run `analysis_500` until a dev method has passed all of:

```text
rewrite/paraphrase usefulness
ordinary near/far locality
same-subject stress
malformed-span budget
compute budget
KL/Pareto budget or chosen dev Pareto point
paired bootstrap sanity checks
target-length breakdown inspection
```

After that, freeze:

```text
method
gate_id
gate thresholds
span policy
normalization
metrics
budgets
report scripts
random seeds
```

Then set:

```bash
export DEV_METHOD_LOCKED=1
```

Only then may `analysis_500` run.

