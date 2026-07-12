# AGENTS.md

Rules for using a local MacBook environment, RunPod SSH, and Codex for the `counterfact_direction1_v1` LLaDA / CounterFact runtime-editing project.

These rules are intentionally operational. They tell Codex what may run locally, what requires RunPod, when the Pod may be started, how GPU jobs must be launched, how Python environments should be handled, and which experimental splits are locked.

---

## 0. Project identity

This repository is for the LLaDA / CounterFact runtime-editing project.

```text
protocol_version = counterfact_direction1_v1
main_model = GSAI-ML/LLaDA-8B-Base
current_research_direction = runtime bridge editing / runtime guided editing
default_tuning_split = dev_tune_200
```

The project studies runtime factual editing for masked diffusion LMs. The default setup is:

```text
Gen_{theta0, B_e}(x; S)
```

where:

```text
theta0 = frozen LLaDA
e = edit request supplied at edit time
B_e = edit-conditioned bridge/controller used during decoding
S = fixed diffusion sampling configuration
```

This is not permanent parameter editing unless a later experiment explicitly says so.

Every run config must record:

```text
protocol_version = counterfact_direction1_v1
edit_access = given_at_edit_time
training_access = none
hyperparameter_access = dev_tune_only
```

Locked split rules:

```text
dev_tune_200 = only split used for tuning thresholds, gates, span policy, hyperparameters, and method selection
analysis_500 = proceed/stop only; no tuning after inspection
final_test_500 = primary locked final result
final_test_full = optional secondary replication only if pre-committed or reported as secondary
```

---

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
- Notebook cleanup
- Report regeneration from existing artifacts

Local-only or CPU-safe examples:

```text
Gate parity audits
Actual-gate activation grids
Report regeneration
Plotting
Unit tests
Small data inspections
CSV/JSON/JSONL validation
```

Do not use the RunPod GPU for cheap CSV/report work unless the Pod is already running for a GPU experiment and the user explicitly approves using it.

---

## 1.1 Local Python environment: use `uv`

On the MacBook/local environment, Codex must use `uv` as the Python project manager.

Local setup commands:

```bash
uv sync
```

Local test commands:

```bash
uv run pytest tests -q
```

Local script commands:

```bash
uv run python scripts/<script_name>.py
```

Local notebook-related commands, if needed:

```bash
uv run jupyter lab
uv run python -m ipykernel install --user --name sb-uv --display-name "SB uv"
```

Local rules:

- Prefer `uv run ...` for all Python commands on the MacBook.
- Prefer `uv sync` to install or update local dependencies.
- Do not manually activate `.venv` locally unless the user explicitly asks.
- Do not use `pip install` directly in the local project environment; update project dependency files and run `uv sync` instead.
- If `uv.lock` or `pyproject.toml` is changed, Codex must show the diff and explain why.
- Local CPU/report tasks should be run with `uv run`, not with the RunPod Python environment.

---

## 2. What should run on RunPod

Use RunPod only for GPU-required work:

- Loading `GSAI-ML/LLaDA-8B-Base`
- Any actual LLaDA model decoding
- `mc_bridge` decoding
- `no_rollout_bridge` or `myopic_score` decoding over many edits
- GPU smoke tests
- Any actual hybrid-gate decode
- `analysis_500` after the dev method is locked
- Final-test evaluation after analysis confirmation
- Direction 2 per-edit adapter training that requires CUDA
- Direction 3 learned controller training that requires CUDA

GPU-required examples:

```text
actual decode with stricter hybrid gate
future analysis_500 confirmation
future final_test_500 locked run
Direction 2 per-edit adapter pilot
Direction 3 learned controller training
```

---

## 2.1 RunPod Python environment: use the Python available there

On RunPod/SSH, Codex should use the Python environment already available in the GPU image/template unless the user explicitly asks for a custom environment.

Default RunPod Python commands:

```bash
python --version
python -m pip --version
python -m pytest tests -q
python scripts/<gpu_script>.py
```

RunPod setup/check commands:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import torch
print('cuda_available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

RunPod rules:

- Do not require `uv` on RunPod.
- Do not use `uv run` on RunPod unless the user explicitly installs and approves `uv` there.
- Do not assume `.venv` exists on RunPod.
- Do not run `source .venv/bin/activate` on RunPod unless a `.venv` has been intentionally created and the user approved using it.
- Prefer the PyTorch/CUDA template Python and `python -m pip` for remote dependency setup.
- Every GPU run must record the Python executable, Python version, CUDA availability, GPU name, torch version, transformers version, and bitsandbytes version when available.

---

## 2.2 Task routing table

| Task type | Default machine | May start RunPod? |
|---|---|---|
| Code editing | MacBook | No |
| Unit tests with fake model/tokenizer | MacBook | No |
| CSV aggregation | MacBook | No |
| Plot generation | MacBook | No |
| Gate-only replay | MacBook | No |
| Gate parity audit | MacBook | No |
| Actual-gate activation grid | MacBook | No |
| Protocol manifest validation | MacBook | No |
| Paired bootstrap over existing files | MacBook | No |
| Report markdown writing | MacBook | No |
| GPU smoke test | RunPod | Yes |
| Actual LLaDA decoding | RunPod | Yes |
| MC bridge decoding | RunPod | Yes |
| Actual hybrid-gate decode | RunPod | Yes |
| `analysis_500` locked evaluation | RunPod | Yes, only if `DEV_METHOD_LOCKED=1` |
| Final-test locked evaluation | RunPod | Yes, only if `FINAL_METHOD_LOCKED=1` |
| Direction 2 adapter training | RunPod | Yes |
| Direction 3 controller training | RunPod | Yes |

---

## 3. RunPod cost discipline

Use RunPod as a temporary GPU workstation, not as always-on storage.

Rules:

1. Start the Pod only when a GPU job is ready.
2. Work inside `tmux` for long runs.
3. Stop the Pod when no GPU job is running.
4. Keep important project data in `/workspace` or a network volume.
5. Back up final artifacts to Git, local machine, Google Drive, S3, or another durable store.
6. Do not rely on container disk for persistent data.
7. Do not leave the Pod running overnight unless an active run is executing.
8. Schedule an automatic stop for long runs only when the user requests it.
9. Never terminate/delete the Pod unless the user explicitly asks.

RunPod stop/start commands must use environment variables, never hard-coded Pod IDs.

```bash
runpodctl pod stop "$RUNPOD_POD_ID"
runpodctl pod start "$RUNPOD_POD_ID"
```

Suggested safety stop after N hours, only if the user requests it:

```bash
sleep 6h; runpodctl pod stop "$RUNPOD_POD_ID" &
```

---

## 3.1 When Codex may start the RunPod Pod

Codex may start the RunPod Pod only for GPU-required work.

GPU-required work includes:

- loading `GSAI-ML/LLaDA-8B-Base`
- actual LLaDA decoding
- MC bridge decoding
- no-rollout/myopic decoding over many edits
- actual hybrid-gate decode
- `analysis_500` after the dev method is locked
- final-test evaluation after analysis confirmation
- adapter/controller training that requires CUDA

Codex must not start the Pod for:

- CSV/report aggregation
- plots
- unit tests with fake models
- gate-only replay
- gate parity audit
- actual-gate activation grid
- markdown/report writing
- paired bootstrap over existing results

Before starting the Pod, Codex must check:

```bash
echo "$RUNPOD_POD_ID"
runpodctl pod list
```

If `$RUNPOD_POD_ID` is empty, Codex must stop and ask the user for the Pod ID.

Start command:

```bash
runpodctl pod start "$RUNPOD_POD_ID"
```

After starting, Codex must verify the Pod is available:

```bash
runpodctl pod list
```

If the Pod starts with zero GPUs or no SSH access, Codex must stop and ask the user to inspect the RunPod console. Do not run a GPU experiment on a zero-GPU Pod.

Codex must never create a new Pod unless the user explicitly asks.
Codex must never terminate/delete a Pod unless the user explicitly asks.

---

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
/workspace/logs/                       # optional logs
```

Never assume data outside `/workspace` will survive a stop.

Before terminating a Pod, back up:

```text
runs/counterfact_direction1_v1/<new_report_or_run_dir>/
```

Do not commit large run artifacts to Git. Commit only scripts, configs, tests, small summaries, and documentation.

---

## 5. SSH and tmux rules

Always connect by SSH for long-running work.

After connecting:

```bash
cd /workspace/SB
tmux new -s sb
```

Inside tmux:

```bash
python --version
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

If a session already exists, use:

```bash
tmux ls
tmux attach -t sb
```

---

## 5.1 Remote GPU execution template

GPU jobs must run inside `tmux` on the RunPod machine.

From the MacBook, first start the Pod if needed:

```bash
runpodctl pod start "$RUNPOD_POD_ID"
runpodctl pod list
```

Then connect by SSH using the current SSH command from the RunPod console or local environment variables:

```bash
ssh -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_SSH_PORT" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
```

Inside the Pod:

```bash
cd /workspace/SB
tmux new -s sb
python --version
nvidia-smi
git status
git pull
python -m pytest tests -q
```

Long GPU jobs must be launched inside tmux and must write logs:

```bash
mkdir -p logs

python scripts/<gpu_script>.py \
  2>&1 | tee logs/<stage_name>_$(date +%Y%m%d_%H%M%S).log
```

For detached execution:

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && python scripts/<gpu_script>.py 2>&1 | tee logs/<stage_name>.log'
```

After launching a detached job, Codex must check:

```bash
tmux ls
```

and provide the user with:

```text
tmux session name
log path
expected output directory
safe stop command
```

---

## 5.2 Stopping the Pod after GPU work

When a GPU job finishes, Codex must verify that no GPU process is still running:

```bash
nvidia-smi
tmux ls
```

If no GPU job is running, Codex should sync or confirm artifacts:

```bash
ls -lah runs/counterfact_direction1_v1/<new_run_dir>/
```

Then Codex may stop the Pod:

```bash
runpodctl pod stop "$RUNPOD_POD_ID"
```

For long jobs, Codex may schedule a safety stop only if the user requested it:

```bash
sleep 6h; runpodctl pod stop "$RUNPOD_POD_ID" &
```

Do not stop the Pod if:

- a Python process is still running
- `nvidia-smi` shows active GPU memory usage from the experiment
- `tmux` contains an active experiment session
- artifacts have not been written or backed up

---

## 6. Git rules

Code moves through Git. Large artifacts do not.

For local MacBook checks, use `uv`:

```bash
git status
uv sync
uv run pytest tests -q
```

For RunPod GPU checks, use the remote Python available there:

```bash
git status
git pull
python -m pytest tests -q
```

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
private SSH keys
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

---

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
RUNPOD_API_KEY
OPENAI_API_KEY
```

Never paste private SSH keys into notebooks, source files, markdown, prompts, or commit history.

---

## 7.1 Required local environment variables for RunPod control

On the MacBook, define:

```bash
export RUNPOD_POD_ID="<pod-id>"
export RUNPOD_SSH_KEY="$HOME/.ssh/runpod_sb"
export RUNPOD_SSH_USER="root"
export RUNPOD_SSH_HOST="<current-runpod-host>"
export RUNPOD_SSH_PORT="<current-runpod-port>"
export REMOTE_REPO_DIR="/workspace/SB"
```

`RUNPOD_SSH_HOST` and `RUNPOD_SSH_PORT` may need to be refreshed from the RunPod console if SSH fails after restart.

Codex must not guess SSH host/port. If missing, ask the user.

---

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

If `analysis_500` fails:

```text
mark counterfact_direction1_v1 failed
create counterfact_direction1_v2
do not tune on analysis_500
```

---

## 9. Run summaries

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
Python executable
Python version
GPU name
CUDA version
torch version
transformers version
bitsandbytes version
GPU minutes per edit
model evals per edit
```

---

## 9.1 GPU job preflight

Before any GPU run, Codex must check:

```bash
cd /workspace/SB
python --version
git status
git pull
python -m pytest tests -q
nvidia-smi
```

Codex must also check that the script will not run locked splits unless explicitly unlocked:

```bash
echo "$DEV_METHOD_LOCKED"
echo "$FINAL_METHOD_LOCKED"
```

For dev tuning jobs, expected values are usually empty.

Before running a GPU script, Codex must print:

```text
stage
split_role
method(s)
gate_id
output_dir
model_id
dtype
use_4bit
expected artifacts
```

Every GPU script must write:

```text
report_summary.json
run_config.json or equivalent config snapshot
logs
```

---

## 10. Overwrite rules

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

Codex must not delete old report directories unless explicitly instructed by the user.

---

## 11. Exit criteria before analysis_500

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
relation_bank_source
relation_bank_fingerprint
```

Then write:

```text
runs/counterfact_direction1_v1/dev_method_lock.json
```

and only then set:

```bash
export DEV_METHOD_LOCKED=1
```

Only then may `analysis_500` run.

---

## 12. GPU command approval rule

Codex must not start a GPU job unless the user has explicitly requested a GPU run or approved the command.

Before starting RunPod, Codex should show:

```text
Reason GPU is needed:
Estimated stage:
Expected output directory:
Command to start Pod:
Command to run experiment:
Command to monitor logs:
Command to stop Pod:
```

Then wait for user approval.

Exception: if the user explicitly says “run the GPU step on RunPod now,” Codex may start the Pod using the approved commands.

---

## 13. RunPod command templates

### List Pods

```bash
runpodctl pod list
```

### Start the configured Pod

```bash
runpodctl pod start "$RUNPOD_POD_ID"
```

After starting:

```bash
runpodctl pod list
```

If the Pod has zero GPUs or SSH fails, stop and ask the user to inspect the RunPod console.

### SSH into the Pod

```bash
ssh -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_SSH_PORT" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
```

### Start tmux session

```bash
cd /workspace/SB
tmux new -s sb
```

### Run GPU job inside tmux

```bash
cd /workspace/SB
python --version
nvidia-smi
python scripts/<gpu_script>.py 2>&1 | tee logs/<stage_name>.log
```

### Run detached GPU job

```bash
tmux new -d -s "<stage_name>" \
  'cd /workspace/SB && python scripts/<gpu_script>.py 2>&1 | tee logs/<stage_name>.log'
```

### Check job

```bash
tmux ls
nvidia-smi
tail -f logs/<stage_name>.log
```

### Stop the Pod

Only stop after verifying no GPU job is running.

```bash
runpodctl pod stop "$RUNPOD_POD_ID"
```

### Schedule safety stop

Only if the user requested it.

```bash
sleep 6h; runpodctl pod stop "$RUNPOD_POD_ID" &
```

### Terminate/delete Pod

Codex must not terminate/delete Pods unless explicitly asked by the user.

```bash
runpodctl pod delete "$RUNPOD_POD_ID"
```

---

## 14. Artifact backup rules

After a GPU job finishes, back up the new report/run directory.

Recommended local pull from MacBook:

```bash
rsync -avz -e "ssh -i $RUNPOD_SSH_KEY -p $RUNPOD_SSH_PORT" \
  "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:/workspace/SB/runs/counterfact_direction1_v1/<new_run_dir>/" \
  "./runs/counterfact_direction1_v1/<new_run_dir>/"
```

Recommended push to remote from MacBook, if needed:

```bash
rsync -avz -e "ssh -i $RUNPOD_SSH_KEY -p $RUNPOD_SSH_PORT" \
  "./scripts/" \
  "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:/workspace/SB/scripts/"
```

Git is preferred for code synchronization.
`rsync` is preferred for run artifacts.

---

## 15. Codex behavior expectations

Codex should:

- Read this file before acting.
- Use `uv` for local MacBook Python commands.
- Use the available Python environment on RunPod/SSH.
- Prefer scripts over notebook-only logic.
- Keep notebooks as orchestration/reporting layers when possible.
- Add or update tests for new logic.
- Avoid running expensive jobs without approval.
- Avoid changing protocol definitions casually.
- Preserve backward compatibility with old artifacts when feasible.
- Report uncertainty when an artifact or config is missing.
- Stop and ask the user when a required environment variable, split lock, SSH detail, or secret is missing.

Codex should not:

- Guess RunPod SSH host/port.
- Start the Pod for CPU/report tasks.
- Run `analysis_500` or final-test splits without lock flags.
- Overwrite old report directories by default.
- Commit secrets or large artifacts.
- Terminate/delete a Pod without explicit instruction.
- Use `uv` on RunPod unless explicitly approved.
- Use `pip install` directly in the local MacBook project environment.

---

## 16. Current research interpretation guardrails

Direction 1 is still active only if a valid dev method passes:

```text
rewrite/paraphrase usefulness
near/far locality
same-subject stress
malformed budget
compute budget
KL/Pareto budget
paired-bootstrap sanity
```

If no runtime gate can satisfy the same-subject stress constraints while preserving rewrite/paraphrase usefulness, then Codex should not keep tuning Direction 1 indefinitely. It should report that Direction 1 is blocked and ask whether to move to:

```text
Direction 2: per-edit bridge adapters
Direction 3: learned edit-conditioned controller
```

---

## 17. Minimal run checklist

Before any dev CPU/report step:

```text
[ ] Confirm split is dev_tune_200.
[ ] Confirm no analysis/final artifacts are being read for tuning.
[ ] Confirm output directory is versioned.
[ ] Run local commands through uv, for example `uv run pytest tests -q`.
[ ] Write report_summary.json.
```

Before any GPU step:

```text
[ ] User approved GPU run.
[ ] RUNPOD_POD_ID is set.
[ ] SSH variables are set.
[ ] Pod is running with GPU.
[ ] Code is committed or diff is understood.
[ ] Remote tests pass with `python -m pytest tests -q`, or skipped with explicit reason.
[ ] tmux session is active.
[ ] nvidia-smi works.
[ ] Output directory is versioned.
[ ] Logs are written.
[ ] Stop/safety-stop plan is known.
```
