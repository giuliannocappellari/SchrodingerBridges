# counterfact_direction1_v1 Implementation Spec

## CLIs

### `llada_counterfact_protocol.py`

Builds official CounterFact manifests and protocol metadata.

Primary outputs:

```text
<output_dir>/dev_tune_200.jsonl
<output_dir>/analysis_500.jsonl
<output_dir>/ablation_500.jsonl
<output_dir>/final_test_500.jsonl
<output_dir>/final_test_full.jsonl
<output_dir>/<split>.metadata.json
<output_dir>/protocol_manifest.json
<output_dir>/split_overlap_report.json
```

For smoke tests, use `--smoke 1`. Smoke mode creates tiny default split sizes
of 10/10/10 train rows and 10 test rows. Official non-smoke builds refuse to
write undersized official manifests.

Each JSONL row remains compatible with the existing `EditExample` loader while
adding protocol fields such as `case_id`, `relation_id`, `subject`,
`rewrite_template`, context-aware token ids, target-length bins, prompt stats,
validity flags, and split role.

### `llada_runtime_editor_eval.py`

Evaluates runtime editors over a protocol manifest.

Sprint-1 methods:

```text
base
target_logit_bias
prompt_memory
target_candidate_insert
myopic_score
no_rollout_bridge
mc_bridge
raw_bridge_gated
```

Every run writes:

```text
run_config.json
per_case_results.jsonl
summary.json
```

Run configs include:

```text
protocol_version = counterfact_direction1_v1
edit_access = given_at_edit_time
training_access = none
hyperparameter_access = dev_tune_only
```

`analysis_500` and final-test runs are refused unless lock requirements are
met in the config.

### `llada_experiment_reports.py`

Aggregates one or more runtime-eval summaries into paper-facing tables:

```text
aggregate metrics
paired bootstrap confidence intervals
candidate-support coverage
target-length breakdowns
sparse guidance-KL summaries
selection-rule diagnostics
```

## Artifact Schema

Per-case result rows include:

```text
protocol_version
split_role
method
edit_id
case_id
bucket
prompt
target
sample_outputs
exact_rate
greedy_exact
token_f1
malformed_rate
target_false_positive_rate
sparse_guidance_kl
base_margin
guided_margin
target_length_bin
relation_id
```

Summary artifacts include aggregate metrics by method, bucket, target-length
bin, and relation id.

## Lock Config

Analysis/final lock configs must include:

```json
{
  "protocol_version": "counterfact_direction1_v1",
  "thresholds_frozen": true,
  "span_policy_frozen": true,
  "gate_policy_frozen": true,
  "normalization_frozen": true,
  "metrics_frozen": true,
  "selected_dev_pareto_point": "required before analysis_500",
  "final_config_locked": true
}
```

`final_config_locked` is required only for final-test runs.

## Tests

Use local fake tokenizer/model stubs where possible. The first test suite covers:

```text
normalization
context-aware target tokenization
split determinism
overlap validation
target-length bins
candidate coverage
exact/F1/pass@k
paired bootstrap by case_id
self-normalized locality
gates
probability margins
sparse guidance KL
old JSONL compatibility
```

GPU/model smoke tests are optional and should run only on 1-2 edits with
`steps=2` and `eval_samples=2`.
