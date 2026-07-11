# counterfact_direction1_v1 Paper Protocol

## Claim

Direction 1 is runtime bridge editing for masked diffusion language models:

```text
Gen_{theta0, B_e}(x; S)
```

`theta0` is frozen LLaDA, `e` is the edit request supplied at edit time, and
`B_e` is an edit-conditioned bridge/controller used only during decoding. This
is not permanent parameter editing.

Every run config must include:

```text
protocol_version = counterfact_direction1_v1
edit_access = given_at_edit_time
training_access = none
hyperparameter_access = dev_tune_only
```

## Split Discipline

Split sources:

```text
HF train split:
  dev_tune_200
  analysis_500
  ablation_500

HF test split:
  final_test_500
  final_test_full
```

Allowed use:

```text
dev_tune_200:
  only split used for tuning thresholds, gates, span policy, and hyperparameters.

ablation_500:
  only for frozen, pre-planned ablation tables.
  cannot affect hyperparameters, gates, span policy, normalization, metrics, or final method selection.

analysis_500:
  proceed/stop only.
  no changes allowed after inspection.

final_test_500:
  primary locked final result.

final_test_full:
  optional secondary replication.
  must be pre-committed before inspecting final_test_500, or reported only as secondary replication.
```

If `analysis_500` fails, mark v1 failed and create `counterfact_direction1_v2`;
do not tune on analysis.

## Lock Enforcement

```text
analysis_500 cannot run unless thresholds, span policy, gate policy, normalization, and metric definitions are frozen in config.

final_test_500 cannot run unless the selected analysis-confirmed config is marked locked.
```

## Core Methods

Core first-paper methods:

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

`path_kl_bridge` is optional during the first dev smoke sprint, but required
before any valid `analysis_500` run.

`constrained_fill_oracle` is an oracle ceiling and excluded from normal method
ranking.

## Experiments

1. Build protocol manifests.
2. Run base/self-consistency.
3. Run G0 candidate-support diagnostics.
4. Run raw MC bridge.
5. Compare runtime baselines.
6. Run bridge mechanism ablations.
7. Run diffusion-step/schedule ablations.
8. Report target-length analysis.
9. Tune gating/locality only on `dev_tune_200`.
10. Select compute-quality Pareto point on `dev_tune_200`.
11. Confirm proceed/stop on `analysis_500`.
12. Run one locked `final_test_500` evaluation.

## Smoke Builds

Smoke builds must not pretend to create the official split sizes. Use
`--smoke 1`, which defaults to:

```text
dev_tune_200 = 10
analysis_500 = 10
ablation_500 = 10
final_test_500 = 10
```

Official non-smoke builds must fail if there are not enough valid records to
fill the requested split sizes.

## Metric Definitions

Context-aware target tokenization:

```text
For each rendered prompt x and target string y:
  tokenize(x + y)
  tokenize(x)
  target_token_ids = suffix difference
```

Standalone target tokenization is stored only as a diagnostic. Primary
target-length bins, candidate coverage, target insertion, span evaluation, and
probability margins use context-aware target tokens.

Self-normalized locality:

```text
SelfLoc_base = agreement between two base generations with different seeds
SelfNormalizedLoc(method) = Loc(method) / max(SelfLoc_base, epsilon)
ClippedSelfNormalizedLoc(method) = min(SelfNormalizedLoc(method), 1.0)
epsilon = 1e-6
```

Report raw locality, `SelfLoc_base`, and self-normalized locality.

Probability margin on the locked answer span:

```text
margin = log P(target_new_tokens | masked prompt)
       - log P(target_true_tokens | masked prompt)
```

For base/unguided methods, probabilities come from the base denoiser. For guided
methods, report guided candidate-support margin when available. Margin is
diagnostic, not the main selection metric.

Sparse-support guidance KL:

```text
KL_t = sum_{v in C_t} p_B(v | x_t) log(p_B(v | x_t) / q_C(v | x_t))
```

`C_t` is shared candidate support and `q_C` is the base distribution
renormalized on that support. This is not full path KL.

Use paired bootstrap CIs resampling by `case_id`, not individual prompt.

## Selection Rule

Before `analysis_500`, select the final method on `dev_tune_200` by maximizing:

```text
H(rewrite_exact, paraphrase_exact, ClippedSelfNormalizedLoc)
```

subject to fixed pre-analysis constraints:

```text
target_false_positive_rate <= base_TFPR + 0.03
malformed_span_rate <= 0.05
gpu_minutes_per_edit <= 2.0
sparse_guidance_KL <= selected dev Pareto point
```

If no method satisfies all constraints, select the best Pareto point and mark
the violated constraint explicitly.

## Interpretation Rules

- `prompt_memory` is a required runtime baseline unless technically blocked.
- `target_candidate_insert` is diagnostic. If it matches MC bridge, the bridge
  claim weakens; if MC bridge beats it, bridge control contributes beyond
  support expansion.
- Memory retrieval scaling, per-edit LoRA, and large side-effect suites are
  appendix or extension work, not the first-paper core.
