# M3 — Schrödinger/Path-KL-Regularized Partial-Mask MEMIT

Protocol: `llada_schrodinger_regularized_memit_v1`

## Objective

Start from the successful partial-mask MEMIT editor and test whether a minimum-intervention path regularizer preserves the edit while reducing deviation from the frozen LLaDA denoising process on identity and same-subject-negative states.

This is an SB-inspired minimum-path-KL editor. Do not call it a full solved Schrödinger bridge.

## Objective

For partial-mask states `x_t`:

```text
L = L_edit
  + lambda_path * sum_t KL_sparse(p_edited(.|x_t) || p_base(.|x_t))
  + lambda_identity * L_identity
  + lambda_weight * ||Delta||^2
```

Where:

```text
L_edit:
  partial-mask target reconstruction loss

KL_sparse:
  KL over a shared support containing base top-k, target_new tokens,
  target_true tokens, and edited top-k

L_identity:
  sparse KL and target-new suppression on training-only locality and
  same-subject-different-relation anchors

Delta:
  MEMIT target residual or resulting weight update
```

## Stage M3.1 — Objective implementation

Tests:

```text
KL=0 when edited/base logits equal
shared support and renormalization correct
gradient flows to target residual
identity prompts do not enter efficacy evaluation
no evaluation prompt leakage
loss components reported separately
```

## Stage M3.2 — Dev grid

Tune only on `cf_sb_dev_200` and `kamel_dev_50_per_length`.

Bounded grid:

```text
lambda_path in {0.01, 0.05, 0.1, 0.25}
lambda_identity in {0.25, 0.5, 1.0}
lambda_weight in {0.0, 0.001, 0.01}
```

Use staged narrowing, not the full Cartesian product:

1. path weight with identity fixed at 0.5 and weight fixed at 0.001;
2. identity weight for top two;
3. weight penalty for top two.

Required baselines:

```text
fully-masked MEMIT
partial-mask MEMIT
partial-mask MEMIT + ordinary L2/update-norm regularization
partial-mask MEMIT + path KL
partial-mask MEMIT + path KL + identity
```

Select at most two Pareto candidates.

## Stage M3.3 — Locked analysis

Run frozen candidates on:

```text
cf_sb_analysis_200
kamel_repro_200_per_length
```

Hard safety metrics:

```text
same-subject TFPR
near/far TFPR
identity sparse KL
classic specificity
```

Positive SB-regularization result requires one candidate to satisfy:

```text
rewrite/efficacy >= partial-mask baseline - 0.05
paraphrase/generalization >= partial-mask baseline - 0.05
and at least one:
  same-subject TFPR reduced >= 25% relative
  identity sparse KL reduced >= 25% relative
  path/intervention cost reduced >= 25% relative
```

Additionally, it must beat the ordinary L2/update-norm baseline on at least one safety/path metric at comparable efficacy.

Strong result:

```text
same-subject TFPR within base + 0.03
while efficacy and generalization remain within 0.05 of partial-mask MEMIT
```

### Bounded M3 rescue

If efficacy collapses:

- use the nearest lower predeclared lambda_path;
- keep identity weight fixed;
- do not add new losses or thresholds;
- run once.

If locality does not improve, no rescue beyond this. Finish M3 negatively and continue M4.

## Mechanism ablations

Report:

```text
path KL only
identity loss only
L2 only
path KL + identity
```

Do not claim SB-specific value if L2 or identity-only matches the result.

## Outputs

```text
runs/masked_diffusion_memit_sb_positive_result_v1/M3_schrodinger_regularized_memit_v1/
  report_summary.json
  dev_grid.csv
  pareto_candidates.json
  analysis_results.csv
  loss_ablation.csv
  identity_stress.csv
  path_kl_table.csv
  paired_bootstrap.csv
  final_track_report.md
```
