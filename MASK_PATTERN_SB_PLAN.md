# M4 — Exact Mask-Pattern Schrödinger Bridge

Protocol: `llada_mask_pattern_sb_v1`

## Objective

Optimize the order in which answer-span positions are revealed during masked-diffusion generation using an exact finite-state Schrödinger/Doob control problem.

This track operates over answer-position reveal states, not the full vocabulary process.

## State space

For a target span of length N:

```text
state S = subset of positions already revealed
start = empty set
terminal = all positions revealed
number of states = 2^N
```

Primary N:

```text
N in {2,3,4}
```

At state S, the current generated tokens at revealed positions remain fixed and the other answer positions remain masked.

## Reference transition Q

Implement reference policies:

```text
uniform among unrevealed positions
base-confidence normalized among unrevealed positions
paper-default random reveal
```

Every transition reveals one additional position.

## Transition cost

For candidate next position i:

```text
c(S,i) = -log p_edited(target_token_i | prompt, revealed target tokens at S,
                                        masks at remaining positions)
```

Use the partial-mask edited model from M2.

Optional identity cost for negative prompts:

```text
c_identity(S,i) = sparse_KL(p_edited || p_base)
```

## Exact controlled process

For beta > 0, compute the backward desirability by dynamic programming:

```text
h(terminal) = 1
h(S) = sum_i Q(i|S) * exp(-beta*c(S,i)) * h(S union {i})
```

Controlled transition:

```text
P*(i|S) proportional to Q(i|S) * exp(-beta*c(S,i)) * h(S union {i})
```

Implement both:

```text
stochastic sampling from P*
greedy max-probability controlled reveal
```

## Stage M4.1 — Analytical tests

Use tiny synthetic N=2/3 examples with known costs.

Tests:

```text
probabilities normalize
terminal reached in N transitions
beta=0 reproduces Q
higher beta favors lower-cost paths
DP equals brute-force trajectory enumeration
no target token is forced; only reveal order is controlled
```

## Stage M4.2 — Integration smoke

On KAMEL smoke, compare:

```text
left-to-right
base confidence
uniform random
best fixed order by dev average
myopic next-position cost
exact mask-pattern SB
```

Run under both ordinary MEMIT and partial-mask MEMIT when feasible.

Acceptance:

```text
all policies generate complete outputs
no forced target filling
trajectory costs logged
model evaluations logged
```

## Stage M4.3 — Dev selection

Tune on `kamel_dev_50_per_length` only.

Bounded grid:

```text
reference in {uniform, base_confidence}
beta in {0.25,0.5,1.0,2.0,4.0}
mode in {stochastic, greedy}
```

Freeze one efficacy candidate and one efficiency/path-cost candidate.

## Stage M4.4 — Locked main evaluation

Evaluate on `kamel_repro_200_per_length`, N=2,3,4.

Primary positive criterion, at least one:

```text
full-target exact improves >= 0.05 absolute over the best fixed/random policy
or
full-target exact is within 0.02 while expected trajectory cost decreases >= 20%
or
full-target exact is within 0.02 while model evaluations decrease >= 20%
```

Mechanism-specific criterion:

```text
beta=0/reference baseline must be weaker than selected controlled process
DP/Doob control must beat one-step myopic reveal on at least one primary axis
```

Safety:

```text
malformed <= 0.05
same-subject/locality behavior not worse than underlying edited model by >0.03 TFPR
```

### Bounded M4 rescue

If no positive result:

- evaluate one state-dependent beta schedule chosen from `{early_strong, late_strong}` on dev;
- no new reference or neural model;
- freeze and rerun once.

Then finish M4 positively or negatively.

## Outputs

```text
runs/masked_diffusion_memit_sb_positive_result_v1/M4_mask_pattern_sb_v1/
  report_summary.json
  analytical_test_report.json
  dev_policy_grid.csv
  main_results_by_length.csv
  trajectory_costs.csv
  reveal_order_examples.jsonl
  mechanism_ablation.csv
  paired_bootstrap.csv
  final_track_report.md
```
