# P1 — Paper-Matched Partial-State MDM-MEMIT Audit

## Objective

Resolve the contradiction between the prior campaign, where partial-mask
schedule controls did not improve multi-token editing, and the source paper,
which reports substantial gains from optimizing across partially revealed
target states.

This track is not passed by obtaining any positive number. It must either:

```text
reproduce the published qualitative trend
or
identify a concrete, tested protocol/implementation difference
```

## Step 1 — Source and code comparison

Build a line-by-line implementation register covering:

```text
model revision
tokenizer revision
edited layer window
MLP target matrix
subject-token index
target-value optimization
learning rate
optimization iterations
clamp norm
KL anchoring
batch update
mask construction
revealed-position schedule
random revealed positions
loss mask
generation schedule
target-length filtering
KAMEL source and templates
```

Use the paper-matched settings when available:

```text
LLaDA-8B-Instruct
early/middle MLP window, with layers 4–7 as the first paper-matched candidate
last subject token
target-value learning rate 0.1
25 optimization steps
clamp-norm factor 0.75
KL factor 0.0625
one answer mask per target token
```

Record any unavoidable deviation.

## Step 2 — Unit and synthetic tests

Required tests:

```text
all mask counts 0..N-1 are visited
the revealed count follows k = optimization_step mod N
revealed positions are resampled
loss is evaluated only on still-masked positions
contextual target tokens align to answer positions
N=1 reduces to ordinary fully masked editing
fixed random seeds reproduce schedules
```

## Step 3 — Controlled dev experiment

On fresh KAMEL development data for N in `{2,3,4}` compare:

```text
ordinary fully masked MDM-MEMIT
partial-state cycle with fixed revealed positions
partial-state cycle with random revealed positions
random mask count
paper-matched schedule
```

Use at least two schedule seeds and three generation seeds.

## Step 4 — Single bounded repair

If the paper trend is absent, one repair is allowed based only on a concrete
source-audit finding, such as:

```text
incorrect loss mask
incorrect subject-token index
wrong layer/module map
wrong target-value baseline
wrong KAMEL rendering
wrong revealed-position update
```

Do not use the dev results to invent a new schedule.

## Acceptance

`reproduced_paper_trend` if:

```text
paper-matched partial-state editing improves efficacy by >= 0.10 absolute
over ordinary fully masked editing on at least two of N={2,3,4}

and

the pooled paired-bootstrap lower bound is > 0

and

malformed rate does not worsen by more than 0.03
```

`concrete_protocol_difference_explained` if the trend is not reproduced but a
specific experimentally verified difference explains the discrepancy.

`unresolved_baseline_discrepancy` otherwise.

## Outputs

```text
runs/mask_pattern_sb_publication_confirmation_v1/
  partial_state_memit_audit_v1/
    report_summary.json
    implementation_difference_register.md
    schedule_unit_test_summary.json
    method_bucket.csv
    paired_bootstrap.csv
    target_length_table.csv
    output_samples.csv
    discrepancy_decision.md
```

The strongest correctly implemented partial-state method becomes a mandatory
baseline for the publication campaign.
