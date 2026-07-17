# P6 — Editor-Generality Evaluation

## Objective

Show that the reveal controller solves a generation-path problem rather than a
quirk of one MEMIT update.

## Mandatory editor conditions

Use the frozen controller with:

```text
E1 ordinary fully-masked MDM-MEMIT
E2 paper-matched partial-state MDM-MEMIT
```

Optional third condition:

```text
E3 prompt-memory edit statement
or
E3 target-logit edit guidance
```

The path controller itself must not be retuned per editor beyond a predeclared
cost normalization fitted on dev.

## Data

Use the same fresh locked LLaDA facts, with editor conditions prepared before
locked outcomes are compared.

## Methods per editor

```text
editor + default reveal
editor + one-step myopic
editor + deterministic global planner
editor + finite-beta controller
```

## Acceptance

Evidence of editor generality if:

```text
finite-beta controller has positive mean rewrite delta
over the best non-SB reveal control in at least two editor conditions

and

at least one editor condition has paired lower bound > 0
```

No condition may have:

```text
same-subject TFPR increase > 0.03
malformed rate > 0.05
```

If the effect exists only with one MDM-MEMIT implementation, call it
editor-specific.

## Outputs

```text
runs/.../editor_generality_v1/
  report_summary.json
  editor_condition_results.csv
  paired_bootstrap.csv
  safety_table.csv
  cost_table.csv
  editor_generality_decision.md
```
