# T0/T1 — TimeROME-DLM Reproduction Plan

## Objective

Validate a faithful temporal causal tracing and low-rank residual-memory implementation before adapting it to CounterFact factual replacement.

## Source audit

Codex must obtain the paper and source code when legally available, record commit/revision hashes, and map every implemented equation and hyperparameter to a source section or clearly labeled local adaptation.

## Reproduction scope

Use the source-paper-compatible checkpoint and task when available. At minimum test:

```text
small smoke subset
source validation subset
retain/utility subset
```

Required components:

```text
temporal indirect effect tracing
per-fact coordinate selection
subject-key extraction
target-delta construction
closed-form ridge residual memory
sparsification q
residual application during every diffusion forward
frozen backbone
```

## Baselines

```text
base
random coordinate residual
static causal coordinate residual
temporal coordinate residual
no-sparsification residual
```

## Required artifacts

```text
source_audit.md
source_revision.json
reproduction_config.json
tie_heatmaps/
coordinate_selection.csv
source_task_results.csv
retain_utility.csv
compute_table.csv
report_summary.json
```

## Acceptance

The reproduction path passes if all implementation invariants hold and at least one temporal residual configuration:

```text
moves the intended source-task metric in the expected direction;
outperforms random-coordinate residual intervention;
shows nontrivial temporal localization;
keeps retain/utility drift within the source-safe operating regime or a predeclared local tolerance;
uses a frozen backbone and finite residual parameters.
```

Exact paper numbers are not required if model/data revisions differ, but the discrepancy must be explained.

## Bounded rescue

One source-integration repair is allowed for dependency, checkpoint, or data-interface mismatch. If the exact source task remains unavailable, write `source_reproduction_technically_infeasible`, preserve the implementation audit, and continue to the CounterFact adaptation only if all component unit tests and synthetic causal/residual tests pass.
