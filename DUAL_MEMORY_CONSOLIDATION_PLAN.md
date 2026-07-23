# C9 — Dual-Memory Fast/Slow Consolidation

## Hypothesis

Immediate factual acquisition and long-term stable consolidation should be separated.

## Architecture

```text
fast memory:
  sparse routed residual or external edit memory
  immediate low-cost updates

slow memory:
  DiffusionGrow function-preserving branch
  periodic consolidation using replay/distillation

base path:
  permanently frozen
```

## Consolidation schedule

Compare consolidation every:

```text
10 edits
25 edits
50 edits
```

## Variants

```text
fast only
slow only
fast + ordinary replay consolidation
fast + dark replay consolidation
fast + bridge replay consolidation
```

## Metrics

```text
immediate acquisition
post-consolidation retention
fast-memory eviction effects
slow-branch forgetting
storage scaling
latency
```

## Pass

Class A, B, C, or D.

## Rescue

One rescue may alter only the consolidation interval among the predeclared values.
