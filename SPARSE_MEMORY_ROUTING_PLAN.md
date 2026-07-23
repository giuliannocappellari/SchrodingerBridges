# C3 — Sparse Routed Residual Memory

## Hypothesis

A dedicated sparse memory can isolate sequential edits better than modifying a shared branch. Query-dependent sparse activation should generalize to paraphrases while suppressing unrelated and same-subject activations.

## Variants

```text
MEMOIR-style sparse residual memory
Sparse Memory Finetuning row selection
timestep-conditioned memory keys
relation-conditioned memory keys
shared memory pool
block-partitioned memory pool
```

## Runtime

\[
h'_t = h_t + \sum_{i\in \mathcal A(x,t)} w_i(x,t) M_i,
\]

where only a small active set is retrieved.

## Required comparisons

```text
dense residual memory
sparse memory
random sparse routing
subject-only routing
subject+relation+timestep routing
```

## Metrics

```text
memory-row overlap between edits
activation sparsity
wrong-edit activation
same-subject activation
storage/edit
retrieval latency
retention across stream
```

## Pass

Class A, B, or D.

## Rescue

One rescue may alter the routing sparsity target or memory-row count within a bounded {64,128,256} row family.
