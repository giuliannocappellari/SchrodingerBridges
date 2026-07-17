# Second Backbone and Scaling Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Part A — Dream-v0-Instruct-7B

### Objective

Test whether causal partial-state locality preservation transfers beyond LLaDA.

### Required adaptation audit

```text
model/tokenizer revision
layer count and MLP module names
shifted prediction convention
mask token and answer-span construction
subject position indexing
editable down-projection matrices
decoding schedule
```

One bounded integration repair is allowed.

### Minimal experiment

Use:

```text
20-edit smoke
100-edit confirmation
KAMEL lengths 2,3,4 where tokenization permits
```

Methods:

```text
Dream MDM-MEMIT
Dream partial-state MDM-MEMIT
Dream AlphaEdit-style MDM-MEMIT
Dream causal partial-state null-space editor
```

### Acceptance

```text
rewrite >= strongest Dream baseline -0.10
same direction of same-subject/locality improvement
at least one locality metric improves with no >0.05 efficacy loss
partial-state trend reported for multi-token targets
```

If Dream remains technically infeasible after one repair, record `dream_integration_infeasible` and run the predeclared LLaDA-8B-Base cross-check. That fallback limits the claim to cross-checkpoint generality.

---

## Part B — Batch and sequential edit scaling

### Edit counts

```text
1
10
50
100
```

Optional 500 only if the implementation naturally supports it and all earlier scales pass.

### Methods

```text
MDM-MEMIT
AlphaEdit-style MDM-MEMIT
TimeROME-DLM-style residual memory
main causal partial-state null-space editor
```

### Metrics

```text
rewrite/paraphrase retention as edit count grows
same-subject/near/far TFPR
inter-edit interference
previous-edit retention
update rank and Frobenius norm
condition number
protected-subspace dimension
storage bytes/edit
edit wall clock
inference overhead
```

### Scaling acceptance

A strong scalable result requires:

```text
50-edit rewrite retains >=80% of single-edit rewrite
same-subject TFPR remains within base +0.05
previous-edit retention remains >=0.80
update statistics remain finite
```

A single-edit locality result can remain valid even if scaling fails, but no mass-edit claim may be made.

---

## Part C — General utility diagnostics

At selected edit counts, run small fixed diagnostics:

```text
perplexity or masked-token loss on unrelated text
BoolQ/HellaSwag-style subset if already supported
basic generation fluency
refusal/safety behavior only if an existing safe benchmark is already integrated
```

These diagnostics are secondary and cannot substitute for CounterFact/KAMEL locality.
