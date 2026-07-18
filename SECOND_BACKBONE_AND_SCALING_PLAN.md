# G — Second Backbone and Edit Scaling

## Edit-count scaling

Evaluate the frozen method at:

```text
1 edit
10 edits
50 edits
100 edits
```

Report:

```text
rewrite/paraphrase/locality
same-subject TFPR
residual-memory rank and bytes
fit time
inference overhead
utility drift
interference with previous edits
```

The scaling track is diagnostic unless the selected primary claim explicitly concerns scalability.

## Second backbone

Primary second model:

```text
Dream-v0-Instruct-7B
```

One bounded integration repair is allowed.

If Dream remains technically infeasible, a predeclared fallback to LLaDA-8B-Base may be run, but it supports only cross-checkpoint evidence, not a broad cross-architecture claim.

## Second-backbone acceptance

At least one positive claim should show the same direction of effect. For a strong generality result:

```text
rewrite/paraphrase effect direction matches LLaDA-Instruct;
same-subject/locality advantage remains;
malformed <= 0.05;
paired lower bound > 0 for at least one primary comparison.
```

Failure of the second backbone does not erase a locked primary-backbone positive result, but limits claim scope.
