# N5 — Joint Answer-Span Coupling Plan

## Hypothesis

Factorized token decisions fail to model dependencies inside multi-token factual objects. A small coupled answer-span model should improve full-span exact match without broadening locality effects.

## Scope

Use fresh KAMEL targets of lengths 2, 3, and 4. Keep the underlying partial-state editor fixed.

## Required variants

```text
factorized independent token scoring
pairwise adjacent-token energy
low-rank pairwise CRF/factor graph
exact inference for lengths <=4
beam-search approximation diagnostic
```

Candidate support must be identical across factorized and coupled methods.

## Training

Fit coupling potentials only on allowed KAMEL train rows. No evaluation target strings or outcomes may enter training.

## Mechanism gate

```text
coupled model improves conditional log likelihood on held-out spans
pairwise mutual-information residual is nonzero
exact and brute-force inference agree on small synthetic tests
```

## Pilot success

```text
full-span exact improves >= 0.10 on at least two target lengths
pooled paired lower CI > 0
token F1 does not decline
malformed <= 0.05
same-subject/locality metrics do not worsen by >0.03
compute <= 2x factorized baseline
```

## Rescue

One rescue only:

```text
increase coupling rank from 32 to 64 or beam width from 4 to 8
```

Do not add a full sequence model.
