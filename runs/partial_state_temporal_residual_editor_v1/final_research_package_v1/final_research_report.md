# Partial-State Temporal Residual Editor Final Report

- Terminal outcome: `formal_negative`
- Claim classification: `formal_bounded_negative`
- Reason: no predeclared positive claim survived the fixed pilot100 comparison
- Historical analysis/final splits opened: `false`

## Stage Evidence

- A0: `True`
- A1: `True`
- B0: `not run`
- C0: `True`
- C1: `True`
- C2: `True`
- D1: `True`
- D2: `True`
- E1: `True`
- E2: `False`
- E3: `not run`
- F1: `not run`
- F2: `not run`
- F3: `not run`
- G1: `not run`
- G2: `not run`

## Key Scientific Evidence

- Stable temporal full-mask pilot: rewrite `0.82`, paraphrase `0.5`, same-subject TFPR `0.07`.
- Random-site control: rewrite `0.31`, paraphrase `0.49`, same-subject TFPR `0.03`.
- Diffusion-specific partial-state criterion: `False`.
- State-conditioning criterion: `False`.
- Strongest fixed-pilot tradeoff: `prompt_memory` with rewrite `0.73`, paraphrase `0.55`, and same-subject TFPR `0.16`.

## Interpretation

No predeclared positive claim survived the bounded pilot. The campaign therefore stops before dev selection and untouched confirmation. The evidence supports a diagnostic result: temporal residuals can produce edit efficacy, but the tested partial-state and state-conditioned variants did not jointly satisfy efficacy and locality.

## Limitations

The source-compatible TimeROME branch was equation-level because no official code source was available. Smoke and pilot evidence use the fresh campaign manifests; skipped locked/scaling/backbone stages are not reported as method failures.
