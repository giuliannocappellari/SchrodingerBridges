# Primary Sources

The implementation and report must cite and distinguish these sources.

## TimeROME-DLM

- **TimeROME-DLM: Temporal Causal Tracing and Low-Rank Inference-Time Knowledge Editing for Masked Diffusion Language Models**
- arXiv:2606.12841
- Core ideas used by this campaign:
  - Temporal Indirect Effect causal tracing over denoising time.
  - A closed-form, low-rank residual edit memory.
  - Ridge regularization and sparsification.
  - Frozen backbone; residual applied during diffusion forwards.
  - Transfer across several masked diffusion language-model backbones.

## Knowledge Editing in Masked Diffusion Language Models

- **Knowledge Editing in Masked Diffusion Language Models**
- arXiv:2606.03924
- Core ideas used by this campaign:
  - Early-to-middle MLP localization at the last subject token.
  - Multi-token degradation caused by partially unmasked inference states.
  - Target optimization across partial-mask states.

## AlphaEdit

- **AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models**
- arXiv:2410.02355
- Used as the protected-subspace/null-space baseline and motivation.
- Its assumptions are not presumed to transfer unchanged to masked diffusion models.

## Historical local evidence

The campaign must preserve and cite the repository's completed evidence showing:

```text
partial-state edit optimization produced large multi-token rewrite gains;
a static null-space projection reduced same-subject leakage but did not satisfy the frozen joint criterion;
rule-based runtime gates and learned value-controller routes were previously closed under their own protocols.
```

Historical results motivate the new protocol but cannot be tuned on or overwritten.
