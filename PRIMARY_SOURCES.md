# Primary Sources

## Knowledge editing in masked diffusion language models

- Haewon Park and Yohan Jo, **Knowledge Editing in Masked Diffusion Language Models**, arXiv:2606.03924 (2026).
- Reproduction facts used by this campaign:
  - primary MDM: LLaDA-8B-Instruct;
  - locate-then-edit/MEMIT adapted by mask-augmenting every forward-pass input distribution;
  - one mask per target token;
  - key and residual baseline from the last subject token under the same masked context;
  - LLaDA selected edit window: layers 4–7 in the paper;
  - target optimization: learning rate 0.1, 25 steps, clamp norm 0.75, KL factor 0.0625, inherited from the LLaMA sibling;
  - paper CounterFact target-length result: efficacy 0.91, generalization 0.56, specificity 0.19;
  - partial-mask augmentation cycles `k = t mod N`, reveals `k` randomly selected target positions, masks the rest, and computes loss on still-masked positions;
  - paper KAMEL results for LLaDA: N=2 efficacy 0.60→0.87, N=3 0.33→0.76, N=4 0.27→0.73.
- The arXiv v1 says code will be released upon publication. Codex must check once for official code; if absent, implement from the paper and record that fact.

## MEMIT

- Meng et al., **Mass-Editing Memory in a Transformer**, arXiv:2210.07229.
- Use official MEMIT or EasyEdit code as the reference implementation, pinned to a commit.

## Categorical Schrodinger bridges

- Ksenofontov and Korotin, **Categorical Schrödinger Bridge Matching**, arXiv:2502.01416.
- This campaign does not claim full-sequence CSBM. The exact mask-pattern track solves a finite bridge over answer-position reveal states.

## Adaptive retrieval guidance fallback

- Kim and Ye, **Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models**, arXiv:2603.17677.
- Fallback only; not part of the main SB claim.
