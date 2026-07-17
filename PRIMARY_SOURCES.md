# Primary sources

Codex must inspect primary papers and official code before implementing or labelling a reproduction.

## Masked-diffusion knowledge editing

### Knowledge Editing in Masked Diffusion Language Models

- arXiv: https://arxiv.org/abs/2606.03924
- Core relevance:
  - causal tracing finds early-to-middle MLP factual recall at the last subject token;
  - locate-then-edit transfers to LLaDA and Dream;
  - multi-token editing fails at partially unmasked intermediate states;
  - optimization over those states restores performance.

### TimeROME-DLM

- arXiv: https://arxiv.org/abs/2606.12841
- Core relevance:
  - temporal indirect effect causal tracing;
  - low-rank residual edit memory applied at a selected coordinate during diffusion forwards;
  - ridge regularization and sparsification for utility preservation;
  - must be treated as a strong recent baseline, not ignored.

## Locate-then-edit foundations

### ROME

- arXiv: https://arxiv.org/abs/2202.05262
- Core relevance: causal localization and rank-one factual weight update.

### MEMIT

- arXiv: https://arxiv.org/abs/2210.07229
- Core relevance: batched multi-layer associative memory update.

### AlphaEdit

- arXiv: https://arxiv.org/abs/2410.02355
- Official code: https://github.com/jianghoucheng/AlphaEdit
- Core relevance: project perturbations into the null space of preserved knowledge; main locality baseline/inspiration.

## Diffusion backbones

### LLaDA

- arXiv: https://arxiv.org/abs/2502.09992
- Primary checkpoint: `GSAI-ML/LLaDA-8B-Instruct`

### Dream

- arXiv: https://arxiv.org/abs/2508.15487
- Official code: https://github.com/DreamLM/Dream
- Secondary checkpoint: `Dream-v0-Instruct-7B`

## Datasets

### CounterFact

- Hugging Face source used in the repository: `azhx/counterfact`
- Standard factual edit, paraphrase, and neighborhood evaluation.

### KAMEL

- Use for controlled multi-token target lengths.
- Record the exact source/version and tokenizer-dependent length bins.

## Source-handling rules

1. Prefer official paper/code repositories.
2. Record commit hashes and model revisions.
3. If official code is unavailable, label the implementation `-style` or `inspired`, not a reproduction.
4. Do not rely on secondary blog summaries for algorithmic details.
5. Preserve all deviations in `source_audit/implementation_gap.md`.
