# Causal and Temporal Localization Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Objective

Determine which internal coordinate causally mediates factual recall in a masked diffusion LM across the denoising states relevant to editing, and whether choosing the edit site from that evidence improves editing/locality over fixed or random sites.

## Stage C0 — Causal-tracing implementation audit

Implement or verify:

```text
clean run
subject-corrupted run
corrupted-with-restoration run
full hidden-state restoration
MLP contribution restoration
attention contribution restoration
contiguous layer-window restoration
```

For masked diffusion models, append the correct number of answer masks and evaluate old-target probability at the answer positions.

### Tests

```text
restoring every corrupted subject embedding recovers the clean run on a toy model
random restoration is near zero in a synthetic causal fixture
normalized AIE is bounded and finite
layer-window indexing matches model config
mask-position indexing is context-aware
```

### Acceptance

```text
all tests pass
clean/corrupt/restored probabilities finite
no target_new or evaluation outcome used for old-fact site selection
```

---

## Stage C1 — Standard causal tracing

### Dataset

Use a fresh development subset with facts the unedited model knows. Stratify by relation and old-target confidence.

### Coordinates

```text
layer: all transformer layers
position:
  first subject token
  last subject token
  relation cue token when identifiable
  first answer mask
module:
  hidden state
  MLP contribution
  attention contribution
```

### Corruption

Add Gaussian noise to subject-token embeddings using a source-aligned scale, e.g. `3 * subject_embedding_std`, then document the exact rule.

### Metrics

```text
P_clean(old_target)
P_corrupted(old_target)
P_restored(old_target)
normalized_AIE = (P_restored - P_corrupted) / max(P_clean - P_corrupted, eps)
```

### Outputs

```text
standard_causal_tracing_v1/report_summary.json
standard_causal_tracing_v1/per_case_effects.csv
standard_causal_tracing_v1/aie_by_layer_position.csv
standard_causal_tracing_v1/causal_heatmap.png
standard_causal_tracing_v1/random_site_comparison.csv
```

### Acceptance

```text
at least one site family exceeds random-site mean normalized AIE by >=0.15
causal peaks are not driven by fewer than 10% of edits
confidence-stratified location is stable enough to report
```

---

## Stage C2 — Temporal Indirect Effect (TIE)

### Definition

At partial state `x_t`, corrupt the subject, restore candidate coordinate `c`, then continue the frozen denoising policy to the end. Define:

```text
TIE(c, x_t) = future log-probability or success recovery of target_true
              relative to the corrupted trajectory
```

Use old-target factual recall only for localization. The new target is used later for editing, not for site discovery.

### State bank

For each target length:

```text
fully masked state
all possible mask counts
multiple random revealed-position subsets
states from default confidence trajectory
```

For single-token facts, use several synthetic surrounding answer-span lengths or fixed one-mask states as appropriate, but do not fabricate multi-token targets.

### Required analyses

```text
TIE by layer, position, and mask count
site rank stability across mask counts
site overlap across rewrite/paraphrase forms
site stability by relation
correlation between AIE/TIE and actual editability
```

### Outputs

```text
temporal_causal_tracing_v1/report_summary.json
temporal_causal_tracing_v1/tie_by_layer_position_state.csv
temporal_causal_tracing_v1/site_stability_by_edit.csv
temporal_causal_tracing_v1/site_overlap_by_mask_count.csv
temporal_causal_tracing_v1/tie_editability_correlation.csv
temporal_causal_tracing_v1/temporal_heatmaps.png
```

### Acceptance

```text
TIE is finite and reproducible across seeds
selected temporal site exceeds random controls
site instability is quantified rather than hidden
no evaluation-only prompt used to select sites
```

---

## Stage C3 — Site-policy candidates

Freeze at most three policies before pilot evaluation:

```text
fixed_global_site:
  best global early/middle last-subject layer window selected on development

per_edit_top_tie_site:
  per-edit coordinate with strongest training-time TIE

stable_temporal_site_set:
  smallest layer/position set covering the high-TIE coordinates across states
```

Required controls:

```text
random_site
late_answer_position
source-paper fixed site
```

## Site-policy acceptance

A causal-site claim requires at least one:

```text
stress-aware aggregate gain >=0.05 over random site

or

same efficacy within 0.02 with >=25% lower update Frobenius norm

or

same efficacy within 0.02 with fewer edited layers
```

If causal localization does not improve editing outcomes, report it as a localization-only finding and use the strongest empirically effective fixed site for the remaining editor experiment.

## Bounded rescue

One rescue may switch among the three frozen site policies. No new localization family or additional metric-driven site search is allowed after pilot inspection.
