# Paper and Reproducibility Plan

Protocol: `diffusion_native_causal_partial_state_editor_v1`

## Final package

Create:

```text
runs/diffusion_native_causal_partial_state_editor_v1/final_research_package_v1/
```

Required artifacts:

```text
report_summary.json
final_research_report.md
paper_claim_recommendation.md
main_results_table.csv
same_subject_stress_table.csv
multi_token_table.csv
causal_localization_table.csv
locality_distribution_table.csv
compute_storage_table.csv
sequential_edit_table.csv
paired_bootstrap.csv
rewrite_locality_pareto.png
causal_heatmap.png
partial_state_plot.png
update_norm_locality_plot.png
failure_cases.csv
artifact_availability_manifest.json
reproducibility_manifest.json
terminal_package_validation.json
```

## Reproducibility manifest

Record:

```text
Git commit and tag
all model/tokenizer revisions
Python/CUDA/PyTorch/Transformers versions
RunPod GPU type
all split and prompt fingerprints
all source-code hashes
all selected hyperparameters
all random seeds
all update/checkpoint hashes
all result artifact hashes
commands for every main table and figure
```

## One-command reproduction

Provide commands such as:

```bash
python reproduce_dnpe_paper.py --table main
python reproduce_dnpe_paper.py --figure causal_heatmap
python reproduce_dnpe_paper.py --validate-terminal-package
```

Add CPU-safe toy tests for:

```text
causal restoration accounting
partial-state enumeration
null-space projector correctness
constrained update closed form
train/eval leakage
paired bootstrap
```

## Main figures

### Figure 1 — Temporal causal localization

Heatmap over layers, token positions, and mask counts.

### Figure 2 — Rewrite/locality Pareto

Show ordinary MDM-MEMIT, partial-state, AlphaEdit-style, TimeROME-style, and the main editor.

### Figure 3 — Multi-token robustness

Target lengths 2–4 with full-span exact and paraphrase.

### Figure 4 — Update geometry

Protected dimension, update norm, and same-subject drift.

### Figure 5 — Sequential scaling

Edit count versus efficacy/locality/interference.

## Claim matrix

Classify each claim as:

```text
supported
partially supported
rejected under bounded protocol
not tested
protocol-infeasible
infrastructure-blocked
```

Potential positive claims:

```text
causal temporal localization improves parametric edit efficiency
partial-state optimization improves multi-token editing
null-space constrained updates improve same-subject locality
full diffusion-native editor improves the joint efficacy/locality trade-off
```

Do not claim a strong editor unless locked analysis/final results pass.

## Failure report requirements

A negative result must separate:

```text
baseline reproduction failure
causal localization failure
partial-state optimization failure
locality projection failure
editability/locality incompatibility
second-backbone integration failure
analysis generalization failure
```

## Final shutdown

After terminal package validation:

```text
mark campaign state terminal
write final Pod/cost state
verify no active tmux/Python GPU job
stop the configured Pod
record stopped status
```
