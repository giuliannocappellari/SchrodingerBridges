# H — Final Paper and Reproducibility Package

## Required reports

```text
main_results_table.csv
multi_token_table.csv
same_subject_stress_table.csv
locality_table.csv
causal_localization_table.csv
state_bucket_ablation.csv
relation_table.csv
compute_storage_table.csv
paired_bootstrap.csv
rewrite_locality_pareto.png
state_bucket_plot.png
multi_token_plot.png
causal_heatmap.png
failure_cases.csv
artifact_availability_manifest.json
reproducibility_manifest.json
final_research_report.md
paper_claim_recommendation.md
terminal_package_validation.json
```

## Claim taxonomy

```text
full_editor_positive
pareto_locality_positive
diffusion_specific_positive
state_conditioning_positive
reproduction_only
diagnostic_negative
formal_bounded_negative
```

## Reproducibility requirements

Record:

```text
Git commit
model/tokenizer revisions
source paper/code revisions
all split hashes
all config files
runtime feature schemas
residual-memory formula and solver tests
raw per-edit outcomes
bootstrap seeds
RunPod image, Python, CUDA, Torch, Transformers versions
commands for every table and figure
artifact hashes
```

Create a cheap synthetic test that verifies the residual-memory closed form and state-bucket routing without loading LLaDA.

## Interpretation requirements

The final report must separate:

```text
TimeROME reproduction
CounterFact adaptation
partial-state mechanism benefit
state-conditioning benefit
full editor success/failure
locked confirmation
second-backbone evidence
```

A near miss remains a formal failure for its frozen claim, but may still support a different predeclared claim class if that class passed independently.
