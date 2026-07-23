# Final Selection and Reporting

## Confirmation rule

Only pilot-eligible candidates run on the fresh 200-edit confirmation stream. No retuning.

## Selection hierarchy

```text
1. full continual editor
2. confirmed SB-specific continual result
3. confirmed retention/locality Pareto improvement
4. confirmed efficiency/scaling result
5. confirmed mechanism-only result
6. no promising continual direction
```

## Tie-breaking

1. lower average forgetting;
2. lower same-subject TFPR;
3. higher past-edit retention;
4. lower base retention loss;
5. lower storage growth;
6. lower compute.

## Required final outputs

```text
final_research_report.md
direction_selection_matrix.csv
plasticity_retention_curves.csv
forgetting_by_block.csv
same_subject_results.csv
base_denoising_retention.csv
multi_token_results.csv
compute_storage_results.csv
paired_bootstrap.csv
track_status_registry.json
next_direction_recommendation.md
SELECTED_CONTINUAL_DIRECTION_FULL_CAMPAIGN_DRAFT.md
terminal_package_validation.json
```

## Wording guard

A track that improves retention but fails factual acquisition is a mechanism result, not a successful editor. A bridge-based track must beat its matched non-SB baseline before receiving an SB-specific claim.
