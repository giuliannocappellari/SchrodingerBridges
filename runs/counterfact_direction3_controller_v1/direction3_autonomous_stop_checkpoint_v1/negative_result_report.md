# Direction 3 Negative Result Report

## Summary

The Direction 3 controller pilot successfully built and audited a deployable frozen-feature cache, trained a strong edit-intent gate, and maintained very low negative guidance on locality/same-subject negatives. However, the learned value controller failed to demonstrate that it was using the intended state/edit representations rather than a target-indicator shortcut.

## Attempt 1

- Full value Spearman: 0.5350
- Teacher top-3 overlap: 0.5208
- Target-indicator-only Spearman: 0.9589
- Gate ROC-AUC: 0.9882
- Negative guidance ratio: 0.000454

## Bounded Rescue

The allowed rescue used `proj_dim=256`, `negative_identity_weight=2.0`, and `target_loss_weight=0.0`. It did not solve the shortcut issue.

- Full value Spearman: 0.5224
- Teacher top-3 overlap: 0.5263
- Target-indicator-only Spearman: 0.9589
- No-target-indicator Spearman: 0.5448
- Gate ROC-AUC: 0.9877
- Negative guidance ratio: 0.000711

## Interpretation

The gate result is encouraging, but the value teacher appears too target-indicator dominated for this controller objective. Because the protocol required the full representation model to beat target-indicator-only and show state sensitivity, Stage 2A actual decoding would be scientifically unjustified.

## Protocol Safety

No analysis or final split was used. No actual D3 decoding was run after the offline hard criteria failed.
