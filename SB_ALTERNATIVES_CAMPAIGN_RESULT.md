# Schrodinger-Bridge Alternatives Campaign Result

Protocol: `counterfact_sb_alternatives_campaign_v1`

Terminal status: `scientific_negative_completion`

All five mandatory bounded pilots completed. No track passed its pilot gate, so
the campaign did not scale a track to common `dev_tune_200`, did not select a
primary candidate, and did not open `analysis_500` or `final_test_500`.

## Track outcomes

| Track | Outcome | Main evidence |
|---|---|---|
| T1 learned gate + raw bridge | Formal negative at actual smoke20 | The gate localized activation, but the best learned-gated method reached 0.05 rewrite exact and 0.00 declarative paraphrase exact. |
| T2 activation-space SB | Formal offline negative | The dynamic bridge improved endpoint MSE by 33.34% and was relation-sensitive, but endpoint cosine was 0.6302, identity drift ratio was 0.5173, path energy failed, and negative-target change was +1.1457. |
| T3 conditional answer-span CSBM | Formal offline negative after the outer-4 rescue | Endpoint accuracy was 0.965, but bridge-state sampling added only 0.005 over ordinary noising, bidirectional fitting added 0.000 over forward-only, and identity sparse KL was 0.2004. |
| T4 unbalanced/partial CSBM | Formal offline negative after temperature calibration | Positive endpoint retention was 98.99% and mass ROC-AUC was 0.9882, but same-subject mean transport mass was 0.0995 and identity sparse KL was 0.1837. |
| T5 parameter-space SB | Formal endpoint-family negative after rank-4 rescue | Direct adapters reached 0.90 rewrite and 0.70 paraphrase exact, but same-subject, near, and far TFPR were 0.85, 0.80, and 0.50. Latent bridge training was therefore not permitted. |

## Claim

The bounded pilots found useful edit pressure, relation sensitivity, and strong
endpoint efficacy in several representations, but no viable
Schrodinger-bridge factual editor under the joint efficacy, locality, identity,
and mechanism-specific advantage criteria.

This is a bounded cross-track negative result, not an impossibility claim.

## Reproducibility

The authoritative compact package is under:

```text
runs/counterfact_sb_alternatives_campaign_v1/final_research_package_v1/
```

The pilot registry lock is:

```text
runs/counterfact_sb_alternatives_campaign_v1/pilot_registry_lock.json
```

Final validation passed with all required files present, five terminal track
statuses, nonblank plots, and `analysis_500_used=false` plus
`final_test_used=false`. Informational estimated spend recorded by the campaign
was `$5.033748` at `$0.45/hour`; monetary guards were disabled by protocol.
