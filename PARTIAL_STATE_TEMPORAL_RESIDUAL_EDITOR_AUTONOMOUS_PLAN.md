# Partial-State Temporal Residual Editor Autonomous Research Plan

Protocol: `partial_state_temporal_residual_editor_v1`

## Central question

Can a temporally localized low-rank residual memory, optimized across partial denoising states and protected with state-conditioned same-subject/locality anchors, improve factual-editing locality at matched efficacy?

## Main hypothesis

A permanent static update is forced to behave correctly across heterogeneous diffusion states. A temporal residual memory can instead intervene dynamically at a causal coordinate during each diffusion forward. Partial-state target construction should improve multi-token efficacy; state-conditioned protection should reduce same-subject leakage.

## Method summary

For state bucket b:

\[
M_b = \arg\min_M
\|M K_{edit,b} - D_b\|_F^2
+ \lambda_r\|M\|_F^2
+ \lambda_p\|M K_{protect,b}\|_F^2.
\]

At inference:

\[
h'_t = h_t + \alpha_b\,\mathrm{Sparse}_{q_b}(M_b k_t).
\]

The backbone remains frozen.

---

# Phase A — Bootstrap, source audit, and persistent state

## A0.1 Campaign state

Create:

```text
runs/partial_state_temporal_residual_editor_v1/autonomous_campaign_v1/
  campaign_state.json
  stage_history.csv
  autonomous_log.md
  cost_state.json
  artifact_registry.json
```

Acceptance:

```text
autonomous_mode = true
active protocol matches root control file
historical protocols read-only
historical analysis/final splits locked
Pod configuration present
```

## A0.2 Start and retain Pod

Start the configured existing Pod, verify GPU/SSH/persistent volume/tests, and keep it running until campaign terminal.

## A0.3 Source and artifact audit

Audit:

```text
TimeROME-DLM paper and code
Knowledge Editing in Masked Diffusion Language Models paper/code
AlphaEdit reference implementation
historical local partial-state and static-nullspace results
available LLaDA and Dream checkpoints
```

Write a source-to-implementation map. Do not silently substitute algorithms.

---

# Phase B — Fresh protocol data

## B0.1 CounterFact manifests

Create fresh disjoint:

```text
cf_trm_localize_50
cf_trm_smoke_20
cf_trm_pilot_100
cf_trm_dev_200
cf_trm_locked_500
cf_trm_scaling_100
```

Stratify by relation, target length, base target probability, and subject ambiguity.

## B0.2 KAMEL manifests

Create fresh:

```text
kamel_trm_dev_50_per_length
kamel_trm_pilot_100_per_length
kamel_trm_locked_200_per_length
```

for lengths 2, 3, and 4. Add 5/6 only as secondary diagnostics when supply permits.

## B0.3 Anchor data

Create training-only same-subject, near/far, attribute, generation, and unrelated anchors. Ensure zero prompt overlap with evaluation.

Acceptance:

```text
zero fact/prompt overlap among roles
historical development fingerprints excluded
source rows and tokenization audited
real prompt provenance reported
```

---

# Phase C — Source reproduction and CounterFact adaptation

## C0. TimeROME source reproduction

Execute `TIMEROME_REPRODUCTION_PLAN.md`.

A technical source reproduction failure does not automatically end the campaign if component tests, equations, and residual-memory invariants validate; it limits reproduction claims.

## C1. Temporal causal localization

Execute the localization section of `COUNTERFACT_TEMPORAL_RESIDUAL_ADAPTATION_PLAN.md` on `cf_trm_localize_50`.

## C2. Full-mask CounterFact temporal residual baseline

Fit and evaluate ordinary full-mask temporal residual editing on smoke20 and pilot100.

Acceptance to continue:

```text
nontrivial rewrite gain over base;
temporal site better than random site or more efficient at matched efficacy;
finite residual parameters;
no catastrophic utility collapse.
```

One site-policy rescue is allowed.

---

# Phase D — Partial states and state-conditioned locality

## D1. Partial-state target-delta construction

Execute `PARTIAL_STATE_TARGET_DELTA_PLAN.md`.

Mandatory variants:

```text
fullmask temporal residual
shared partial-state residual
mask-count cycling residual
trajectory-sampled residual
state-bucketed residual
```

## D2. State-conditioned protection

Execute `STATE_CONDITIONED_LOCALITY_PLAN.md`.

Mandatory variants:

```text
unprotected temporal residual
static null-space projection
shared soft preservation penalty
state-conditioned preservation
state-conditioned sparsification
```

Run the relation-conditioned rescue only if its trigger is met.

---

# Phase E — Pilot ladder

## E1. Smoke20

Use only for integration and the bounded `alpha/lambda/q` calibration defined in the plans.

Red failure conditions:

```text
no rewrite gain
malformed > 0.05
same-subject TFPR > 0.30
runtime schema mismatch
residual causes numerical instability
```

A red failure after rescue ends the campaign negatively.

## E2. Pilot100

Compare the complete method registry. Select at most three candidates corresponding to predeclared claim classes.

Advance if at least one candidate supports:

```text
full editor
Pareto locality
diffusion-specific partial-state
state-conditioning
```

and passes deployment, leakage, malformed, and utility checks.

## E3. KAMEL multi-token pilot

Run lengths 2/3/4 with at least two state-pattern seeds and paired evaluation.

---

# Phase F — Development selection and locked confirmation

## F1. Dev200

Freeze the selected architecture families before dev. Run a bounded staged sweep:

```text
1. alpha/lambda/q
2. state bucket thresholds
3. protection strength
4. optional relation rescue if already triggered
```

No architecture expansion.

Select one primary and up to two secondary claim candidates.

## F2. Lock

Write `dev_method_lock.json` with all method/config/split/code hashes.

## F3. Fresh locked confirmation

Execute `LOCKED_CONFIRMATION_PLAN.md` exactly once. No tuning after inspection.

A failed locked claim is terminal for that claim and v1.

---

# Phase G — Scaling and second backbone

## G1. Edit-count scaling

Run 1, 10, 50, and 100 edit batches for the frozen primary candidate and strongest baseline.

## G2. Dream

Run the second-backbone plan if the primary method supports a positive claim or if the final diagnostic report requires a cross-backbone check. One integration repair is allowed.

---

# Phase H — Final package and shutdown

Execute `PAPER_REPRODUCIBILITY_PLAN.md`.

The final validator must verify:

```text
all required artifacts exist and are nonempty
all recorded hashes match
all track statuses are terminal
historical analysis/final splits unused
claim classification follows frozen evidence
Pod is idle before shutdown
```

After validation:

```text
mark campaign completed or formal_negative
stop the Pod
report the consolidated result
```

---

# Terminal decision hierarchy

## Full positive

A locked candidate satisfies the full editor criteria.

## Pareto positive

A locked candidate significantly reduces same-subject leakage at matched efficacy.

## Diffusion-specific positive

Partial-state temporal residual editing produces robust multi-token gains.

## State-conditioning positive

State-conditioned preservation outperforms shared/global preservation at matched efficacy.

## Reproduction-only

Source method or partial-state mechanism reproduces, but the novel adaptation does not pass.

## Formal bounded negative

No predeclared positive claim survives the bounded pilot/locked path.
