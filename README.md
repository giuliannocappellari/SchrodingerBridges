# Masked-Diffusion MEMIT + Schrödinger Goal Bundle

Place every file in this bundle at the repository root.

## Active goal

Obtain a strong positive factual-editing result in a masked diffusion language model, then test whether Schrödinger-style path control improves locality or denoising trajectory quality.

## Required execution order

```text
M1 MDM-MEMIT reproduction
M2 partial-mask MEMIT
M3 Schrodinger/path-KL regularization
M4 exact mask-pattern SB
F1 adaptive edit-memory fallback, only if M1 fails
F2 toy text CSBM fallback, only if M3 and M4 both fail
final cross-track package
```

## Autonomous Pod policy

The Pod stays running from campaign start until the entire goal is complete or an unrecoverable Pod/infrastructure issue remains after retries. Monetary cost does not stop or reorder the work.

## Launch

Set the RunPod variables and:

```bash
export MDM_MEMIT_SB_AUTONOMOUS_MODE=1
export MDM_MEMIT_SB_MAX_INFRA_RETRIES=3
```

Then use Codex Goal mode and paste `START_MDM_MEMIT_SB_GOAL.md`.
