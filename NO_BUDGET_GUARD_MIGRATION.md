# No-Budget-Guard Campaign Migration

The prior autonomous campaign stopped because of a monetary budget guard. That stop is superseded.

Rules now in force:

- Monetary budget variables are not required.
- Monetary cost cannot block, pause, reorder, or skip any planned stage or track.
- Existing validated stages remain valid and must not be rerun unnecessarily.
- Resume from the first incomplete scientifically valid stage in persisted campaign state.
- Keep the existing RunPod Pod running between all stages and tracks.
- Stop the Pod only after the entire campaign goal and final package are complete, or after an unrecoverable Pod/infrastructure issue prevents execution. Scientific or data-integrity failures must first produce a validated formal terminal package.
- Cost may still be reported when available, but it is informational only.
