# TRM Autonomous Log

- 2026-07-18T17:50:48.746014+00:00: campaign initialized.
- 2026-07-18T17:50:49.487030+00:00: A0_bootstrap -> passed; acceptance=True; Pod/GPU/tests verified and historical artifacts snapshotted read-only.
- 2026-07-18T17:50:58.715975+00:00: A1_source_audit -> passed; acceptance=True; Paper sources hashed; unavailable official TimeROME code explicitly limits reproduction claims.
- 2026-07-18T18:04:36.869710+00:00: B0_fresh_protocol -> passed; acceptance=True; Fresh disjoint CounterFact/KAMEL manifests and train-only protection anchors.
- 2026-07-18T18:08:57.258695+00:00: C0_timerome_source_reproduction -> passed_component_branch; acceptance=True; Official TimeROME code/checkpoint unavailable; equation and synthetic causal/residual invariants validated.
- 2026-07-18T18:25:59.922848+00:00: C1_temporal_localization -> passed; acceptance=True; 50-edit localization; stable-vs-random proxy delta=0.0398
- 2026-07-18T18:44:34.177562+00:00: C2_fullmask_temporal_residual -> passed; acceptance=True; rewrite gain=0.8200; stable-random stress delta=0.2456
- 2026-07-18T19:31:22.652851+00:00: D1_partial_state_target_delta -> passed; acceptance=True; selected=state_bucketed_delta; diffusion_specific_pass=False
- 2026-07-18T19:38:21.988854+00:00: D2_state_conditioned_protection -> passed; acceptance=True; selected=state_conditioned_sparsification; state_conditioning_pass=False; rescue_triggered=False
- 2026-07-18T19:56:43.794487+00:00: E1_smoke20 -> passed; acceptance=True; selected=prompt_memory; missing=[]
- 2026-07-18T21:04:43.619266+00:00: E2_pilot100 -> failed; acceptance=False; positive_classes={'full_editor': False, 'pareto_locality': False, 'diffusion_specific_partial_state': False, 'state_conditioning': False}; candidates=[]
- 2026-07-18T21:09:30.981428+00:00: H1_final_package -> passed; acceptance=True; Terminal formal_negative package assembled.
