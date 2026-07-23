# Continual-Learning Alternatives Catalog

This catalog lists the broad design space. The master campaign tests the highest-value families directly and keeps the remaining ideas as conditional extensions.

## Core diffusion-LM alternatives

1. **DiffusionGrow function-preserving growth**  
   Add timestep-conditioned trainable branches while retaining the frozen pretrained path. Zero initialization gives exact base behavior before adaptation.

2. **Shared growth branch with sequential updates**  
   One branch is updated across edit blocks; tests raw catastrophic forgetting.

3. **Block-specific growth branches**  
   Add one branch per edit block and route by prompt/edit relevance.

4. **Progressive branch compression**  
   Periodically distill multiple branches into a smaller shared branch.

5. **Small-block continual pretraining**  
   Use small diffusion blocks during consolidation to preserve informative contexts.

6. **Partial-state replay**  
   Replay old edits at fully masked, early, middle, and late denoising states.

7. **Trajectory-balanced replay**  
   Balance replay by timestep, active-mask count, relation, and target length.

8. **Dark experience replay**  
   Store top-k logits or branch outputs for old edit states instead of only target labels.

9. **Learning without forgetting across mask states**  
   Distill the previous denoiser on new-task inputs and training-only anchors.

10. **Gradient episodic memory / A-GEM**  
    Project new-edit gradients so they do not increase loss on stored old-edit states.

## Parameter-isolation and routing alternatives

11. **C-LoRA-style self-regularized continual LoRA**  
    Penalize interference between sequential low-rank branches.

12. **GainLoRA-style gated branch integration**  
    Add a branch per block and learn gates that suppress new branches on old tasks.

13. **O-Edit orthogonal subspace updates**  
    Orthogonalize each new update against earlier edit directions.

14. **FGGM Fisher-guided gradient masking**  
    Freeze or downweight parameters important to the base and previous edits.

15. **NuSA-style null-space adaptation**  
    Restrict new low-rank updates to an approximate protected null space.

16. **Progressive networks / expandable experts**  
    Freeze old branches and add lateral connections for new edit blocks.

17. **PackNet/dynamic sparse masks**  
    Allocate disjoint sparse parameter subsets to sequential edit groups.

18. **Relation- or subject-specific experts**  
    Route edits to specialized branches; requires careful same-subject gating.

## Memory alternatives

19. **MEMOIR sparse residual memory**  
    Use sparse activation masks to isolate edits in a dedicated memory module.

20. **Sparse Memory Finetuning**  
    Update only highly accessed memory rows.

21. **Fast episodic memory + slow semantic branch**  
    Apply edits immediately in external memory, then consolidate safely.

22. **Reservoir replay**  
    Maintain a bounded unbiased buffer of old edit states.

23. **Interference-prioritized replay**  
    Replay edits with high gradient conflict or high observed forgetting.

24. **Coreset selection**  
    Choose representative old states by relation, hidden-state geometry, or Fisher leverage.

25. **Free-text causal memory**  
    Store causal abstractions or edit descriptions instead of parameter changes.

## Schrödinger-bridge combinations

26. **Bridge generative replay**  
    Sample old-edit partial states from a reference bridge conditioned on old endpoints instead of ordinary random masking.

27. **Bridge replay with previous-model distillation**  
    Distill the previous branch on bridge-sampled states.

28. **Multi-marginal SB over edit blocks**  
    Treat successive old/new function distributions as multiple marginals and find a minimum-KL consolidation path.

29. **Function-space Schrödinger barycenter**  
    Consolidate old and new denoiser distributions at each state through an entropic KL barycenter.

30. **Unbalanced SB replay**  
    Allocate transport mass only to edits judged at risk of forgetting.

31. **Parameter-space SB over adapter latents**  
    Bridge from the previous adapter state to the new optimum in a low-dimensional adapter manifold.

32. **Fisher-metric parameter bridge**  
    Use a behavioral metric rather than Euclidean adapter distance.

33. **SB-guided branch merging**  
    Compare entropy-regularized consolidation with linear averaging, EMA, and task arithmetic.

34. **Bridge-based rehearsal scheduler**  
    Use bridge potential or path cost to prioritize which old edits need replay.

35. **Doob-transform retention control**  
    Reweight continual training trajectories toward states that retain old edits while acquiring the new block.

## Statistical and Bayesian alternatives

36. **EWC / online EWC**  
    Penalize changes to Fisher-important parameters.

37. **Synaptic Intelligence / MAS**  
    Accumulate online parameter importance without full replay.

38. **Online Laplace continual adapters**  
    Maintain a posterior precision over adapter parameters.

39. **Kalman-filter branch updates**  
    Treat the adapter state as a latent dynamical system with noisy edit observations.

40. **Hierarchical Bayesian relation adapters**  
    Share statistical strength across relations while retaining edit-specific corrections.

41. **CVaR forgetting optimization**  
    Optimize the worst-forgotten tail rather than only average retention.

42. **Conformal selective editing**  
    Abstain or route to external memory when safe continual adaptation cannot be certified.

## Post-hoc consolidation and repair

43. **Spectral unforgetting**  
    Remove low-signal/noise components from accumulated parameter deltas.

44. **Weight interpolation / WiSE-style merging**  
    Interpolate adapted and base branches to trade plasticity for retention.

45. **TIES/DARE-style task-vector merging**  
    Resolve sign conflicts before merging sequential branches.

46. **Knowledge-driven parameter fusion**  
    Weight branch contributions according to edit relevance and retention risk.

47. **Teacher-student periodic consolidation**  
    Train a clean student from the base plus all accepted edit memories.

## Recommended priority

Highest probability:

```text
DiffusionGrow + partial-state dark replay
DiffusionGrow + sparse routed residual memory
DiffusionGrow + O-Edit/FGGM protection
dual-memory fast/slow consolidation
```

Highest Schrödinger-specific value:

```text
bridge generative replay
multi-marginal/function-space SB consolidation
Fisher-metric parameter-space SB
```

Highest engineering risk:

```text
full multi-marginal SB
Bayesian parameter bridge
large progressive expert systems
```
