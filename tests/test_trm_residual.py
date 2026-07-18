from __future__ import annotations

import torch

from scripts.trm_residual import fit_residual_memory, fit_state_bucketed_memories, state_bucket


def test_dual_residual_solve_matches_primal_ridge_solution() -> None:
    torch.manual_seed(11)
    keys = torch.randn(7, 5)
    deltas = torch.randn(7, 4)
    ridge = 0.2
    memory = fit_residual_memory(keys, deltas, ridge=ridge)
    expected = torch.linalg.solve(keys.T @ keys + ridge * torch.eye(5), keys.T @ deltas)
    assert torch.allclose(memory.weight, expected, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(memory.weight).all()


def test_preservation_penalty_reduces_protected_key_drift() -> None:
    torch.manual_seed(23)
    edit_keys = torch.randn(8, 6)
    deltas = torch.randn(8, 5)
    protect = torch.randn(12, 6)
    plain = fit_residual_memory(edit_keys, deltas, ridge=0.1)
    protected = fit_residual_memory(
        edit_keys,
        deltas,
        ridge=0.1,
        protect_keys=protect,
        preservation_strength=10.0,
    )
    assert protected.predict(protect).norm() < plain.predict(protect).norm()


def test_state_routing_and_bucketed_memory_contract() -> None:
    assert state_bucket(step_index=0, total_steps=4, active_mask_count=4, span_length=4) == "early"
    assert state_bucket(step_index=1, total_steps=4, active_mask_count=2, span_length=4) == "middle"
    assert state_bucket(step_index=3, total_steps=4, active_mask_count=0, span_length=4) == "late"
    torch.manual_seed(7)
    banks = {
        name: (torch.randn(4, 3), torch.randn(4, 2), torch.randn(5, 3))
        for name in ("early", "middle", "late")
    }
    memories = fit_state_bucketed_memories(banks, ridge=0.2, preservation_strength=1.0)
    assert set(memories) == {"early", "middle", "late"}
    assert all(memory.state_bucket_name == name for name, memory in memories.items())


def test_top_q_sparsification_keeps_exactly_requested_coordinates() -> None:
    keys = torch.eye(3)
    deltas = torch.tensor([[1.0, 4.0, 2.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    memory = fit_residual_memory(keys, deltas, ridge=0.01)
    prediction = memory.predict(keys[:1], top_q=1)
    assert int((prediction != 0).sum()) == 1
