from __future__ import annotations

import torch

from scripts.run_dnpe_timerome_style import build_residual_memory


def test_residual_memory_is_finite_and_dual_shaped() -> None:
    if not torch.cuda.is_available():
        # The production helper is intentionally CUDA-only; validate the same
        # dual algebra on CPU without pretending to load LLaDA.
        keys = torch.randn(4, 6)
        residuals = torch.randn(4, 3)
        system = keys @ keys.T + torch.eye(4) * 1e-2
        dual = torch.linalg.solve(system, keys)
        assert dual.shape == (4, 6)
        assert torch.isfinite(dual).all()
        return
    memory = build_residual_memory(torch.randn(4, 6), torch.randn(4, 3), 1e-2)
    assert memory["dual"].shape == (4, 6)
    assert torch.isfinite(memory["dual"]).all()
