from __future__ import annotations

import torch

from scripts.run_trm_fullmask_baseline import frozen_policy_layers
from scripts.trm_editor import (
    FactorizedResidualMemory,
    fit_factorized_residual_memory,
    harmonic_mean,
    install_factorized_residual_memory,
)


def test_factorized_memory_matches_closed_form_ridge() -> None:
    torch.manual_seed(13)
    keys = torch.randn(5, 7)
    residuals = torch.randn(5, 3)
    ridge = 0.2
    memory = fit_factorized_residual_memory(keys, residuals, ridge=ridge)
    expected = keys.T @ torch.linalg.solve(
        keys @ keys.T + ridge * torch.eye(5), residuals
    )
    probe = torch.randn(4, 7)
    assert torch.allclose(memory.predict(probe), probe @ expected, atol=1e-5, rtol=1e-5)


def test_factorized_memory_payload_round_trip() -> None:
    torch.manual_seed(19)
    memory = fit_factorized_residual_memory(
        torch.randn(4, 6), torch.randn(4, 5), ridge=0.1
    )
    loaded = FactorizedResidualMemory.from_payload(memory.cpu_payload(), device="cpu")
    probes = torch.randn(3, 6)
    assert torch.allclose(memory.predict(probes), loaded.predict(probes))
    assert loaded.edit_row_count == 4
    assert loaded.protect_row_count == 0


def test_factorized_memory_top_q_and_runtime_hook() -> None:
    keys = torch.eye(3)
    residuals = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 2.0]])
    memory = fit_factorized_residual_memory(keys, residuals, ridge=0.01)
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.data.zero_()
    inputs = torch.tensor([[[1.0, 0.0, 0.0]]])
    with install_factorized_residual_memory(module, memory, alpha=1.0, top_q=1) as state:
        output = module(inputs)
    assert int((output != 0).sum()) == 1
    assert state["hook_calls"] == 1
    assert state["nonzero_delta_coordinates"] == 1


def test_harmonic_mean_has_zero_boundary() -> None:
    assert harmonic_mean((1.0, 1.0, 1.0)) == 1.0
    assert harmonic_mean((0.5, 0.0, 1.0)) == 0.0


def test_frozen_policy_layers_reads_only_mlp_last_subject(tmp_path) -> None:
    path = tmp_path / "site_policy_comparison.csv"
    path.write_text(
        "policy_id,candidate_ids_json\n"
        'stable_temporal_site_set,"[""L06:mlp:last_subject""]"\n'
        'random_site,"[""L09:mlp:last_subject""]"\n',
        encoding="utf-8",
    )
    assert frozen_policy_layers(tmp_path) == {
        "stable_temporal_top1": 6,
        "random_site_top1": 9,
    }
