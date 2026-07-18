from __future__ import annotations

import torch

from scripts.run_trm_source_reproduction import run_component_reproduction, synthetic_tie


def test_synthetic_tie_has_stable_nontrivial_peak() -> None:
    tie = synthetic_tie()
    peak = int(tie.argmax())
    layer, step = divmod(peak, tie.shape[1])
    assert (layer, step) == (3, 1)
    assert float(tie[layer, step] - tie.mean()) >= 0.15


def test_component_reproduction_validates_residual_equations() -> None:
    report = run_component_reproduction()
    assert report["temporal_residual_target_mse"] < report["base_target_mse"]
    assert report["temporal_residual_target_mse"] < report["random_coordinate_target_mse"]
    assert report["ridge_dual_primal_max_abs_error"] <= 1e-4
    assert report["sparse_q4_retain_rms_drift"] <= report["dense_retain_rms_drift"] + 1e-8
    assert report["residual_parameters_finite"]
    assert torch.isfinite(torch.tensor(list(value for value in report.values() if isinstance(value, float)))).all()
