from __future__ import annotations

import pytest

from scripts.dnpe_editor import normalized_aie
from scripts.run_dnpe_causal_trace import aggregate_effects


def test_normalized_aie_is_bounded() -> None:
    assert normalized_aie(0.8, 0.2, 2.0) == 1.0
    assert normalized_aie(0.8, 0.2, -2.0) == -1.0


def test_causal_aggregate_counts_edits() -> None:
    rows = [
        {"case_id": "a", "layer": 4, "component": "mlp", "position": "last_subject", "revealed_count": 0, "normalized_aie": 0.4},
        {"case_id": "b", "layer": 4, "component": "mlp", "position": "last_subject", "revealed_count": 0, "normalized_aie": 0.2},
    ]
    result = aggregate_effects(rows)[0]
    assert result["num_edits"] == 2
    assert result["mean_normalized_aie"] == pytest.approx(0.3)
    assert result["positive_edit_fraction"] == 1.0
