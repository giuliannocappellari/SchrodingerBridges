import torch

from scripts.run_nds_selective_conformal import feature_matrix


def test_confirmation_features_preserve_frozen_schema_order():
    ids, matrix = feature_matrix(
        [
            {
                "case_id": "a",
                "base_target_rank": "2",
                "base_target_margin": "-1.0",
                "base_target_probability": "0.1",
                "target_length": "1",
            }
        ]
    )
    assert ids == ["a"]
    assert torch.equal(matrix, torch.tensor([[2.0, -1.0, 0.1, 1.0]]))
