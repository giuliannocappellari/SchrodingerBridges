import torch

from scripts.nds_methods import LowRankPairwiseCoupler
from scripts.run_nds_joint_span import (
    calibration_strength,
    exact_decode_support,
    token_f1,
)


def test_exact_joint_decode_uses_identical_support_and_can_change_sequence():
    embeddings = torch.eye(5)
    coupler = LowRankPairwiseCoupler(5, 2)
    with torch.no_grad():
        coupler.left.weight.zero_()
        coupler.right.weight.zero_()
        coupler.bias.fill_(1.0)
    ids = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    log_probs = [torch.tensor([-0.1, -1.0]), torch.tensor([-0.1, -1.0])]
    sequence, score = exact_decode_support(ids, log_probs, embeddings, coupler, 1.0)
    assert sequence == [1, 3]
    assert score > 0


def test_token_f1_and_calibration_selection():
    assert token_f1([1, 2], [1, 2]) == 1.0
    rows = {
        0.25: [{"method": "coupled", "bucket": "rewrite", "exact": False, "token_f1": 0.5}],
        0.5: [{"method": "coupled", "bucket": "rewrite", "exact": True, "token_f1": 1.0}],
    }
    selected, summary = calibration_strength(rows)
    assert selected == 0.5
    assert len(summary) == 2
