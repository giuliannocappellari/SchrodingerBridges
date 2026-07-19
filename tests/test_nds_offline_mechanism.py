import torch

from scripts.run_nds_offline_mechanism import (
    n1_report,
    n2_report,
    n3_report,
    nearest_relation_accuracy,
)


def _payload(seed=1):
    generator = torch.Generator().manual_seed(seed)
    edit = torch.randn(20, 8, generator=generator)
    anchor = torch.randn(20, 8, generator=generator) * 0.3
    protected = {
        name: torch.randn(20, 8, generator=generator)
        for name in ("same_subject", "near", "far", "unrelated")
    }
    basis = torch.linalg.qr(torch.randn(8, 8, generator=generator)).Q
    return {
        "edit_keys": edit,
        "subject_anchor_keys": anchor,
        "relation_ids": [f"P{index % 2}" for index in range(20)],
        "relation_global_mean": edit.mean(dim=0),
        "relation_means": {
            "P0": edit[::2].mean(dim=0),
            "P1": edit[1::2].mean(dim=0),
        },
        "protected_keys": protected,
        "fisher_diagonal": torch.rand(8, generator=generator) + 0.2,
        "fisher_basis": basis,
        "fisher_eigenvalues": torch.linspace(2.0, 0.1, 8),
    }


def test_relation_classifier_and_reports_are_finite():
    train = _payload(1)
    calibration = _payload(2)
    classification = nearest_relation_accuracy(
        train["edit_keys"], train["relation_ids"], calibration["edit_keys"], calibration["relation_ids"]
    )
    assert 0 <= classification["accuracy"] <= 1
    for report in (
        n1_report(train, calibration, 6),
        n2_report(train, calibration, 6),
        n3_report(train, calibration, 6),
    ):
        assert report["track_id"] in {"N1", "N2", "N3"}
        assert isinstance(report["mechanism_pass"], bool)
        assert isinstance(report["checks"], dict)
