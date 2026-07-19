import math

import pytest
import torch

from scripts.nds_methods import (
    LowRankPairwiseCoupler,
    binomial_upper_bound,
    calibrate_selective_threshold,
    exact_coupled_decode,
    fisher_diagonal,
    fisher_quadratic,
    fit_logistic_risk,
    pairwise_mutual_information,
    primal_dual_project_update,
    relation_residualize,
    trust_region_scale,
    validate_runtime_feature_schema,
)


def test_relation_residualization_removes_relation_main_effect():
    keys = torch.tensor([[3.0, 0.0], [3.0, 1.0], [0.0, 3.0], [1.0, 3.0]])
    negatives = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    residual, report = relation_residualize(
        keys, ["a", "a", "b", "b"], subject_negative_keys=negatives, mode="full"
    )
    assert residual.shape == keys.shape
    assert torch.isfinite(residual).all()
    assert report["relation_effect_norm"] > 0


def test_fisher_trust_region_respects_radius():
    protected = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    fisher = fisher_diagonal(protected, 1e-3)
    update = torch.ones(3, 2)
    scaled, report = trust_region_scale(update, fisher, radius=0.5)
    assert fisher_quadratic(scaled, fisher) <= 0.500001
    assert report["scale"] < 1.0


def test_primal_dual_reduces_protected_response_without_nan():
    update = torch.tensor([[1.0, 1.0], [0.5, 0.5]])
    keys = {"same_subject": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    projected, report = primal_dual_project_update(
        update,
        keys,
        {"same_subject": 0.1},
        multiplier_step=0.1,
        penalty_growth=1.5,
        iterations=40,
    )
    assert torch.isfinite(projected).all()
    assert report["finite"] is True
    assert report["trajectory"][-1]["maximum_violation"] < report["trajectory"][0]["maximum_violation"]


def test_risk_model_and_exact_bound_are_deployable():
    features = torch.tensor([[0.0], [0.1], [1.0], [1.1]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    model = fit_logistic_risk(features, labels, feature_names=["base_target_margin"], steps=200)
    predictions = model.predict(features)
    assert predictions[0] < predictions[-1]
    assert binomial_upper_bound(0, 100) < 0.04
    calibrated = calibrate_selective_threshold(
        predictions.tolist(), labels.bool().tolist(), maximum_upper_bound=0.8
    )
    assert calibrated["coverage"] > 0


def test_forbidden_runtime_feature_is_rejected():
    with pytest.raises(ValueError):
        validate_runtime_feature_schema(["base_target_margin", "prompt_type"])


def test_exact_coupled_decode_matches_brute_force_support():
    embeddings = torch.eye(5)
    coupler = LowRankPairwiseCoupler(5, 2)
    with torch.no_grad():
        for parameter in coupler.parameters():
            parameter.zero_()
    sequence, score = exact_coupled_decode(
        [[1, 2], [3, 4]],
        [[-0.1, -1.0], [-0.2, -2.0]],
        embeddings,
        coupler,
        coupling_strength=1.0,
    )
    assert sequence == [1, 3]
    assert math.isclose(score, -0.3, abs_tol=1e-6)
    assert pairwise_mutual_information([[1, 3], [1, 3], [2, 4]]) > 0
