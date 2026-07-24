from __future__ import annotations

from scripts.run_cl_bounded_rescues import frozen_rescue_specs


def unsafe_candidate(**overrides):
    row = {
        "confirmation_eligible": "False",
        "current_rewrite_exact": "0.85",
        "current_paraphrase_exact": "0.50",
        "past_retention": "0.70",
        "average_forgetting": "0.12",
        "same_subject_tfpr": "0.06",
        "near_tfpr": "0.01",
        "far_tfpr": "0.0",
        "base_retention_loss_fraction": "0.04",
    }
    row.update(overrides)
    return row


def test_c1_and_c3_rescue_only_use_frozen_router_thresholds() -> None:
    c1 = frozen_rescue_specs("C1", unsafe_candidate())
    c3 = frozen_rescue_specs("C3", unsafe_candidate())
    assert [row["extra_args"][-1] for row in c1] == ["0.35", "0.5"]
    assert [row["extra_args"][-1] for row in c3] == ["0.35", "0.5"]
    assert {row["method"] for row in c1} == {"growth_block_gate"}
    assert {row["method"] for row in c3} == {"sparse_routed_memory"}


def test_c4_and_c5_use_only_predeclared_rescues() -> None:
    c4 = frozen_rescue_specs("C4", unsafe_candidate())
    c5 = frozen_rescue_specs("C5", unsafe_candidate())
    assert [row["method"] for row in c4] == ["gated_adapter_shared_basis"]
    assert [row["extra_args"][-1] for row in c5] == ["16", "32"]


def test_no_rescue_for_failed_acquisition_or_already_eligible() -> None:
    assert not frozen_rescue_specs(
        "C1", unsafe_candidate(current_rewrite_exact="0.50")
    )
    assert not frozen_rescue_specs(
        "C1", unsafe_candidate(confirmation_eligible="True")
    )
