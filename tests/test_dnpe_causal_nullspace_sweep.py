from __future__ import annotations

from scripts.run_dnpe_causal_nullspace_sweep import d4_id, hard_checks


def test_d4_id_is_stable() -> None:
    assert d4_id(0.95, 1e-3, 1.0) == "variance0p95_ridge1e-03_identity1p0"


def test_hard_checks_apply_frozen_thresholds() -> None:
    report = {
        "rewrite_exact": 0.75,
        "declarative_paraphrase_exact": 0.40,
        "same_subject_tfpr": 0.03,
        "near_tfpr": 0.03,
        "far_tfpr": 0.03,
        "malformed_rate": 0.05,
        "base_summary": {
            "same_subject": {"target_new_tfpr_or_exact": 0.0},
            "near_locality": {"target_new_tfpr_or_exact": 0.0},
            "far_locality": {"target_new_tfpr_or_exact": 0.0},
        },
    }
    assert all(hard_checks(report).values())
