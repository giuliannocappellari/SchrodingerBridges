from __future__ import annotations

from pathlib import Path

from scripts.run_cl_confirmation_suite import (
    _csv_truthy,
    confirmation_acceptance,
    frozen_editor_command,
)


def test_confirmation_requires_same_class_and_frozen_safety() -> None:
    row = {
        "success_classes": "A",
        "same_subject_tfpr": 0.02,
        "near_tfpr": 0.01,
        "far_tfpr": 0.0,
        "base_retention_loss_fraction": 0.04,
        "malformed_rate": 0.0,
    }
    assert confirmation_acceptance(["A"], row) == (True, [])
    row["same_subject_tfpr"] = 0.04
    passed, reasons = confirmation_acceptance(["A"], row)
    assert not passed
    assert "same_subject_tfpr_above_0.03" in reasons


def test_confirmation_rejects_class_drift() -> None:
    row = {
        "success_classes": "B",
        "same_subject_tfpr": 0.0,
        "near_tfpr": 0.0,
        "far_tfpr": 0.0,
        "base_retention_loss_fraction": 0.0,
        "malformed_rate": 0.0,
    }
    passed, reasons = confirmation_acceptance(["A"], row)
    assert not passed
    assert "pilot_success_class_not_reproduced" in reasons


def test_frozen_confirmation_command_uses_pilot_hyperparameters(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "run_config.json").write_text(
        """{
          "method": "growth_shared",
          "model_id": "model",
          "model_revision": "revision",
          "layers": [4, 5],
          "covariance_representation": "diagonal",
          "lowrank_rank": 8,
          "protected_basis_rank": 32,
          "lora_rank": 8,
          "lora_steps": 25,
          "lora_learning_rate": 0.001,
          "replay_items_per_block": 10,
          "relation_overlap_threshold": 0.35,
          "memit": {
            "target_optimization_steps": 25,
            "learning_rate": 0.1,
            "covariance_weight": 15000.0
          }
        }""",
        encoding="utf-8",
    )
    command = frozen_editor_command(
        pilot_dir=pilot,
        manifest=tmp_path / "cf_cl_confirmation_200.jsonl",
        output_dir=tmp_path / "output",
    )
    joined = " ".join(command)
    assert "--allow_confirmation 1" in joined
    assert "--lowrank_rank 8" in joined
    assert "--protected_basis_rank 32" in joined
    assert "--layers 4,5" in joined
    assert "--relation_overlap_threshold 0.35" in joined
    assert "analysis_500" not in joined.casefold()
    assert "final_test" not in joined.casefold()


def test_csv_truth_parser_does_not_treat_false_text_as_true() -> None:
    assert _csv_truthy(True)
    assert _csv_truthy("True")
    assert not _csv_truthy("False")
