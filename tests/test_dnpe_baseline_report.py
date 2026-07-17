from __future__ import annotations

import json
from pathlib import Path

from scripts.report_dnpe_baselines import report_b1


def _run(path: Path, edits: int, rewrite: float, paraphrase: float) -> None:
    path.mkdir(parents=True)
    report = {
        "campaign_id": "diffusion_native_causal_partial_state_editor_v1",
        "num_edits": edits,
        "rewrite_exact": rewrite,
        "declarative_paraphrase_exact": paraphrase,
        "pre_edit_target_new_rewrite_exact": 0.0,
        "same_subject_tfpr": 0.01,
        "near_tfpr": 0.01,
        "far_tfpr": 0.0,
        "malformed_rate": 0.0,
        "gpu_minutes_per_edit": 0.1,
        "rollback_checksum_pass": True,
        "acceptance_pass": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    for name, value in (
        ("report_summary.json", report),
        ("validation_report.json", {"acceptance_pass": True}),
        ("run_config.json", {"campaign_id": report["campaign_id"]}),
    ):
        (path / name).write_text(json.dumps(value))


def test_b1_report_applies_frozen_thresholds(tmp_path: Path) -> None:
    _run(tmp_path / "smoke20_v1", 20, 0.8, 0.5)
    _run(tmp_path / "pilot100_v1", 100, 0.8, 0.5)
    _run(tmp_path / "dev200_v1", 200, 0.75, 0.4)
    report = report_b1(tmp_path, repair_used=False)
    assert report["acceptance_pass"] is True
    assert report["acceptance"]["rewrite_at_least_0_75"] is True
