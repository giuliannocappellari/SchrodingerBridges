from __future__ import annotations

import json
from pathlib import Path

from scripts.report_dnpe_timerome_baseline import validate


def test_timerome_report_requires_finite_complete_runs(tmp_path: Path) -> None:
    for name, edits in (("smoke20_v1", 20), ("pilot100_v1", 100)):
        path = tmp_path / name
        path.mkdir()
        report = {
            "num_edits": edits,
            "rewrite_exact": 0.5,
            "declarative_paraphrase_exact": 0.4,
            "same_subject_tfpr": 0.01,
            "near_tfpr": 0.01,
            "far_tfpr": 0.0,
            "malformed_rate": 0.0,
            "residual_memory_rank": edits,
            "residual_memory_finite": True,
            "storage_bytes": 10,
            "gpu_minutes_per_edit": 0.1,
            "analysis_500_used": False,
            "final_test_used": False,
        }
        (path / "report_summary.json").write_text(json.dumps(report), encoding="utf-8")
        (path / "validation_report.json").write_text(
            json.dumps({"metrics_complete": True}), encoding="utf-8"
        )
    assert validate(tmp_path)["acceptance_pass"] is True
