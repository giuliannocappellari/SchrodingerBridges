from __future__ import annotations

import json
from pathlib import Path

from scripts.update_dnpe_cost_state import collect_runtime_rows


def test_cost_state_collects_only_completed_runtime_reports(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "report_summary.json").write_text(
        json.dumps({"runtime_seconds": 12.5, "stage": "test"}), encoding="utf-8"
    )
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "report_summary.json").write_text("{}", encoding="utf-8")
    rows = collect_runtime_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["runtime_seconds"] == 12.5
