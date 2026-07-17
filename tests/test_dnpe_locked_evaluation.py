from __future__ import annotations

import csv
from pathlib import Path

from scripts.run_dnpe_locked_evaluation import leakage_bootstrap


def _write(path: Path, values: list[int]) -> None:
    path.mkdir()
    with (path / "edited_per_prompt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "bucket", "target_new_hit"))
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow({"case_id": f"c{index}", "bucket": "same_subject", "target_new_hit": bool(value)})


def test_leakage_bootstrap_is_paired_by_case(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(left, [0, 0, 0, 0])
    _write(right, [1, 1, 1, 1])
    report = leakage_bootstrap(left, right, "same_subject")
    assert report["delta"] == -1.0
    assert report["ci_high"] == -1.0
