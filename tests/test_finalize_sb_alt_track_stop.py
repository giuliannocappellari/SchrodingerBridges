from pathlib import Path

import pytest

from scripts.finalize_sb_alt_track_stop import parse_evidence


def test_parse_evidence_keeps_role_and_path() -> None:
    assert parse_evidence(["offline=runs/example/report_summary.json"]) == [
        ("offline", "runs/example/report_summary.json")
    ]


def test_parse_evidence_rejects_unlabelled_path() -> None:
    with pytest.raises(ValueError):
        parse_evidence(["runs/example/report_summary.json"])
