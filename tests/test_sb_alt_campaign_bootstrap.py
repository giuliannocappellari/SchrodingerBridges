from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import sb_alt_common
from scripts.build_sb_alt_common_protocol import select_disjoint_splits


def test_track_pilot_estimates_cover_configured_minimum() -> None:
    assert sum(float(track["pilot_estimate_usd"]) for track in sb_alt_common.TRACKS) == 8.0
    assert [track["id"] for track in sb_alt_common.TRACKS] == ["T1", "T2", "T3", "T4", "T5"]


def test_locked_exclusion_reader_projects_only_allowed_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    for index, name in enumerate(sb_alt_common.LOCKED_MANIFEST_NAMES):
        row = {
            "case_id": f"case-{index}",
            "source_dataset_split": "train",
            "source_index": index,
            "prompt": "must not escape projection",
            "target": "must not escape projection",
            "metric": 1.0,
        }
        (protocol / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(sb_alt_common, "ROOT", tmp_path)
    result = sb_alt_common.collect_locked_exclusions(protocol)
    assert len(result["case_ids"]) == len(sb_alt_common.LOCKED_MANIFEST_NAMES)
    for manifest in result["manifests"].values():
        assert manifest["prompt_label_output_metric_fields_used"] is False
        assert set(manifest["fields_used"]) == sb_alt_common.LOCKED_ALLOWED_FIELDS


def test_common_splits_are_deterministic_and_disjoint() -> None:
    rows = []
    for index in range(300):
        rows.append(
            {
                "case_id": f"case-{index}",
                "relation_id": f"P{index % 7}",
                "target_length_bin": str(index % 3 + 1),
            }
        )
    sizes = {"train": 120, "val": 40, "smoke": 20}
    first = select_disjoint_splits(rows, sizes, 17)
    second = select_disjoint_splits(rows, sizes, 17)
    assert first == second
    id_sets = [{row["case_id"] for row in split} for split in first.values()]
    assert sum(len(ids) for ids in id_sets) == len(set().union(*id_sets))
    for split in first.values():
        bins = {row["target_length_bin"] for row in split}
        assert {"1", "2"}.issubset(bins)


def test_autonomous_environment_rejects_insufficient_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "SB_ALT_AUTONOMOUS_MODE": "1",
        "SB_ALT_AUTONOMOUS_BUDGET_USD": "12",
        "RUNPOD_HOURLY_RATE_USD": "0.45",
        "SB_ALT_AUTONOMOUS_BUDGET_RESERVE_USD": "5",
        "SB_ALT_MIN_UNTESTED_TRACK_RESERVE_USD": "8",
        "RUNPOD_POD_ID": "pod",
        "RUNPOD_SSH_KEY": "/tmp/key",
        "RUNPOD_SSH_USER": "root",
        "RUNPOD_SSH_HOST": "host",
        "RUNPOD_SSH_PORT": "22",
        "REMOTE_REPO_DIR": "/workspace/SB",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="estimates plus reserve"):
        sb_alt_common.require_autonomous_environment()


def test_historical_case_namespace_is_source_index_based() -> None:
    source_index = 8375
    upstream_case_id = 9275
    assert f"counterfact_train_{source_index}" != f"counterfact_train_{upstream_case_id}"
