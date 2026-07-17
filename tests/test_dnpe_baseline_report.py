from __future__ import annotations

import json
from pathlib import Path

from scripts.report_dnpe_baselines import report_b1, report_b2


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


def _b2_run(
    path: Path,
    *,
    length: int,
    policy: str,
    rewrite_hits: list[int],
    paraphrase_hits: list[int],
) -> None:
    path.mkdir(parents=True)
    report = {
        "campaign_id": "diffusion_native_causal_partial_state_editor_v1",
        "manifest": f"dnpe_kamel_dev_100_n{length}.jsonl",
        "method": f"partial_state_mdm_memit__{policy}",
        "rewrite_exact": sum(rewrite_hits) / len(rewrite_hits),
        "declarative_paraphrase_exact": sum(paraphrase_hits) / len(paraphrase_hits),
        "same_subject_tfpr": 0.0,
        "malformed_rate": 0.0,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    for name, value in (
        ("report_summary.json", report),
        ("validation_report.json", {"acceptance_pass": True}),
        ("run_config.json", {"campaign_id": report["campaign_id"]}),
    ):
        (path / name).write_text(json.dumps(value), encoding="utf-8")
    rows = []
    for bucket, hits in (
        ("rewrite", rewrite_hits),
        ("declarative_paraphrase", paraphrase_hits),
    ):
        for index, hit in enumerate(hits):
            rows.append(f"case-{index},{bucket},{bool(hit)}\n")
    (path / "edited_per_prompt.csv").write_text(
        "case_id,bucket,expected_hit\n" + "".join(rows), encoding="utf-8"
    )


def test_b2_requires_paraphrase_gain_or_positive_pooled_ci(tmp_path: Path) -> None:
    for length in (2, 3, 4):
        _b2_run(
            tmp_path / f"dev_n{length}_fully_masked_only",
            length=length,
            policy="fully_masked_only",
            rewrite_hits=[0] * 20,
            paraphrase_hits=[0] * 20,
        )
        _b2_run(
            tmp_path / f"dev_n{length}_partial",
            length=length,
            policy="partial",
            rewrite_hits=[1] * 20,
            paraphrase_hits=[0] * 20,
        )
    report = report_b2(tmp_path)
    assert report["positive_lengths_at_rewrite_gain_0_15"] == 3
    assert report["strong_lengths_with_paraphrase_gain_0_08"] == 0
    assert report["acceptance_pass"] is False
