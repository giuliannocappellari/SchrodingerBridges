#!/usr/bin/env python3
"""Freeze the bounded DNPE causal-site policies before pilot editing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import CAMPAIGN_ID, CAMPAIGN_ROOT, git_commit, now_utc, read_json, record_stage, write_json


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_dir", type=Path, default=CAMPAIGN_ROOT / "standard_causal_tracing_v1")
    parser.add_argument("--temporal_dir", type=Path, default=CAMPAIGN_ROOT / "temporal_causal_tracing_v1")
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "site_policy_lock_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    standard_report = read_json(args.standard_dir / "report_summary.json")
    temporal_report = read_json(args.temporal_dir / "report_summary.json")
    standard = read_csv(args.standard_dir / "aie_by_layer_position.csv")
    temporal = read_csv(args.temporal_dir / "tie_aggregate.csv")
    if standard_report.get("new_target_used_for_localization") or temporal_report.get("new_target_used_for_localization"):
        raise RuntimeError("New-target leakage in causal localization")
    standard_sorted = sorted(standard, key=lambda row: float(row["mean_normalized_aie"]), reverse=True)
    temporal_sorted = sorted(temporal, key=lambda row: float(row["mean_normalized_aie"]), reverse=True)
    best_standard = standard_sorted[0]
    best_temporal = temporal_sorted[0]
    editable_standard = [
        row
        for row in standard_sorted
        if row["component"] == "mlp" and row["position"] == "last_subject"
    ]
    editable_temporal = [
        row
        for row in temporal_sorted
        if row["component"] == "mlp" and row["position"] == "last_subject"
    ]
    if not editable_standard or not editable_temporal:
        raise RuntimeError("Causal tracing did not produce an editable MLP/last-subject coordinate")
    best_editable_standard = editable_standard[0]
    best_editable_temporal = editable_temporal[0]
    top_temporal = editable_temporal[: max(3, min(12, len(editable_temporal)))]
    layer_counts = Counter(int(row["layer"]) for row in top_temporal)
    stable_layers = sorted(layer for layer, _count in layer_counts.most_common(4))
    fixed_center = int(best_editable_standard["layer"])
    fixed_window = sorted({max(0, min(31, fixed_center + offset)) for offset in (-1, 0, 1, 2)})
    policies = [
        {
            "policy_id": "fixed_global_site",
            "layers": fixed_window,
            "position": "last_subject",
            "component": "mlp",
            "selection_source": "standard_causal_tracing_dev_only",
        },
        {
            "policy_id": "per_edit_top_tie_site",
            "layers": [int(best_editable_temporal["layer"])],
            "position": "last_subject",
            "component": "mlp",
            "selection_source": "temporal_causal_tracing_dev_only",
            "runtime_rule": "per-edit argmax within frozen temporal coordinates",
        },
        {
            "policy_id": "stable_temporal_site_set",
            "layers": stable_layers,
            "position": "last_subject",
            "component": "mlp",
            "selection_source": "top temporal coordinate coverage",
        },
    ]
    controls = [
        {"policy_id": "random_site", "rule": "seeded random layer from 0..31"},
        {"policy_id": "late_answer_position", "layers": [28, 29, 30, 31], "position": "first_answer_mask"},
        {"policy_id": "source_paper_fixed_site", "layers": [3, 4, 5, 6], "position": "last_subject", "component": "mlp"},
    ]
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "policies": policies,
        "best_overall_standard_site": best_standard,
        "best_overall_temporal_site": best_temporal,
        "edit_compatibility_filter": "MLP contribution at last subject token",
        "controls": controls,
        "policy_count": len(policies),
        "no_more_site_policies_after_pilot_inspection": True,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": len(policies) <= 3,
    }
    write_json(args.output_dir / "site_policy_lock.json", payload)
    write_json(args.output_dir / "report_summary.json", payload)
    write_json(args.output_dir / "validation_report.json", {"at_most_three_policies": len(policies) <= 3, "new_target_used": False, "acceptance_pass": len(policies) <= 3})
    record_stage(
        "C3_site_policy_lock",
        status="passed",
        acceptance_pass=True,
        output_dir=args.output_dir,
        started_at_utc=now_utc(),
        notes="Three causal site policies frozen before editor pilot.",
        next_stage="D1_state_banks",
    )
    print(json.dumps({"acceptance_pass": True, "policies": policies}, sort_keys=True))


if __name__ == "__main__":
    main()
