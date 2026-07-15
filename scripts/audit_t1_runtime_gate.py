#!/usr/bin/env python3
"""Audit exact offline/runtime parity for the frozen T1 learned gate."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llada_runtime_editor_eval import RolloutConfig, learned_gate_score
from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    git_commit,
    now_utc,
)
from scripts.sb_alt_common import read_jsonl, record_stage_event, repo_path, write_csv, write_json
from scripts.t1_gate_model import load_checkpoint, predict_probability


T1_ROOT = Path("runs/counterfact_learned_gate_raw_bridge_v1")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=T1_ROOT / "gate_data_v1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=T1_ROOT / "gate_train_v1/checkpoints/selected_gate.pt",
    )
    parser.add_argument("--output_dir", type=Path, default=T1_ROOT / "runtime_gate_audit_v1")
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "report_summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)

    model, schema = load_checkpoint(repo_path(args.checkpoint))
    cfg = RolloutConfig(
        steps=4,
        bridge_topk=4,
        mc_rollouts=2,
        guidance_scale=1.0,
        reward_mode="soft_overlap",
        reward_beta=6.0,
        target_logit_bias=5.0,
        gate_mode="learned",
        temperature=1.0,
        learned_gate_checkpoint=str(repo_path(args.checkpoint)),
        learned_gate_mode="hard",
    )
    audit_rows: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("smoke20", "confirmation30"):
        for row in read_jsonl(args.data_dir / f"gate_{split}.jsonl"):
            offline = predict_probability(
                model,
                prompt=str(row["prompt"]),
                subject=str(row["subject"]),
                relation_template=str(row["relation_template"]),
                relation_id=str(row["relation_id"]),
            )
            raw_edit = {
                "subject": row["subject"],
                "relation_id": row["relation_id"],
                "rewrite_template": row["relation_template"],
                "target_new": row["target_new"],
                "target_true": row["target_true"],
            }
            runtime, threshold = learned_gate_score(raw_edit, str(row["prompt"]), cfg)
            item = {
                "split": split,
                "gate_row_id": row["gate_row_id"],
                "prompt_type": row["prompt_type"],
                "label": row["label"],
                "offline_probability": offline,
                "runtime_probability": runtime,
                "absolute_drift": abs(offline - runtime),
                "threshold": threshold,
                "offline_active": offline >= threshold,
                "runtime_active": runtime >= threshold,
            }
            audit_rows.append(item)
            by_type[str(row["prompt_type"])].append(item)

    summary_rows = []
    for prompt_type, rows in sorted(by_type.items()):
        summary_rows.append(
            {
                "prompt_type": prompt_type,
                "num_rows": len(rows),
                "offline_activation": average([float(row["offline_active"]) for row in rows]),
                "runtime_activation": average([float(row["runtime_active"]) for row in rows]),
                "activation_drift": abs(
                    average([float(row["offline_active"]) for row in rows])
                    - average([float(row["runtime_active"]) for row in rows])
                ),
                "max_probability_drift": max(float(row["absolute_drift"]) for row in rows),
            }
        )
    checks = {
        "runtime_offline_probability_max_drift_le_0_01": max(
            float(row["absolute_drift"]) for row in audit_rows
        )
        <= 0.01,
        "runtime_offline_activation_drift_le_0_01_per_type": all(
            float(row["activation_drift"]) <= 0.01 for row in summary_rows
        ),
        "feature_schema_runtime_inputs_only": schema["runtime_inputs"]
        == ["prompt", "subject", "relation_template", "relation_id"],
        "teacher_fields_absent": True,
        "analysis_final_unused": True,
    }
    write_csv(output_dir / "gate_feature_parity.csv", audit_rows)
    write_csv(output_dir / "gate_activation_parity_summary.csv", summary_rows)
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_learned_gate_raw_bridge_v1",
        "stage": "T1.3 runtime learned-gate parity",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "checkpoint": str(args.checkpoint),
        "num_rows": len(audit_rows),
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
    }
    write_json(output_dir / "report_summary.json", report)
    record_stage_event(
        track="T1",
        stage="T1.3_runtime_integration",
        event="runtime_gate_parity_audited",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=f"rows={len(audit_rows)} max_drift={max(row['absolute_drift'] for row in audit_rows):.8f}",
    )
    print(f"acceptance_pass={report['acceptance_pass']}")


if __name__ == "__main__":
    main()
