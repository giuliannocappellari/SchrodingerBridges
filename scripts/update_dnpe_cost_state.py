#!/usr/bin/env python3
"""Refresh informational DNPE GPU/runtime accounting from completed artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import CAMPAIGN_ID, CAMPAIGN_ROOT, STATE_ROOT, now_utc, write_json


def collect_runtime_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/report_summary.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        runtime = report.get("runtime_seconds")
        if runtime is None:
            continue
        value = float(runtime)
        if value < 0:
            raise ValueError(f"Negative runtime in {path}")
        display_path = (
            path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.relative_to(root)
        )
        rows.append(
            {
                "artifact": str(display_path),
                "stage": str(report.get("stage") or path.parent.parent.name),
                "runtime_seconds": value,
                "gpu_minutes_per_edit": report.get("gpu_minutes_per_edit"),
                "num_edits": report.get("num_edits"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CAMPAIGN_ROOT)
    args = parser.parse_args()
    rows = collect_runtime_rows(args.root)
    total_seconds = sum(row["runtime_seconds"] for row in rows)
    rate_text = os.environ.get("RUNPOD_HOURLY_RATE_USD")
    rate = float(rate_text) if rate_text else None
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "monetary_budget_guard_enabled": False,
        "cost_logging_informational_only": True,
        "pod_hourly_rate_usd": rate,
        "estimated_gpu_runtime_hours": total_seconds / 3600.0,
        "estimated_cost_usd": total_seconds / 3600.0 * rate if rate is not None else None,
        "stage_costs": rows,
        "completed_runtime_artifact_count": len(rows),
        "updated_at_utc": now_utc(),
    }
    write_json(STATE_ROOT / "cost_state.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
