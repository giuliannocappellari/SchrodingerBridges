#!/usr/bin/env python3
"""Run the bounded B2 partial-state policy sweep and selected dev comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import CAMPAIGN_ROOT, read_json, write_json


POLICIES = {
    "fully_masked_only": ("fully_masked", "random", 0.0, 0.0),
    "all_mask_counts_random_positions": ("cycle", "random", 0.1, 0.25),
    "uniform_mask_count_states": ("uniform", "random", 0.1, 0.25),
    "confidence_trajectory_states": ("cycle", "base_confidence", 0.1, 0.25),
}


def run_one(*, manifest: Path, output: Path, policy: str) -> None:
    schedule, reveal, consistency, old_suppress = POLICIES[policy]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_dnpe_editor.py"),
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output),
        "--method",
        f"partial_state_mdm_memit__{policy}",
        "--layers",
        "3,4,5,6",
        "--target_optimization_steps",
        "25",
        "--learning_rate",
        "0.1",
        "--partial_mask_schedule",
        schedule,
        "--reveal_policy",
        reveal,
        "--state_consistency_weight",
        str(consistency),
        "--old_target_suppression_weight",
        str(old_suppress),
        "--include_locality",
        "1",
        "--decode_batch_size",
        "16",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def should_run(output: Path, *, resume: bool) -> bool:
    if not output.exists():
        return True
    if resume and (output / "report_summary.json").exists():
        return False
    raise FileExistsError(output)


def select_smoke(root: Path) -> dict[str, Any]:
    selection = {}
    table = []
    for length in (2, 3, 4):
        candidates = []
        for policy in POLICIES:
            path = root / f"smoke_n{length}_{policy}" / "report_summary.json"
            report = read_json(path)
            row = {
                "target_length": length,
                "policy": policy,
                "rewrite_exact": float(report["rewrite_exact"]),
                "paraphrase_exact": float(report["declarative_paraphrase_exact"]),
                "malformed_rate": float(report["malformed_rate"]),
                "path": str(path.parent.relative_to(ROOT)),
            }
            table.append(row)
            if policy != "fully_masked_only":
                candidates.append(row)
        best = max(
            candidates,
            key=lambda row: (
                row["rewrite_exact"] + row["paraphrase_exact"],
                row["rewrite_exact"],
                -row["malformed_rate"],
                row["policy"],
            ),
        )
        selection[str(length)] = best["policy"]
    payload = {
        "selection_source": "fresh_KAMEL_smoke_only",
        "selected_policy_by_length": selection,
        "rows": table,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(root / "smoke_policy_selection.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "dev"), required=True)
    parser.add_argument("--root", type=Path, default=CAMPAIGN_ROOT / "B2_partial_state_mdm_memit_v1")
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    protocol = CAMPAIGN_ROOT / "protocol_v1"
    if args.phase == "smoke":
        for length in (2, 3, 4):
            manifest = protocol / f"dnpe_kamel_smoke_20_n{length}.jsonl"
            for policy in POLICIES:
                output = args.root / f"smoke_n{length}_{policy}"
                if should_run(output, resume=bool(args.resume)):
                    run_one(manifest=manifest, output=output, policy=policy)
        payload = select_smoke(args.root)
        print(json.dumps(payload["selected_policy_by_length"], sort_keys=True))
        return
    selection = read_json(args.root / "smoke_policy_selection.json")["selected_policy_by_length"]
    for length in (2, 3, 4):
        manifest = protocol / f"dnpe_kamel_dev_100_n{length}.jsonl"
        for policy in ("fully_masked_only", str(selection[str(length)])):
            output = args.root / f"dev_n{length}_{policy}"
            if should_run(output, resume=bool(args.resume)):
                run_one(manifest=manifest, output=output, policy=policy)
    print(json.dumps({"dev_runs_complete": True, "selected_policy_by_length": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
