#!/usr/bin/env python3
"""Build disjoint standard and same-subject runtime manifests for T1 pilots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llada_sb_common import looks_like_qa_prompt
from scripts.sb_alt_common import COMMON_ROOT, git_commit, now_utc, read_jsonl, write_json, write_jsonl


T1_ROOT = Path("runs/counterfact_learned_gate_raw_bridge_v1")


def standard_row(row: dict[str, Any], all_rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    unrelated = all_rows[(index + 1) % len(all_rows)]
    near_prompts = list(row.get("near_locality_prompts") or [])[:1]
    far_prompts = [unrelated["rewrite_prompt"]]
    paraphrases = list(row.get("paraphrase_prompts") or [])[:2]
    return {
        **row,
        "protocol_version": "counterfact_learned_gate_raw_bridge_v1",
        "prompt": row["rewrite_prompt"],
        "target": row["target_new"],
        "aliases": [row["target_new"]],
        "paraphrase_prompts": [],
        "declarative_paraphrase_prompts": [
            prompt for prompt in paraphrases if not looks_like_qa_prompt(prompt)
        ],
        "qa_paraphrase_prompts": [
            prompt for prompt in paraphrases if looks_like_qa_prompt(prompt)
        ],
        "near_locality_cases": [
            {
                "id": f"{row['case_id']}_near_{prompt_index}",
                "prompt": prompt,
                "target": row["target_new"],
                "aliases": [row["target_new"]],
            }
            for prompt_index, prompt in enumerate(near_prompts)
        ],
        "far_locality_cases": [
            {
                "id": f"{row['case_id']}_far_{prompt_index}",
                "prompt": prompt,
                "target": row["target_new"],
                "aliases": [row["target_new"]],
            }
            for prompt_index, prompt in enumerate(far_prompts)
        ],
    }


def stress_row(edit: dict[str, Any], gate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stress = next(
        row
        for row in gate_rows
        if row["case_id"] == edit["case_id"]
        and row["prompt_type"] == "same_subject_different_relation"
    )
    stress_id = f"{edit['case_id']}__same_subject_stress"
    return {
        **edit,
        "id": stress_id,
        "case_id": stress_id,
        "original_edit_id": edit["case_id"],
        "protocol_version": "counterfact_learned_gate_raw_bridge_v1",
        "split_role": f"{edit['split_role']}_same_subject_stress",
        "prompt": stress["prompt"],
        "target": edit["target_new"],
        "aliases": [edit["target_new"]],
        "paraphrase_prompts": [],
        "declarative_paraphrase_prompts": [],
        "qa_paraphrase_prompts": [],
        "near_locality_cases": [],
        "far_locality_cases": [],
        "stress_eval": True,
        "stress_name": "same_subject_different_relation_target_new_over_injection",
        "target_semantics": "target_new_over_injection",
        "synthetic_from_metadata": bool(stress["synthetic_from_metadata"]),
        "stress_prompt_provenance": stress["prompt_provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_dir", type=Path, default=COMMON_ROOT)
    parser.add_argument("--gate_data_dir", type=Path, default=T1_ROOT / "gate_data_v2")
    parser.add_argument("--output_dir", type=Path, default=T1_ROOT / "runtime_manifests_v2")
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    report_path = args.output_dir / "report_summary.json"
    if report_path.exists() and not args.allow_overwrite:
        raise FileExistsError(report_path)
    summaries = {}
    for split, common_name, gate_name in (
        ("smoke20", "track_smoke_20.jsonl", "gate_smoke20.jsonl"),
        ("confirmation30", "track_confirmation_30.jsonl", "gate_confirmation30.jsonl"),
    ):
        edits = read_jsonl(args.common_dir / common_name)
        gate_rows = read_jsonl(args.gate_data_dir / gate_name)
        standard = [standard_row(row, edits, index) for index, row in enumerate(edits)]
        stress = [stress_row(row, gate_rows) for row in edits]
        write_jsonl(args.output_dir / f"{split}_standard.jsonl", standard)
        write_jsonl(args.output_dir / f"{split}_same_subject_stress.jsonl", stress)
        write_jsonl(args.output_dir / f"{split}_combined.jsonl", standard + stress)
        summaries[split] = {
            "num_edits": len(edits),
            "standard_rows": len(standard),
            "stress_rows": len(stress),
            "combined_rows": len(standard) + len(stress),
            "standard_case_ids": len({row["case_id"] for row in standard}),
            "stress_case_ids": len({row["case_id"] for row in stress}),
        }
    report = {
        "track_protocol": "counterfact_learned_gate_raw_bridge_v1",
        "stage": "T1 runtime pilot manifests",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "summaries": summaries,
        "same_subject_target_semantics": "target_new_over_injection",
        "supersedes_invalid_manifest": "runtime_manifests_v1 (legacy paraphrase fields leaked into stress rows)",
        "prompt_materialization_repair_used": True,
        "acceptance_pass": summaries["smoke20"]["num_edits"] == 20
        and summaries["confirmation30"]["num_edits"] == 30,
    }
    write_json(report_path, report)
    print(f"acceptance_pass={report['acceptance_pass']}")


if __name__ == "__main__":
    main()
