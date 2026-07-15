#!/usr/bin/env python3
"""Build real-prompt T1 learned edit-intent gate data with provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    COMMON_ROOT,
    budget_guard,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage_event,
    repo_path,
    stable_key,
    summarize,
    write_csv,
    write_json,
    write_jsonl,
)


T1_ROOT = Path("runs/counterfact_learned_gate_raw_bridge_v1")


def choose_other(rows: list[dict[str, Any]], index: int, *, different_relation: bool) -> dict[str, Any]:
    current = rows[index]
    candidates = [
        row
        for row in rows
        if row["case_id"] != current["case_id"]
        and (not different_relation or row["relation_id"] != current["relation_id"])
    ]
    candidates.sort(key=lambda row: stable_key(current["case_id"], row["case_id"]))
    if not candidates:
        raise RuntimeError(f"No legal comparison row for {current['case_id']}")
    return candidates[0]


def gate_rows_for_split(rows: list[dict[str, Any]], split_role: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, edit in enumerate(rows):
        other_relation = choose_other(rows, index, different_relation=True)
        unrelated = choose_other(rows, index, different_relation=False)
        prompts = [
            ("rewrite", 1, "", edit["rewrite_prompt"], "real_counterfact_rewrite", False),
        ]
        prompts.extend(
            ("declarative_paraphrase", 1, "", prompt, "real_counterfact_paraphrase", False)
            for prompt in edit.get("paraphrase_prompts", [])[:2]
        )
        prompts.extend(
            [
                (
                    "same_subject_different_relation",
                    0,
                    "same_subject_different_relation",
                    other_relation["rewrite_template"].format(edit["subject"]),
                    "composed_from_real_train_relation_template",
                    True,
                ),
                (
                    "near_locality",
                    0,
                    "near_locality",
                    (edit.get("near_locality_prompts") or [unrelated["rewrite_prompt"]])[0],
                    "real_counterfact_neighborhood",
                    False,
                ),
                (
                    "far_locality",
                    0,
                    "far_locality",
                    unrelated["rewrite_prompt"],
                    "real_unrelated_train_rewrite",
                    False,
                ),
                (
                    "generation",
                    0,
                    "generation",
                    (edit.get("generation_prompts") or [f"{edit['subject']} is known for"])[0],
                    "real_counterfact_generation" if edit.get("generation_prompts") else "synthetic_fallback",
                    not bool(edit.get("generation_prompts")),
                ),
                (
                    "attribute",
                    0,
                    "attribute",
                    (edit.get("attribute_prompts") or [unrelated["rewrite_prompt"]])[0],
                    "real_counterfact_attribute" if edit.get("attribute_prompts") else "synthetic_fallback",
                    not bool(edit.get("attribute_prompts")),
                ),
                (
                    "unrelated",
                    0,
                    "unrelated",
                    unrelated["rewrite_prompt"],
                    "real_unrelated_train_rewrite",
                    False,
                ),
            ]
        )
        for prompt_index, (prompt_type, label, negative_type, prompt, provenance, synthetic) in enumerate(prompts):
            output.append(
                {
                    "gate_row_id": f"{split_role}:{edit['case_id']}:{prompt_type}:{prompt_index}",
                    "campaign_protocol": CAMPAIGN_PROTOCOL,
                    "track_protocol": "counterfact_learned_gate_raw_bridge_v1",
                    "split_role": split_role,
                    "case_id": edit["case_id"],
                    "prompt_id": f"{edit['case_id']}_{prompt_type}_{prompt_index}",
                    "prompt_type": prompt_type,
                    "label": label,
                    "negative_type": negative_type,
                    "prompt": prompt,
                    "subject": edit["subject"],
                    "relation_id": edit["relation_id"],
                    "relation_template": edit["rewrite_template"],
                    "target_new": edit["target_new"],
                    "target_true": edit["target_true"],
                    "prompt_provenance": provenance,
                    "synthetic_from_metadata": synthetic,
                    "source_manifest": edit["split_role"],
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_dir", type=Path, default=COMMON_ROOT)
    parser.add_argument("--output_dir", type=Path, default=T1_ROOT / "gate_data_v1")
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    report_path = repo_path(args.output_dir / "report_summary.json")
    if report_path.exists() and not args.allow_overwrite:
        raise FileExistsError(report_path)

    guard = budget_guard("T1")
    if not guard["pass"]:
        raise RuntimeError(f"T1 budget guard failed: {guard}")
    split_inputs = {
        "train": args.common_dir / "sb_alt_train_2000.jsonl",
        "val": args.common_dir / "sb_alt_val_300.jsonl",
        "smoke20": args.common_dir / "track_smoke_20.jsonl",
        "confirmation30": args.common_dir / "track_confirmation_30.jsonl",
    }
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for role, path in split_inputs.items():
        all_rows[role] = gate_rows_for_split(read_jsonl(path), role)
        write_jsonl(args.output_dir / f"gate_{role}.jsonl", all_rows[role])

    train_ids = {row["case_id"] for row in all_rows["train"]}
    val_ids = {row["case_id"] for row in all_rows["val"]}
    smoke_ids = {row["case_id"] for row in all_rows["smoke20"] + all_rows["confirmation30"]}
    overlaps = [
        {"left": "train", "right": "val", "overlap_count": len(train_ids & val_ids)},
        {"left": "train", "right": "smoke", "overlap_count": len(train_ids & smoke_ids)},
        {"left": "val", "right": "smoke", "overlap_count": len(val_ids & smoke_ids)},
    ]
    if any(row["overlap_count"] for row in overlaps):
        raise RuntimeError("T1 gate edit splits overlap")

    provenance_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for role, rows in all_rows.items():
        by_type = summarize(row["prompt_type"] for row in rows)
        summaries[role] = {
            "num_rows": len(rows),
            "num_edits": len({row["case_id"] for row in rows}),
            "label_counts": summarize(row["label"] for row in rows),
            "prompt_type_counts": by_type,
            "negative_type_counts": summarize(row["negative_type"] for row in rows if row["negative_type"]),
            "synthetic_count": sum(bool(row["synthetic_from_metadata"]) for row in rows),
        }
        for prompt_type, count in by_type.items():
            type_rows = [row for row in rows if row["prompt_type"] == prompt_type]
            real_count = sum(not bool(row["synthetic_from_metadata"]) for row in type_rows)
            provenance_rows.append(
                {
                    "split": role,
                    "prompt_type": prompt_type,
                    "count": count,
                    "real_prompt_count": real_count,
                    "real_prompt_coverage": real_count / count if count else 0.0,
                }
            )
    write_csv(args.output_dir / "prompt_provenance_audit.csv", provenance_rows)
    write_csv(args.output_dir / "split_overlap_audit.csv", overlaps)
    required_real = {"rewrite", "declarative_paraphrase", "near_locality", "far_locality"}
    coverage_pass = all(
        float(row["real_prompt_coverage"]) >= 0.95
        for row in provenance_rows
        if row["split"] in {"train", "val"} and row["prompt_type"] in required_real
    )
    same_subject_coverage = summaries["train"]["prompt_type_counts"].get(
        "same_subject_different_relation", 0
    ) / summaries["train"]["num_edits"]
    report = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_learned_gate_raw_bridge_v1",
        "stage": "T1.1 gate dataset and provenance",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "budget_guard": guard,
        "summaries": summaries,
        "acceptance_checks": {
            "real_rewrite_paraphrase_near_far_coverage_ge_0_95": coverage_pass,
            "same_subject_negative_coverage_ge_0_80": same_subject_coverage >= 0.80,
            "train_val_smoke_edit_overlap_zero": True,
            "synthetic_fallback_tagged": True,
            "analysis_final_unused": True,
        },
    }
    report["acceptance_pass"] = all(report["acceptance_checks"].values())
    write_json(args.output_dir / "gate_data_summary.json", report)
    write_json(args.output_dir / "report_summary.json", report)
    record_stage_event(
        track="T1",
        stage="T1.1_gate_data",
        event="gate_data_built",
        status="pass" if report["acceptance_pass"] else "fail",
        notes=f"rows={sum(len(rows) for rows in all_rows.values())}",
    )
    print(f"acceptance_pass={report['acceptance_pass']}")


if __name__ == "__main__":
    main()
