#!/usr/bin/env python3
"""Materialize a historical locked manifest into the DNPE evaluation schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import CAMPAIGN_ID, CAMPAIGN_ROOT, read_json, read_jsonl, sha256_file, write_json, write_jsonl


def text_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("text") or value.get("str") or "")
    return str(value or "")


def ids_value(value: Any, fallback: Any) -> list[int]:
    if isinstance(value, Mapping):
        values = value.get("context_token_ids") or value.get("token_ids")
        if values:
            return list(map(int, values))
    return list(map(int, fallback or []))


def adapt_row(row: Mapping[str, Any], split_role: str) -> dict[str, Any]:
    target_new = text_value(row.get("target_new") or row.get("target"))
    target_true = text_value(row.get("target_true") or row.get("old_target"))
    near_cases = list(row.get("near_locality_cases") or [])
    far_cases = list(row.get("far_locality_cases") or [])
    attributes = list(row.get("attribute_prompts") or [])
    return {
        "campaign_id": CAMPAIGN_ID,
        "split_role": split_role,
        "case_id": str(row.get("case_id") or row.get("id")),
        "source_split": str(row.get("source_dataset_split") or "train"),
        "source_index": int(row["source_index"]),
        "relation_id": str(row["relation_id"]),
        "subject": str(row["subject"]),
        "rewrite_template": str(row.get("rewrite_template") or "{}"),
        "rewrite_prompt": str(row.get("prompt") or ""),
        "target_new": target_new,
        "target_true": target_true,
        "target_new_token_ids": ids_value(
            row.get("target_new"), row.get("target_new_token_ids")
        ),
        "target_true_token_ids": ids_value(
            row.get("target_true"), row.get("target_true_token_ids")
        ),
        "target_length": int(
            row.get("target_new_token_len")
            or len(ids_value(row.get("target_new"), row.get("target_new_token_ids")))
        ),
        "paraphrase_prompts": list(
            row.get("declarative_paraphrase_prompts")
            or row.get("paraphrase_prompts")
            or []
        ),
        "near_locality_prompts": [
            str(case["prompt"]) for case in near_cases
        ],
        "near_locality_cases": [
            {"prompt": str(case["prompt"]), "target": text_value(case.get("target"))}
            for case in near_cases
        ],
        "far_locality_cases": [
            {"prompt": str(case["prompt"]), "target": text_value(case.get("target"))}
            for case in far_cases
        ],
        "same_subject_prompts": attributes,
        "attribute_prompts": attributes,
        "generation_prompts": list(row.get("generation_prompts") or []),
        "identity_prompts": attributes[:2],
        "locked_source_protocol": str(row.get("protocol_version")),
    }


def materialize(source: Path, output: Path, *, split_name: str) -> dict[str, Any]:
    required_env = (
        "DEV_METHOD_LOCKED" if split_name == "analysis_500" else "FINAL_METHOD_LOCKED"
    )
    if os.environ.get(required_env) != "1":
        raise PermissionError(f"{split_name} requires {required_env}=1")
    registry = read_json(
        CAMPAIGN_ROOT / "protocol_v1" / "locked_manifest_registry.json"
    )["locked_manifests"][split_name]
    if sha256_file(source) != registry["sha256"]:
        raise RuntimeError(f"Locked manifest hash mismatch: {source}")
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(source)
    adapted = [adapt_row(row, f"dnpe_{split_name}") for row in rows]
    if len(adapted) != int(registry["count"]):
        raise RuntimeError("Locked manifest row count mismatch")
    write_jsonl(output, adapted)
    report = {
        "campaign_id": CAMPAIGN_ID,
        "split_name": split_name,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": registry["sha256"],
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256_file(output),
        "num_rows": len(adapted),
        "materialized_after_lock": True,
        "used_for_tuning": False,
    }
    write_json(output.with_suffix(".summary.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("analysis_500", "final_test_500"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (
        ROOT
        / "runs"
        / "counterfact_direction1_v1"
        / "protocol"
        / f"{args.split}.jsonl"
    )
    print(json.dumps(materialize(source, args.output, split_name=args.split), sort_keys=True))


if __name__ == "__main__":
    main()
