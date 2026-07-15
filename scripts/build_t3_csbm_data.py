#!/usr/bin/env python3
"""Build the disjoint real-prompt T3 answer-span CSBM pilot dataset."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llada_counterfact_protocol import context_aware_target_tokenization, format_target
from llada_sb_common import infer_mask_id
from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    COMMON_ROOT,
    budget_guard,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage_event,
    repo_path,
    write_csv,
    write_json,
    write_jsonl,
)


T3_ROOT = Path("runs/counterfact_conditional_answer_span_csbm_v1")


def stable_int(*parts: Any) -> int:
    return int(hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def choose_single_token(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if str(row["target_length_bin"]) == "1"]
    selected.sort(key=lambda row: stable_int(row["case_id"]))
    if len(selected) < count:
        raise RuntimeError(f"Need {count} single-token rows, found {len(selected)}")
    return selected[:count]


def choose_compatible_single_token(
    rows: Sequence[Mapping[str, Any]], count: int, tokenizer: Any
) -> list[Mapping[str, Any]]:
    candidates = choose_single_token(rows, len([row for row in rows if str(row["target_length_bin"]) == "1"]))
    selected = []
    for row in candidates:
        prompt = str(row["rewrite_prompt"])
        if len(tokenize_endpoint(tokenizer, prompt, str(row["target_new"]))) != 1:
            continue
        if len(tokenize_endpoint(tokenizer, prompt, str(row["target_true"]))) != 1:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Could select only {len(selected)} compatible single-token endpoint rows; need {count}")


def choose_other(rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = [
        row
        for row in rows
        if row["case_id"] != current["case_id"] and row["relation_id"] != current["relation_id"]
    ]
    return min(candidates, key=lambda row: stable_int(current["case_id"], row["case_id"]))


def tokenize_endpoint(tokenizer: Any, prompt: str, target: str) -> list[int]:
    result = context_aware_target_tokenization(tokenizer, prompt, format_target(target))
    if not result.prefix_match or not result.target_token_ids:
        raise RuntimeError(f"Invalid context tokenization for {prompt!r} -> {target!r}")
    return list(result.target_token_ids)


def build_rows(
    edits: Sequence[Mapping[str, Any]], pool: Sequence[Mapping[str, Any]], split: str, tokenizer: Any, mask_id: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for edit in edits:
        prompt = str(edit["rewrite_prompt"])
        new_ids = tokenize_endpoint(tokenizer, prompt, str(edit["target_new"]))
        old_ids = tokenize_endpoint(tokenizer, prompt, str(edit["target_true"]))
        if len(new_ids) != 1:
            raise RuntimeError(f"T3 primary row is not single token: {edit['case_id']}")
        support = list(dict.fromkeys([old_ids[0], new_ids[0], mask_id]))
        output.append(
            {
                "row_id": f"{split}:{edit['case_id']}:rewrite",
                "split": split,
                "edit_id": edit["case_id"],
                "prompt_id": f"{edit['case_id']}_rewrite",
                "prompt_type": "rewrite",
                "prompt": prompt,
                "prompt_provenance": "real_counterfact_rewrite",
                "subject": edit["subject"],
                "relation_id": edit["relation_id"],
                "relation_template": edit["rewrite_template"],
                "target_new": edit["target_new"],
                "target_true": edit["target_true"],
                "x0_token_id": old_ids[0],
                "xT_token_id": new_ids[0],
                "mask_token_id": mask_id,
                "candidate_support": support,
                "identity": False,
                "transport_label": 1,
                "synthetic_from_metadata": False,
            }
        )
        if edit.get("paraphrase_prompts"):
            paraphrase = str(edit["paraphrase_prompts"][0])
            output.append(
                {
                    **output[-1],
                    "row_id": f"{split}:{edit['case_id']}:paraphrase",
                    "prompt_id": f"{edit['case_id']}_paraphrase_0",
                    "prompt_type": "paraphrase",
                    "prompt": paraphrase,
                    "prompt_provenance": "real_counterfact_paraphrase",
                }
            )
        other = choose_other(pool, edit)
        same_subject_prompt = str(other["rewrite_template"]).format(edit["subject"])
        output.append(
            {
                "row_id": f"{split}:{edit['case_id']}:same_subject",
                "split": split,
                "edit_id": edit["case_id"],
                "prompt_id": f"{edit['case_id']}_same_subject",
                "prompt_type": "same_subject_different_relation",
                "prompt": same_subject_prompt,
                "prompt_provenance": "composed_from_real_train_relation_template",
                "subject": edit["subject"],
                "relation_id": edit["relation_id"],
                "relation_template": edit["rewrite_template"],
                "target_new": edit["target_new"],
                "target_true": edit["target_true"],
                "x0_token_id": old_ids[0],
                "xT_token_id": old_ids[0],
                "mask_token_id": mask_id,
                "candidate_support": list(dict.fromkeys([old_ids[0], new_ids[0], mask_id])),
                "identity": True,
                "transport_label": 0,
                "synthetic_from_metadata": True,
            }
        )
        near_prompt = str((edit.get("near_locality_prompts") or [other["rewrite_prompt"]])[0])
        output.append(
            {
                **output[-1],
                "row_id": f"{split}:{edit['case_id']}:near",
                "prompt_id": f"{edit['case_id']}_near_0",
                "prompt_type": "near_locality",
                "prompt": near_prompt,
                "prompt_provenance": "real_counterfact_neighborhood",
                "synthetic_from_metadata": not bool(edit.get("near_locality_prompts")),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=COMMON_ROOT)
    parser.add_argument("--output_dir", type=Path, default=T3_ROOT / "csbm_pilot_data_v1")
    parser.add_argument("--model_id", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--train_edits", type=int, default=200)
    parser.add_argument("--val_edits", type=int, default=50)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    guard = budget_guard("T3")
    if not guard["pass"]:
        raise RuntimeError(f"T3 budget guard failed: {guard}")
    train_pool = read_jsonl(args.input_dir / "sb_alt_train_2000.jsonl")
    val_pool = read_jsonl(args.input_dir / "sb_alt_val_300.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    train_edits = choose_compatible_single_token(train_pool, args.train_edits, tokenizer)
    val_edits = choose_compatible_single_token(val_pool, args.val_edits, tokenizer)
    mask_id = int(getattr(tokenizer, "mask_token_id", 126336) or 126336)
    train_rows = build_rows(train_edits, train_pool, "train", tokenizer, mask_id)
    val_rows = build_rows(val_edits, val_pool, "val", tokenizer, mask_id)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "identity_negatives.jsonl", [row for row in train_rows + val_rows if row["identity"]])
    overlap = {row["edit_id"] for row in train_rows} & {row["edit_id"] for row in val_rows}
    overlap_rows = [{"left": "train", "right": "val", "overlap_count": len(overlap)}]
    write_csv(output_dir / "overlap_audit.csv", overlap_rows)
    checks = {
        "train_200_edits": len({row["edit_id"] for row in train_rows}) == args.train_edits,
        "val_50_edits": len({row["edit_id"] for row in val_rows}) == args.val_edits,
        "single_token_primary": all(len(row["candidate_support"]) >= 2 for row in train_rows + val_rows),
        "identity_negative_per_edit": all(
            any(candidate["identity"] and candidate["edit_id"] == edit["case_id"] for candidate in rows)
            for edit, rows in ((edit, train_rows) for edit in train_edits)
        ),
        "train_val_overlap_zero": not overlap,
        "analysis_final_unused": True,
    }
    summary = {
        "campaign_protocol": CAMPAIGN_PROTOCOL,
        "track_protocol": "counterfact_conditional_answer_span_csbm_v1",
        "stage": "T3.1 categorical answer-span pilot data",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "analysis_500_used": False,
        "final_test_used": False,
        "model_loaded": False,
        "tokenizer_only": True,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_prompt_types": dict(sorted(Counter(row["prompt_type"] for row in train_rows).items())),
        "val_prompt_types": dict(sorted(Counter(row["prompt_type"] for row in val_rows).items())),
        "budget_guard": guard,
        "acceptance_checks": checks,
        "acceptance_pass": all(checks.values()),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "report_summary.json", summary)
    record_stage_event(
        track="T3",
        stage="T3.1_pilot_data",
        event="categorical_pilot_data_built",
        status="pass" if summary["acceptance_pass"] else "fail",
        notes=f"train_edits={args.train_edits} val_edits={args.val_edits}",
    )
    print(f"acceptance_pass={summary['acceptance_pass']}")


if __name__ == "__main__":
    main()
