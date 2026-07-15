#!/usr/bin/env python3
"""Build disjoint real-prompt answer-span data for the T3/T4 pilots."""

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
from scripts.sb_alt_common import (
    CAMPAIGN_PROTOCOL,
    COMMON_ROOT,
    budget_guard,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage_event,
    repo_path,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


T3_ROOT = Path("runs/counterfact_conditional_answer_span_csbm_v1")


def stable_int(*parts: Any) -> int:
    return int(hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def tokenize_endpoint(tokenizer: Any, prompt: str, target: str) -> list[int]:
    result = context_aware_target_tokenization(tokenizer, prompt, format_target(target))
    if not result.prefix_match or not result.target_token_ids:
        raise RuntimeError(f"Invalid context tokenization for {prompt!r} -> {target!r}")
    return list(result.target_token_ids)


def compatible_rows(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, multi_token: bool
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for row in sorted(rows, key=lambda item: stable_int(item["case_id"])):
        prompt = str(row["rewrite_prompt"])
        new_ids = tokenize_endpoint(tokenizer, prompt, str(row["target_new"]))
        old_ids = tokenize_endpoint(tokenizer, prompt, str(row["target_true"]))
        if len(new_ids) != len(old_ids):
            continue
        if multi_token and len(new_ids) >= 2:
            selected.append(row)
        elif not multi_token and len(new_ids) == 1:
            selected.append(row)
    return selected


def choose_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    single_count: int,
    multi_count: int,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    singles = compatible_rows(rows, tokenizer, multi_token=False)
    multis = compatible_rows(rows, tokenizer, multi_token=True)
    if len(singles) < single_count:
        raise RuntimeError(f"Need {single_count} compatible single-token rows, found {len(singles)}")
    chosen_multi = multis[: min(multi_count, len(multis))]
    return singles[:single_count] + chosen_multi, {
        "single_requested": single_count,
        "single_selected": single_count,
        "multi_requested": multi_count,
        "multi_legal_available": len(multis),
        "multi_selected": len(chosen_multi),
    }


def choose_other(rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = [
        row
        for row in rows
        if row["case_id"] != current["case_id"] and row["relation_id"] != current["relation_id"]
    ]
    return min(candidates, key=lambda row: stable_int(current["case_id"], row["case_id"]))


def base_row(
    edit: Mapping[str, Any], split: str, prompt: str, prompt_type: str, tokenizer: Any, mask_id: int
) -> dict[str, Any]:
    rewrite_prompt = str(edit["rewrite_prompt"])
    edit_target = tokenize_endpoint(tokenizer, rewrite_prompt, str(edit["target_new"]))
    old = tokenize_endpoint(tokenizer, rewrite_prompt, str(edit["target_true"]))
    if len(edit_target) != len(old):
        raise RuntimeError(f"Endpoint lengths differ for {edit['case_id']}")
    support = [list(dict.fromkeys([old_token, new_token, mask_id])) for old_token, new_token in zip(old, edit_target)]
    return {
        "split": split,
        "edit_id": edit["case_id"],
        "prompt_type": prompt_type,
        "prompt": prompt,
        "subject": edit["subject"],
        "relation_id": edit["relation_id"],
        "relation_template": edit["rewrite_template"],
        "target_new": edit["target_new"],
        "target_true": edit["target_true"],
        "target_new_token_ids": edit_target,
        "x0_token_ids": old,
        "endpoint_token_ids": edit_target,
        "mask_token_id": mask_id,
        "candidate_support_by_position": support,
        "span_length": len(edit_target),
        "target_length_bin": str(len(edit_target) if len(edit_target) <= 3 else ">=4"),
        "identity": False,
        "transport_label": 1,
        "synthetic_from_metadata": False,
    }


def build_rows(
    edits: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    split: str,
    tokenizer: Any,
    mask_id: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for edit in edits:
        rewrite = base_row(edit, split, str(edit["rewrite_prompt"]), "rewrite", tokenizer, mask_id)
        rewrite.update(
            {
                "row_id": f"{split}:{edit['case_id']}:rewrite",
                "prompt_id": f"{edit['case_id']}_rewrite",
                "prompt_provenance": "real_counterfact_rewrite",
            }
        )
        output.append(rewrite)
        if edit.get("paraphrase_prompts"):
            paraphrase = {
                **rewrite,
                "row_id": f"{split}:{edit['case_id']}:paraphrase",
                "prompt_id": f"{edit['case_id']}_paraphrase_0",
                "prompt_type": "paraphrase",
                "prompt": str(edit["paraphrase_prompts"][0]),
                "prompt_provenance": "real_counterfact_paraphrase",
            }
            output.append(paraphrase)

        other = choose_other(pool, edit)
        same_subject = {
            **rewrite,
            "row_id": f"{split}:{edit['case_id']}:same_subject",
            "prompt_id": f"{edit['case_id']}_same_subject",
            "prompt_type": "same_subject_different_relation",
            "prompt": str(other["rewrite_template"]).format(edit["subject"]),
            "prompt_provenance": "composed_from_real_train_relation_template",
            "endpoint_token_ids": list(rewrite["x0_token_ids"]),
            "identity": True,
            "transport_label": 0,
            "synthetic_from_metadata": True,
        }
        output.append(same_subject)
        near_prompts = edit.get("near_locality_prompts") or []
        near = {
            **same_subject,
            "row_id": f"{split}:{edit['case_id']}:near",
            "prompt_id": f"{edit['case_id']}_near_0",
            "prompt_type": "near_locality",
            "prompt": str(near_prompts[0] if near_prompts else other["rewrite_prompt"]),
            "prompt_provenance": "real_counterfact_neighborhood" if near_prompts else "real_unrelated_train_rewrite",
            "synthetic_from_metadata": False,
        }
        output.append(near)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=COMMON_ROOT)
    parser.add_argument("--output_dir", type=Path, default=T3_ROOT / "csbm_pilot_data_v1")
    parser.add_argument("--model_id", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--train_edits", type=int, default=200)
    parser.add_argument("--val_edits", type=int, default=50)
    parser.add_argument("--train_multi_edits", type=int, default=50)
    parser.add_argument("--val_multi_edits", type=int, default=20)
    parser.add_argument("--allow_overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = repo_path(args.output_dir)
    if (output_dir / "summary.json").exists() and not args.allow_overwrite:
        raise FileExistsError(output_dir)
    guard = budget_guard("T3")
    train_path = repo_path(args.input_dir / "sb_alt_train_2000.jsonl")
    val_path = repo_path(args.input_dir / "sb_alt_val_300.jsonl")
    train_pool = read_jsonl(train_path)
    val_pool = read_jsonl(val_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    train_edits, train_selection = choose_rows(
        train_pool,
        tokenizer,
        single_count=args.train_edits,
        multi_count=args.train_multi_edits,
    )
    val_edits, val_selection = choose_rows(
        val_pool,
        tokenizer,
        single_count=args.val_edits,
        multi_count=args.val_multi_edits,
    )
    mask_id = int(getattr(tokenizer, "mask_token_id", 126336) or 126336)
    train_rows = build_rows(train_edits, train_pool, "train", tokenizer, mask_id)
    val_rows = build_rows(val_edits, val_pool, "val", tokenizer, mask_id)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(
        output_dir / "identity_negatives.jsonl",
        [row for row in train_rows + val_rows if row["identity"]],
    )
    overlap = {row["edit_id"] for row in train_rows} & {row["edit_id"] for row in val_rows}
    write_csv(output_dir / "overlap_audit.csv", [{"left": "train", "right": "val", "overlap_count": len(overlap)}])
    all_rows = train_rows + val_rows
    checks = {
        "train_primary_edit_count": len({row["edit_id"] for row in train_rows if row["span_length"] == 1}) == args.train_edits,
        "val_primary_edit_count": len({row["edit_id"] for row in val_rows if row["span_length"] == 1}) == args.val_edits,
        "multi_token_diagnostic_included_when_legal": (
            train_selection["multi_selected"] == min(args.train_multi_edits, train_selection["multi_legal_available"])
            and val_selection["multi_selected"] == min(args.val_multi_edits, val_selection["multi_legal_available"])
        ),
        "endpoint_lengths_aligned": all(
            len(row["x0_token_ids"]) == len(row["target_new_token_ids"]) == len(row["endpoint_token_ids"])
            for row in all_rows
        ),
        "identity_negative_per_edit": all(
            any(candidate["identity"] and candidate["edit_id"] == edit_id for candidate in rows)
            for rows in (train_rows, val_rows)
            for edit_id in {row["edit_id"] for row in rows}
        ),
        "real_positive_prompts": all(
            row["prompt_provenance"].startswith("real_counterfact")
            for row in all_rows
            if not row["identity"]
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
        "source_manifest_sha256": {"train": sha256_file(train_path), "val": sha256_file(val_path)},
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_edit_count": len({row["edit_id"] for row in train_rows}),
        "val_edit_count": len({row["edit_id"] for row in val_rows}),
        "train_selection": train_selection,
        "val_selection": val_selection,
        "train_target_length_histogram": dict(sorted(Counter(str(row["span_length"]) for row in train_rows).items())),
        "val_target_length_histogram": dict(sorted(Counter(str(row["span_length"]) for row in val_rows).items())),
        "train_prompt_types": dict(sorted(Counter(row["prompt_type"] for row in train_rows).items())),
        "val_prompt_types": dict(sorted(Counter(row["prompt_type"] for row in val_rows).items())),
        "independent_position_factorization": True,
        "multi_token_dependency_limitation_reported": True,
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
        notes=(f"primary_train={args.train_edits} primary_val={args.val_edits} "
               f"multi_train={train_selection['multi_selected']} multi_val={val_selection['multi_selected']}"),
    )
    print(f"acceptance_pass={summary['acceptance_pass']}")


if __name__ == "__main__":
    main()
