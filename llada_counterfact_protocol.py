#!/usr/bin/env python3
"""Build the frozen CounterFact protocol manifests for Direction 1.

The JSONL output is intentionally backward-compatible with the existing
``EditExample`` loader while adding the metadata needed by
``counterfact_direction1_v1``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import string
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROTOCOL_VERSION = "counterfact_direction1_v1"
DEFAULT_DATASET = "azhx/counterfact"
DEFAULT_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"

OFFICIAL_TRAIN_SPLIT_SPECS = (
    ("dev_tune_200", 200),
    ("analysis_500", 500),
    ("ablation_500", 500),
)
OFFICIAL_TEST_SPLIT_SPECS = (
    ("final_test_500", 500),
)

SMOKE_TRAIN_SPLIT_SPECS = (
    ("dev_tune_200", 10),
    ("analysis_500", 10),
    ("ablation_500", 10),
)
SMOKE_TEST_SPLIT_SPECS = (
    ("final_test_500", 10),
)


@dataclass(frozen=True)
class ContextTokenization:
    target_token_ids: List[int]
    standalone_token_ids: List[int]
    full_token_ids: List[int]
    prompt_token_ids: List[int]
    prefix_match: bool
    common_prefix_len: int
    mismatch: bool


def normalize_counterfact_text(text: str) -> str:
    """Normalize target strings for filtering/reporting, not for generation."""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized.strip(string.punctuation + " \t\n\r")


def make_aliases(text: str) -> List[str]:
    stripped = str(text).strip()
    aliases = [f" {stripped}", stripped]
    return list(dict.fromkeys(alias for alias in aliases if alias))


def format_target(text: str) -> str:
    stripped = str(text).strip()
    return f" {stripped}" if stripped else ""


def render_counterfact_prompt(prompt_template: str, subject: str) -> str:
    template = str(prompt_template)
    if "{}" in template:
        return template.format(subject)
    if "{subject}" in template:
        return template.replace("{subject}", subject)
    return template.replace("{", "").replace("}", "").strip() + f" {subject}"


def looks_like_qa_prompt(prompt: str) -> bool:
    lowered = str(prompt).strip().lower()
    return "?" in lowered or "answer:" in lowered or lowered.endswith(":")


def tokenizer_encode(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(map(int, encoded["input_ids"]))


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def context_aware_target_tokenization(
    tokenizer: Any,
    rendered_prompt: str,
    target_string: str,
) -> ContextTokenization:
    """Tokenize target as the suffix of ``rendered_prompt + target_string``.

    Exact prefix subtraction is preferred. If the tokenizer changes the prompt
    boundary token, we fall back to the suffix after the longest common prefix
    and mark the row as a mismatch for reporting.
    """
    prompt_ids = tokenizer_encode(tokenizer, rendered_prompt)
    full_ids = tokenizer_encode(tokenizer, rendered_prompt + target_string)
    standalone_ids = tokenizer_encode(tokenizer, target_string)
    prefix_match = len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids
    common_prefix_len = _common_prefix_len(prompt_ids, full_ids)
    if prefix_match:
        target_ids = full_ids[len(prompt_ids) :]
    else:
        target_ids = full_ids[common_prefix_len:]
    return ContextTokenization(
        target_token_ids=list(target_ids),
        standalone_token_ids=list(standalone_ids),
        full_token_ids=list(full_ids),
        prompt_token_ids=list(prompt_ids),
        prefix_match=bool(prefix_match),
        common_prefix_len=int(common_prefix_len),
        mismatch=list(target_ids) != list(standalone_ids) or not prefix_match,
    )


def target_length_bin(token_count: int) -> str:
    if token_count <= 1:
        return "1"
    if token_count == 2:
        return "2"
    if token_count == 3:
        return "3"
    return ">=4"


def coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Expected mapping, got {type(value).__name__}")


def coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                return [value]
        if isinstance(parsed, list):
            return parsed
    return [value]


def build_pool_entry(row: Dict[str, Any], source_split: str, source_index: int) -> Dict[str, Any]:
    rewrite = coerce_mapping(row["requested_rewrite"])
    prompt = render_counterfact_prompt(rewrite["prompt"], rewrite["subject"])
    target_true = rewrite["target_true"]["str"]
    return {
        "id": f"counterfact_{source_split}_pool_{source_index}",
        "prompt": prompt,
        "target": format_target(target_true),
        "aliases": make_aliases(target_true),
        "source_index": int(source_index),
    }


def split_paraphrases(paraphrase_prompts: Sequence[str]) -> Tuple[List[str], List[str]]:
    declarative: List[str] = []
    qa: List[str] = []
    for prompt in paraphrase_prompts:
        if looks_like_qa_prompt(prompt):
            qa.append(str(prompt))
        else:
            declarative.append(str(prompt))
    return declarative, qa


def choose_disjoint_cases(
    rng: random.Random,
    candidate_pool: Sequence[Dict[str, Any]],
    counts: Sequence[Tuple[str, int]],
) -> Dict[str, List[Dict[str, Any]]]:
    shuffled = list(candidate_pool)
    rng.shuffle(shuffled)
    selected: Dict[str, List[Dict[str, Any]]] = {}
    offset = 0
    for name, count in counts:
        selected[name] = shuffled[offset : offset + max(0, int(count))]
        offset += max(0, int(count))
    return selected


def convert_counterfact_row(
    *,
    row: Dict[str, Any],
    source_split: str,
    source_index: int,
    tokenizer: Any,
    anchor_pool: Sequence[Dict[str, Any]],
    rng: random.Random,
    split_role: str,
    anchor_cases_per_edit: int,
    far_locality_cases_per_edit: int,
    eval_anchor_cases_per_edit: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    validity: Dict[str, Any] = {
        "source_split": source_split,
        "source_index": int(source_index),
        "split_role": split_role,
        "valid": False,
        "invalid_reasons": [],
    }
    try:
        rewrite = coerce_mapping(row["requested_rewrite"])
        subject = str(rewrite["subject"])
        relation_id = str(rewrite.get("relation_id", ""))
        rewrite_template = str(rewrite["prompt"])
        target_new = str(rewrite["target_new"]["str"])
        target_true = str(rewrite["target_true"]["str"])
        rendered_prompt = render_counterfact_prompt(rewrite_template, subject)
    except Exception as exc:
        validity["invalid_reasons"].append(f"missing_required_fields:{exc}")
        return None, validity

    target_new_text = format_target(target_new)
    target_true_text = format_target(target_true)
    new_tok = context_aware_target_tokenization(tokenizer, rendered_prompt, target_new_text)
    true_tok = context_aware_target_tokenization(tokenizer, rendered_prompt, target_true_text)
    if not new_tok.target_token_ids:
        validity["invalid_reasons"].append("empty_target_new_tokens")
    if not true_tok.target_token_ids:
        validity["invalid_reasons"].append("empty_target_true_tokens")

    normalized_new = normalize_counterfact_text(target_new)
    normalized_true = normalize_counterfact_text(target_true)
    if normalized_new == normalized_true:
        validity["target_new_equals_target_true_normalized"] = True

    paraphrase_prompts = [str(p) for p in coerce_list(row.get("paraphrase_prompts"))]
    declarative_paraphrases, qa_paraphrases = split_paraphrases(paraphrase_prompts)
    neighborhood_prompts = [str(p) for p in coerce_list(row.get("neighborhood_prompts"))]
    near_locality_cases = [
        {
            "id": f"counterfact_{source_split}_{source_index}_near_{idx}",
            "prompt": prompt,
            "target": target_true_text,
            "aliases": make_aliases(target_true),
        }
        for idx, prompt in enumerate(neighborhood_prompts)
    ]

    candidate_pool = [
        item
        for item in anchor_pool
        if int(item.get("source_index", -1)) != int(source_index)
        and item["target"].strip() not in {target_new.strip(), target_true.strip()}
    ]
    chosen_cases = choose_disjoint_cases(
        rng=rng,
        candidate_pool=candidate_pool,
        counts=[
            ("anchor_cases", anchor_cases_per_edit),
            ("far_locality_cases", far_locality_cases_per_edit),
            ("eval_anchor_cases", eval_anchor_cases_per_edit),
        ],
    )

    prompt_token_len = len(new_tok.prompt_token_ids)
    edit_id = f"counterfact_{source_split}_{source_index}"
    validity.update(
        {
            "valid": not validity["invalid_reasons"],
            "case_id": edit_id,
            "relation_id": relation_id,
            "subject": subject,
            "target_new_context_token_len": len(new_tok.target_token_ids),
            "target_true_context_token_len": len(true_tok.target_token_ids),
            "target_new_standalone_token_len": len(new_tok.standalone_token_ids),
            "target_true_standalone_token_len": len(true_tok.standalone_token_ids),
            "target_new_context_mismatch": new_tok.mismatch,
            "target_true_context_mismatch": true_tok.mismatch,
            "prompt_token_len": prompt_token_len,
            "neighborhood_prompt_count": len(neighborhood_prompts),
            "paraphrase_prompt_count": len(paraphrase_prompts),
        }
    )

    if not validity["valid"]:
        return None, validity

    record = {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "split_role": split_role,
        "source_dataset_split": source_split,
        "source_index": int(source_index),
        "id": edit_id,
        "case_id": edit_id,
        "relation_id": relation_id,
        "subject": subject,
        "rewrite_template": rewrite_template,
        "prompt": rendered_prompt,
        "target": target_new_text,
        "aliases": make_aliases(target_new),
        "old_target": target_true_text,
        "old_aliases": make_aliases(target_true),
        "target_new": {
            "text": target_new_text,
            "normalized": normalized_new,
            "context_token_ids": new_tok.target_token_ids,
            "standalone_token_ids": new_tok.standalone_token_ids,
            "context_prefix_match": new_tok.prefix_match,
            "context_token_len": len(new_tok.target_token_ids),
            "standalone_token_len": len(new_tok.standalone_token_ids),
        },
        "target_true": {
            "text": target_true_text,
            "normalized": normalized_true,
            "context_token_ids": true_tok.target_token_ids,
            "standalone_token_ids": true_tok.standalone_token_ids,
            "context_prefix_match": true_tok.prefix_match,
            "context_token_len": len(true_tok.target_token_ids),
            "standalone_token_len": len(true_tok.standalone_token_ids),
        },
        "target_new_token_len": len(new_tok.target_token_ids),
        "target_true_token_len": len(true_tok.target_token_ids),
        "target_length_bin": target_length_bin(len(new_tok.target_token_ids)),
        "prompt_token_len": prompt_token_len,
        "context_tokenization": {
            "target_new_mismatch": new_tok.mismatch,
            "target_true_mismatch": true_tok.mismatch,
            "target_new_common_prefix_len": new_tok.common_prefix_len,
            "target_true_common_prefix_len": true_tok.common_prefix_len,
        },
        "declarative_paraphrase_prompts": declarative_paraphrases,
        "qa_paraphrase_prompts": qa_paraphrases,
        "near_locality_cases": near_locality_cases,
        "far_locality_cases": chosen_cases["far_locality_cases"],
        "anchor_cases": chosen_cases["anchor_cases"],
        "eval_anchor_cases": chosen_cases["eval_anchor_cases"],
        "generation_prompts": [str(p) for p in coerce_list(row.get("generation_prompts"))],
        "attribute_prompts": [str(p) for p in coerce_list(row.get("attribute_prompts"))],
        "requested_rewrite": rewrite,
        "validity": validity,
    }
    return record, validity


def stratified_order(records: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("relation_id", "")), str(record.get("target_length_bin", "")))
        groups.setdefault(key, []).append(record)

    rng = random.Random(seed)
    for key, group in groups.items():
        rng.shuffle(group)
    ordered: List[Dict[str, Any]] = []
    keys = sorted(groups)
    while keys:
        next_keys: List[Tuple[str, str]] = []
        for key in keys:
            group = groups[key]
            if group:
                ordered.append(group.pop())
            if group:
                next_keys.append(key)
        keys = next_keys
    return ordered


def assign_disjoint_roles(
    records: Sequence[Dict[str, Any]],
    specs: Sequence[Tuple[str, int]],
    seed: int,
    *,
    allow_undersized: bool = False,
    source_label: str = "",
) -> Dict[str, List[Dict[str, Any]]]:
    ordered = stratified_order(records, seed=seed)
    required = sum(size for _, size in specs)
    if len(ordered) < required and not allow_undersized:
        label = f" for {source_label}" if source_label else ""
        raise ValueError(
            f"Not enough valid records{label}: need {required}, found {len(ordered)}. "
            "Use --smoke 1 or explicit smaller --*_size values for smoke runs."
        )
    assigned: Dict[str, List[Dict[str, Any]]] = {}
    offset = 0
    for role, size in specs:
        selected = [dict(item, split_role=role) for item in ordered[offset : offset + size]]
        if len(selected) < size and not allow_undersized:
            raise ValueError(
                f"Could not fill split {role}: requested {size}, got {len(selected)}."
            )
        assigned[role] = selected
        offset += size
    return assigned


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    relation_hist: Dict[str, int] = {}
    length_hist: Dict[str, int] = {}
    prefix_mismatches = 0
    for row in rows:
        relation = str(row.get("relation_id", ""))
        relation_hist[relation] = relation_hist.get(relation, 0) + 1
        length_bin = str(row.get("target_length_bin", ""))
        length_hist[length_bin] = length_hist.get(length_bin, 0) + 1
        if row.get("context_tokenization", {}).get("target_new_mismatch"):
            prefix_mismatches += 1
    return {
        "count": len(rows),
        "relation_id_histogram": dict(sorted(relation_hist.items())),
        "target_length_bin_histogram": dict(sorted(length_hist.items())),
        "target_new_context_mismatch_count": prefix_mismatches,
    }


def overlap_report(split_rows: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    roles = sorted(split_rows)
    reports: List[Dict[str, Any]] = []
    for i, left in enumerate(roles):
        left_ids = {row["case_id"] for row in split_rows[left]}
        left_sources = {
            (row.get("source_dataset_split"), int(row.get("source_index", -1)))
            for row in split_rows[left]
        }
        for right in roles[i + 1 :]:
            right_ids = {row["case_id"] for row in split_rows[right]}
            right_sources = {
                (row.get("source_dataset_split"), int(row.get("source_index", -1)))
                for row in split_rows[right]
            }
            id_overlap = sorted(left_ids & right_ids)
            source_overlap = sorted(left_sources & right_sources)
            allowed = {left, right} == {"final_test_500", "final_test_full"}
            reports.append(
                {
                    "left": left,
                    "right": right,
                    "id_overlap_count": len(id_overlap),
                    "source_overlap_count": len(source_overlap),
                    "allowed_overlap": allowed,
                    "id_overlap": id_overlap[:100],
                    "source_overlap": source_overlap[:100],
                }
            )
    return {
        "pairwise": reports,
        "has_disallowed_overlap": any(
            (row["id_overlap_count"] or row["source_overlap_count"])
            and not row["allowed_overlap"]
            for row in reports
        ),
    }


def load_hf_dataset_rows(dataset_name: str, split: str, max_rows: int = 0) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    rows = list(dataset)
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def load_tokenizer(model_id: str):
    if model_id in {"", "none", "simple"}:
        return SimpleWhitespaceTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


class SimpleWhitespaceTokenizer:
    """Small fallback tokenizer for smoke tests; not for paper runs."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        del add_special_tokens
        tokens = re.findall(r"\s+\S+|\S+", text)
        ids: List[int] = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
            ids.append(self.vocab[token])
        return {"input_ids": ids}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(int(i), f"<{i}>") for i in ids).strip()


def build_records_for_source_split(
    *,
    rows: Sequence[Dict[str, Any]],
    source_split: str,
    tokenizer: Any,
    seed: int,
    anchor_cases_per_edit: int,
    far_locality_cases_per_edit: int,
    eval_anchor_cases_per_edit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    anchor_pool = [build_pool_entry(row, source_split, idx) for idx, row in enumerate(rows)]
    valid: List[Dict[str, Any]] = []
    validity_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        record, validity = convert_counterfact_row(
            row=row,
            source_split=source_split,
            source_index=idx,
            tokenizer=tokenizer,
            anchor_pool=anchor_pool,
            rng=rng,
            split_role="unassigned",
            anchor_cases_per_edit=anchor_cases_per_edit,
            far_locality_cases_per_edit=far_locality_cases_per_edit,
            eval_anchor_cases_per_edit=eval_anchor_cases_per_edit,
        )
        validity_rows.append(validity)
        if record is not None:
            valid.append(record)
    return valid, validity_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer_model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default="runs/counterfact_direction1_v1/protocol")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Use tiny default split sizes suitable for preprocessing/tokenization smoke tests.",
    )
    parser.add_argument("--dev_size", type=int, default=-1)
    parser.add_argument("--analysis_size", type=int, default=-1)
    parser.add_argument("--ablation_size", type=int, default=-1)
    parser.add_argument("--final_test_size", type=int, default=-1)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_test_rows", type=int, default=0)
    parser.add_argument("--anchor_cases_per_edit", type=int, default=3)
    parser.add_argument("--far_locality_cases_per_edit", type=int, default=3)
    parser.add_argument("--eval_anchor_cases_per_edit", type=int, default=3)
    parser.add_argument("--skip_final_test_full", type=int, default=0)
    return parser.parse_args()


def resolve_split_specs(args: argparse.Namespace) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    train_defaults = SMOKE_TRAIN_SPLIT_SPECS if bool(args.smoke) else OFFICIAL_TRAIN_SPLIT_SPECS
    test_defaults = SMOKE_TEST_SPLIT_SPECS if bool(args.smoke) else OFFICIAL_TEST_SPLIT_SPECS
    size_overrides = {
        "dev_tune_200": int(args.dev_size),
        "analysis_500": int(args.analysis_size),
        "ablation_500": int(args.ablation_size),
        "final_test_500": int(args.final_test_size),
    }

    def apply_overrides(specs: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
        resolved: List[Tuple[str, int]] = []
        for role, default_size in specs:
            override = size_overrides.get(role, -1)
            resolved.append((role, int(default_size if override < 0 else override)))
        return resolved

    return apply_overrides(train_defaults), apply_overrides(test_defaults)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer_model_id)
    train_specs, test_specs = resolve_split_specs(args)

    train_rows = load_hf_dataset_rows(args.dataset_name, "train", max_rows=args.max_train_rows)
    test_rows = load_hf_dataset_rows(args.dataset_name, "test", max_rows=args.max_test_rows)

    train_valid, train_validity = build_records_for_source_split(
        rows=train_rows,
        source_split="train",
        tokenizer=tokenizer,
        seed=args.seed,
        anchor_cases_per_edit=args.anchor_cases_per_edit,
        far_locality_cases_per_edit=args.far_locality_cases_per_edit,
        eval_anchor_cases_per_edit=args.eval_anchor_cases_per_edit,
    )
    test_valid, test_validity = build_records_for_source_split(
        rows=test_rows,
        source_split="test",
        tokenizer=tokenizer,
        seed=args.seed + 1,
        anchor_cases_per_edit=args.anchor_cases_per_edit,
        far_locality_cases_per_edit=args.far_locality_cases_per_edit,
        eval_anchor_cases_per_edit=args.eval_anchor_cases_per_edit,
    )

    split_rows = assign_disjoint_roles(
        train_valid,
        train_specs,
        seed=args.seed,
        source_label="HF train",
    )
    test_assigned = assign_disjoint_roles(
        test_valid,
        test_specs,
        seed=args.seed + 2,
        source_label="HF test",
    )
    split_rows.update(test_assigned)
    if not bool(args.skip_final_test_full):
        split_rows["final_test_full"] = [dict(item, split_role="final_test_full") for item in test_valid]

    artifacts: Dict[str, Dict[str, Any]] = {}
    for role, rows in split_rows.items():
        jsonl_path = os.path.join(args.output_dir, f"{role}.jsonl")
        metadata_path = os.path.join(args.output_dir, f"{role}.metadata.json")
        write_jsonl(jsonl_path, rows)
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_name": args.dataset_name,
            "tokenizer_model_id": args.tokenizer_model_id,
            "split_role": role,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "summary": summarize_rows(rows),
            "jsonl_path": jsonl_path,
        }
        write_json(metadata_path, metadata)
        artifacts[role] = {
            "jsonl_path": jsonl_path,
            "metadata_path": metadata_path,
            "jsonl_sha256": sha256_file(jsonl_path),
            "metadata_sha256": sha256_file(metadata_path),
            "summary": metadata["summary"],
        }

    overlaps = overlap_report(split_rows)
    write_json(os.path.join(args.output_dir, "split_overlap_report.json"), overlaps)
    validity_payload = {
        "train": train_validity,
        "test": test_validity,
        "train_valid_count": len(train_valid),
        "test_valid_count": len(test_valid),
        "train_total_count": len(train_rows),
        "test_total_count": len(test_rows),
    }
    write_json(os.path.join(args.output_dir, "validity_report.json"), validity_payload)
    protocol_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "tokenizer_model_id": args.tokenizer_model_id,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "split_sizes": {
            **{role: size for role, size in train_specs},
            **{role: size for role, size in test_specs},
            "final_test_full": 0 if bool(args.skip_final_test_full) else len(test_valid),
        },
        "split_sources": {
            "dev_tune_200": "HF train",
            "analysis_500": "HF train",
            "ablation_500": "HF train",
            "final_test_500": "HF test",
            "final_test_full": "HF test",
        },
        "artifacts": artifacts,
        "overlap_report": os.path.join(args.output_dir, "split_overlap_report.json"),
        "validity_report": os.path.join(args.output_dir, "validity_report.json"),
    }
    write_json(os.path.join(args.output_dir, "protocol_manifest.json"), protocol_manifest)

    print(f"[INFO] Wrote protocol manifests to {args.output_dir}")
    if overlaps["has_disallowed_overlap"]:
        raise SystemExit("[ERROR] Disallowed split overlap detected; inspect split_overlap_report.json")


if __name__ == "__main__":
    main()
