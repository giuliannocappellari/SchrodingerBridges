#!/usr/bin/env python3
"""Build Direction 3 teacher-cache artifacts.

Only fake mode is implemented locally. Real teacher-cache generation requires
RunPod/GPU approval and should be added behind the same schema.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import (
    D3_PROTOCOL_VERSION,
    D3_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    repo_path,
    softmax,
    stable_float,
    stable_int,
    summarize_counter,
    write_json,
    write_jsonl,
)


PROMPT_TYPES = [
    "rewrite",
    "declarative_paraphrase",
    "same_subject_different_relation",
    "near_locality",
    "far_locality",
    "generation",
]
STEPS = [0, 1, 2]
TOP_K = 8


def fake_candidate_ids(edit_id: str, target_length_bin: str) -> List[int]:
    base = 1000 + stable_int(edit_id, 50000)
    ids = [base + i for i in range(TOP_K)]
    # The first candidate is treated as target_new for positives.
    return ids


def as_float_list(values: Sequence[Any]) -> List[float]:
    return [float(x) for x in values]


def fixed_width_candidates(
    *,
    base_probs: Any,
    target_ids: Sequence[int],
    rel_pos: int,
    top_k: int,
) -> List[int]:
    import torch

    width = min(int(top_k), int(base_probs.numel()))
    candidates = torch.topk(base_probs, k=width).indices.detach().cpu().tolist()
    if rel_pos < len(target_ids):
        target_id = int(target_ids[rel_pos])
        if target_id not in candidates:
            if len(candidates) >= width:
                candidates[-1] = target_id
            else:
                candidates.append(target_id)
    return list(dict.fromkeys(map(int, candidates)))[:width]


def prompt_specs_for_record(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "prompt_type": "rewrite",
            "prompt_id": f"{record['case_id']}::rewrite",
            "prompt": record["prompt"],
            "label": 1,
        }
    ]
    declarative = list(record.get("declarative_paraphrase_prompts") or [])
    if declarative:
        specs.append(
            {
                "prompt_type": "declarative_paraphrase",
                "prompt_id": f"{record['case_id']}::declarative_paraphrase",
                "prompt": declarative[0],
                "label": 1,
            }
        )
    generation = list(record.get("generation_prompts") or [])
    if generation:
        specs.append(
            {
                "prompt_type": "same_subject_different_relation",
                "prompt_id": f"{record['case_id']}::same_subject_different_relation",
                "prompt": generation[0],
                "label": 0,
            }
        )
        specs.append(
            {
                "prompt_type": "generation",
                "prompt_id": f"{record['case_id']}::generation",
                "prompt": generation[min(1, len(generation) - 1)],
                "label": 0,
            }
        )
    else:
        specs.append(
            {
                "prompt_type": "same_subject_different_relation",
                "prompt_id": f"{record['case_id']}::same_subject_different_relation",
                "prompt": f"{record['subject']} is known for",
                "label": 0,
            }
        )
    near_cases = list(record.get("near_locality_cases") or [])
    if near_cases:
        specs.append(
            {
                "prompt_type": "near_locality",
                "prompt_id": near_cases[0].get("id") or f"{record['case_id']}::near_locality",
                "prompt": near_cases[0]["prompt"],
                "label": 0,
            }
        )
    far_cases = list(record.get("far_locality_cases") or [])
    if far_cases:
        specs.append(
            {
                "prompt_type": "far_locality",
                "prompt_id": far_cases[0].get("id") or f"{record['case_id']}::far_locality",
                "prompt": far_cases[0]["prompt"],
                "label": 0,
            }
        )
    return specs


def load_selected_counterfact_records(
    *,
    manifest_rows: Sequence[Dict[str, Any]],
    split_role: str,
    tokenizer: Any,
    dataset_name: str,
    seed: int,
) -> List[Dict[str, Any]]:
    from llada_counterfact_protocol import build_pool_entry, convert_counterfact_row
    from llada_counterfact_protocol import load_hf_dataset_rows

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    anchor_pool_by_source: Dict[str, List[Dict[str, Any]]] = {}
    selected_records: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for item in manifest_rows:
        source_split = str(item["source_dataset_split"])
        source_index = int(item["source_index"])
        if source_split not in by_source:
            rows = load_hf_dataset_rows(dataset_name, source_split)
            by_source[source_split] = rows
            anchor_pool_by_source[source_split] = [
                build_pool_entry(row, source_split, idx)
                for idx, row in enumerate(rows)
            ]
        source_rows = by_source[source_split]
        if source_index < 0 or source_index >= len(source_rows):
            raise IndexError(f"source_index out of range for {source_split}: {source_index}")
        record, validity = convert_counterfact_row(
            row=source_rows[source_index],
            source_split=source_split,
            source_index=source_index,
            tokenizer=tokenizer,
            anchor_pool=anchor_pool_by_source[source_split],
            rng=rng,
            split_role=split_role,
            anchor_cases_per_edit=3,
            far_locality_cases_per_edit=3,
            eval_anchor_cases_per_edit=3,
        )
        if record is None:
            raise AssertionError(f"Selected D3 source row failed materialization: {item['case_id']} {validity}")
        record["split_role"] = split_role
        selected_records.append(record)
    return selected_records


def real_teacher_rows_for_record(
    *,
    model: Any,
    tokenizer: Any,
    record: Dict[str, Any],
    split_role: str,
    steps: int,
    top_k: int,
    mc_rollouts: int,
    reward_mode: str,
    reward_beta: float,
) -> List[Dict[str, Any]]:
    import torch
    from llada_sb_common import (
        build_initial_state,
        build_transfer_schedule,
        decode_ids,
        default_aliases_for_text,
        endpoint_reward,
        estimate_candidate_bridge_scores,
        get_model_device,
        infer_mask_id,
        normalize_probability_vector,
        run_logits,
        sanitize_logits_row,
        tokenize_alias_lists_for_target_length,
        tokenize_prompt,
    )

    device = get_model_device(model)
    mask_id = infer_mask_id(model)
    target_text = str(record["target"])
    target_alias_pool = list(dict.fromkeys(list(record.get("aliases") or [target_text]) + default_aliases_for_text(target_text)))
    prompt_specs = prompt_specs_for_record(record)
    rows: List[Dict[str, Any]] = []

    for prompt_spec in prompt_specs:
        prompt_text = str(prompt_spec["prompt"])
        prompt_ids = tokenize_prompt(tokenizer, prompt_text)
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        answer_len = len(target_ids)
        if answer_len <= 0:
            continue
        alias_token_lists = tokenize_alias_lists_for_target_length(tokenizer, target_alias_pool, answer_len)
        x = build_initial_state(prompt_ids, answer_len, mask_id, device)
        prompt_len = len(prompt_ids)
        answer_abs_positions = list(range(prompt_len, prompt_len + answer_len))
        schedule = build_transfer_schedule(answer_len, steps)

        for step_index, num_transfer in enumerate(schedule):
            masked_positions = [
                rel_pos
                for rel_pos, abs_pos in enumerate(answer_abs_positions)
                if int(x[0, abs_pos].item()) == int(mask_id)
            ]
            if not masked_positions:
                break
            logits = run_logits(model, x)[0]
            proposal_scores: List[Tuple[float, int, int]] = []
            remaining_steps = max(0, int(steps) - step_index - 1)
            active_mask_count = len(masked_positions)

            for rel_pos in masked_positions:
                abs_pos = answer_abs_positions[rel_pos]
                row_logits = sanitize_logits_row(logits[abs_pos])
                base_probs_full = normalize_probability_vector(torch.softmax(row_logits, dim=-1))
                candidate_ids = fixed_width_candidates(
                    base_probs=base_probs_full,
                    target_ids=target_ids,
                    rel_pos=rel_pos,
                    top_k=top_k,
                )
                cand_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=row_logits.device)
                base_logits = row_logits[cand_tensor].detach().cpu().tolist()
                base_probs = normalize_probability_vector(base_probs_full[cand_tensor]).detach().cpu().tolist()
                target_token_ids_for_pos = {
                    int(alias[rel_pos])
                    for alias in alias_token_lists
                    if rel_pos < len(alias)
                }
                myopic_scores = [
                    math.exp(float(reward_beta)) if int(token_id) in target_token_ids_for_pos else 1.0
                    for token_id in candidate_ids
                ]
                no_rollout_scores: List[float] = []
                for token_id in candidate_ids:
                    x_tent = x.clone()
                    x_tent[0, abs_pos] = int(token_id)
                    partial = x_tent[0, prompt_len : prompt_len + answer_len].detach().cpu().tolist()
                    no_rollout_scores.append(
                        float(endpoint_reward(partial, alias_token_lists, reward_mode=reward_mode, reward_beta=reward_beta))
                    )
                mc_rewards = estimate_candidate_bridge_scores(
                    model=model,
                    x=x,
                    prompt_len=prompt_len,
                    answer_len=answer_len,
                    rel_pos=rel_pos,
                    candidate_ids=candidate_ids,
                    alias_token_lists=alias_token_lists,
                    remaining_steps=remaining_steps,
                    reward_mode=reward_mode,
                    reward_beta=reward_beta,
                    mc_rollouts=mc_rollouts,
                    mask_id=mask_id,
                )
                chosen_idx = max(range(len(candidate_ids)), key=lambda idx: mc_rewards[idx])
                chosen_token = int(candidate_ids[chosen_idx])
                proposal_scores.append((float(mc_rewards[chosen_idx]), rel_pos, chosen_token))
                state_ids = x[0].detach().cpu().tolist()
                target_len_bin = str(record.get("target_length_bin") or ("2" if answer_len == 2 else "1" if answer_len == 1 else "3"))
                output_ids = list(state_ids[prompt_len : prompt_len + answer_len])
                output_ids[rel_pos] = chosen_token
                final_output = decode_ids(tokenizer, output_ids)
                is_positive = int(prompt_spec["label"]) == 1
                final_edit_success = bool(is_positive and chosen_token in target_token_ids_for_pos)
                final_locality_success = bool((not is_positive) and chosen_token not in target_token_ids_for_pos)
                teacher_row = {
                    "protocol_version": D3_PROTOCOL_VERSION,
                    "schema_version": 1,
                    "fake_model": False,
                    "case_id": str(record["case_id"]),
                    "edit_id": str(record["case_id"]),
                    "split_role": split_role,
                    "prompt_id": str(prompt_spec["prompt_id"]),
                    "prompt_type": str(prompt_spec["prompt_type"]),
                    "prompt_text": prompt_text,
                    "subject": record.get("subject"),
                    "relation_id": record.get("relation_id"),
                    "target_new": target_text,
                    "target_true": str(record.get("old_target") or ""),
                    "target_token_ids": list(map(int, target_ids)),
                    "target_length_bin": target_len_bin,
                    "step_index": int(step_index),
                    "timestep": int(step_index),
                    "mask_ratio": float(active_mask_count / max(1, answer_len)),
                    "active_mask_count": int(active_mask_count),
                    "current_state": state_ids,
                    "selected_mask_positions": [int(pos) for pos in masked_positions],
                    "top_k_candidate_token_ids": list(map(int, candidate_ids)),
                    "base_logits_top_k": as_float_list(base_logits),
                    "base_probabilities_top_k": as_float_list(base_probs),
                    "raw_bridge_scores_top_k": as_float_list(mc_rewards),
                    "myopic_scores_top_k": as_float_list(myopic_scores),
                    "no_rollout_scores_top_k": as_float_list(no_rollout_scores),
                    "mc_rollout_rewards_top_k": as_float_list(mc_rewards),
                    "chosen_token_id": chosen_token,
                    "final_decoded_output": final_output,
                    "final_edit_success": final_edit_success,
                    "final_locality_success": final_locality_success,
                    "sparse_guidance_kl": 0.0,
                    "malformed": False,
                    "label": int(prompt_spec["label"]),
                    "fake_state": state_ids,
                    "selected_mask_position": int(rel_pos),
                    "top_k_candidate_ids": list(map(int, candidate_ids)),
                    "base_logits": as_float_list(base_logits),
                    "base_probs": as_float_list(base_probs),
                    "raw_bridge_scores": as_float_list(mc_rewards),
                    "myopic_scores": as_float_list(myopic_scores),
                    "no_rollout_scores": as_float_list(no_rollout_scores),
                    "mc_rollout_rewards": as_float_list(mc_rewards),
                    "chosen_token": chosen_token,
                }
                rows.append(teacher_row)

            if num_transfer <= 0 or not proposal_scores:
                continue
            proposal_scores.sort(key=lambda item: item[0], reverse=True)
            for _, rel_pos, token_id in proposal_scores[: int(num_transfer)]:
                x[0, prompt_len + rel_pos] = int(token_id)
    return rows


def fake_teacher_row(item: Dict[str, Any], split_role: str, prompt_type: str, step: int) -> Dict[str, Any]:
    edit_id = str(item["case_id"])
    prompt_id = f"{edit_id}::{prompt_type}"
    target_length_bin = str(item.get("target_length_bin", "1"))
    target_len = 2 if target_length_bin == "2" else 1
    candidate_ids = fake_candidate_ids(edit_id, target_length_bin)
    key = f"{split_role}:{edit_id}:{prompt_type}:{step}"
    active_mask_count = 1 + (step % 2)
    base_logits = [stable_float(f"{key}:base:{i}", -2.0, 2.0) for i in range(TOP_K)]
    myopic_scores = [x + stable_float(f"{key}:myopic:{i}", -0.5, 0.5) for i, x in enumerate(base_logits)]
    no_rollout_scores = [x + stable_float(f"{key}:no_rollout:{i}", -0.35, 0.65) for i, x in enumerate(base_logits)]
    positive = prompt_type in {"rewrite", "declarative_paraphrase"}
    bridge_boost = 1.5 if positive else -0.2
    raw_bridge_scores = [
        x + (bridge_boost if i == 0 else stable_float(f"{key}:bridge:{i}", -0.4, 0.4))
        for i, x in enumerate(base_logits)
    ]
    mc_rewards = [
        raw_bridge_scores[i] + stable_float(f"{key}:reward:{i}", -0.2, 0.2)
        for i in range(TOP_K)
    ]
    chosen_idx = max(range(TOP_K), key=lambda i: raw_bridge_scores[i])
    base_probs = softmax(base_logits)
    current_state = [stable_int(f"{key}:state:{i}", 32000) for i in range(4 + active_mask_count)]
    selected_mask_positions = list(range(active_mask_count))
    chosen_token_id = candidate_ids[chosen_idx]
    row = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "schema_version": 1,
        "fake_model": True,
        "case_id": edit_id,
        "edit_id": edit_id,
        "split_role": split_role,
        "prompt_id": prompt_id,
        "prompt_type": prompt_type,
        "subject": item.get("subject"),
        "relation_id": item.get("relation_id"),
        "target_new": f"fake_target_new_{target_length_bin}",
        "target_true": "fake_target_true",
        "target_token_ids": candidate_ids[:target_len],
        "target_length_bin": target_length_bin,
        "step_index": step,
        "timestep": step,
        "mask_ratio": round((len(STEPS) - step) / len(STEPS), 4),
        "active_mask_count": active_mask_count,
        "current_state": current_state,
        "selected_mask_positions": selected_mask_positions,
        "top_k_candidate_token_ids": candidate_ids,
        "base_logits_top_k": base_logits,
        "base_probabilities_top_k": base_probs,
        "raw_bridge_scores_top_k": raw_bridge_scores,
        "myopic_scores_top_k": myopic_scores,
        "no_rollout_scores_top_k": no_rollout_scores,
        "mc_rollout_rewards_top_k": mc_rewards,
        "chosen_token_id": chosen_token_id,
        "final_decoded_output": "fake_target_new" if positive and chosen_idx == 0 else "fake_other",
        "final_edit_success": bool(positive and chosen_idx == 0),
        "final_locality_success": bool((not positive) and chosen_idx != 0),
        "sparse_guidance_kl": abs(raw_bridge_scores[0] - base_logits[0]) / 10.0,
        "malformed": False,
        "label": 1 if positive else 0,
    }
    # Stage 1A aliases mirror the planned real teacher-cache schema while
    # preserving earlier scaffold field names consumed by fake training.
    row.update(
        {
            "fake_state": current_state,
            "selected_mask_position": selected_mask_positions[0],
            "top_k_candidate_ids": candidate_ids,
            "base_logits": base_logits,
            "base_probs": base_probs,
            "raw_bridge_scores": raw_bridge_scores,
            "myopic_scores": myopic_scores,
            "no_rollout_scores": no_rollout_scores,
            "mc_rollout_rewards": mc_rewards,
            "chosen_token": chosen_token_id,
        }
    )
    return row


def build_fake_cache_rows(manifest_rows: Sequence[Dict[str, Any]], split_role: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in manifest_rows:
        for prompt_type in PROMPT_TYPES:
            for step in STEPS:
                rows.append(fake_teacher_row(item, split_role, prompt_type, step))
    return rows


def validate_cache(rows: Sequence[Dict[str, Any]], top_k: int = TOP_K) -> Dict[str, Any]:
    steps = sorted({int(row["step_index"]) for row in rows})
    active_counts = sorted({int(row["active_mask_count"]) for row in rows})
    target_bins = sorted({str(row["target_length_bin"]) for row in rows})
    prompt_types = sorted({str(row["prompt_type"]) for row in rows})
    if len(steps) < 3:
        raise AssertionError("Fake teacher cache must include at least 3 distinct denoising steps")
    if not any(count > 1 for count in active_counts):
        raise AssertionError("Fake teacher cache must include active_mask_count > 1")
    if "1" not in target_bins or "2" not in target_bins:
        raise AssertionError("Fake teacher cache must include target length bins 1 and 2")
    if not any(t in {"rewrite", "declarative_paraphrase"} for t in prompt_types):
        raise AssertionError("Fake teacher cache must include positive prompt rows")
    if "same_subject_different_relation" not in prompt_types:
        raise AssertionError("Fake teacher cache must include same-subject negative prompt rows")
    if not any(t in {"near_locality", "far_locality"} for t in prompt_types):
        raise AssertionError("Fake teacher cache must include locality negative prompt rows")
    if not any(t in {"same_subject_different_relation", "near_locality", "far_locality", "generation"} for t in prompt_types):
        raise AssertionError("Fake teacher cache must include negative prompt rows")
    required_alias_fields = [
        "fake_state",
        "selected_mask_position",
        "top_k_candidate_ids",
        "base_logits",
        "base_probs",
        "raw_bridge_scores",
        "myopic_scores",
        "no_rollout_scores",
        "mc_rollout_rewards",
        "chosen_token",
    ]
    for row in rows:
        for field in required_alias_fields:
            if field not in row:
                raise AssertionError(f"Missing fake teacher schema alias field: {field}")
        if len(row["top_k_candidate_token_ids"]) != top_k:
            raise AssertionError("Invalid top-k candidate width")
        for key in [
            "base_logits_top_k",
            "base_probabilities_top_k",
            "raw_bridge_scores_top_k",
            "myopic_scores_top_k",
            "no_rollout_scores_top_k",
            "mc_rollout_rewards_top_k",
            "base_logits",
            "base_probs",
            "raw_bridge_scores",
            "myopic_scores",
            "no_rollout_scores",
            "mc_rollout_rewards",
        ]:
            if len(row[key]) != top_k:
                raise AssertionError(f"Invalid score width for {key}")
            if not all(math.isfinite(float(value)) for value in row[key]):
                raise AssertionError(f"Non-finite score in {key}")
        if len(row["top_k_candidate_ids"]) != top_k:
            raise AssertionError("Invalid top_k_candidate_ids width")
    return {
        "num_rows": len(rows),
        "step_histogram": summarize_counter(row["step_index"] for row in rows),
        "active_mask_count_histogram": summarize_counter(row["active_mask_count"] for row in rows),
        "target_len_histogram": summarize_counter(row["target_length_bin"] for row in rows),
        "prompt_type_histogram": summarize_counter(row["prompt_type"] for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=D3_ROOT)
    parser.add_argument("--output_dir", type=Path, default=D3_ROOT / "fake_teacher_cache_v1")
    parser.add_argument("--fake_model", type=int, default=0)
    parser.add_argument("--split_train", type=str, default="controller_train_100.jsonl")
    parser.add_argument("--split_val", type=str, default="controller_val_50.jsonl")
    parser.add_argument("--dataset_name", type=str, default="azhx/counterfact")
    parser.add_argument("--model_id", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--use_4bit", type=int, default=1)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--steps", type=int, default=len(STEPS))
    parser.add_argument("--mc_rollouts", type=int, default=2)
    parser.add_argument("--methods", type=str, default="base,myopic_score,no_rollout_bridge,mc_bridge")
    parser.add_argument("--reward_mode", type=str, default="soft_overlap")
    parser.add_argument("--reward_beta", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    repo_path(out_dir).mkdir(parents=True, exist_ok=True)
    if bool(args.fake_model):
        train_manifest = read_jsonl(args.input_dir / "controller_train_100.jsonl")
        val_manifest = read_jsonl(args.input_dir / "controller_val_50.jsonl")
        train_rows = build_fake_cache_rows(train_manifest, "controller_train_100")
        val_rows = build_fake_cache_rows(val_manifest, "controller_val_50")
        top_k = TOP_K
        split_train = "controller_train_100.jsonl"
        split_val = "controller_val_50.jsonl"
        methods = ["fake_base", "fake_myopic_score", "fake_no_rollout_bridge", "fake_mc_bridge"]
        model_id = "fake"
    else:
        from llada_sb_common import load_llada_model_and_tokenizer, reset_model_eval_counter, get_model_eval_counter

        if "analysis_500" in str(args.split_train) or "final_test" in str(args.split_train):
            raise AssertionError("Locked analysis/final split is not allowed for D3 teacher-cache smoke")
        if "analysis_500" in str(args.split_val) or "final_test" in str(args.split_val):
            raise AssertionError("Locked analysis/final split is not allowed for D3 teacher-cache smoke")
        reset_model_eval_counter()
        model, tokenizer = load_llada_model_and_tokenizer(
            args.model_id,
            dtype_name=args.dtype,
            use_4bit=bool(args.use_4bit),
            device_map=args.device_map,
        )
        train_manifest = read_jsonl(args.input_dir / args.split_train)
        val_manifest = read_jsonl(args.input_dir / args.split_val)
        train_records = load_selected_counterfact_records(
            manifest_rows=train_manifest,
            split_role=Path(args.split_train).stem,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            seed=args.seed,
        )
        val_records = load_selected_counterfact_records(
            manifest_rows=val_manifest,
            split_role=Path(args.split_val).stem,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            seed=args.seed + 1,
        )
        train_rows = []
        for record in train_records:
            train_rows.extend(
                real_teacher_rows_for_record(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    split_role=Path(args.split_train).stem,
                    steps=int(args.steps),
                    top_k=int(args.top_k),
                    mc_rollouts=int(args.mc_rollouts),
                    reward_mode=args.reward_mode,
                    reward_beta=float(args.reward_beta),
                )
            )
        val_rows = []
        for record in val_records:
            val_rows.extend(
                real_teacher_rows_for_record(
                    model=model,
                    tokenizer=tokenizer,
                    record=record,
                    split_role=Path(args.split_val).stem,
                    steps=int(args.steps),
                    top_k=int(args.top_k),
                    mc_rollouts=int(args.mc_rollouts),
                    reward_mode=args.reward_mode,
                    reward_beta=float(args.reward_beta),
                )
            )
        top_k = int(args.top_k)
        split_train = args.split_train
        split_val = args.split_val
        methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
        model_id = args.model_id
        model_eval_count = get_model_eval_counter()

    all_rows = train_rows + val_rows
    validation = validate_cache(all_rows, top_k=top_k)

    write_jsonl(out_dir / "teacher_states_train.jsonl", train_rows)
    write_jsonl(out_dir / "teacher_states_val.jsonl", val_rows)
    teacher_summary = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 fake teacher cache" if bool(args.fake_model) else "Direction 3 teacher-cache smoke",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": bool(args.fake_model),
        "llada_loaded": not bool(args.fake_model),
        "analysis_500_used": False,
        "final_test_used": False,
        "dataset_name": args.dataset_name,
        "model_id": model_id,
        "dtype": args.dtype if not bool(args.fake_model) else "fake",
        "use_4bit": bool(args.use_4bit) if not bool(args.fake_model) else False,
        "device_map": args.device_map if not bool(args.fake_model) else "fake",
        "split_train": split_train,
        "split_val": split_val,
        "top_k": top_k,
        "prompt_types": PROMPT_TYPES,
        "steps": list(range(int(args.steps))) if not bool(args.fake_model) else STEPS,
        "mc_rollouts": int(args.mc_rollouts) if not bool(args.fake_model) else 0,
        "methods": methods,
        "model_eval_count": locals().get("model_eval_count", 0),
        **validation,
        "artifacts": {
            "teacher_states_train": str(out_dir / "teacher_states_train.jsonl"),
            "teacher_states_val": str(out_dir / "teacher_states_val.jsonl"),
        },
    }
    write_json(out_dir / "teacher_summary.json", teacher_summary)
    write_json(out_dir / "report_summary.json", teacher_summary)
    print(f"[INFO] Wrote fake Direction 3 teacher cache to {out_dir}")


if __name__ == "__main__":
    main()
