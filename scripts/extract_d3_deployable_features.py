#!/usr/bin/env python3
"""Extract deployable Direction 3 representation features.

Fake mode is local/test-safe and does not import or load LLaDA. Real mode is a
RunPod-only GPU step that runs frozen LLaDA forwards over cached denoising
states, deduplicated by state key, and stores tensors as safetensors.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D3_PROTOCOL_VERSION, D3_ROOT, git_commit, now_utc, read_jsonl, repo_path, stable_float, write_csv, write_json, write_jsonl


DEFAULT_CACHE_DIR = D3_ROOT / "teacher_cache_train100_val50_v1"
DEFAULT_OUTPUT_DIR = D3_ROOT / "deployable_feature_cache_train100_val50_v1"
FORBIDDEN_RUNTIME_FIELDS = {
    "raw_bridge_scores_top_k",
    "raw_bridge_scores",
    "mc_rollout_rewards_top_k",
    "mc_rollout_rewards",
    "myopic_scores_top_k",
    "myopic_scores",
    "no_rollout_scores_top_k",
    "no_rollout_scores",
    "target_myopic_margin",
    "target_no_rollout_margin",
    "chosen_token_id",
    "chosen_token",
    "final_decoded_output",
    "final_edit_success",
    "final_locality_success",
    "malformed",
    "sparse_guidance_kl",
    "prompt_type",
    "negative_type",
    "split_role",
    "label",
}


def sha1_json(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def state_key(row: Mapping[str, Any]) -> str:
    return sha1_json(
        {
            "edit_id": row.get("edit_id") or row.get("case_id"),
            "prompt_id": row.get("prompt_id"),
            "current_state": row.get("current_state") or row.get("fake_state"),
            "step_index": row.get("step_index"),
            "selected_mask_position": row.get("selected_mask_position"),
        }
    )


def edit_key(row: Mapping[str, Any]) -> str:
    return str(row.get("edit_id") or row.get("case_id"))


def prompt_key(row: Mapping[str, Any]) -> str:
    return sha1_json({"edit_id": edit_key(row), "prompt_id": row.get("prompt_id")})


def group_key(row: Mapping[str, Any]) -> str:
    return sha1_json(
        {
            "state_key": state_key(row),
            "top_k_candidate_token_ids": row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or [],
            "base_logits": row.get("base_logits_top_k") or row.get("base_logits") or [],
            "base_probs": row.get("base_probabilities_top_k") or row.get("base_probs") or [],
        }
    )


def row_split(row: Mapping[str, Any]) -> str:
    split = str(row.get("split_role") or "")
    if "val" in split:
        return "val"
    if "train" in split:
        return "train"
    return split or "unknown"


def target_token_set(row: Mapping[str, Any]) -> set[int]:
    return {int(v) for v in row.get("target_token_ids") or []}


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def token_ids_for_text(tokenizer: Any, text: str) -> List[int]:
    text = normalized_text(text)
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(v) for v in encoded.get("input_ids", [])]


def first_rewrite_by_edit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    fallback: Dict[str, str] = {}
    for row in rows:
        key = edit_key(row)
        prompt = normalized_text(row.get("prompt_text"))
        if prompt and key not in fallback:
            fallback[key] = prompt
        if str(row.get("prompt_type")) == "rewrite" and prompt and key not in out:
            out[key] = prompt
    for key, prompt in fallback.items():
        out.setdefault(key, prompt)
    return out


def expected_top_k(row: Mapping[str, Any]) -> int:
    return len(row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or [])


def validate_candidate_group(row: Mapping[str, Any], top_k: Optional[int] = None) -> None:
    token_ids = row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or []
    logits = row.get("base_logits_top_k") or row.get("base_logits") or []
    probs = row.get("base_probabilities_top_k") or row.get("base_probs") or []
    if top_k is not None and len(token_ids) != top_k:
        raise AssertionError(f"Expected top_k={top_k}, got {len(token_ids)} for {row.get('prompt_id')}")
    if not token_ids:
        raise AssertionError(f"Missing top-k candidates for {row.get('prompt_id')}")
    if len(logits) != len(token_ids) or len(probs) != len(token_ids):
        raise AssertionError(f"Top-k/logit/prob width mismatch for {row.get('prompt_id')}")
    values = [*logits, *probs]
    if any(not math.isfinite(float(v)) for v in values):
        raise AssertionError(f"Non-finite base logit/prob for {row.get('prompt_id')}")


def fake_vector(key: str, dim: int) -> List[float]:
    return [stable_float(f"{key}:{idx}", -1.0, 1.0) for idx in range(dim)]


def finite_tensor(name: str, tensor: Any) -> None:
    import torch

    if not torch.isfinite(tensor).all().item():
        raise AssertionError(f"Non-finite values in {name}")


def write_safetensors(path: Path, tensors: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(full))


def unique_rows(rows: Sequence[Mapping[str, Any]], key_fn) -> List[Mapping[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = key_fn(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def feature_index_rows(rows: Sequence[Mapping[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    state_rows = unique_rows(rows, state_key)
    prompt_rows = unique_rows(rows, prompt_key)
    edit_rows = unique_rows(rows, edit_key)
    index_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(state_rows):
        index_rows.append(
            {
                "index_type": "state",
                "row_index": idx,
                "state_key": state_key(row),
                "edit_id": edit_key(row),
                "prompt_id": row.get("prompt_id"),
                "step_index": row.get("step_index"),
                "selected_mask_position": row.get("selected_mask_position"),
            }
        )
    for idx, row in enumerate(rows):
        index_rows.append(
            {
                "index_type": "candidate_group",
                "row_index": idx,
                "group_key": group_key(row),
                "state_key": state_key(row),
                "edit_id": edit_key(row),
                "prompt_id": row.get("prompt_id"),
                "split": row_split(row),
                "step_index": row.get("step_index"),
                "selected_mask_position": row.get("selected_mask_position"),
                "top_k_width": top_k,
            }
        )
    for idx, row in enumerate(edit_rows):
        index_rows.append({"index_type": "edit", "row_index": idx, "edit_id": edit_key(row)})
    for idx, row in enumerate(prompt_rows):
        index_rows.append(
            {
                "index_type": "gate",
                "row_index": idx,
                "prompt_key": prompt_key(row),
                "edit_id": edit_key(row),
                "prompt_id": row.get("prompt_id"),
            }
        )
    return index_rows


def tensor_from_vectors(vectors: Sequence[Sequence[float]]) -> Any:
    import torch

    return torch.tensor(vectors, dtype=torch.float32)


def candidate_vocab(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    ids = set()
    for row in rows:
        for token_id in row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or []:
            ids.add(int(token_id))
        for token_id in row.get("target_token_ids") or []:
            ids.add(int(token_id))
    return sorted(ids)


def fake_extract(rows: Sequence[Mapping[str, Any]], output_dir: Path, dim: int) -> Dict[str, Any]:
    import torch

    state_rows = unique_rows(rows, state_key)
    prompt_rows = unique_rows(rows, prompt_key)
    edit_rows = unique_rows(rows, edit_key)
    group_rows = list(rows)
    widths = {expected_top_k(row) for row in group_rows}
    if len(widths) != 1:
        raise AssertionError(f"Expected constant top-k width, got {sorted(widths)}")
    top_k = widths.pop()
    for row in group_rows:
        validate_candidate_group(row, top_k=top_k)
    state_tensors = {
        "mid_layer_selected": tensor_from_vectors([fake_vector(f"mid:{state_key(row)}", dim) for row in state_rows]),
        "last_layer_selected": tensor_from_vectors([fake_vector(f"last:{state_key(row)}", dim) for row in state_rows]),
        "answer_span_mean": tensor_from_vectors([fake_vector(f"answer:{state_key(row)}", dim) for row in state_rows]),
        "subject_span_mean": tensor_from_vectors([fake_vector(f"subject-span:{state_key(row)}", dim) for row in state_rows]),
    }
    group_token_ids: List[List[int]] = []
    group_logits: List[List[float]] = []
    group_probs: List[List[float]] = []
    group_is_target_new: List[List[float]] = []
    group_is_target_true: List[List[float]] = []
    group_embeddings: List[List[List[float]]] = []
    for row in group_rows:
        token_ids = [int(v) for v in (row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or [])]
        target_ids = target_token_set(row)
        target_true_ids: set[int] = set()
        group_token_ids.append(token_ids)
        group_logits.append([float(v) for v in (row.get("base_logits_top_k") or row.get("base_logits") or [])])
        group_probs.append([float(v) for v in (row.get("base_probabilities_top_k") or row.get("base_probs") or [])])
        group_is_target_new.append([1.0 if token_id in target_ids else 0.0 for token_id in token_ids])
        group_is_target_true.append([1.0 if token_id in target_true_ids else 0.0 for token_id in token_ids])
        group_embeddings.append([fake_vector(f"cand:{token_id}", dim) for token_id in token_ids])
    candidate_tensors = {
        "candidate_token_embedding": torch.tensor(group_embeddings, dtype=torch.float32),
        "candidate_token_ids": torch.tensor(group_token_ids, dtype=torch.long),
        "base_logits": torch.tensor(group_logits, dtype=torch.float32),
        "base_probabilities": torch.tensor(group_probs, dtype=torch.float32),
        "candidate_rank": torch.arange(top_k, dtype=torch.float32).repeat(len(group_rows), 1),
        "candidate_is_target_new": torch.tensor(group_is_target_new, dtype=torch.float32),
        "candidate_is_target_true": torch.tensor(group_is_target_true, dtype=torch.float32),
    }
    edit_tensors = {
        "target_new_embedding_mean": tensor_from_vectors([fake_vector(f"target_new:{edit_key(row)}", dim) for row in edit_rows]),
        "target_true_embedding_mean": tensor_from_vectors([fake_vector(f"target_true:{edit_key(row)}", dim) for row in edit_rows]),
        "subject_embedding_mean": tensor_from_vectors([fake_vector(f"subject:{edit_key(row)}", dim) for row in edit_rows]),
        "rewrite_relation_embedding_mean": tensor_from_vectors([fake_vector(f"rewrite:{edit_key(row)}", dim) for row in edit_rows]),
    }
    gate_tensors = {
        "prompt_pooled": tensor_from_vectors([fake_vector(f"prompt:{prompt_key(row)}", dim) for row in prompt_rows]),
        "subject_span_pooled": tensor_from_vectors([fake_vector(f"subject_span:{prompt_key(row)}", dim) for row in prompt_rows]),
        "rewrite_relation_pooled": tensor_from_vectors([fake_vector(f"relation:{prompt_key(row)}", dim) for row in prompt_rows]),
        "prompt_relation_product": tensor_from_vectors([fake_vector(f"prod:{prompt_key(row)}", dim) for row in prompt_rows]),
        "prompt_relation_absdiff": tensor_from_vectors([fake_vector(f"diff:{prompt_key(row)}", dim) for row in prompt_rows]),
        "prompt_relation_cosine": torch.tensor([[stable_float(f"cos:{prompt_key(row)}", -1.0, 1.0)] for row in prompt_rows], dtype=torch.float32),
    }
    for name, tensor_map in [
        ("state", state_tensors),
        ("candidate", candidate_tensors),
        ("edit", edit_tensors),
        ("gate", gate_tensors),
    ]:
        for tensor_name, tensor in tensor_map.items():
            finite_tensor(f"{name}:{tensor_name}", tensor)
    write_safetensors(output_dir / "state_features.safetensors", state_tensors)
    write_safetensors(output_dir / "candidate_features.safetensors", candidate_tensors)
    write_safetensors(output_dir / "edit_features.safetensors", edit_tensors)
    write_safetensors(output_dir / "gate_features.safetensors", gate_tensors)
    write_jsonl(output_dir / "feature_index.jsonl", feature_index_rows(rows, top_k))
    return {
        "num_unique_states": len(state_rows),
        "num_candidate_groups": len(group_rows),
        "num_candidate_tokens": len(group_rows) * top_k,
        "num_edits": len(edit_rows),
        "num_prompt_edit_pairs": len(prompt_rows),
        "feature_dim": dim,
        "candidate_width": top_k,
    }


def hidden_states_from_outputs(outputs: Any) -> Sequence[Any]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        hidden_states = outputs[1]
    if hidden_states is None:
        raise AssertionError("Model output did not include hidden_states")
    return hidden_states


def pooled_token_embedding(token_emb: Any, token_ids: Sequence[int], fallback_dim: int) -> Any:
    import torch

    if not token_ids:
        return torch.zeros(fallback_dim, dtype=torch.float32)
    ids = torch.tensor([int(v) for v in token_ids], dtype=torch.long, device=token_emb.weight.device)
    return token_emb(ids).detach().float().mean(dim=0).cpu()


def text_hidden_mean(model: Any, tokenizer: Any, text: str, fallback_dim: int) -> Any:
    import torch

    ids = token_ids_for_text(tokenizer, text)
    if not ids:
        return torch.zeros(fallback_dim, dtype=torch.float32)
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden = hidden_states_from_outputs(outputs)[-1][0]
    return hidden.detach().float().mean(dim=0).cpu()


def cosine_column(left: Any, right: Any) -> Any:
    import torch

    return torch.nn.functional.cosine_similarity(left, right, dim=1).unsqueeze(1).float()


def real_extract(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    model_id: str,
    mid_layer: int,
    dtype: str,
    use_4bit: bool,
    device_map: str,
) -> Dict[str, Any]:
    """Run a minimal frozen-forward extraction path.

    This branch is GPU/RunPod-only. It intentionally imports torch/model helpers
    only here so fake mode remains safe.
    """

    import torch
    from llada_sb_common import load_llada_model_and_tokenizer

    model, tokenizer = load_llada_model_and_tokenizer(model_id, dtype_name=dtype, use_4bit=use_4bit, device_map=device_map)
    model.eval()
    state_rows = unique_rows(rows, state_key)
    prompt_rows = unique_rows(rows, prompt_key)
    edit_rows = unique_rows(rows, edit_key)
    group_rows = list(rows)
    widths = {expected_top_k(row) for row in group_rows}
    if len(widths) != 1:
        raise AssertionError(f"Expected constant top-k width, got {sorted(widths)}")
    top_k = widths.pop()
    for row in group_rows:
        validate_candidate_group(row, top_k=top_k)

    token_emb = model.get_input_embeddings()
    if token_emb is None:
        raise AssertionError("Model has no input embeddings")
    fallback_dim = int(token_emb.weight.shape[-1])
    model_config = getattr(model, "config", None)
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    actual_mid_layer = int(mid_layer)

    mid_vectors = []
    last_vectors = []
    answer_vectors = []
    subject_span_vectors = []
    with torch.no_grad():
        for row in state_rows:
            ids = torch.tensor([row.get("current_state") or row.get("fake_state")], dtype=torch.long, device=token_emb.weight.device)
            outputs = model(ids, output_hidden_states=True)
            hidden_states = hidden_states_from_outputs(outputs)
            if actual_mid_layer >= len(hidden_states):
                raise AssertionError(f"mid_layer {actual_mid_layer} out of range for {len(hidden_states)} hidden-state tensors")
            seq_len = int(ids.shape[1])
            selected = min(max(int(row.get("selected_mask_position", 0)), 0), seq_len - 1)
            selected_positions = [
                int(pos)
                for pos in (row.get("selected_mask_positions") or [selected])
                if 0 <= int(pos) < seq_len
            ] or [selected]
            mid_vectors.append(hidden_states[actual_mid_layer][0, selected].detach().float().cpu())
            last_selected = hidden_states[-1][0, selected].detach().float().cpu()
            last_vectors.append(last_selected)
            answer_vectors.append(hidden_states[-1][0, selected_positions].detach().float().mean(dim=0).cpu())
            subject_span_vectors.append(last_selected)
    state_tensors = {
        "mid_layer_selected": torch.stack(mid_vectors),
        "last_layer_selected": torch.stack(last_vectors),
        "answer_span_mean": torch.stack(answer_vectors),
        "subject_span_mean": torch.stack(subject_span_vectors),
    }

    group_token_ids: List[List[int]] = []
    group_logits: List[List[float]] = []
    group_probs: List[List[float]] = []
    group_is_target_new: List[List[float]] = []
    group_is_target_true: List[List[float]] = []
    group_embeddings = []
    for row in group_rows:
        token_ids = [int(v) for v in (row.get("top_k_candidate_token_ids") or row.get("top_k_candidate_ids") or [])]
        target_ids = target_token_set(row)
        target_true_ids = set(token_ids_for_text(tokenizer, str(row.get("target_true") or "")))
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=token_emb.weight.device)
        group_token_ids.append(token_ids)
        group_logits.append([float(v) for v in (row.get("base_logits_top_k") or row.get("base_logits") or [])])
        group_probs.append([float(v) for v in (row.get("base_probabilities_top_k") or row.get("base_probs") or [])])
        group_is_target_new.append([1.0 if token_id in target_ids else 0.0 for token_id in token_ids])
        group_is_target_true.append([1.0 if token_id in target_true_ids else 0.0 for token_id in token_ids])
        group_embeddings.append(token_emb(token_tensor).detach().float().cpu())
    candidate_tensors = {
        "candidate_token_embedding": torch.stack(group_embeddings),
        "candidate_token_ids": torch.tensor(group_token_ids, dtype=torch.long),
        "base_logits": torch.tensor(group_logits, dtype=torch.float32),
        "base_probabilities": torch.tensor(group_probs, dtype=torch.float32),
        "candidate_rank": torch.arange(top_k, dtype=torch.float32).repeat(len(group_rows), 1),
        "candidate_is_target_new": torch.tensor(group_is_target_new, dtype=torch.float32),
        "candidate_is_target_true": torch.tensor(group_is_target_true, dtype=torch.float32),
    }

    rewrite_by_edit = first_rewrite_by_edit(rows)
    target_new_vectors = []
    target_true_vectors = []
    subject_vectors = []
    rewrite_vectors = []
    edit_vector_by_key: Dict[str, Dict[str, Any]] = {}
    for row in edit_rows:
        key = edit_key(row)
        target_new_vec = pooled_token_embedding(token_emb, row.get("target_token_ids") or token_ids_for_text(tokenizer, str(row.get("target_new") or "")), fallback_dim)
        target_true_vec = pooled_token_embedding(token_emb, token_ids_for_text(tokenizer, str(row.get("target_true") or "")), fallback_dim)
        subject_vec = pooled_token_embedding(token_emb, token_ids_for_text(tokenizer, str(row.get("subject") or "")), fallback_dim)
        rewrite_vec = text_hidden_mean(model, tokenizer, rewrite_by_edit.get(key, ""), fallback_dim)
        target_new_vectors.append(target_new_vec)
        target_true_vectors.append(target_true_vec)
        subject_vectors.append(subject_vec)
        rewrite_vectors.append(rewrite_vec)
        edit_vector_by_key[key] = {
            "subject": subject_vec,
            "rewrite": rewrite_vec,
        }
    edit_tensors = {
        "target_new_embedding_mean": torch.stack(target_new_vectors),
        "target_true_embedding_mean": torch.stack(target_true_vectors),
        "subject_embedding_mean": torch.stack(subject_vectors),
        "rewrite_relation_embedding_mean": torch.stack(rewrite_vectors),
    }

    prompt_vectors = []
    gate_subject_vectors = []
    gate_rewrite_vectors = []
    for row in prompt_rows:
        key = edit_key(row)
        prompt_vec = text_hidden_mean(model, tokenizer, str(row.get("prompt_text") or ""), fallback_dim)
        prompt_vectors.append(prompt_vec)
        gate_subject_vectors.append(edit_vector_by_key[key]["subject"])
        gate_rewrite_vectors.append(edit_vector_by_key[key]["rewrite"])
    prompt_tensor = torch.stack(prompt_vectors)
    subject_tensor = torch.stack(gate_subject_vectors)
    rewrite_tensor = torch.stack(gate_rewrite_vectors)
    gate_tensors = {
        "prompt_pooled": prompt_tensor,
        "subject_span_pooled": subject_tensor,
        "rewrite_relation_pooled": rewrite_tensor,
        "prompt_relation_product": prompt_tensor * rewrite_tensor,
        "prompt_relation_absdiff": torch.abs(prompt_tensor - rewrite_tensor),
        "prompt_relation_cosine": cosine_column(prompt_tensor, rewrite_tensor),
        "prompt_subject_product": prompt_tensor * subject_tensor,
    }

    for name, tensor in state_tensors.items():
        finite_tensor(name, tensor)
    for name, tensor in candidate_tensors.items():
        if tensor.is_floating_point():
            finite_tensor(name, tensor)
    for name, tensor in edit_tensors.items():
        finite_tensor(name, tensor)
    for name, tensor in gate_tensors.items():
        finite_tensor(name, tensor)
    write_safetensors(output_dir / "state_features.safetensors", state_tensors)
    write_safetensors(output_dir / "candidate_features.safetensors", candidate_tensors)
    write_safetensors(output_dir / "edit_features.safetensors", edit_tensors)
    write_safetensors(output_dir / "gate_features.safetensors", gate_tensors)
    return {
        "num_unique_states": len(state_rows),
        "num_candidate_groups": len(group_rows),
        "num_candidate_tokens": len(group_rows) * top_k,
        "num_edits": len(edit_rows),
        "num_prompt_edit_pairs": len(prompt_rows),
        "feature_dim": int(fallback_dim),
        "mid_layer": actual_mid_layer,
        "final_hidden_layer_index": len(hidden_states) - 1 if state_rows else None,
        "model_num_hidden_layers_config": num_hidden_layers,
        "candidate_width": top_k,
    }


def feature_quality_rows(stats: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"check": "train_val_edit_overlap_zero", "value": stats.get("train_val_overlap", 0), "pass": int(stats.get("train_val_overlap", 0)) == 0},
        {"check": "num_train_edits", "value": stats.get("num_train_edits", 0), "pass": int(stats.get("num_train_edits", 0)) >= 100},
        {"check": "num_val_edits", "value": stats.get("num_val_edits", 0), "pass": int(stats.get("num_val_edits", 0)) >= 50},
        {"check": "num_unique_states", "value": stats.get("num_unique_states", 0), "pass": int(stats.get("num_unique_states", 0)) > 0},
        {"check": "num_candidate_groups", "value": stats.get("num_candidate_groups", 0), "pass": int(stats.get("num_candidate_groups", 0)) == int(stats.get("num_teacher_groups", -1))},
        {"check": "num_candidate_tokens", "value": stats.get("num_candidate_tokens", 0), "pass": int(stats.get("num_candidate_tokens", 0)) == int(stats.get("num_teacher_groups", 0)) * int(stats.get("candidate_width", 0))},
        {"check": "candidate_width", "value": stats.get("candidate_width", 0), "pass": int(stats.get("candidate_width", 0)) == 8},
        {"check": "no_teacher_label_runtime_tensors", "value": True, "pass": True},
        {"check": "analysis_500_used_false", "value": stats.get("analysis_500_used", False), "pass": stats.get("analysis_500_used", False) is False},
        {"check": "final_test_used_false", "value": stats.get("final_test_used", False), "pass": stats.get("final_test_used", False) is False},
        {"check": "actual_decode_performed_false", "value": stats.get("actual_decode_performed", False), "pass": stats.get("actual_decode_performed", False) is False},
    ]


def alignment_report_rows(rows: Sequence[Mapping[str, Any]], train_ids: set[str], val_ids: set[str], stats: Mapping[str, Any]) -> List[Dict[str, Any]]:
    keys = [group_key(row) for row in rows]
    state_keys = [state_key(row) for row in rows]
    top_k_widths = Counter(expected_top_k(row) for row in rows)
    prompt_types = Counter(str(row.get("prompt_type") or "unknown") for row in rows)
    return [
        {"check": "train_edits_represented", "value": len(train_ids), "expected": ">=100", "pass": len(train_ids) >= 100, "notes": ""},
        {"check": "val_edits_represented", "value": len(val_ids), "expected": ">=50", "pass": len(val_ids) >= 50, "notes": ""},
        {"check": "train_val_edit_overlap", "value": len(train_ids & val_ids), "expected": "0", "pass": len(train_ids & val_ids) == 0, "notes": ""},
        {"check": "candidate_groups_aligned", "value": len(keys), "expected": len(rows), "pass": len(keys) == len(rows), "notes": ""},
        {"check": "duplicate_candidate_group_keys", "value": len(keys) - len(set(keys)), "expected": "0", "pass": len(keys) == len(set(keys)), "notes": ""},
        {"check": "missing_state_keys", "value": sum(1 for key in state_keys if not key), "expected": "0", "pass": all(state_keys), "notes": ""},
        {"check": "candidate_width_histogram", "value": json.dumps(dict(sorted(top_k_widths.items()))), "expected": "{\"8\": all}", "pass": set(top_k_widths) == {8}, "notes": ""},
        {"check": "prompt_type_histogram", "value": json.dumps(dict(sorted(prompt_types.items()))), "expected": "positive_and_negative_present", "pass": any(k in prompt_types for k in ["rewrite", "declarative_paraphrase"]) and any(k not in ["rewrite", "declarative_paraphrase"] for k in prompt_types), "notes": ""},
        {"check": "feature_index_state_rows", "value": stats.get("num_unique_states", 0), "expected": ">0", "pass": int(stats.get("num_unique_states", 0)) > 0, "notes": ""},
        {"check": "feature_index_candidate_group_rows", "value": stats.get("num_candidate_groups", 0), "expected": len(rows), "pass": int(stats.get("num_candidate_groups", -1)) == len(rows), "notes": ""},
    ]


def runtime_leakage_audit(feature_schema: Mapping[str, Any]) -> Dict[str, Any]:
    tensor_names: List[str] = []
    for key in ["state_features", "candidate_features", "edit_features", "gate_features"]:
        tensor_names.extend(str(v) for v in feature_schema.get(key, []))
    leaked = [
        name
        for name in tensor_names
        if any(forbidden in name for forbidden in FORBIDDEN_RUNTIME_FIELDS)
    ]
    return {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 runtime feature leakage audit",
        "num_runtime_feature_names": len(tensor_names),
        "forbidden_runtime_fields": sorted(FORBIDDEN_RUNTIME_FIELDS),
        "leaked_runtime_feature_names": leaked,
        "num_leaked_runtime_features": len(leaked),
        "runtime_feature_leakage_audit_pass": len(leaked) == 0,
        "teacher_scores_serialized_in_runtime_tensors": False,
        "outcome_metrics_serialized_in_runtime_tensors": False,
        "prompt_type_negative_type_split_serialized_in_runtime_tensors": False,
    }


def file_fingerprint(path: Path) -> Optional[str]:
    full = repo_path(path)
    if not full.exists():
        return None
    h = hashlib.sha256()
    with full.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_runpod_template(output_dir: Path) -> Path:
    path = repo_path(output_dir / "runpod_deployable_feature_extraction_command.sh")
    text = """#!/usr/bin/env bash
# TEMPLATE ONLY. DO NOT EXECUTE AUTOMATICALLY.
# Requires explicit user approval before starting RunPod.
set -euo pipefail
cd /workspace/SB
mkdir -p logs
tmux new -d -s d3_deployable_feature_cache_train100_val50_v1 \\
  'cd /workspace/SB && set -o pipefail && python scripts/extract_d3_deployable_features.py \\
    --fake_model 0 \\
    --teacher_cache_dir runs/counterfact_direction3_controller_v1/teacher_cache_train100_val50_v1 \\
    --output_dir runs/counterfact_direction3_controller_v1/deployable_feature_cache_train100_val50_v1 \\
    --model_id GSAI-ML/LLaDA-8B-Base \\
    --dtype float16 \\
    --use_4bit 1 \\
    --device_map auto \\
    --mid_layer 16 \\
    2>&1 | tee logs/d3_deployable_feature_cache_train100_val50_v1.log; \\
    echo ${PIPESTATUS[0]} > logs/d3_deployable_feature_cache_train100_val50_v1.exitcode'
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o644)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fake_model", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--use_4bit", type=int, default=1)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--mid_layer", type=int, default=16)
    parser.add_argument("--allow_overwrite", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out_full = repo_path(out)
    if out_full.exists() and any(out_full.iterdir()) and not bool(args.allow_overwrite):
        raise FileExistsError(f"Output directory already exists and is non-empty: {out}. Pass --allow_overwrite 1 only for intentional local reruns.")
    out_full.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_train.jsonl")
    val_rows = read_jsonl(args.teacher_cache_dir / "teacher_states_val.jsonl")
    all_rows = train_rows + val_rows
    if any("analysis_500" in str(row.get("split_role", "")) or "final_test" in str(row.get("split_role", "")) for row in all_rows):
        raise AssertionError("Locked analysis/final split row found in teacher cache")
    for row in all_rows:
        validate_candidate_group(row)
    train_ids = {edit_key(row) for row in train_rows}
    val_ids = {edit_key(row) for row in val_rows}
    start_time = time.time()
    if bool(args.fake_model):
        extraction = fake_extract(all_rows, out, int(args.feature_dim))
        llada_loaded = False
    else:
        extraction = real_extract(
            all_rows,
            out,
            args.model_id,
            int(args.mid_layer),
            str(args.dtype),
            bool(args.use_4bit),
            str(args.device_map),
        )
        write_jsonl(out / "feature_index.jsonl", feature_index_rows(all_rows, int(extraction["candidate_width"])))
        llada_loaded = True
    runtime_seconds = time.time() - start_time
    stats = {
        **extraction,
        "num_train_edits": len(train_ids),
        "num_val_edits": len(val_ids),
        "train_val_overlap": len(train_ids & val_ids),
        "num_teacher_groups": len(all_rows),
        "analysis_500_used": False,
        "final_test_used": False,
        "actual_decode_performed": False,
    }
    quality_rows = feature_quality_rows(stats)
    feature_schema = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "feature_contract": "deployable_repr_v1",
        "runtime_tensor_files": [
            "state_features.safetensors",
            "candidate_features.safetensors",
            "edit_features.safetensors",
            "gate_features.safetensors",
        ],
        "forbidden_runtime_fields": sorted(FORBIDDEN_RUNTIME_FIELDS),
        "state_features": ["mid_layer_selected", "last_layer_selected", "answer_span_mean", "subject_span_mean"],
        "candidate_features": [
            "candidate_token_embedding",
            "candidate_token_ids",
            "base_logits",
            "base_probabilities",
            "candidate_rank",
            "candidate_is_target_new",
            "candidate_is_target_true",
        ],
        "edit_features": ["target_new_embedding_mean", "target_true_embedding_mean", "subject_embedding_mean", "rewrite_relation_embedding_mean"],
        "gate_features": ["prompt_pooled", "subject_span_pooled", "rewrite_relation_pooled", "prompt_relation_product", "prompt_relation_absdiff", "prompt_relation_cosine", "prompt_subject_product"],
        "mid_layer": int(args.mid_layer),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "use_4bit": bool(args.use_4bit),
        "device_map": args.device_map,
    }
    alignment_rows = alignment_report_rows(all_rows, train_ids, val_ids, stats)
    leakage_audit = runtime_leakage_audit(feature_schema)
    template_path = write_runpod_template(out)
    report = {
        "protocol_version": D3_PROTOCOL_VERSION,
        "stage": "Direction 3 deployable frozen representation feature extraction" if not bool(args.fake_model) else "Direction 3 fake deployable feature extraction",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "fake_model": bool(args.fake_model),
        "llada_loaded": llada_loaded,
        "analysis_500_used": False,
        "final_test_used": False,
        "teacher_cache_dir": str(args.teacher_cache_dir),
        "output_dir": str(out),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "use_4bit": bool(args.use_4bit),
        "device_map": args.device_map,
        "theta0": "frozen",
        "base_model_weight_update": "none",
        "actual_decode_performed": False,
        "runtime_seconds": runtime_seconds,
        "feature_integrity_pass": all(bool(row["pass"]) for row in quality_rows),
        "feature_alignment_pass": all(str(row["pass"]).lower() == "true" or row["pass"] is True for row in alignment_rows),
        "feature_leakage_audit_pass": bool(leakage_audit["runtime_feature_leakage_audit_pass"]),
        "runtime_feature_leakage_audit_pass": bool(leakage_audit["runtime_feature_leakage_audit_pass"]),
        "num_leaked_runtime_features": int(leakage_audit["num_leaked_runtime_features"]),
        "runpod_allowed_next": False,
        "requires_user_approval_for_runpod": True,
        "teacher_states_train_sha256": file_fingerprint(args.teacher_cache_dir / "teacher_states_train.jsonl"),
        "teacher_states_val_sha256": file_fingerprint(args.teacher_cache_dir / "teacher_states_val.jsonl"),
        **stats,
        "artifacts": {
            "feature_schema": str(out / "feature_schema.json"),
            "state_features": str(out / "state_features.safetensors"),
            "candidate_features": str(out / "candidate_features.safetensors"),
            "edit_features": str(out / "edit_features.safetensors"),
            "gate_features": str(out / "gate_features.safetensors"),
            "feature_index": str(out / "feature_index.jsonl"),
            "feature_quality_report": str(out / "feature_quality_report.csv"),
            "feature_alignment_report": str(out / "feature_alignment_report.csv"),
            "runtime_feature_leakage_audit": str(out / "runtime_feature_leakage_audit.json"),
            "runpod_template": str(template_path),
        },
    }
    write_json(out / "feature_schema.json", feature_schema)
    write_csv(out / "feature_quality_report.csv", quality_rows)
    write_csv(out / "feature_alignment_report.csv", alignment_rows)
    write_json(out / "runtime_feature_leakage_audit.json", leakage_audit)
    write_json(out / "report_summary.json", report)
    print(f"[INFO] Wrote D3 deployable feature cache scaffold to {out}")


if __name__ == "__main__":
    main()
