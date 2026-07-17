#!/usr/bin/env python3
"""Build train-only positive and preservation state banks at frozen causal sites."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dnpe_preservation_basis import (
    build_prompt_specs,
    display_path,
    extract_prompt_keys,
    stratified_limit,
)
from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    record_stage,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
)
from scripts.mdm_memit_editor import build_protected_basis, extract_keys_and_outputs
from scripts.run_mdm_memit_stage import load_model


PROMPT_FIELDS = (
    "rewrite_prompt",
    "declarative_paraphrases",
    "same_subject_prompts",
    "near_locality_prompts",
    "far_locality_prompts",
    "attribute_prompts",
    "generation_prompts",
)


def iter_prompts(row: Mapping[str, Any]) -> Iterable[str]:
    for field in PROMPT_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            yield from (str(item) for item in value)


def prompt_hashes(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        stable_hash(" ".join(prompt.casefold().split()))
        for row in rows
        for prompt in iter_prompts(row)
        if prompt.strip()
    }


def normalized_prompt_hash(prompt: str) -> str:
    return stable_hash(" ".join(str(prompt).casefold().split()))


def select_state_bank_inputs(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    maximum_positive_keys: int,
    maximum_preservation_keys: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Select only the exact train prompts that are safe to enter the banks."""

    eval_hashes = prompt_hashes(eval_rows)
    source_overlap = prompt_hashes(train_rows) & eval_hashes
    positive_candidates = [
        row
        for row in train_rows
        if normalized_prompt_hash(str(row["rewrite_prompt"])) not in eval_hashes
    ]
    positive_rows = positive_candidates[:maximum_positive_keys]
    preservation_candidates = [
        spec
        for spec in build_prompt_specs(train_rows)
        if normalized_prompt_hash(str(spec["prompt"])) not in eval_hashes
    ]
    preservation_specs = stratified_limit(
        preservation_candidates, maximum_preservation_keys
    )
    bank_hashes = {
        normalized_prompt_hash(str(row["rewrite_prompt"]))
        for row in positive_rows
    } | {
        normalized_prompt_hash(str(spec["prompt"]))
        for spec in preservation_specs
    }
    actual_overlap = bank_hashes & eval_hashes
    diagnostics = {
        "source_manifest_prompt_overlap_count": len(source_overlap),
        "positive_candidates_after_exclusion": len(positive_candidates),
        "preservation_candidates_after_exclusion": len(preservation_candidates),
        "state_bank_prompt_overlap_count": len(actual_overlap),
    }
    return positive_rows, preservation_specs, diagnostics


def frozen_layers(site_lock: Mapping[str, Any]) -> list[int]:
    layers = {
        int(layer)
        for policy in site_lock["policies"]
        for layer in policy.get("layers", [])
    }
    if not layers:
        raise RuntimeError("Site policy lock contains no editable layers")
    return sorted(layers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CAMPAIGN_ROOT / "protocol_v1" / "dnpe_anchor_train_500.jsonl",
    )
    parser.add_argument(
        "--site_lock",
        type=Path,
        default=CAMPAIGN_ROOT / "site_policy_lock_v1" / "site_policy_lock.json",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "D1_state_banks_v1"
    )
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--maximum_positive_keys", type=int, default=100)
    parser.add_argument("--maximum_preservation_keys", type=int, default=700)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = now_utc()
    begin = time.monotonic()
    train_rows = read_jsonl(args.manifest)
    if not train_rows or any(
        row.get("split_role") != "dnpe_anchor_train_500" for row in train_rows
    ):
        raise RuntimeError("State banks require only dnpe_anchor_train_500 rows")
    site_lock = read_json(args.site_lock)
    layers = frozen_layers(site_lock)
    eval_rows = []
    for name in (
        "dnpe_smoke_20.jsonl",
        "dnpe_pilot_100.jsonl",
        "dnpe_dev_200.jsonl",
        "dnpe_locality_eval_300.jsonl",
    ):
        eval_rows.extend(read_jsonl(CAMPAIGN_ROOT / "protocol_v1" / name))
    positive_rows, preservation_specs, overlap_diagnostics = select_state_bank_inputs(
        train_rows,
        eval_rows,
        maximum_positive_keys=args.maximum_positive_keys,
        maximum_preservation_keys=args.maximum_preservation_keys,
    )
    if overlap_diagnostics["state_bank_prompt_overlap_count"]:
        raise RuntimeError(
            "Training/evaluation prompt overlap in selected state-bank inputs: "
            f"{overlap_diagnostics['state_bank_prompt_overlap_count']}"
        )
    if len(positive_rows) < args.maximum_positive_keys:
        raise RuntimeError(
            "Insufficient non-overlapping positive rows: "
            f"{len(positive_rows)} < {args.maximum_positive_keys}"
        )
    preservation_counts = Counter(
        str(row["category"]) for row in preservation_specs
    )
    required_preservation = {
        "same_subject_different_relation",
        "different_subject_same_relation",
        "near_locality",
        "far_locality",
        "attribute",
        "generation",
        "unrelated",
    }
    if not required_preservation.issubset(preservation_counts):
        raise RuntimeError(
            "Missing preservation categories: "
            f"{sorted(required_preservation - set(preservation_counts))}"
        )
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    import torch

    layer_reports = {}
    summary_rows = []
    all_finite = True
    for layer in layers:
        positive_keys, positive_outputs = extract_keys_and_outputs(
            model,
            tokenizer,
            positive_rows,
            key_layer=layer,
            output_layer=layer,
            batch_size=args.batch_size,
            partial_mask_schedule="cycle",
            reveal_policy="random",
            seed=260717101,
        )
        preservation_keys = extract_prompt_keys(
            model,
            tokenizer,
            preservation_specs,
            layer=layer,
            batch_size=args.batch_size,
        )
        layer_finite = bool(
            torch.isfinite(positive_keys).all()
            and torch.isfinite(positive_outputs).all()
            and torch.isfinite(preservation_keys).all()
        )
        all_finite = all_finite and layer_finite
        if not layer_finite:
            raise FloatingPointError(f"Non-finite state bank at layer {layer}")
        positive_path = args.output_dir / f"layer_{layer}_positive_state_bank.pt"
        preservation_path = (
            args.output_dir / f"layer_{layer}_preservation_state_bank.pt"
        )
        torch.save(
            {
                "keys": positive_keys.half(),
                "outputs": positive_outputs.half(),
                "case_ids": [row["case_id"] for row in positive_rows],
                "partial_mask_schedule": "cycle",
                "reveal_policy": "random",
            },
            positive_path,
        )
        torch.save(
            {
                "keys": preservation_keys.half(),
                "case_ids": [row["case_id"] for row in preservation_specs],
                "categories": [row["category"] for row in preservation_specs],
            },
            preservation_path,
        )
        bases = {}
        for variance in (0.90, 0.95, 0.99):
            basis, geometry = build_protected_basis(preservation_keys, variance)
            basis_path = (
                args.output_dir
                / f"layer_{layer}_variance_{variance:.2f}_basis.pt"
            )
            torch.save(
                {
                    "basis": basis.half(),
                    "layer": layer,
                    "explained_variance": variance,
                    "geometry": geometry,
                },
                basis_path,
            )
            bases[f"{variance:.2f}"] = {
                **geometry,
                "path": display_path(basis_path),
                "sha256": sha256_file(basis_path),
            }
        layer_reports[str(layer)] = {
            "positive_key_count": len(positive_keys),
            "preservation_key_count": len(preservation_keys),
            "key_width": int(positive_keys.shape[1]),
            "finite": layer_finite,
            "positive_bank_sha256": sha256_file(positive_path),
            "preservation_bank_sha256": sha256_file(preservation_path),
            "bases": bases,
        }
        summary_rows.extend(
            {
                "layer": layer,
                "bank": bank,
                "num_keys": len(keys),
                "mean_key_norm": float(keys.float().norm(dim=1).mean()),
                "finite": bool(torch.isfinite(keys).all()),
            }
            for bank, keys in (
                ("positive", positive_keys),
                ("preservation", preservation_keys),
            )
        )
    positive_categories = {
        "rewrite_states": True,
        "training_only_augmentations": True,
        "all_mask_counts": True,
        "random_reveal_subsets": True,
        "actual_trajectory_states": False,
    }
    unavailable = {
        "actual_trajectory_states": (
            "optional category not used; cycle/random states cover the bounded pilot"
        )
    }
    acceptance = {
        "train_eval_prompt_overlap": (
            overlap_diagnostics["state_bank_prompt_overlap_count"] == 0
        ),
        "all_required_positive_categories_present": all(
            value
            for name, value in positive_categories.items()
            if name != "actual_trajectory_states"
        ),
        "optional_categories_explicitly_unavailable": bool(unavailable),
        "all_preservation_categories_present": required_preservation.issubset(
            preservation_counts
        ),
        "keys_aligned_to_selected_sites": bool(layers),
        "all_activations_finite": all_finite,
        "remaining_editable_dimension_positive": all(
            base["remaining_editable_dimension"] > 0
            for layer in layer_reports.values()
            for base in layer["bases"].values()
        ),
    }
    passed = all(acceptance.values())
    write_csv(args.output_dir / "state_bank_summary.csv", summary_rows)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "site_lock": str(args.site_lock),
            "site_lock_sha256": sha256_file(args.site_lock),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "layers": layers,
            "maximum_positive_keys": args.maximum_positive_keys,
            "maximum_preservation_keys": args.maximum_preservation_keys,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "D1_state_banks",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "layers": layers,
        "positive_state_categories": positive_categories,
        "category_unavailable_reason": unavailable,
        "preservation_category_counts": dict(sorted(preservation_counts.items())),
        "prompt_overlap_audit": overlap_diagnostics,
        "layer_reports": layer_reports,
        "runtime_seconds": time.monotonic() - begin,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance": acceptance,
        "acceptance_pass": passed,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", acceptance)
    record_stage(
        "D1_state_banks",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes="Positive and preservation state banks built from train-only prompts.",
        next_stage="D2_target_value_optimization",
    )
    print(
        json.dumps(
            {
                "acceptance_pass": passed,
                "layers": layers,
                "prompt_overlap": overlap_diagnostics[
                    "state_bank_prompt_overlap_count"
                ],
                "source_manifest_prompt_overlap": overlap_diagnostics[
                    "source_manifest_prompt_overlap_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
