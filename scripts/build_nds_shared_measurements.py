#!/usr/bin/env python3
"""Build training/calibration-only key, Fisher, and risk-feature caches."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_editor import (
    contextual_target_ids,
    extract_keys_and_outputs,
    infer_mask_id,
    render_masked_input,
)
from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
)
from scripts.nds_editor import fit_relation_key_statistics
from scripts.nds_methods import fisher_diagonal, fisher_low_rank
from scripts.run_mdm_memit_stage import load_model


ALLOWED_SPLIT_TOKENS = ("statistics_train", "calibration")
PROTECTED_FAMILIES = ("same_subject", "near", "far", "unrelated")


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def validate_training_manifest(path: Path) -> None:
    lower = path.name.casefold()
    if not any(token in lower for token in ALLOWED_SPLIT_TOKENS):
        raise PermissionError(f"measurement manifest is not training/calibration: {path}")
    if any(token in lower for token in ("analysis_500", "final_test", "confirmation")):
        raise PermissionError(f"locked manifest is forbidden for measurements: {path}")


def relation_template_bank(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    bank: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        template = str(row.get("rewrite_template") or "").strip()
        if template and "{}" in template:
            bank[str(row.get("relation_id") or "")].add(template)
    return {key: sorted(values) for key, values in sorted(bank.items()) if values}


def build_subject_anchor_requests(
    rows: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    relations = sorted(bank)
    if len(relations) < 2:
        raise ValueError("subject anchors require at least two training relations")
    output = []
    for row in rows:
        own = str(row.get("relation_id") or "")
        alternatives = [relation for relation in relations if relation != own]
        selected = alternatives[
            int(stable_hash("subject-anchor", row["case_id"]), 16) % len(alternatives)
        ]
        templates = list(bank[selected])
        template = templates[
            int(stable_hash("subject-template", row["case_id"], selected), 16)
            % len(templates)
        ]
        prompt = template.format(str(row["subject"]))
        output.append(
            {
                **dict(row),
                "case_id": f"{row['case_id']}::subject_anchor",
                "rewrite_prompt": prompt,
                "rewrite_template": template,
                "relation_id": selected,
                "prompt_provenance": "training_relation_template_runtime_subject",
                "evaluation_prompt_used": False,
            }
        )
    return output


def _first_prompt(row: Mapping[str, Any], family: str) -> str:
    if family == "same_subject":
        values = list(row.get("same_subject_prompts") or [])
    elif family == "near":
        values = list(row.get("near_locality_prompts") or [])
    elif family == "far":
        values = [item["prompt"] for item in list(row.get("far_locality_cases") or [])]
    elif family == "unrelated":
        values = list(row.get("attribute_prompts") or []) + list(
            row.get("generation_prompts") or []
        )
    else:
        raise ValueError(f"unknown protected family: {family}")
    if not values:
        raise ValueError(f"missing {family} protected prompt for {row['case_id']}")
    return str(values[0])


def build_protected_requests(
    rows: Sequence[Mapping[str, Any]], family: str
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "case_id": f"{row['case_id']}::protected::{family}",
            "rewrite_prompt": _first_prompt(row, family),
            "prompt_provenance": f"allowed_{family}_training_prompt",
        }
        for row in rows
    ]


@torch.no_grad()
def pre_edit_rank_margin(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    mask_id = infer_mask_id(model)
    output = []
    for row in rows:
        prompt = str(row["rewrite_prompt"])
        target_ids = list(
            row.get("target_new_token_ids")
            or contextual_target_ids(tokenizer, prompt, str(row["target_new"]))
        )
        rendered = render_masked_input(tokenizer, prompt, target_ids, mask_id)
        ids = torch.tensor([rendered["input_ids"]], dtype=torch.long, device=device)
        logits = model(input_ids=ids).logits[0, rendered["answer_positions"][0]].float()
        target = int(target_ids[0])
        target_logit = float(logits[target])
        best_other = float(torch.cat((logits[:target], logits[target + 1 :])).max())
        rank = int((logits > logits[target]).sum()) + 1
        probability = float(F.softmax(logits, dim=-1)[target])
        output.append(
            {
                "case_id": row["case_id"],
                "relation_id": row.get("relation_id"),
                "target_length": row.get("target_length"),
                "base_target_rank": rank,
                "base_target_margin": target_logit - best_other,
                "base_target_probability": probability,
            }
        )
    return output


def fake_keys(rows: Sequence[Mapping[str, Any]], layer: int, width: int = 16) -> torch.Tensor:
    values = []
    for row in rows:
        seed = int(stable_hash("fake-key", layer, row["case_id"]), 16) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        values.append(torch.randn(width, generator=generator))
    return torch.stack(values)


def _extract(
    model: torch.nn.Module | None,
    tokenizer: Any | None,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    *,
    fake_model: bool,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fake_model:
        keys = fake_keys(rows, layer)
        return keys, keys[:, :8]
    assert model is not None and tokenizer is not None
    return extract_keys_and_outputs(
        model,
        tokenizer,
        rows,
        key_layer=layer,
        output_layer=layer,
        batch_size=batch_size,
        partial_mask_schedule="cycle",
        reveal_policy="random",
        seed=260719201,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--statistics_manifest",
        type=Path,
        default=PROTOCOL_ROOT / "cf_nds_statistics_train_500.jsonl",
    )
    parser.add_argument(
        "--calibration_manifest",
        type=Path,
        default=PROTOCOL_ROOT / "cf_nds_calibration_200.jsonl",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "S1_shared_measurements_v1"
    )
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--fisher_rank", type=int, default=64)
    parser.add_argument("--fisher_damping", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fake_model", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    validate_training_manifest(args.statistics_manifest)
    validate_training_manifest(args.calibration_manifest)
    args.output_dir.mkdir(parents=True)
    splits = {
        "statistics_train": read_jsonl(args.statistics_manifest),
        "calibration": read_jsonl(args.calibration_manifest),
    }
    if not all(splits.values()):
        raise RuntimeError("measurement manifests must be nonempty")
    bank = relation_template_bank(splits["statistics_train"])
    model = tokenizer = None
    if not args.fake_model:
        model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, "float16")
    feature_rows: list[dict[str, Any]] = []
    cache_index: dict[str, Any] = {}
    for split_name, rows in splits.items():
        split_dir = args.output_dir / split_name
        split_dir.mkdir()
        anchors = build_subject_anchor_requests(rows, bank)
        cache_index[split_name] = {}
        layer_keys = []
        for layer in args.layers:
            edit_keys, outputs = _extract(
                model,
                tokenizer,
                rows,
                layer,
                fake_model=bool(args.fake_model),
                batch_size=args.batch_size,
            )
            anchor_keys, _ = _extract(
                model,
                tokenizer,
                anchors,
                layer,
                fake_model=bool(args.fake_model),
                batch_size=args.batch_size,
            )
            protected = {}
            for family in PROTECTED_FAMILIES:
                requests = build_protected_requests(rows, family)
                keys, _ = _extract(
                    model,
                    tokenizer,
                    requests,
                    layer,
                    fake_model=bool(args.fake_model),
                    batch_size=args.batch_size,
                )
                protected[family] = keys.float().cpu()
            combined = torch.cat(list(protected.values()), dim=0)
            relation_stats = fit_relation_key_statistics(
                edit_keys, [str(row["relation_id"]) for row in rows]
            )
            fisher_diag = fisher_diagonal(combined, args.fisher_damping)
            low_rank = fisher_low_rank(combined, args.fisher_rank, args.fisher_damping)
            covariance = edit_keys.float().var(dim=0, unbiased=False).clamp_min(1e-4)
            payload = {
                "case_ids": [str(row["case_id"]) for row in rows],
                "relation_ids": [str(row["relation_id"]) for row in rows],
                "edit_keys": edit_keys.float().cpu(),
                "subject_anchor_keys": anchor_keys.float().cpu(),
                "outputs": outputs.float().cpu(),
                "protected_keys": protected,
                "covariance_diagonal": covariance.cpu(),
                "fisher_diagonal": fisher_diag.cpu(),
                "fisher_basis": low_rank["basis"].cpu(),
                "fisher_eigenvalues": low_rank["eigenvalues"].cpu(),
                "fisher_damping": float(low_rank["damping"]),
                "relation_global_mean": relation_stats.global_mean.cpu(),
                "relation_means": {
                    key: value.cpu() for key, value in relation_stats.relation_means.items()
                },
            }
            path = split_dir / f"layer_{layer}_measurements.pt"
            torch.save(payload, path)
            cache_index[split_name][str(layer)] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "num_edits": len(rows),
                "key_width": int(edit_keys.shape[1]),
                "fisher_rank": int(low_rank["basis"].shape[1]),
                "protected_rows": int(combined.shape[0]),
            }
            layer_keys.append(edit_keys.float())
        if args.fake_model:
            rank_margin = [
                {
                    "case_id": row["case_id"],
                    "relation_id": row["relation_id"],
                    "target_length": row["target_length"],
                    "base_target_rank": index + 1,
                    "base_target_margin": -float(index % 7),
                    "base_target_probability": 1.0 / (index + 2),
                }
                for index, row in enumerate(rows)
            ]
        else:
            assert model is not None and tokenizer is not None
            rank_margin = pre_edit_rank_margin(model, tokenizer, rows)
        for index, row in enumerate(rank_margin):
            stability = torch.stack([keys[index] for keys in layer_keys]).float().std(dim=0).mean()
            feature_rows.append(
                {
                    **row,
                    "split_role": split_name,
                    "causal_site_stability": float(1.0 / (1.0 + stability)),
                    "mean_key_norm": float(
                        torch.stack([keys[index].norm() for keys in layer_keys]).mean()
                    ),
                    "runtime_feature_schema": json.dumps(
                        [
                            "base_target_rank",
                            "base_target_margin",
                            "base_target_probability",
                            "causal_site_stability",
                            "mean_key_norm",
                            "target_length",
                        ]
                    ),
                }
            )
    write_csv(args.output_dir / "pre_edit_features.csv", feature_rows)
    write_json(args.output_dir / "relation_template_bank.json", bank)
    write_json(args.output_dir / "cache_index.json", cache_index)
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "S1_shared_measurements",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": PRIMARY_MODEL_ID,
        "model_revision": PRIMARY_MODEL_REVISION,
        "layers": list(args.layers),
        "statistics_edits": len(splits["statistics_train"]),
        "calibration_edits": len(splits["calibration"]),
        "protected_families": list(PROTECTED_FAMILIES),
        "relation_count": len(bank),
        "runtime_feature_schema": [
            "base_target_rank",
            "base_target_margin",
            "base_target_probability",
            "causal_site_stability",
            "mean_key_norm",
            "target_length",
        ],
        "teacher_only_runtime_inputs": False,
        "evaluation_outcome_runtime_inputs": False,
        "evaluation_prompts_used_for_subject_anchors": False,
        "fake_model": bool(args.fake_model),
        "llada_loaded": not bool(args.fake_model),
        "analysis_500_used": False,
        "final_test_used": False,
        "runtime_seconds": runtime,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if not args.fake_model else None,
            "gpu": torch.cuda.get_device_name(0) if not args.fake_model else None,
        },
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "run_config.json",
        {
            "statistics_manifest": str(args.statistics_manifest),
            "statistics_manifest_sha256": sha256_file(args.statistics_manifest),
            "calibration_manifest": str(args.calibration_manifest),
            "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
            "layers": list(args.layers),
            "fisher_rank": args.fisher_rank,
            "fisher_damping": args.fisher_damping,
            "fake_model": bool(args.fake_model),
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    record_stage(
        "S1_shared_measurements",
        status="passed",
        acceptance_pass=True,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"Cached training/calibration keys and Fisher sketches for {len(args.layers)} layers.",
        next_stage="N1_pilot",
    )
    print(json.dumps({"acceptance_pass": True, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
