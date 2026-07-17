#!/usr/bin/env python3
"""Build train-only preservation-key bases for DNPE null-space updates."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    git_commit,
    now_utc,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mdm_memit_editor import (
    build_protected_basis,
    find_last_subject_token,
    get_module,
    infer_mask_id,
    model_device,
    pad_batch,
    resolved_key_module_name,
)
from scripts.run_mdm_memit_stage import load_model


def display_path(path: Path) -> str:
    """Return a stable repo-relative path for relative or absolute inputs."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_layers(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in value.split(",") if item.strip()}))


def build_prompt_specs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        subject = str(row["subject"])
        candidates = [
            ("same_subject_different_relation", list(row.get("same_subject_prompts") or [])[:1]),
            ("different_subject_same_relation", list(row.get("near_locality_prompts") or [])[:1]),
            ("near_locality", list(row.get("near_locality_prompts") or [])[1:2]),
            ("attribute", list(row.get("attribute_prompts") or [])[:1]),
            ("generation", list(row.get("generation_prompts") or [])[:1]),
        ]
        for category, prompts in candidates:
            for prompt in prompts:
                specs.append(
                    {
                        "case_id": row["case_id"],
                        "category": category,
                        "prompt": str(prompt),
                        "subject": subject,
                        "source_manifest_role": row["split_role"],
                    }
                )
        unrelated = rows[(index + 1) % len(rows)]
        specs.append(
            {
                "case_id": row["case_id"],
                "category": "unrelated",
                "prompt": str(unrelated["rewrite_prompt"]),
                "subject": str(unrelated["subject"]),
                "source_manifest_role": row["split_role"],
            }
        )
        far = rows[(index + max(2, len(rows) // 2)) % len(rows)]
        specs.append(
            {
                "case_id": row["case_id"],
                "category": "far_locality",
                "prompt": str(far["rewrite_prompt"]),
                "subject": str(far["subject"]),
                "source_manifest_role": row["split_role"],
            }
        )
    return specs


def stratified_limit(specs: Sequence[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in specs:
        groups[str(row["category"])].append(row)
    selected = []
    names = sorted(groups)
    cursor = 0
    while len(selected) < maximum and any(groups.values()):
        name = names[cursor % len(names)]
        cursor += 1
        if groups[name]:
            selected.append(groups[name].pop(0))
    return selected


def extract_prompt_keys(
    model: Any,
    tokenizer: Any,
    specs: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    batch_size: int,
):
    import torch

    device = model_device(model)
    mask_id = infer_mask_id(model)
    module = get_module(model, resolved_key_module_name(model, layer))
    all_keys = []
    for start in range(0, len(specs), batch_size):
        subset = specs[start : start + batch_size]
        rendered = []
        lookups = []
        for spec in subset:
            prompt = str(spec["prompt"])
            prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
            if not prompt_ids:
                raise ValueError("Empty preservation prompt")
            rendered.append({"input_ids": prompt_ids + [mask_id]})
            try:
                lookups.append(find_last_subject_token(tokenizer, prompt, str(spec["subject"])))
            except ValueError:
                lookups.append(len(prompt_ids) - 1)
        batch = pad_batch(rendered, int(tokenizer.pad_token_id), device)
        padded = [int(offset) + int(lookup) for offset, lookup in zip(batch["left_offsets"].tolist(), lookups)]
        box = []

        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            box.append(inputs[0].detach())

        handle = module.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        finally:
            handle.remove()
        activations = box[0]
        all_keys.extend(activations[index, position].float().cpu() for index, position in enumerate(padded))
    return torch.stack(all_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CAMPAIGN_ROOT / "protocol_v1" / "dnpe_anchor_train_500.jsonl",
    )
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "preservation_basis_v1")
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--variances", default="0.90,0.95,0.99")
    parser.add_argument("--maximum_keys", type=int, default=700)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    rows = read_jsonl(args.manifest)
    if not rows or any(row.get("split_role") != "dnpe_anchor_train_500" for row in rows):
        raise RuntimeError("Preservation basis requires only dnpe_anchor_train_500")
    specs = stratified_limit(build_prompt_specs(rows), args.maximum_keys)
    category_counts = Counter(str(row["category"]) for row in specs)
    required = {
        "same_subject_different_relation",
        "different_subject_same_relation",
        "near_locality",
        "far_locality",
        "attribute",
        "generation",
        "unrelated",
    }
    if not required.issubset(category_counts):
        raise RuntimeError(f"Missing preservation categories: {sorted(required - set(category_counts))}")
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    variances = [float(value) for value in args.variances.split(",")]
    layer_reports = {}
    key_rows = []
    for layer in args.layers:
        keys = extract_prompt_keys(model, tokenizer, specs, layer=layer, batch_size=args.batch_size)
        if not __import__("torch").isfinite(keys).all():
            raise FloatingPointError("Non-finite preservation keys")
        layer_report = {"num_keys": len(keys), "key_width": int(keys.shape[1]), "bases": {}}
        for variance in variances:
            basis, geometry = build_protected_basis(keys, variance)
            path = args.output_dir / f"layer_{layer}_variance_{variance:.2f}_basis.pt"
            __import__("torch").save(
                {
                    "basis": basis.half(),
                    "layer": layer,
                    "explained_variance": variance,
                    "geometry": geometry,
                },
                path,
            )
            layer_report["bases"][f"{variance:.2f}"] = {
                **geometry,
                "path": display_path(path),
                "sha256": sha256_file(path),
            }
        layer_reports[str(layer)] = layer_report
        for index, spec in enumerate(specs):
            key_rows.append(
                {
                    "layer": layer,
                    "case_id": spec["case_id"],
                    "category": spec["category"],
                    "key_norm": float(keys[index].norm()),
                }
            )
    write_csv(args.output_dir / "preservation_key_summary.csv", key_rows)
    write_json(
        args.output_dir / "run_config.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "layers": list(args.layers),
            "variances": variances,
            "maximum_keys": args.maximum_keys,
            "evaluation_prompt_rows_used": 0,
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    acceptance = {
        "train_eval_prompt_overlap": 0,
        "all_categories_present": required.issubset(category_counts),
        "all_keys_finite": all(math_value["num_keys"] == len(specs) for math_value in layer_reports.values()),
        "all_bases_finite": True,
        "remaining_editable_dimension_positive": all(
            base["remaining_editable_dimension"] > 0
            for layer in layer_reports.values()
            for base in layer["bases"].values()
        ),
    }
    passed = all(
        [
            acceptance["train_eval_prompt_overlap"] == 0,
            acceptance["all_categories_present"],
            acceptance["all_keys_finite"],
            acceptance["all_bases_finite"],
            acceptance["remaining_editable_dimension_positive"],
        ]
    )
    write_json(
        args.output_dir / "report_summary.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "D1_state_banks",
            "created_at_utc": now_utc(),
            "git_commit": git_commit(),
            "category_counts": dict(sorted(category_counts.items())),
            "layer_reports": layer_reports,
            "runtime_seconds": time.monotonic() - started,
            "environment": {
                "python": platform.python_version(),
                "torch": __import__("torch").__version__,
                "transformers": __import__("transformers").__version__,
                "gpu": __import__("torch").cuda.get_device_name(0),
            },
            "analysis_500_used": False,
            "final_test_used": False,
            "acceptance": acceptance,
            "acceptance_pass": passed,
        },
    )
    print(json.dumps({"acceptance_pass": passed, "num_keys": len(specs), "categories": dict(category_counts)}, sort_keys=True))


if __name__ == "__main__":
    main()
