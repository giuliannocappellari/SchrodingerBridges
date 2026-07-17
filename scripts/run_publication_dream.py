#!/usr/bin/env python3
"""P5 Dream integration, development, and fresh locked confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mask_pattern_publication_runtime import (
    PlannerSpec,
    build_full_cost_tables,
    build_prompt_items,
    decode_with_planner,
    planner_spec_from_label,
)
from scripts.mask_pattern_publication_stats import paired_bootstrap, paired_values
from scripts.mdm_memit_editor import (
    MemitConfig,
    apply_memit_batch,
    find_last_subject_token,
    get_module,
    infer_mask_id,
    pad_batch,
    render_masked_input,
)
from scripts.run_mdm_memit_stage import load_model
from scripts.run_publication_locked_confirmation import (
    GENERATION_SEEDS,
    RANDOM_SEEDS,
    _aggregate,
    _attach_base,
    _mean_delta,
    _seed_rows,
)


LAYERS = (4, 5, 6, 7)
BLOCK_TEMPLATE = "model.layers.{layer}"
KEY_TEMPLATE = "model.layers.{layer}.mlp.down_proj"
REFERENCES = ("uniform", "edited_target_confidence", "edited_max_confidence")


def _module_map(model: Any) -> dict[str, Any]:
    layers = []
    for layer in LAYERS:
        block_name = BLOCK_TEMPLATE.format(layer=layer)
        key_name = KEY_TEMPLATE.format(layer=layer)
        block = get_module(model, block_name)
        key = get_module(model, key_name)
        layers.append(
            {
                "layer": layer,
                "block_module": block_name,
                "block_type": type(block).__name__,
                "key_module": key_name,
                "key_type": type(key).__name__,
                "editable_weight_shape": list(key.weight.shape),
                "editable_weight_dtype": str(key.weight.dtype),
                "editable_weight_floating_point": bool(key.weight.is_floating_point()),
            }
        )
    return {
        "model_id": SECONDARY_MODEL_ID,
        "model_revision": SECONDARY_MODEL_REVISION,
        "architecture": type(model).__name__,
        "mask_token_id": infer_mask_id(model),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "block_module_template": BLOCK_TEMPLATE,
        "key_module_template": KEY_TEMPLATE,
        "module_mapping_repair": (
            "Use the architecture-faithful MLP down projection with a positive diagonal "
            "second-moment covariance and an exact Woodbury solve; a dense intermediate-width "
            "covariance is computationally infeasible."
        ),
        "covariance_representation": "positive_diagonal_second_moment",
        "linear_solve": "exact_woodbury_for_diagonal_plus_low_rank_edit_keys",
        "bounded_module_mapping_repair_used": True,
        "layers": layers,
        "acceptance_pass": all(row["editable_weight_floating_point"] for row in layers),
    }


def _build_covariance_cache(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    rendered = []
    lookups = []
    mask_id = infer_mask_id(model)
    for row in rows:
        prompt = str(row["rewrite_prompt"])
        target_ids = list(map(int, row["target_new_token_ids"]))
        rendered.append(render_masked_input(tokenizer, prompt, target_ids, mask_id))
        lookups.append(find_last_subject_token(tokenizer, prompt, str(row["subject"])))
    reports = []
    for layer in LAYERS:
        module_name = KEY_TEMPLATE.format(layer=layer)
        module = get_module(model, module_name)
        width = int(module.weight.shape[1])
        accumulator = torch.zeros(width, dtype=torch.float64, device=device)
        count = 0
        for start in range(0, len(rendered), batch_size):
            subset = rendered[start : start + batch_size]
            batch = pad_batch(subset, int(tokenizer.pad_token_id), device)
            offsets = batch["left_offsets"].tolist()
            capture: list[Any] = []

            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                capture.append(inputs[0])

            handle = module.register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
            finally:
                handle.remove()
            hidden = capture[0]
            vectors = torch.stack(
                [
                    hidden[index, int(offsets[index]) + int(lookups[start + index])]
                    .float()
                    .to(torch.float64)
                    for index in range(len(subset))
                ]
            )
            accumulator.add_(vectors.square().sum(dim=0))
            count += vectors.shape[0]
        covariance = (accumulator / max(count, 1)).float()
        ridge = max(1e-5, float(covariance.mean()) * 1e-4)
        covariance.add_(ridge)
        path = output_dir / f"layer_{layer}_covariance.pt"
        torch.save(covariance.cpu(), path)
        reports.append(
            {
                "layer": layer,
                "module": module_name,
                "num_contexts": count,
                "key_width": width,
                "ridge": ridge,
                "sha256": sha256_file(path),
            }
        )
        del accumulator, covariance
        torch.cuda.empty_cache()
    write_csv(output_dir / "covariance_summary.csv", reports)
    return reports


def _load_covariance(cache_dir: Path, layer: int):
    import torch

    path = cache_dir / f"layer_{layer}_covariance.pt"
    return torch.load(path, map_location="cpu", weights_only=True).to("cuda")


def _partial_config(llada_lock: Mapping[str, Any]) -> MemitConfig:
    editor = llada_lock["editor"]
    return MemitConfig(
        layers=LAYERS,
        learning_rate=float(llada_lock["target_value_config"]["learning_rate"]),
        target_optimization_steps=int(llada_lock["target_value_config"]["steps"]),
        clamp_norm_factor=float(llada_lock["target_value_config"]["clamp_norm_factor"]),
        kl_factor=float(llada_lock["target_value_config"]["kl_factor"]),
        partial_mask_schedule=str(editor["partial_mask_schedule"]),
        reveal_policy=str(editor["reveal_policy"]),
        seed=260_717_810,
        block_module_template=BLOCK_TEMPLATE,
        key_module_template=KEY_TEMPLATE,
    )


def _planner_specs(
    *,
    stage: str,
    n: int,
    llada_lock: Mapping[str, Any],
    dream_lock: Mapping[str, Any] | None,
) -> list[PlannerSpec]:
    beta = float(llada_lock["beta"])
    fixed = llada_lock["fixed_orders"].get(str(n), list(range(n)))
    non_sb = str(llada_lock["best_non_sb_planner"])
    specs = [
        PlannerSpec("dream_default", "default_confidence"),
        PlannerSpec("one_step_myopic", "myopic"),
        PlannerSpec("deterministic_global", "deterministic_global"),
    ]
    if non_sb == "uniform_random":
        specs.extend(
            PlannerSpec(f"uniform_random_seed{seed}", "uniform_random", seed=seed)
            for seed in RANDOM_SEEDS
        )
    elif non_sb not in {"one_step_myopic", "deterministic_global"}:
        specs.append(
            planner_spec_from_label(non_sb, n=n, fixed_order=fixed, seed=260_717_811)
        )
    if stage in {"smoke", "dev"}:
        specs.extend(
            PlannerSpec(
                f"finite_{reference}_beta{beta:g}",
                "finite_beta",
                beta=beta,
                reference=reference,
            )
            for reference in REFERENCES
        )
    else:
        if dream_lock is None:
            raise RuntimeError("Dream locked run requires a validated Dream dev lock")
        specs.append(
            PlannerSpec(
                str(dream_lock["finite_controller_label"]),
                "finite_beta",
                beta=float(dream_lock["beta"]),
                reference=str(dream_lock["reference_process"]),
            )
        )
    return specs


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregate = _aggregate(rows)
    output = []
    for family in sorted({str(row["family"]) for row in aggregate}):
        selected = [row for row in aggregate if row["family"] == family]
        for bucket in ("rewrite", "paraphrase", "same_subject_stress", "far_locality"):
            values = [row for row in selected if row["bucket"] == bucket]
            if not values:
                continue
            output.append(
                {
                    "family": family,
                    "bucket": bucket,
                    "num_edits": sum(int(row["num_edits"]) for row in values),
                    "exact": sum(float(row["full_target_exact"]) * int(row["num_edits"]) for row in values)
                    / sum(int(row["num_edits"]) for row in values),
                    "target_token_f1": sum(float(row["target_token_f1"]) * int(row["num_edits"]) for row in values)
                    / sum(int(row["num_edits"]) for row in values),
                    "malformed_rate": max(float(row["malformed"]) for row in values),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "dev", "locked"), required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--limit_per_length", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    defaults = {
        "smoke": CAMPAIGN_ROOT / "dream_integration_smoke_v1",
        "dev": CAMPAIGN_ROOT / "dream_confirmation_dev_v1",
        "locked": CAMPAIGN_ROOT / "dream_confirmation_v1",
    }
    args.output_dir = args.output_dir or defaults[args.stage]
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    if args.output_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    wall_start = time.monotonic()
    llada_lock_path = CAMPAIGN_ROOT / "dev_method_lock.json"
    llada_lock = read_json(llada_lock_path)
    if not llada_lock.get("validation_pass"):
        raise RuntimeError("Dream track requires the frozen LLaDA dev lock")
    dream_lock_path = CAMPAIGN_ROOT / "dream_confirmation_dev_v1" / "dream_dev_lock.json"
    dream_lock = read_json(dream_lock_path) if args.stage == "locked" else None
    if dream_lock is not None and not dream_lock.get("validation_pass"):
        raise RuntimeError("Dream dev lock did not validate")
    prefix = "dream_pub_locked" if args.stage == "locked" else "dream_pub_dev"
    manifests = {n: PROTOCOL_ROOT / f"{prefix}_n{n}.jsonl" for n in (3, 4, 5)}
    protocol = read_json(PROTOCOL_ROOT / "report_summary.json")
    expected_hashes = {
        n: protocol["splits"][f"{prefix}_n{n}"]["sha256"] for n in manifests
    }
    for n, path in manifests.items():
        if sha256_file(path) != expected_hashes[n]:
            raise RuntimeError(f"Dream manifest hash mismatch at N={n}")
    if args.stage == "locked":
        write_json(
            args.output_dir / "locked_open_record.json",
            {
                "dream_dev_lock_sha256": sha256_file(dream_lock_path),
                "manifest_hashes": expected_hashes,
                "selection_complete_before_open": True,
            },
        )

    model, tokenizer = load_model(SECONDARY_MODEL_ID, SECONDARY_MODEL_REVISION, args.dtype)
    module_map = _module_map(model)
    if not module_map["acceptance_pass"]:
        raise RuntimeError("Dream module mapping failed")
    write_json(args.output_dir / "model_module_map.json", module_map)
    selected_rows: dict[int, list[dict[str, Any]]] = {}
    for n, manifest in manifests.items():
        rows = read_jsonl(manifest)
        limit = args.limit_per_length or (5 if args.stage == "smoke" else 0)
        selected_rows[n] = rows[:limit] if limit else rows

    if args.stage == "locked":
        covariance_dir = CAMPAIGN_ROOT / "dream_confirmation_dev_v1" / "covariance_cache"
        covariance_reports = list(
            csv.DictReader((covariance_dir / "covariance_summary.csv").open(newline=""))
        )
    else:
        covariance_dir = args.output_dir / "covariance_cache"
        covariance_reports = _build_covariance_cache(
            model,
            tokenizer,
            [row for n in (3, 4, 5) for row in selected_rows[n]],
            covariance_dir,
        )

    all_rows: list[dict[str, Any]] = []
    compute = []
    for n in (3, 4, 5):
        rows = selected_rows[n]
        items = build_prompt_items(rows, include_stress=True)
        base_tables, base_account = build_full_cost_tables(model, tokenizer, items)
        base = decode_with_planner(
            model,
            tokenizer,
            items,
            base_tables,
            PlannerSpec("base_default_confidence", "default_confidence"),
        )
        base_seeded = _seed_rows(base)
        all_rows.extend(_attach_base(base_seeded, base_seeded))
        rollback, _ = apply_memit_batch(
            model,
            tokenizer,
            rows,
            _partial_config(llada_lock),
            lambda layer: _load_covariance(covariance_dir, layer),
            target_cache_dir=args.output_dir / "target_value_cache" / f"n{n}",
        )
        try:
            tables, edited_account = build_full_cost_tables(model, tokenizer, items)
            for spec in _planner_specs(
                stage=args.stage, n=n, llada_lock=llada_lock, dream_lock=dream_lock
            ):
                decoded = decode_with_planner(model, tokenizer, items, tables, spec)
                all_rows.extend(_attach_base(_seed_rows(decoded), base_seeded))
            compute.append(
                {
                    "target_length": n,
                    "num_edits": len(rows),
                    "base_cost_table": json.dumps(base_account, sort_keys=True),
                    "edited_cost_table": json.dumps(edited_account, sort_keys=True),
                }
            )
        finally:
            rollback.rollback()
        if not rollback.checksum_matches(atol=0.0):
            raise RuntimeError(f"Dream rollback failed at N={n}")

    aggregate = _aggregate(all_rows)
    compact = _summary(all_rows)
    write_csv(args.output_dir / "dev_results.csv", aggregate if args.stage != "locked" else [])
    write_csv(args.output_dir / "locked_results.csv", aggregate if args.stage == "locked" else [])
    write_csv(args.output_dir / "compute_table.csv", compute)
    write_csv(args.output_dir / "tokenizer_alignment.csv", [
        {
            "target_length": n,
            "num_rows": len(selected_rows[n]),
            "all_contextual_lengths_match": all(int(row["target_length"]) == n for row in selected_rows[n]),
            "tokenizer_model_id": SECONDARY_MODEL_ID,
            "tokenizer_revision": SECONDARY_MODEL_REVISION,
        }
        for n in (3, 4, 5)
    ])
    write_csv(args.output_dir / "dream_memit_smoke.csv", aggregate if args.stage == "smoke" else [])

    if args.stage in {"smoke", "dev"}:
        finite_candidates = sorted(
            {
                str(row["family"])
                for row in aggregate
                if str(row["family"]).startswith("finite_")
            }
        )
        scores = {}
        for family in finite_candidates:
            values = [
                row
                for row in compact
                if row["family"] == family and row["bucket"] in {"rewrite", "paraphrase"}
            ]
            scores[family] = sum(float(row["exact"]) for row in values)
        selected_finite = max(finite_candidates, key=lambda family: (scores[family], family))
        reference = selected_finite.removeprefix("finite_").rsplit("_beta", 1)[0]
        dream_dev_lock = {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "stage": args.stage,
            "model_id": SECONDARY_MODEL_ID,
            "model_revision": SECONDARY_MODEL_REVISION,
            "llada_dev_lock_sha256": sha256_file(llada_lock_path),
            "module_map_sha256": sha256_file(args.output_dir / "model_module_map.json"),
            "covariance_hashes": {str(row["layer"]): row["sha256"] for row in covariance_reports},
            "reference_process": reference,
            "beta": float(llada_lock["beta"]),
            "finite_controller_label": selected_finite,
            "best_non_sb_planner": llada_lock["best_non_sb_planner"],
            "manifest_hashes": expected_hashes,
            "locked_split_opened": False,
            "validation_pass": args.stage == "dev",
        }
        write_json(args.output_dir / "dream_dev_lock.json", dream_dev_lock)
        acceptance = bool(module_map["acceptance_pass"]) and bool(all_rows)
        classification = "integration_smoke_passed" if args.stage == "smoke" else "dream_dev_locked"
        bootstrap_rows: list[dict[str, Any]] = []
    else:
        finite = str(dream_lock["finite_controller_label"])
        baseline = str(dream_lock["best_non_sb_planner"])
        bootstrap_rows = []
        for n in (3, 4, 5):
            result = paired_bootstrap(
                paired_values(
                    all_rows,
                    left=finite,
                    right=baseline,
                    bucket="rewrite",
                    metric="full_target_exact",
                    lengths={n},
                ),
                resamples=10_000,
                seed=260_717_850 + n,
            )
            bootstrap_rows.append({"target_length": n, "left": finite, "right": baseline, **result})
        pooled = paired_bootstrap(
            paired_values(
                all_rows,
                left=finite,
                right=baseline,
                bucket="rewrite",
                metric="full_target_exact",
                lengths={3, 4},
            ),
            resamples=10_000,
            seed=260_717_860,
        )
        bootstrap_rows.append({"target_length": "pooled_3_4", "left": finite, "right": baseline, **pooled})
        stress_delta, _, _ = _mean_delta(
            all_rows,
            left=finite,
            right="base_default_confidence",
            bucket="same_subject_stress",
            metric="full_target_exact",
            lengths={3, 4, 5},
        )
        malformed = max(
            float(row["malformed"])
            for row in aggregate
            if row["family"] == finite
        )
        llada_report = read_json(CAMPAIGN_ROOT / "llada_locked_confirmation_v1" / "report_summary.json")
        llada_delta = float(llada_report["pooled_primary_bootstrap"]["mean_delta"])
        same_direction = float(pooled["mean_delta"]) * llada_delta > 0
        one_length = any(
            int(row["target_length"]) in {3, 4, 5}
            and float(row["mean_delta"]) >= 0.03
            and float(row["ci95_low"]) > 0
            for row in bootstrap_rows
            if row["target_length"] != "pooled_3_4"
        )
        acceptance = same_direction and one_length and stress_delta <= 0.03 and malformed <= 0.05
        classification = "cross_backbone_pass" if acceptance else "cross_backbone_not_confirmed"
        selected_finite = finite
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    interpretation = f"""# Dream Cross-Backbone Interpretation

Stage: `{args.stage}`. Classification: `{classification}`.

The Dream model/tokenizer revision and module map were pinned before outcomes.
Only the documented model-specific module mapping and reference-process choice
were fit on Dream development data; the LLaDA-selected beta and scientific
thresholds were unchanged.
"""
    (args.output_dir / "cross_backbone_interpretation.md").write_text(
        interpretation, encoding="utf-8"
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P5",
        "stage": f"P5_dream_{args.stage}",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": SECONDARY_MODEL_ID,
        "model_revision": SECONDARY_MODEL_REVISION,
        "profile": args.stage,
        "manifest_hashes": expected_hashes,
        "num_edits_by_length": {str(n): len(selected_rows[n]) for n in (3, 4, 5)},
        "module_mapping_repair_used": True,
        "selected_finite_controller": selected_finite,
        "classification": classification,
        "locked_outcomes_used_for_tuning": False,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "runtime_seconds": time.monotonic() - wall_start,
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "acceptance_pass": acceptance,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage=str(report["stage"]),
        track="P5",
        status=classification,
        output_dir=args.output_dir,
        acceptance_pass=acceptance,
        started_at_utc=started,
        notes=f"profile={args.stage}; classification={classification}",
        next_stage="P6_editor_generality" if args.stage == "locked" else f"P5_dream_{'dev' if args.stage == 'smoke' else 'locked'}",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
