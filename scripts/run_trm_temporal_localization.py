#!/usr/bin/env python3
"""Run coarse-to-fine temporal causal localization on fresh CounterFact edits."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mdm_memit_stage import load_model
from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage,
    record_stage_cost,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.trm_localization import (
    TraceCandidate,
    aggregate_coordinates,
    build_site_policy_rows,
    candidate_grid,
    confidence_order,
    shortlist_candidates,
    stability_summary,
    temporal_state_specs,
    trace_candidates_batched,
)


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def stratified_prefix(rows: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_length: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_length.setdefault(int(row["target_length"]), []).append(dict(row))
    selected: list[dict[str, Any]] = []
    for length in sorted(by_length):
        selected.append(by_length[length][0])
    cursor = {length: 1 for length in by_length}
    while len(selected) < min(int(limit), len(rows)):
        progressed = False
        for length in sorted(by_length):
            index = cursor[length]
            if index < len(by_length[length]) and len(selected) < int(limit):
                selected.append(by_length[length][index])
                cursor[length] += 1
                progressed = True
        if not progressed:
            break
    return selected[: int(limit)]


def annotate(
    rows: Sequence[dict[str, Any]],
    *,
    source: Mapping[str, Any],
    target_role: str,
    prompt_type: str,
    prompt_index: int,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "relation_id": source.get("relation_id"),
                "target_role": target_role,
                "prompt_type": prompt_type,
                "prompt_index": int(prompt_index),
                "state_label": state["state_label"],
                "effective_state_signature": state["effective_state_signature"],
            }
        )
    return output


def environment_report() -> dict[str, Any]:
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "gpu_count": torch.cuda.device_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROTOCOL_ROOT / "cf_trm_localize_50.jsonl")
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "C1_temporal_localization_v1")
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--layers", type=parse_ints, default=tuple(range(32)))
    parser.add_argument("--components", default="hidden,mlp,attention")
    parser.add_argument("--positions", default="first_subject,last_subject,first_answer_mask")
    parser.add_argument("--coarse_limit", type=int, default=12)
    parser.add_argument("--fine_limit", type=int, default=50)
    parser.add_argument("--fine_candidate_count", type=int, default=16)
    parser.add_argument("--max_paraphrases", type=int, default=1)
    parser.add_argument("--seeds", type=parse_ints, default=(260718301, 260718302))
    parser.add_argument("--noise_scale", type=float, default=3.0)
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()
    started = now_utc()
    begin = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = read_jsonl(args.manifest)
    if len(rows) != 50 or {row.get("split_role") for row in rows} != {"cf_trm_localize_50"}:
        raise RuntimeError("C1 requires the complete fresh cf_trm_localize_50 manifest")
    if any(token in str(args.manifest).casefold() for token in ("analysis_500", "final_test_500", "final_test_full")):
        raise RuntimeError("Locked historical evaluation manifest is forbidden")
    components = tuple(item.strip() for item in args.components.split(",") if item.strip())
    positions = tuple(item.strip() for item in args.positions.split(",") if item.strip())
    seeds = tuple(args.seeds)
    all_candidates = candidate_grid(args.layers, components, positions)
    config = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "C1_temporal_localization",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "layers": list(args.layers),
        "components": list(components),
        "positions": list(positions),
        "coarse_limit": args.coarse_limit,
        "fine_limit": args.fine_limit,
        "fine_candidate_count": args.fine_candidate_count,
        "max_paraphrases": args.max_paraphrases,
        "seeds": list(seeds),
        "noise_scale": args.noise_scale,
        "chunk_size": args.chunk_size,
        "coarse_design": "all layers/components/positions on a target-length-stratified subset",
        "fine_design": "frozen shortlist across all edits, rewrite plus one paraphrase, five semantic states, two seeds",
        "analysis_500_used": False,
        "final_test_used": False,
    }
    write_json(args.output_dir / "run_config.json", config)
    model, tokenizer = load_model(args.model_id, args.model_revision, args.dtype)
    model.eval()
    environment = environment_report()
    coarse_rows = stratified_prefix(rows, args.coarse_limit)
    raw_coarse: list[dict[str, Any]] = []
    full_state = {
        "state_label": "fully_masked",
        "revealed_positions": [],
        "effective_state_signature": "all_masked",
    }
    for index, source in enumerate(coarse_rows, start=1):
        for target_role, field in (
            ("target_true", "target_true_token_ids"),
            ("target_new", "target_new_token_ids"),
        ):
            traced = trace_candidates_batched(
                model,
                tokenizer,
                case_id=str(source["case_id"]),
                prompt=str(source["rewrite_prompt"]),
                subject=str(source["subject"]),
                target_ids=source[field],
                candidates=all_candidates,
                noise_scale=args.noise_scale,
                seed=seeds[0],
                chunk_size=args.chunk_size,
            )
            raw_coarse.extend(
                annotate(
                    traced,
                    source=source,
                    target_role=target_role,
                    prompt_type="rewrite",
                    prompt_index=0,
                    state=full_state,
                )
            )
        write_csv(args.output_dir / "coarse_trace_rows.partial.csv", raw_coarse)
        print(f"C1 coarse {index}/{len(coarse_rows)}", flush=True)
    coarse_summary = aggregate_coordinates(raw_coarse)
    shortlist = shortlist_candidates(
        coarse_summary,
        all_candidates=all_candidates,
        limit=args.fine_candidate_count,
        seed=seeds[0],
    )
    write_csv(args.output_dir / "causal_trace_summary.csv", coarse_summary)
    write_json(
        args.output_dir / "fine_candidate_shortlist.json",
        {
            "selection_split": "cf_trm_localize_50_coarse_subset",
            "target_role_weighting": {"target_new": 0.65, "target_true": 0.35},
            "candidates": [candidate.to_dict() for candidate in shortlist],
        },
    )
    raw_fine: list[dict[str, Any]] = []
    fine_sources = list(rows[: args.fine_limit])
    for edit_index, source in enumerate(fine_sources, start=1):
        prompts = [("rewrite", str(source["rewrite_prompt"]))]
        prompts.extend(
            ("declarative_paraphrase", str(prompt))
            for prompt in list(source.get("paraphrase_prompts", []))[: args.max_paraphrases]
        )
        target_ids = list(map(int, source["target_new_token_ids"]))
        for prompt_index, (prompt_type, prompt) in enumerate(prompts):
            order = confidence_order(model, tokenizer, prompt=prompt, target_ids=target_ids)
            for seed in seeds:
                for state in temporal_state_specs(
                    len(target_ids), seed=seed, confidence_order=order
                ):
                    traced = trace_candidates_batched(
                        model,
                        tokenizer,
                        case_id=str(source["case_id"]),
                        prompt=prompt,
                        subject=str(source["subject"]),
                        target_ids=target_ids,
                        candidates=shortlist,
                        revealed_positions=state["revealed_positions"],
                        noise_scale=args.noise_scale,
                        seed=seed,
                        chunk_size=args.chunk_size,
                    )
                    raw_fine.extend(
                        annotate(
                            traced,
                            source=source,
                            target_role="target_new",
                            prompt_type=prompt_type,
                            prompt_index=prompt_index,
                            state=state,
                        )
                    )
        if edit_index % 2 == 0 or edit_index == len(fine_sources):
            write_csv(args.output_dir / "fine_trace_rows.partial.csv", raw_fine)
        if edit_index % 5 == 0 or edit_index == len(fine_sources):
            print(f"C1 fine {edit_index}/{len(fine_sources)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    temporal_summary = aggregate_coordinates(
        raw_fine,
        group_fields=(
            "candidate_id",
            "layer",
            "component",
            "position",
            "state_label",
            "prompt_type",
        ),
    )
    stability = stability_summary(raw_fine)
    policies = build_site_policy_rows(
        stability, raw_fine, num_layers=max(args.layers) + 1, seed=seeds[0]
    )
    write_csv(args.output_dir / "temporal_trace_summary.csv", temporal_summary)
    write_csv(args.output_dir / "site_stability.csv", stability)
    write_csv(args.output_dir / "site_policy_comparison.csv", policies)
    write_json(
        args.output_dir / "site_policy_lock.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "selection_manifest": str(args.manifest),
            "selection_manifest_sha256": sha256_file(args.manifest),
            "policies": policies,
            "primary_policy_for_C2": "stable_temporal_site_set",
            "required_control_for_C2": "random_site",
            "site_policy_rescue_unused": True,
            "no_pilot_or_locked_outcomes_used": True,
        },
    )
    write_json(
        args.output_dir / "residual_memory_schema.json",
        {
            "schema_version": 1,
            "key": "hidden state at frozen layer/component/position and partial-state bucket",
            "value": "optimized residual delta increasing target_new and suppressing target_true",
            "fit": "D K^T (K K^T + lambda I)^-1",
            "runtime": "h_t_prime = h_t + alpha * Sparse_q(M k_t)",
            "routing_inputs": [
                "edit_id supplied at edit time",
                "current hidden state",
                "active mask count",
                "denoising step index",
            ],
            "forbidden_runtime_inputs": [
                "evaluation bucket",
                "future outcome",
                "analysis or final label",
                "teacher-only score",
            ],
            "backbone_update": "none",
        },
    )
    finite = all(
        bool(row["all_finite"])
        for row in coarse_summary
    ) and all(bool(row["all_finite"]) for row in temporal_summary)
    observed_states = {str(row["state_label"]) for row in raw_fine}
    observed_prompts = {str(row["prompt_type"]) for row in raw_fine}
    observed_seeds = {int(row["noise_seed"]) for row in raw_fine}
    required_states = {
        "fully_masked",
        "early",
        "middle",
        "late",
        "actual_confidence_trajectory",
    }
    stable_policy = next(row for row in policies if row["policy_id"] == "stable_temporal_site_set")
    random_policy = next(row for row in policies if row["policy_id"] == "random_site")
    integrity = {
        "all_50_localization_edits_covered": len({row["case_id"] for row in raw_fine}) == 50,
        "all_requested_layers_covered_coarsely": {int(row["layer"]) for row in raw_coarse} == set(args.layers),
        "all_module_families_covered": {row["component"] for row in raw_coarse} == set(components),
        "all_positions_covered": {row["position"] for row in raw_coarse} == set(positions),
        "all_temporal_state_labels_covered": observed_states == required_states,
        "rewrite_and_paraphrase_covered": observed_prompts == {"rewrite", "declarative_paraphrase"},
        "both_seeds_covered": observed_seeds == set(seeds),
        "target_length_bins_1_and_2_present": {1, 2}.issubset({int(row["target_length"]) for row in raw_fine}),
        "all_metrics_finite": finite,
        "stable_mlp_last_subject_policy_nonempty": bool(json.loads(stable_policy["candidate_ids_json"])),
        "analysis_500_used": False,
        "final_test_used": False,
    }
    acceptance_pass = all(
        value for key, value in integrity.items() if key not in {"analysis_500_used", "final_test_used"}
    ) and not integrity["analysis_500_used"] and not integrity["final_test_used"]
    runtime = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "C1_temporal_localization",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "num_localization_edits": len(fine_sources),
        "coarse_edit_count": len(coarse_rows),
        "coarse_candidate_count": len(all_candidates),
        "fine_candidate_count": len(shortlist),
        "fine_trace_row_count": len(raw_fine),
        "target_length_histogram": dict(sorted(Counter(int(row["target_length"]) for row in fine_sources).items())),
        "best_stable_site": stability[0],
        "stable_policy_localization_proxy": stable_policy["localization_proxy"],
        "random_policy_localization_proxy": random_policy["localization_proxy"],
        "localization_proxy_delta_vs_random": float(stable_policy["localization_proxy"]) - float(random_policy["localization_proxy"]),
        "C2_policy_acceptance_pending": True,
        "temporal_localization_acceptance_rule": "site policy efficacy/efficiency is tested against random in C2 and pilot, not inferred from tracing alone",
        "integrity_checks": integrity,
        "environment": environment,
        "runtime_seconds": runtime,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": acceptance_pass,
    }
    write_json(args.output_dir / "validation_report.json", {"integrity_checks": integrity, "acceptance_pass": acceptance_pass})
    write_json(args.output_dir / "report_summary.json", report)
    write_csv(args.output_dir / "coarse_trace_rows.csv", raw_coarse)
    write_csv(args.output_dir / "fine_trace_rows.csv", raw_fine)
    for partial in (args.output_dir / "coarse_trace_rows.partial.csv", args.output_dir / "fine_trace_rows.partial.csv"):
        if partial.exists():
            partial.unlink()
    record_stage_cost(
        "C1_temporal_localization",
        runtime_seconds=runtime,
        gpu_count=1,
        notes="A40 coarse-to-fine temporal causal localization",
    )
    record_stage(
        "C1_temporal_localization",
        status="passed" if acceptance_pass else "failed",
        acceptance_pass=acceptance_pass,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"50-edit localization; stable-vs-random proxy delta={report['localization_proxy_delta_vs_random']:.4f}",
        next_stage="C2_fullmask_temporal_residual" if acceptance_pass else None,
    )
    if not acceptance_pass:
        raise SystemExit(2)
    print(json.dumps({"acceptance_pass": True, "best_stable_site": stability[0]}, sort_keys=True))


if __name__ == "__main__":
    main()
