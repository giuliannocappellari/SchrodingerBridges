#!/usr/bin/env python3
"""Run standard or temporal causal localization for the DNPE campaign."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
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
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.dnpe_editor import state_bank, trace_case_grid
from scripts.run_mdm_memit_stage import load_model


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in value.split(",") if item.strip()}))


def aggregate_effects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["layer"]),
            str(row["component"]),
            str(row["position"]),
            int(row.get("revealed_count", 0)),
        )
        groups[key].append(row)
    output = []
    for (layer, component, position, revealed_count), values in groups.items():
        effects = [float(row["normalized_aie"]) for row in values]
        output.append(
            {
                "layer": layer,
                "component": component,
                "position": position,
                "revealed_count": revealed_count,
                "num_edits": len({row["case_id"] for row in values}),
                "mean_normalized_aie": mean(effects),
                "positive_edit_fraction": sum(value > 0 for value in effects) / len(effects),
                "finite": all(math.isfinite(value) for value in effects),
            }
        )
    return sorted(output, key=lambda row: (row["revealed_count"], -row["mean_normalized_aie"]))


def plot_heatmap(path: Path, aggregate: Sequence[Mapping[str, Any]], *, title: str) -> None:
    import matplotlib.pyplot as plt

    positions = sorted({str(row["position"]) for row in aggregate})
    layers = sorted({int(row["layer"]) for row in aggregate})
    values = {
        (int(row["layer"]), str(row["position"])): float(row["mean_normalized_aie"])
        for row in aggregate
        if row["component"] == "hidden" and int(row["revealed_count"]) == 0
    }
    matrix = [[values.get((layer, position), float("nan")) for position in positions] for layer in layers]
    figure, axis = plt.subplots(figsize=(8, 9))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(positions)), labels=positions, rotation=30, ha="right")
    axis.set_yticks(range(len(layers)), labels=layers)
    axis.set_xlabel("Position")
    axis.set_ylabel("Layer")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Mean normalized AIE")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("standard", "temporal"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=PRIMARY_MODEL_ID)
    parser.add_argument("--model_revision", default=PRIMARY_MODEL_REVISION)
    parser.add_argument("--layers", type=parse_ints, default=tuple(range(32)))
    parser.add_argument("--components", default="hidden,mlp,attention")
    parser.add_argument("--positions", default="first_subject,last_subject,relation_cue,first_answer_mask")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--noise_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=260717101)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = now_utc()
    begin = time.monotonic()
    rows = read_jsonl(args.manifest)[: args.limit]
    if not rows:
        raise RuntimeError("No tracing rows")
    model, tokenizer = load_model(args.model_id, args.model_revision, "float16")
    components = tuple(item for item in args.components.split(",") if item)
    positions = tuple(item for item in args.positions.split(",") if item)
    all_effects = []
    base_rows = []
    for row_index, row in enumerate(rows):
        states = [{"revealed_positions": []}]
        if args.mode == "temporal":
            states = state_bank(
                row["target_true_token_ids"],
                policy="all_mask_counts_random_positions",
                seed=args.seed + row_index,
            )
        for state in states:
            effects, base = trace_case_grid(
                model,
                tokenizer,
                row,
                layers=args.layers,
                components=components,
                position_names=positions,
                noise_scale=args.noise_scale,
                seed=args.seed + row_index,
                revealed_positions=state["revealed_positions"],
            )
            all_effects.extend(effects)
            base_rows.append(
                {
                    "case_id": row["case_id"],
                    "revealed_positions": json.dumps(state["revealed_positions"]),
                    **base,
                }
            )
    aggregate = aggregate_effects(all_effects)
    rng = random.Random(args.seed)
    random_rows = []
    for row in rows:
        values = [effect for effect in all_effects if effect["case_id"] == row["case_id"] and int(effect["revealed_count"]) == 0]
        if values:
            choice = rng.choice(values)
            random_rows.append(choice)
    random_mean = mean(float(row["normalized_aie"]) for row in random_rows)
    best = max(aggregate, key=lambda row: float(row["mean_normalized_aie"]))
    peak_minus_random = float(best["mean_normalized_aie"]) - random_mean
    supported = float(best["positive_edit_fraction"]) >= 0.10
    passed = (
        all(bool(row["finite"]) for row in aggregate)
        and peak_minus_random >= 0.15
        and supported
    )
    effects_name = "per_case_effects.csv" if args.mode == "standard" else "tie_by_layer_position_state.csv"
    aggregate_name = "aie_by_layer_position.csv" if args.mode == "standard" else "tie_aggregate.csv"
    write_csv(args.output_dir / effects_name, all_effects)
    write_csv(args.output_dir / aggregate_name, aggregate)
    write_csv(args.output_dir / "random_site_comparison.csv", random_rows)
    write_csv(args.output_dir / "clean_corrupt_summary.csv", base_rows)
    plot_heatmap(
        args.output_dir / ("causal_heatmap.png" if args.mode == "standard" else "temporal_heatmaps.png"),
        aggregate,
        title="Standard causal tracing" if args.mode == "standard" else "Temporal partial-state tracing",
    )
    stage = "C1_standard_causal_tracing" if args.mode == "standard" else "C2_temporal_causal_tracing"
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": stage,
        "mode": args.mode,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "num_edits": len(rows),
        "layers": list(args.layers),
        "components": list(components),
        "positions": list(positions),
        "noise_scale_rule": f"{args.noise_scale}x_subject_embedding_std",
        "best_site": best,
        "random_mean_normalized_aie": random_mean,
        "peak_minus_random": peak_minus_random,
        "runtime_seconds": time.monotonic() - begin,
        "analysis_500_used": False,
        "final_test_used": False,
        "new_target_used_for_localization": False,
        "acceptance_pass": passed,
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "run_config.json", {key: report[key] for key in ("campaign_id", "mode", "manifest", "manifest_sha256", "layers", "components", "positions", "noise_scale_rule")})
    write_json(args.output_dir / "validation_report.json", {"all_finite": all(row["finite"] for row in aggregate), "peak_minus_random_at_least_0_15": peak_minus_random >= 0.15, "peak_supported_by_at_least_10_percent": supported, "acceptance_pass": passed})
    record_stage(
        stage,
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"peak_minus_random={peak_minus_random:.4f}; best={best['layer']}/{best['position']}/{best['component']}",
        next_stage="C2_temporal_causal_tracing" if args.mode == "standard" else "C3_site_policy_lock",
    )
    print(json.dumps({"acceptance_pass": passed, "best_site": best, "peak_minus_random": peak_minus_random}, sort_keys=True))


if __name__ == "__main__":
    main()
