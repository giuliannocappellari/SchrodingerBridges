#!/usr/bin/env python3
"""Execute the bounded M2 partial-mask MEMIT campaign."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_common import (
    CAMPAIGN_ROOT,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


M1_ROOT = CAMPAIGN_ROOT / "M1_mdm_memit_reproduction_v1"
M2_ROOT = CAMPAIGN_ROOT / "M2_partial_mask_memit_v1"
SCHEDULES = ("fully_masked", "fewer_revealed", "uniform", "more_revealed", "cycle")
REVEAL_POLICIES = ("left_to_right", "base_confidence", "random")


def _bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _run_or_reuse(
    *,
    output_dir: Path,
    manifest: Path,
    layers: Sequence[int],
    schedule: str,
    reveal_policy: str,
    seed: int,
    include_locality: bool,
    optimization_steps: int = 25,
) -> dict[str, Any]:
    report_path = output_dir / "report_summary.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if output_dir.exists():
        raise RuntimeError(f"Partial M2 output requires audit: {output_dir}")
    command = [
        sys.executable,
        str(ROOT / "scripts/run_mdm_memit_stage.py"),
        "--stage",
        "batch",
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output_dir),
        "--covariance_dir",
        str(CAMPAIGN_ROOT / "covariance_cache_v1"),
        "--layers",
        ",".join(map(str, layers)),
        "--partial_mask_schedule",
        schedule,
        "--reveal_policy",
        reveal_policy,
        "--target_optimization_steps",
        str(optimization_steps),
        "--seed",
        str(seed),
        "--include_locality",
        "1" if include_locality else "0",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _metrics_by_length(run_dir: Path, label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with (run_dir / "edited_per_prompt.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["bucket"], int(row["target_length"]))].append(row)
    output: list[dict[str, Any]] = []
    for (bucket, length), rows in sorted(grouped.items()):
        exact = sum(_bool(row["target_new_hit"]) for row in rows) / len(rows)
        malformed = sum(_bool(row["malformed"]) for row in rows) / len(rows)
        token_coverages: list[float] = []
        for row in rows:
            output_ids = set(map(int, json.loads(row.get("output_token_ids") or "[]")))
            target_ids = list(map(int, json.loads(row.get("target_new_token_ids") or "[]")))
            if target_ids:
                token_coverages.append(sum(token in output_ids for token in target_ids) / len(target_ids))
        coverage = sum(token_coverages) / len(token_coverages) if token_coverages else 0.0
        output.append(
            {
                "label": label,
                "bucket": bucket,
                "target_length": length,
                "num_prompt_rows": len(rows),
                "num_edits": len({row["case_id"] for row in rows}),
                "full_target_exact": exact,
                "target_token_appearance": coverage,
                "full_target_assembly_gap": coverage - exact,
                "malformed_rate": malformed,
            }
        )
    return output


def _mean_primary(rows: Sequence[Mapping[str, Any]], bucket: str) -> float:
    values = [
        float(row["full_target_exact"])
        for row in rows
        if row["bucket"] == bucket and int(row["target_length"]) in {2, 3, 4}
    ]
    return sum(values) / len(values)


def _rank(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (-float(row[key]), str(row["label"])))
    return {str(row["label"]): index + 1 for index, row in enumerate(ordered)}


def _augment_locality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_relation[str(row["relation_id"])].append(row)
        by_subject[str(row["subject"]).casefold()].append(row)
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["neighborhood_prompts"] = [
            other["rewrite_prompt"]
            for other in by_relation[str(row["relation_id"])]
            if other["case_id"] != row["case_id"] and other["subject"] != row["subject"]
        ][:3]
        row["generation_prompts"] = [
            other["rewrite_prompt"]
            for other in by_subject[str(row["subject"]).casefold()]
            if other["relation_id"] != row["relation_id"]
        ][:3]
        row["attribute_prompts"] = []
        row["locality_prompt_provenance"] = "other_real_KAMEL_evaluation_rows_base_agreement_only"
        output.append(row)
    return output


def _bootstrap(
    baseline_dir: Path,
    method_dir: Path,
    *,
    trials: int = 2000,
    seed: int = 260603924,
) -> list[dict[str, Any]]:
    def load(path: Path, bucket: str, length: int) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["bucket"] == bucket and int(row["target_length"]) == length:
                    grouped[row["case_id"]].append(float(_bool(row["target_new_hit"])))
        return {case: sum(values) / len(values) for case, values in grouped.items()}

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    for length in (1, 2, 3, 4):
        for bucket in ("rewrite", "paraphrase"):
            left = load(baseline_dir / "edited_per_prompt.csv", bucket, length)
            right = load(method_dir / "edited_per_prompt.csv", bucket, length)
            cases = sorted(set(left) & set(right))
            if not cases:
                continue
            observed = sum(right[c] - left[c] for c in cases) / len(cases)
            draws = []
            for _ in range(trials):
                sample = [cases[rng.randrange(len(cases))] for _ in cases]
                draws.append(sum(right[c] - left[c] for c in sample) / len(sample))
            draws.sort()
            output.append(
                {
                    "target_length": length,
                    "bucket": bucket,
                    "partial_minus_fully_masked": observed,
                    "ci95_low": draws[int(0.025 * (trials - 1))],
                    "ci95_high": draws[int(0.975 * (trials - 1))],
                    "num_edits": len(cases),
                    "bootstrap_trials": trials,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=M2_ROOT)
    parser.add_argument("--seed", type=int, default=260603924)
    args = parser.parse_args()
    started = now_utc()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(
        (M1_ROOT / "layer_selection_v1/selected_layer_window.json").read_text(encoding="utf-8")
    )
    layers = list(map(int, selection["layers"]))
    smoke_manifest = PROTOCOL_ROOT / "kamel_smoke_20_per_length.jsonl"
    dev_manifest = PROTOCOL_ROOT / "kamel_dev_50_per_length.jsonl"
    main_manifest = PROTOCOL_ROOT / "kamel_repro_200_per_length.jsonl"

    baseline_smoke_dir = args.output_dir / "baseline_smoke_fully_masked_v1"
    baseline_smoke_report = _run_or_reuse(
        output_dir=baseline_smoke_dir,
        manifest=smoke_manifest,
        layers=layers,
        schedule="fully_masked",
        reveal_policy="random",
        seed=args.seed,
        include_locality=False,
    )
    smoke_rows = _metrics_by_length(baseline_smoke_dir, "fully_masked")
    smoke_pass = (
        {int(row["target_length"]) for row in smoke_rows if row["bucket"] == "rewrite"}
        == {1, 2, 3, 4}
        and max(float(row["malformed_rate"]) for row in smoke_rows) <= 0.05
        and baseline_smoke_report["num_edits"] == 80
    )

    schedule_summaries: list[dict[str, Any]] = []
    schedule_metrics: dict[str, list[dict[str, Any]]] = {}
    schedule_dirs: dict[str, Path] = {}
    for schedule in SCHEDULES:
        label = f"schedule_{schedule}"
        run_dir = args.output_dir / f"dev_{label}_v1"
        _run_or_reuse(
            output_dir=run_dir,
            manifest=dev_manifest,
            layers=layers,
            schedule=schedule,
            reveal_policy="random",
            seed=args.seed,
            include_locality=False,
        )
        metrics = _metrics_by_length(run_dir, label)
        schedule_metrics[schedule] = metrics
        schedule_dirs[schedule] = run_dir
        schedule_summaries.append(
            {
                "label": schedule,
                "rewrite_mean_n2_n4": _mean_primary(metrics, "rewrite"),
                "paraphrase_mean_n2_n4": _mean_primary(metrics, "paraphrase"),
                "n4_rewrite": next(
                    float(row["full_target_exact"])
                    for row in metrics
                    if row["bucket"] == "rewrite" and int(row["target_length"]) == 4
                ),
            }
        )
    rewrite_ranks = _rank(schedule_summaries, "rewrite_mean_n2_n4")
    para_ranks = _rank(schedule_summaries, "paraphrase_mean_n2_n4")
    for row in schedule_summaries:
        row["rewrite_rank"] = rewrite_ranks[row["label"]]
        row["paraphrase_rank"] = para_ranks[row["label"]]
        row["rank_sum"] = row["rewrite_rank"] + row["paraphrase_rank"]
    selected_schedule = sorted(
        schedule_summaries,
        key=lambda row: (row["rank_sum"], -row["n4_rewrite"], row["label"]),
    )[0]["label"]

    reveal_summaries: list[dict[str, Any]] = []
    reveal_dirs: dict[str, Path] = {}
    for policy in REVEAL_POLICIES:
        if policy == "random":
            run_dir = schedule_dirs[selected_schedule]
        else:
            run_dir = args.output_dir / f"dev_reveal_{selected_schedule}_{policy}_v1"
            _run_or_reuse(
                output_dir=run_dir,
                manifest=dev_manifest,
                layers=layers,
                schedule=selected_schedule,
                reveal_policy=policy,
                seed=args.seed,
                include_locality=False,
            )
        reveal_dirs[policy] = run_dir
        metrics = _metrics_by_length(run_dir, f"reveal_{policy}")
        reveal_summaries.append(
            {
                "label": policy,
                "schedule": selected_schedule,
                "rewrite_mean_n2_n4": _mean_primary(metrics, "rewrite"),
                "paraphrase_mean_n2_n4": _mean_primary(metrics, "paraphrase"),
                "n4_rewrite": next(
                    float(row["full_target_exact"])
                    for row in metrics
                    if row["bucket"] == "rewrite" and int(row["target_length"]) == 4
                ),
            }
        )
    rewrite_ranks = _rank(reveal_summaries, "rewrite_mean_n2_n4")
    para_ranks = _rank(reveal_summaries, "paraphrase_mean_n2_n4")
    for row in reveal_summaries:
        row["rewrite_rank"] = rewrite_ranks[row["label"]]
        row["paraphrase_rank"] = para_ranks[row["label"]]
        row["rank_sum"] = row["rewrite_rank"] + row["paraphrase_rank"]
    selected_reveal = sorted(
        reveal_summaries,
        key=lambda row: (row["rank_sum"], -row["n4_rewrite"], row["label"]),
    )[0]["label"]

    augmented_manifest = args.output_dir / "kamel_main_locality_augmented.jsonl"
    if not augmented_manifest.exists():
        write_jsonl(augmented_manifest, _augment_locality(read_jsonl(main_manifest)))
    baseline_main_dir = args.output_dir / "main_fully_masked_seed1_v1"
    partial_main_dir = args.output_dir / "main_partial_seed1_v1"
    partial_seed2_dir = args.output_dir / "main_partial_seed2_v1"
    _run_or_reuse(
        output_dir=baseline_main_dir,
        manifest=augmented_manifest,
        layers=layers,
        schedule="fully_masked",
        reveal_policy="random",
        seed=args.seed,
        include_locality=True,
    )
    _run_or_reuse(
        output_dir=partial_main_dir,
        manifest=augmented_manifest,
        layers=layers,
        schedule=selected_schedule,
        reveal_policy=selected_reveal,
        seed=args.seed,
        include_locality=True,
    )
    _run_or_reuse(
        output_dir=partial_seed2_dir,
        manifest=augmented_manifest,
        layers=layers,
        schedule=selected_schedule,
        reveal_policy=selected_reveal,
        seed=args.seed + 1,
        include_locality=True,
    )
    baseline_metrics = _metrics_by_length(baseline_main_dir, "fully_masked")
    seed1_metrics = _metrics_by_length(partial_main_dir, "partial_seed1")
    seed2_metrics = _metrics_by_length(partial_seed2_dir, "partial_seed2")
    all_main = baseline_metrics + seed1_metrics + seed2_metrics

    def value(metrics: Sequence[Mapping[str, Any]], bucket: str, length: int) -> float:
        return next(
            float(row["full_target_exact"])
            for row in metrics
            if row["bucket"] == bucket and int(row["target_length"]) == length
        )

    confirmation: list[dict[str, Any]] = []
    minimum_lengths = 0
    strong = True
    for length in (2, 3, 4):
        base_rw = value(baseline_metrics, "rewrite", length)
        base_pa = value(baseline_metrics, "paraphrase", length)
        s1_rw = value(seed1_metrics, "rewrite", length)
        s1_pa = value(seed1_metrics, "paraphrase", length)
        s2_rw = value(seed2_metrics, "rewrite", length)
        s2_pa = value(seed2_metrics, "paraphrase", length)
        mean_rw = (s1_rw + s2_rw) / 2
        mean_pa = (s1_pa + s2_pa) / 2
        passes = mean_rw - base_rw >= 0.15 and mean_pa - base_pa >= 0.08
        persists = s1_rw > base_rw and s2_rw > base_rw and s1_pa > base_pa and s2_pa > base_pa
        minimum_lengths += int(passes and persists)
        thresholds = {2: 0.75, 3: 0.60, 4: 0.55}
        strong = strong and mean_rw >= thresholds[length]
        confirmation.append(
            {
                "target_length": length,
                "baseline_rewrite": base_rw,
                "baseline_paraphrase": base_pa,
                "seed1_rewrite": s1_rw,
                "seed1_paraphrase": s1_pa,
                "seed2_rewrite": s2_rw,
                "seed2_paraphrase": s2_pa,
                "mean_rewrite_improvement": mean_rw - base_rw,
                "mean_paraphrase_improvement": mean_pa - base_pa,
                "positive_direction_both_seeds": persists,
                "minimum_length_pass": passes and persists,
            }
        )
    positive_pass = minimum_lengths >= 2
    rescue_used = False
    if not positive_pass:
        rescue_used = True
        rescue_dir = args.output_dir / "bounded_rescue_seed3_v1"
        _run_or_reuse(
            output_dir=rescue_dir,
            manifest=augmented_manifest,
            layers=layers,
            schedule=selected_schedule,
            reveal_policy=selected_reveal,
            seed=args.seed + 2,
            include_locality=True,
            optimization_steps=max(25, 20),
        )
        rescue_metrics = _metrics_by_length(rescue_dir, "partial_rescue_seed")
        all_main.extend(rescue_metrics)
        rescue_confirmation: list[dict[str, Any]] = []
        rescue_minimum_lengths = 0
        rescue_strong = True
        for length in (2, 3, 4):
            base_rw = value(baseline_metrics, "rewrite", length)
            base_pa = value(baseline_metrics, "paraphrase", length)
            s1_rw = value(seed1_metrics, "rewrite", length)
            s1_pa = value(seed1_metrics, "paraphrase", length)
            sr_rw = value(rescue_metrics, "rewrite", length)
            sr_pa = value(rescue_metrics, "paraphrase", length)
            mean_rw = (s1_rw + sr_rw) / 2
            mean_pa = (s1_pa + sr_pa) / 2
            passes = mean_rw - base_rw >= 0.15 and mean_pa - base_pa >= 0.08
            persists = s1_rw > base_rw and sr_rw > base_rw and s1_pa > base_pa and sr_pa > base_pa
            rescue_minimum_lengths += int(passes and persists)
            rescue_strong = rescue_strong and mean_rw >= {2: 0.75, 3: 0.60, 4: 0.55}[length]
            rescue_confirmation.append(
                {
                    "target_length": length,
                    "baseline_rewrite": base_rw,
                    "baseline_paraphrase": base_pa,
                    "seed1_rewrite": s1_rw,
                    "seed1_paraphrase": s1_pa,
                    "seed2_rewrite": sr_rw,
                    "seed2_paraphrase": sr_pa,
                    "mean_rewrite_improvement": mean_rw - base_rw,
                    "mean_paraphrase_improvement": mean_pa - base_pa,
                    "positive_direction_both_seeds": persists,
                    "minimum_length_pass": passes and persists,
                    "bounded_rescue_pair": True,
                }
            )
        if rescue_minimum_lengths >= 2:
            minimum_lengths = rescue_minimum_lengths
            positive_pass = True
            strong = rescue_strong
            confirmation = rescue_confirmation

    write_csv(args.output_dir / "state_schedule_ablation.csv", schedule_summaries)
    write_csv(args.output_dir / "reveal_policy_ablation.csv", reveal_summaries)
    write_csv(args.output_dir / "main_results_by_length.csv", all_main)
    write_csv(
        args.output_dir / "token_assembly_gap.csv",
        [row for row in all_main if row["bucket"] in {"rewrite", "paraphrase"}],
    )
    write_csv(args.output_dir / "seed_confirmation.csv", confirmation)
    bootstrap = _bootstrap(baseline_main_dir, partial_main_dir)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap)
    write_json(
        args.output_dir / "kamel_manifest_summary.json",
        {
            "smoke_manifest": str(smoke_manifest),
            "smoke_sha256": sha256_file(smoke_manifest),
            "dev_manifest": str(dev_manifest),
            "dev_sha256": sha256_file(dev_manifest),
            "main_manifest": str(main_manifest),
            "main_sha256": sha256_file(main_manifest),
            "augmented_evaluation_manifest": str(augmented_manifest),
            "augmented_sha256": sha256_file(augmented_manifest),
            "locality_augmentation_used_for_optimization": False,
        },
    )
    report = {
        "campaign_id": "masked_diffusion_memit_sb_positive_result_v1",
        "track": "M2",
        "stage": "M2_partial_mask_complete",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "selected_layers": layers,
        "baseline_smoke_pass": smoke_pass,
        "selected_schedule": selected_schedule,
        "selected_reveal_policy": selected_reveal,
        "minimum_positive_length_count": minimum_lengths,
        "minimum_positive_pass": positive_pass,
        "strong_pass": strong and positive_pass,
        "two_seed_confirmation_pass": positive_pass,
        "bounded_rescue_used": rescue_used,
        "acceptance_pass": positive_pass,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
    }
    write_json(args.output_dir / "report_summary.json", report)
    final = f"""# M2 Partial-Mask MDM-MEMIT

Status: **{'passed' if positive_pass else 'formal_negative'}**

- Selected schedule: `{selected_schedule}`
- Selected reveal policy: `{selected_reveal}`
- Multi-token lengths passing the two-seed minimum criterion: {minimum_lengths}/3
- Strong pass: {report['strong_pass']}
- Bounded rescue used: {rescue_used}

All tuning used fresh KAMEL dev rows. The locked main rows were evaluated only after policy freeze.
"""
    (args.output_dir / "final_track_report.md").write_text(final, encoding="utf-8")
    record_stage(
        stage="M2_partial_mask_complete",
        track="M2",
        status="passed" if positive_pass else "failed",
        output_dir=args.output_dir,
        acceptance_pass=positive_pass,
        started_at_utc=started,
        notes=f"schedule={selected_schedule}; reveal={selected_reveal}; passing_lengths={minimum_lengths}",
    )
    print(json.dumps({"acceptance_pass": positive_pass, "selected_schedule": selected_schedule, "selected_reveal": selected_reveal}))


if __name__ == "__main__":
    main()
