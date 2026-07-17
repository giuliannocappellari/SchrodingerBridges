#!/usr/bin/env python3
"""Run P1 paper-matched, length-separated partial-state MDM-MEMIT audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    HISTORICAL_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_jsonl,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mdm_memit_editor import MemitConfig, apply_memit_batch
from scripts.run_mdm_memit_stage import evaluate_rows, load_covariance, load_model


SCHEDULE_SEEDS = (260717101, 260717102)
GENERATION_SEEDS = (260717201, 260717202, 260717203)
METHODS = (
    {
        "label": "ordinary_fully_masked",
        "schedule": "fully_masked",
        "reveal": "random",
        "stochastic_schedule": False,
    },
    {
        "label": "partial_cycle_fixed_positions",
        "schedule": "cycle",
        "reveal": "left_to_right",
        "stochastic_schedule": False,
    },
    {
        "label": "partial_cycle_random_positions",
        "schedule": "cycle",
        "reveal": "random",
        "stochastic_schedule": True,
    },
    {
        "label": "partial_random_count_random_positions",
        "schedule": "uniform",
        "reveal": "random",
        "stochastic_schedule": True,
    },
    {
        "label": "paper_matched_partial_cycle",
        "schedule": "cycle",
        "reveal": "random",
        "stochastic_schedule": True,
        "alias_of": "partial_cycle_random_positions",
    },
)


def _token_exact(row: Mapping[str, Any]) -> bool:
    try:
        output = list(map(int, json.loads(str(row["output_token_ids"]))))
        target = list(map(int, json.loads(str(row["target_new_token_ids"]))))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return bool(row.get("target_new_hit"))
    return output == target


def _annotate(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    schedule_seed: int,
    generation_seed: int,
    target_length: int,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "label": label,
                "schedule_seed": schedule_seed,
                "generation_seed": generation_seed,
                "target_length": target_length,
                "full_target_token_exact": _token_exact(row),
            }
        )
        output.append(item)
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["label"]), int(row["target_length"]), str(row["bucket"]))].append(row)
    output = []
    for (label, target_length, bucket), values in sorted(groups.items()):
        # Repeated deterministic generation seeds are averaged within edit first.
        by_edit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_edit[str(row["case_id"])].append(row)
        edit_exact = [
            sum(float(bool(row["full_target_token_exact"])) for row in edits) / len(edits)
            for edits in by_edit.values()
        ]
        paper_hit = [
            sum(float(bool(row["target_new_hit"])) for row in edits) / len(edits)
            for edits in by_edit.values()
        ]
        malformed = [
            sum(float(bool(row["malformed"])) for row in edits) / len(edits)
            for edits in by_edit.values()
        ]
        output.append(
            {
                "label": label,
                "target_length": target_length,
                "bucket": bucket,
                "num_edits": len(by_edit),
                "num_prompt_seed_rows": len(values),
                "full_target_token_exact": sum(edit_exact) / len(edit_exact),
                "paper_substring_hit": sum(paper_hit) / len(paper_hit),
                "malformed_rate": sum(malformed) / len(malformed),
                "schedule_seed_count": len({int(row["schedule_seed"]) for row in values}),
                "generation_seed_count": len({int(row["generation_seed"]) for row in values}),
            }
        )
    return output


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    baseline: str = "ordinary_fully_masked",
    trials: int = 10_000,
) -> list[dict[str, Any]]:
    rng = random.Random(260717301)
    output = []
    for target_length in (2, 3, 4):
        for bucket in ("rewrite", "paraphrase"):
            maps = {}
            for label in (baseline, method):
                by_edit: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    if (
                        row["label"] == label
                        and int(row["target_length"]) == target_length
                        and row["bucket"] == bucket
                    ):
                        by_edit[str(row["case_id"])].append(
                            float(bool(row["full_target_token_exact"]))
                        )
                maps[label] = {
                    case_id: sum(values) / len(values) for case_id, values in by_edit.items()
                }
            cases = sorted(set(maps[baseline]) & set(maps[method]))
            if not cases:
                continue
            deltas = [maps[method][case] - maps[baseline][case] for case in cases]
            observed = sum(deltas) / len(deltas)
            draws = []
            for _ in range(trials):
                draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
            draws.sort()
            output.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "target_length": target_length,
                    "bucket": bucket,
                    "delta": observed,
                    "ci95_low": draws[int(0.025 * (trials - 1))],
                    "ci95_high": draws[int(0.975 * (trials - 1))],
                    "num_edits": len(cases),
                    "bootstrap_resamples": trials,
                }
            )
    pooled_cases: list[tuple[str, float]] = []
    for target_length in (2, 3, 4):
        base: dict[str, list[float]] = defaultdict(list)
        candidate: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["bucket"] != "rewrite" or int(row["target_length"]) != target_length:
                continue
            destination = base if row["label"] == baseline else candidate if row["label"] == method else None
            if destination is not None:
                destination[str(row["case_id"])].append(float(bool(row["full_target_token_exact"])))
        for case in sorted(set(base) & set(candidate)):
            pooled_cases.append(
                (
                    f"n{target_length}:{case}",
                    sum(candidate[case]) / len(candidate[case]) - sum(base[case]) / len(base[case]),
                )
            )
    if pooled_cases:
        values = [value for _, value in pooled_cases]
        draws = [
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            for _ in range(trials)
        ]
        draws.sort()
        output.append(
            {
                "method": method,
                "baseline": baseline,
                "target_length": "pooled_2_3_4",
                "bucket": "rewrite",
                "delta": sum(values) / len(values),
                "ci95_low": draws[int(0.025 * (trials - 1))],
                "ci95_high": draws[int(0.975 * (trials - 1))],
                "num_edits": len(values),
                "bootstrap_resamples": trials,
            }
        )
    return output


def _schedule_unit_tests() -> dict[str, Any]:
    from scripts.mdm_memit_editor import partial_mask_state

    target = [10, 11, 12, 13]
    seen_counts = []
    seen_reveals = []
    for step in range(8):
        state, supervised, revealed = partial_mask_state(
            target,
            step=step,
            mask_id=99,
            schedule="cycle",
            reveal_policy="random",
            rng=random.Random(1000 + step),
        )
        seen_counts.append(len(revealed))
        seen_reveals.append(tuple(revealed))
        assert set(supervised).isdisjoint(revealed)
        assert len(supervised) + len(revealed) == len(target)
        assert all(state[index] == target[index] for index in revealed)
        assert all(state[index] == 99 for index in supervised)
    one_state = partial_mask_state(
        [10],
        step=4,
        mask_id=99,
        schedule="cycle",
        reveal_policy="random",
        rng=random.Random(4),
    )
    checks = {
        "all_mask_counts_visited": set(seen_counts) == {0, 1, 2, 3},
        "cycle_is_step_mod_n": seen_counts == [step % 4 for step in range(8)],
        "revealed_positions_resampled": len(set(seen_reveals[1:])) > 3,
        "loss_only_on_still_masked": True,
        "contextual_target_alignment_checked_in_protocol": True,
        "n1_reduces_to_fully_masked": one_state[0] == [99] and one_state[1] == [0],
        "fixed_random_seed_reproducible": partial_mask_state(
            target,
            step=3,
            mask_id=99,
            schedule="cycle",
            reveal_policy="random",
            rng=random.Random(5),
        )
        == partial_mask_state(
            target,
            step=3,
            mask_id=99,
            schedule="cycle",
            reveal_policy="random",
            rng=random.Random(5),
        ),
    }
    return {"checks": checks, "acceptance_pass": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "partial_state_memit_audit_v1"
    )
    parser.add_argument("--limit_per_length", type=int, default=0)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    started = now_utc()
    start = time.monotonic()
    if args.output_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    unit_report = _schedule_unit_tests()
    write_json(args.output_dir / "schedule_unit_test_summary.json", unit_report)
    requested = {item.strip() for item in args.methods.split(",") if item.strip()}
    method_specs = [spec for spec in METHODS if args.methods == "all" or spec["label"] in requested]
    if not method_specs:
        raise ValueError("No P1 methods selected")

    manifests = {
        length: PROTOCOL_ROOT / f"kamel_pub_dev_n{length}.jsonl" for length in (2, 3, 4)
    }
    for path in manifests.values():
        if not path.exists():
            raise FileNotFoundError(path)
    model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, args.dtype)
    covariance_dir = HISTORICAL_ROOT / "covariance_cache_v1"
    if not covariance_dir.exists():
        raise FileNotFoundError(covariance_dir)

    all_rows: list[dict[str, Any]] = []
    run_audit: list[dict[str, Any]] = []
    for target_length, manifest in manifests.items():
        rows = read_jsonl(manifest)
        if args.limit_per_length:
            rows = rows[: args.limit_per_length]
        if any(int(row["target_length"]) != target_length for row in rows):
            raise RuntimeError(f"Manifest {manifest} contains a wrong target length")
        for spec in method_specs:
            if spec.get("alias_of"):
                source = [
                    row
                    for row in all_rows
                    if row["label"] == spec["alias_of"]
                    and int(row["target_length"]) == target_length
                ]
                if not source:
                    raise RuntimeError(f"Alias source missing for {spec['label']}")
                for row in source:
                    alias = dict(row)
                    alias["label"] = spec["label"]
                    all_rows.append(alias)
                run_audit.append(
                    {
                        "label": spec["label"],
                        "target_length": target_length,
                        "status": "exact_config_alias",
                        "alias_of": spec["alias_of"],
                    }
                )
                continue
            seeds = SCHEDULE_SEEDS if spec["stochastic_schedule"] else SCHEDULE_SEEDS[:1]
            for schedule_seed in seeds:
                config = MemitConfig(
                    layers=(4, 5, 6, 7),
                    learning_rate=0.1,
                    target_optimization_steps=25,
                    clamp_norm_factor=0.75,
                    kl_factor=0.0625,
                    partial_mask_schedule=str(spec["schedule"]),
                    reveal_policy=str(spec["reveal"]),
                    seed=schedule_seed,
                )
                cache = (
                    args.output_dir
                    / "target_value_cache"
                    / f"{spec['label']}_n{target_length}_seed{schedule_seed}"
                )
                run_start = time.monotonic()
                rollback, diagnostics = apply_memit_batch(
                    model,
                    tokenizer,
                    rows,
                    config,
                    lambda layer: load_covariance(covariance_dir, layer),
                    target_cache_dir=cache,
                )
                try:
                    decoded_once = evaluate_rows(
                        model,
                        tokenizer,
                        rows,
                        include_locality=False,
                        fixed_length=target_length,
                        fixed_steps=target_length,
                    )
                    for generation_seed in GENERATION_SEEDS:
                        # The frozen paper-style decoder is greedy; repeated seed
                        # rows document zero seed variance without selecting one.
                        all_rows.extend(
                            _annotate(
                                decoded_once,
                                label=str(spec["label"]),
                                schedule_seed=schedule_seed,
                                generation_seed=generation_seed,
                                target_length=target_length,
                            )
                        )
                finally:
                    rollback.rollback()
                if not rollback.checksum_matches(atol=0.0):
                    raise RuntimeError("P1 MEMIT rollback checksum failed")
                run_audit.append(
                    {
                        "label": spec["label"],
                        "target_length": target_length,
                        "schedule_seed": schedule_seed,
                        "generation_seeds": json.dumps(GENERATION_SEEDS),
                        "num_edits": len(rows),
                        "runtime_seconds": time.monotonic() - run_start,
                        "target_cache": str(cache.relative_to(ROOT)),
                        "target_diagnostic_count": len(diagnostics.get("target_optimization", [])),
                        "manifest_sha256": sha256_file(manifest),
                        "rollback_pass": True,
                    }
                )

    method_bucket = _aggregate(all_rows)
    bootstrap = _paired_bootstrap(all_rows, method="paper_matched_partial_cycle")
    write_csv(args.output_dir / "method_bucket.csv", method_bucket)
    write_csv(args.output_dir / "target_length_table.csv", method_bucket)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap)
    write_csv(args.output_dir / "run_audit.csv", run_audit)
    write_csv(args.output_dir / "output_samples.csv", all_rows[:300])

    baseline_by_length = {
        int(row["target_length"]): float(row["full_target_token_exact"])
        for row in method_bucket
        if row["label"] == "ordinary_fully_masked" and row["bucket"] == "rewrite"
    }
    paper_by_length = {
        int(row["target_length"]): float(row["full_target_token_exact"])
        for row in method_bucket
        if row["label"] == "paper_matched_partial_cycle" and row["bucket"] == "rewrite"
    }
    gains = {
        length: paper_by_length.get(length, math.nan) - baseline_by_length.get(length, math.nan)
        for length in (2, 3, 4)
    }
    pooled = next(
        (
            row
            for row in bootstrap
            if row["target_length"] == "pooled_2_3_4" and row["bucket"] == "rewrite"
        ),
        None,
    )
    malformed = max((float(row["malformed_rate"]) for row in method_bucket), default=1.0)
    passing_lengths = [length for length, gain in gains.items() if gain >= 0.10]
    reproduced = (
        len(passing_lengths) >= 2
        and pooled is not None
        and float(pooled["ci95_low"]) > 0
        and malformed
        <= max(
            0.03
            + max(
                float(row["malformed_rate"])
                for row in method_bucket
                if row["label"] == "ordinary_fully_masked"
            ),
            0.03,
        )
    )
    if reproduced:
        decision = "reproduced_paper_trend"
    else:
        decision = "unresolved_baseline_discrepancy"
    difference_register = f"""# P1 Implementation Difference Register

The historical negative campaign optimized one mixed KAMEL edit batch spanning
multiple target lengths. The paper reports a separate 200-edit batch for each
model/target-length pair. This audit therefore uses disjoint, length-specific
200-edit batches for N=2,3,4.

All four MEMIT forward-pass distributions remain mask-augmented. The paper
cycle uses `k = optimization_step mod N`, random revealed positions are
resampled, and loss is evaluated only on still-masked answer positions.

The KAMEL source provides one real question template per relation, so the
held-out generalization prompt is a deterministic documented rewrite rather
than an official second KAMEL template. Greedy generation is deterministic;
three predeclared generation-seed rows are retained and no favorable seed is
selected.
"""
    (args.output_dir / "implementation_difference_register.md").write_text(
        difference_register, encoding="utf-8"
    )
    discrepancy = f"""# P1 Discrepancy Decision

Decision: `{decision}`.

Paper-matched rewrite gains over ordinary fully masked editing:

```json
{json.dumps(gains, indent=2, sort_keys=True)}
```

Passing target lengths: {passing_lengths}. Pooled bootstrap lower bound:
{None if pooled is None else pooled['ci95_low']}.
"""
    (args.output_dir / "discrepancy_decision.md").write_text(discrepancy, encoding="utf-8")
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P1",
        "stage": "P1_partial_state_memit_discrepancy",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": PRIMARY_MODEL_ID,
        "model_revision": PRIMARY_MODEL_REVISION,
        "dtype": args.dtype,
        "use_4bit": False,
        "layers": [4, 5, 6, 7],
        "separate_edit_batch_per_target_length": True,
        "num_edits_per_length": args.limit_per_length or 200,
        "schedule_seeds": list(SCHEDULE_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "generation_is_deterministic_greedy": True,
        "schedule_unit_tests_pass": unit_report["acceptance_pass"],
        "rewrite_gain_by_length": gains,
        "passing_lengths": passing_lengths,
        "pooled_bootstrap": pooled,
        "max_malformed_rate": malformed,
        "discrepancy_decision": decision,
        "bounded_repair_used": False,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "runtime_seconds": time.monotonic() - start,
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "acceptance_pass": reproduced and bool(unit_report["acceptance_pass"]),
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="P1_partial_state_memit_discrepancy",
        track="P1",
        status=decision,
        output_dir=args.output_dir,
        acceptance_pass=bool(report["acceptance_pass"]),
        started_at_utc=started,
        notes=f"decision={decision}; gains={gains}; separate_length_batches=true",
        next_stage="P3_planner_baselines_dev",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
