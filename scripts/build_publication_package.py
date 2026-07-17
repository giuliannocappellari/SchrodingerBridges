#!/usr/bin/env python3
"""P8 statistics, reproducibility, and terminal publication-readiness package."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduce_paper import check_dp, reproduce_figure, reproduce_table
from scripts.mask_pattern_kl_control import solve_exact_kl_control
from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    STATE_ROOT,
    git_commit,
    now_utc,
    read_json,
    secondary_backbone_terminal_dir,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.mask_pattern_publication_runtime import reference_table


REQUIRED_OUTPUTS = (
    "report_summary.json",
    "top_tier_readiness.json",
    "main_results_table.csv",
    "compute_matched_table.csv",
    "second_backbone_table.csv",
    "editor_generality_table.csv",
    "target_length_table.csv",
    "beta_ablation.csv",
    "planner_ablation.csv",
    "same_subject_stress_table.csv",
    "malformed_and_locality_table.csv",
    "paired_bootstrap.csv",
    "holm_corrected_tests.csv",
    "power_analysis.json",
    "theory_statement.md",
    "naming_decision.md",
    "complexity_analysis.md",
    "trajectory_examples.md",
    "failure_cases.csv",
    "artifact_availability.json",
    "reproducibility_manifest.json",
    "final_research_report.md",
    "paper_outline.md",
    "paper_claim_recommendation.md",
)


def _copy(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_cost_tables(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row.pop("item_key"))] = row
    return output


def _trajectory_examples(p3_dir: Path, lock: Mapping[str, Any]) -> str:
    rows = _read_gzip_jsonl(p3_dir / "per_prompt_results.jsonl.gz")
    finite = str(lock["finite_controller_label"])
    methods = ("default_confidence", "one_step_myopic", "deterministic_global", finite)
    finite_rows = [
        row
        for row in rows
        if row["family"] == finite
        and row["bucket"] == "rewrite"
        and int(row["target_length"]) in {3, 4}
    ]
    selected_cases = []
    for desired in (True, False):
        match = next(
            (row for row in finite_rows if bool(row["full_target_exact"]) is desired),
            None,
        )
        if match is not None:
            selected_cases.append((str(match["case_id"]), int(match["target_length"]), desired))
    sections = ["# Controlled Trajectory Examples", ""]
    tables_by_length = {
        n: _read_cost_tables(p3_dir / f"edited_cost_tables_n{n}.jsonl.gz")
        for n in {length for _, length, _ in selected_cases}
    }
    for case_id, n, success in selected_cases:
        table = tables_by_length[n][f"{case_id}::rewrite"]
        costs = {
            tuple(map(int, key.split(":"))): float(value)
            for key, value in table["costs"].items()
        }
        reference = reference_table(table, str(lock["reference_process"]))
        solution = solve_exact_kl_control(
            costs, n, beta=float(lock["beta"]), reference=reference
        )
        sections.extend(
            [
                f"## {case_id}",
                "",
                f"Finite-controller exact success: `{success}`; target length: `{n}`.",
                "",
            ]
        )
        for family in methods:
            row = next(
                (
                    item
                    for item in rows
                    if str(item["case_id"]) == case_id
                    and item["bucket"] == "rewrite"
                    and item["family"] == family
                ),
                None,
            )
            if row is None:
                continue
            sections.extend(
                [
                    f"### {family}",
                    "",
                    "| step | mask | chosen position | target probability | immediate cost | log backward partition | finite-policy probability | remaining free energy |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for step in json.loads(row["trajectory"]):
                mask = int(step["mask_before"])
                position = int(step["chosen_position"])
                key = f"{mask}:{position}"
                free_energy = (
                    -float(solution.log_partition[mask]) / float(lock["beta"])
                    if float(lock["beta"]) > 0
                    else math.nan
                )
                sections.append(
                    "| {step} | {mask} | {position} | {prob:.6f} | {cost:.6f} | "
                    "{log_z:.6f} | {policy:.6f} | {free:.6f} |".format(
                        step=int(step["step"]),
                        mask=mask,
                        position=position,
                        prob=float(table["target_probabilities"][key]),
                        cost=float(table["costs"][key]),
                        log_z=float(solution.log_partition[mask]),
                        policy=float(solution.policy[mask][position]),
                        free=free_energy,
                    )
                )
            sections.extend(
                [
                    "",
                    f"Output: `{row['output_text']}`; exact: `{row['full_target_exact']}`; "
                    f"trajectory target cost: `{float(row['trajectory_target_cost']):.6f}`.",
                    "",
                ]
            )
    if not selected_cases:
        sections.append("No eligible trajectory rows were available.")
    return "\n".join(sections) + "\n"


def _mechanism_diagnostics(p3_dir: Path) -> list[dict[str, Any]]:
    rows = _read_gzip_jsonl(p3_dir / "per_prompt_results.jsonl.gz")
    output = []
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["bucket"] == "rewrite":
            by_family[str(row["family"])].append(row)
    for family, values in sorted(by_family.items()):
        costs = [float(row["trajectory_target_cost"]) for row in values]
        success = [float(row["full_target_exact"]) for row in values]
        correlation = (
            statistics.correlation(costs, success)
            if len(costs) > 1 and len(set(costs)) > 1 and len(set(success)) > 1
            else math.nan
        )
        output.append(
            {
                "family": family,
                "num_rows": len(values),
                "mean_trajectory_cost": sum(costs) / len(costs),
                "rewrite_exact": sum(success) / len(success),
                "trajectory_cost_success_pearson": correlation,
                "mean_effective_reveal_orders": math.exp(
                    sum(float(row["path_entropy"]) for row in values) / len(values)
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=CAMPAIGN_ROOT / "final_publication_package_v1"
    )
    parser.add_argument("--allow_overwrite", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    if args.output_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    p1 = read_json(CAMPAIGN_ROOT / "partial_state_memit_audit_v1" / "report_summary.json")
    p2 = read_json(CAMPAIGN_ROOT / "theory_and_naming_v1" / "report_summary.json")
    p3_dir = CAMPAIGN_ROOT / "planner_baselines_dev_v1"
    p3 = read_json(p3_dir / "report_summary.json")
    p4_dir = CAMPAIGN_ROOT / "llada_locked_confirmation_v1"
    p4 = read_json(p4_dir / "report_summary.json")
    p5_dir = secondary_backbone_terminal_dir()
    p5 = read_json(p5_dir / "report_summary.json")
    p6_dir = CAMPAIGN_ROOT / "editor_generality_v1"
    p6 = read_json(p6_dir / "report_summary.json")
    p7_dir = CAMPAIGN_ROOT / "approximate_solver_v1"
    p7 = read_json(p7_dir / "report_summary.json")
    lock = read_json(CAMPAIGN_ROOT / "dev_method_lock.json")

    copies = {
        p4_dir / "main_results.csv": args.output_dir / "main_results_table.csv",
        p4_dir / "compute_matched_results.csv": args.output_dir / "compute_matched_table.csv",
        p5_dir / "locked_results.csv": args.output_dir / "second_backbone_table.csv",
        p6_dir / "editor_condition_results.csv": args.output_dir / "editor_generality_table.csv",
        p4_dir / "target_length_results.csv": args.output_dir / "target_length_table.csv",
        p3_dir / "beta_sweep.csv": args.output_dir / "beta_ablation.csv",
        p3_dir / "planner_results.csv": args.output_dir / "planner_ablation.csv",
        p4_dir / "same_subject_stress.csv": args.output_dir / "same_subject_stress_table.csv",
        p4_dir / "locality_malformed.csv": args.output_dir / "malformed_and_locality_table.csv",
        p4_dir / "paired_bootstrap.csv": args.output_dir / "paired_bootstrap.csv",
        p4_dir / "holm_corrected_tests.csv": args.output_dir / "holm_corrected_tests.csv",
        p3_dir / "power_analysis.json": args.output_dir / "power_analysis.json",
        CAMPAIGN_ROOT / "theory_and_naming_v1" / "proposition_and_proof.md": args.output_dir / "theory_statement.md",
        CAMPAIGN_ROOT / "theory_and_naming_v1" / "naming_decision.md": args.output_dir / "naming_decision.md",
        p7_dir / "approximation_decision.md": args.output_dir / "complexity_analysis.md",
        p4_dir / "failure_cases.csv": args.output_dir / "failure_cases.csv",
    }
    for source, destination in copies.items():
        _copy(source, destination)
    (args.output_dir / "trajectory_examples.md").write_text(
        _trajectory_examples(p3_dir, lock), encoding="utf-8"
    )
    write_csv(args.output_dir / "mechanism_diagnostics.csv", _mechanism_diagnostics(p3_dir))
    dp_report = check_dp()
    write_json(args.output_dir / "dp_reproduction_check.json", dp_report)

    criteria = {
        "partial_state_discrepancy_resolved": bool(p1["acceptance_pass"]),
        "fresh_locked_llada_primary_pass": bool(p4["acceptance_pass"]),
        "compute_matched_non_sb_beaten": bool(p3["finite_beta_mechanism_pass"])
        and bool(p4["acceptance_pass"]),
        "finite_beta_beyond_limits": bool(p3["finite_beta_mechanism_pass"]),
        "second_backbone_consistent": bool(p5["acceptance_pass"]),
        "two_editor_conditions_positive": bool(p6["acceptance_pass"]),
        "formal_naming_defensible": bool(p2["acceptance_pass"]),
        "locked_locality_and_malformed_pass": (
            float(p4["same_subject_tfpr_delta"]) <= 0.03
            and float(p4["finite_malformed_rate"]) <= 0.05
        ),
        "exact_dp_reproduction_pass": bool(dp_report["acceptance_pass"]),
    }
    if not criteria["fresh_locked_llada_primary_pass"]:
        classification = "fresh_confirmation_failed"
    elif all(criteria.values()):
        classification = "top_tier_ready"
    elif criteria["compute_matched_non_sb_beaten"]:
        classification = "narrow_method_ready"
    else:
        classification = "diagnostic_only"
    readiness = {
        "campaign_id": CAMPAIGN_ID,
        "classification": classification,
        "criteria": criteria,
        "finite_controller": lock["finite_controller_label"],
        "compute_matched_baseline": lock["best_non_sb_planner"],
        "formal_method_name": p2["naming_decision"],
        "classical_schrodinger_bridge_claim_allowed": False,
    }
    write_json(args.output_dir / "top_tier_readiness.json", readiness)
    reproducibility = {
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "models": {
            "llada": {"id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION},
            "dream": {"id": SECONDARY_MODEL_ID, "revision": SECONDARY_MODEL_REVISION},
            "secondary_backbone_terminal": {
                "id": p5.get("model_id", SECONDARY_MODEL_ID),
                "revision": p5.get("model_revision", SECONDARY_MODEL_REVISION),
                "backbone_profile": p5.get("backbone_profile", "dream"),
                "evidence_tier": p5.get("evidence_tier", "top_tier_secondary_backbone"),
                "classification": p5.get("classification", "unknown"),
            },
        },
        "dev_method_lock_sha256": sha256_file(CAMPAIGN_ROOT / "dev_method_lock.json"),
        "protocol_report_sha256": sha256_file(CAMPAIGN_ROOT / "protocol_v1" / "report_summary.json"),
        "memit_configuration": lock["target_value_config"],
        "layers": lock["layers"],
        "reference_process": lock["reference_process"],
        "beta": lock["beta"],
        "controller_action_rule": lock["controller_action_rule"],
        "span_policy": lock["span_policy"],
        "random_seeds": lock["random_policy_seeds"],
        "bootstrap_resamples": lock["bootstrap_resamples"],
        "commands": [
            "python reproduce_paper.py --table main",
            "python reproduce_paper.py --figure main",
            "python reproduce_paper.py --check-dp",
        ],
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
    }
    write_json(args.output_dir / "reproducibility_manifest.json", reproducibility)
    claim = {
        "top_tier_ready": "A finite-beta Doob-transformed mask-pattern controller provides fresh, compute-matched, cross-backbone, editor-general factual-edit gains.",
        "narrow_method_ready": "A finite-beta Doob-transformed mask-pattern controller improves fresh LLaDA multi-token realization under the bounded tested protocol.",
        "diagnostic_only": "Reveal-order planning matters, but finite-beta control does not establish an advantage over the strongest compute-matched planner.",
        "fresh_confirmation_failed": "The development effect did not survive the fresh locked LLaDA confirmation.",
    }[classification]
    (args.output_dir / "paper_claim_recommendation.md").write_text(
        f"# Paper Claim Recommendation\n\nClassification: `{classification}`.\n\n{claim}\n",
        encoding="utf-8",
    )
    (args.output_dir / "paper_outline.md").write_text(
        "# Paper Outline\n\n1. Multi-token factual editing as controlled mask-pattern generation\n"
        "2. Exact Doob-transformed KL path control\n3. Fresh KAMEL protocol and compute matching\n"
        "4. Locked LLaDA confirmation\n5. Dream and editor-generality evidence\n"
        "6. Exact/approximate scaling\n7. Limitations and failure cases\n",
        encoding="utf-8",
    )
    report_text = f"""# Final Research Report

## Decision

`{classification}`

## Primary Result

The frozen controller `{lock['finite_controller_label']}` was compared with
`{lock['best_non_sb_planner']}` on fresh locked KAMEL facts. The pooled N=3/N=4
rewrite delta was {float(p4['pooled_primary_bootstrap']['mean_delta']):.6f}
with 95% CI [{float(p4['pooled_primary_bootstrap']['ci95_low']):.6f},
{float(p4['pooled_primary_bootstrap']['ci95_high']):.6f}].

## Scope

The mathematically defensible name is `{p2['naming_decision']}`. This is a
Doob-transformed entropy/KL path-control process, not a classical
endpoint-constrained Schrödinger bridge. Dream classification:
`{p5['classification']}`. Editor-generality decision: `{p6['decision']}`.
Approximation decision: `{p7['decision']}`.

## Limitations

The KAMEL source supplies one real cloze template per relation, so paraphrase
prompts are documented held-out deterministic rewrites. Exact planning scales
exponentially in target length. The N=7 through N=10 extension was not run
because no fresh manifests for those lengths were precommitted.
"""
    (args.output_dir / "final_research_report.md").write_text(report_text, encoding="utf-8")

    # The figure and table commands must reproduce successfully before the
    # package can be declared valid.
    reproduce_table()
    reproduce_figure()
    preliminary_required = [path for path in REQUIRED_OUTPUTS if path not in {"report_summary.json", "artifact_availability.json"}]
    missing = [name for name in preliminary_required if not (args.output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing final package outputs: {missing}")
    artifacts = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_availability.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_json(
        args.output_dir / "artifact_availability.json",
        {"all_required_available": True, "artifacts": artifacts},
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P8",
        "stage": "P8_publication_package",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "classification": classification,
        "criteria": criteria,
        "scientific_acceptance_pass": classification in {"top_tier_ready", "narrow_method_ready"},
        "package_validation_pass": True,
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    if any(not (args.output_dir / name).exists() for name in REQUIRED_OUTPUTS):
        raise RuntimeError("Final package failed its post-write completeness check")
    record_stage(
        stage="P8_publication_package",
        track="P8",
        status=classification,
        output_dir=args.output_dir,
        acceptance_pass=True,
        started_at_utc=started,
        notes=f"classification={classification}; package_validation=true",
        next_stage="terminal_pod_stop",
    )
    state_path = STATE_ROOT / "campaign_state.json"
    state = read_json(state_path)
    state.update(
        {
            "campaign_status": "completed",
            "publication_readiness": classification,
            "current_stage": "P8_publication_package",
            "next_stage": "terminal_pod_stop",
            "pod_status": "stop_pending_after_terminal_validation",
            "updated_at_utc": now_utc(),
            "last_git_commit": git_commit(),
        }
    )
    write_json(state_path, state)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
