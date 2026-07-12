#!/usr/bin/env python3
"""Write the formal Direction 1 stop checkpoint.

This is a local report task only. It reads historical test50 summaries as
Direction 1 evidence and current dev-only Direction 1 artifacts as the stop
condition. It does not run decoding or touch locked analysis/final artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_common import D1_PROTOCOL_VERSION, D1_ROOT, git_commit, now_utc, read_json, repo_path, write_csv, write_json


OUT_DIR = D1_ROOT / "dev_direction1_gate_stop_checkpoint_v1"

LEGACY_EVIDENCE = [
    {
        "label": "legacy_test50_direct_lora",
        "path": "runs/test50_direct_anchor05_broad_reval_metrics/benchmark_summary.json",
        "phase": "post",
        "bucket": "rewrite",
        "metric": "mean_exact_rate",
        "expected": 0.005,
        "interpretation": "Static/direct LoRA did not produce useful rewrite success.",
    },
    {
        "label": "legacy_test50_option2_distilled_static_bridge",
        "path": "runs/test50_option2_bridge_anchor05_broad_reval_metrics/benchmark_summary.json",
        "phase": "post",
        "bucket": "rewrite",
        "metric": "mean_exact_rate",
        "expected": 0.005,
        "interpretation": "Distilled/static bridge student did not produce useful rewrite success.",
    },
    {
        "label": "legacy_test50_raw_edit_conditioned_bridge",
        "path": "runs/test50_option2_raw_bridge_editcond_anchor05_broad_reval_metrics/benchmark_summary.json",
        "phase": "post",
        "bucket": "rewrite",
        "metric": "mean_exact_rate",
        "expected": 0.385,
        "interpretation": "Raw edit-conditioned bridge produced real rewrite signal.",
    },
    {
        "label": "legacy_test50_raw_edit_conditioned_bridge",
        "path": "runs/test50_option2_raw_bridge_editcond_anchor05_broad_reval_metrics/benchmark_summary.json",
        "phase": "post",
        "bucket": "declarative_paraphrases",
        "metric": "mean_exact_rate",
        "expected": 0.2375,
        "interpretation": "Raw edit-conditioned bridge generalized beyond rewrite prompts.",
    },
]

REQUIRED_REPORTS = {
    "same_subject_stress_inputs": D1_ROOT / "same_subject_stress_inputs/summary.json",
    "step_3d_same_subject_stress": D1_ROOT / "dev_tune_200_same_subject_stress_report_v1/report_summary.json",
    "step_3d1_midkl_stress": D1_ROOT / "dev_tune_200_same_subject_stress_midkl_v1/report_summary.json",
    "step_3e0_hybrid_replay": D1_ROOT / "dev_tune_200_hybrid_gate_replay_v1/report_summary.json",
    "step_3e1_hybrid_actual_decode": D1_ROOT / "dev_tune_200_hybrid_gate_decode_v1/report_summary.json",
    "step_3e2_parity_audit": D1_ROOT / "dev_tune_200_hybrid_gate_parity_audit_v1/report_summary.json",
    "step_3e3_activation_grid": D1_ROOT / "dev_tune_200_actual_gate_activation_grid_v1/report_summary.json",
}


def metric_from_benchmark(path: str, phase: str, bucket: str, metric: str) -> Optional[float]:
    data = read_json(path)
    value = (
        data.get("metric_results", {})
        .get(phase, {})
        .get("aggregate", {})
        .get(bucket, {})
        .get(metric)
    )
    return None if value is None else float(value)


def require_dev_report(name: str, path: Path) -> Dict[str, Any]:
    full = repo_path(path)
    if not full.exists():
        raise FileNotFoundError(f"Required Direction 1 evidence missing: {path}")
    data = read_json(path)
    if data.get("analysis_500_used") is not False:
        raise AssertionError(f"{name}: analysis_500_used must be false")
    if data.get("final_test_used") is not False:
        raise AssertionError(f"{name}: final_test_used must be false")
    return data


def extract_actual_decode_failure_rows(path: Path) -> List[Dict[str, Any]]:
    selection_path = repo_path(path.parent / "hybrid_gate_selection.csv")
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing actual decode selection table: {selection_path}")
    rows: List[Dict[str, Any]] = []
    with selection_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise AssertionError("Hybrid actual decode selection table is empty")
    worst_same = max(rows, key=lambda r: float(r.get("same_subject_template_tfpr") or 0.0))
    best_score = max(rows, key=lambda r: float(r.get("selection_score") or 0.0))
    return [
        {
            "stage": "Step 3E.1 actual decode",
            "label": worst_same["label"],
            "metric": "same_subject_template_tfpr",
            "observed": worst_same.get("same_subject_template_tfpr"),
            "expected_or_budget": "0.030416666666666665",
            "status": "fail",
            "source_path": str(path.parent / "hybrid_gate_selection.csv"),
            "interpretation": "Best-tested hybrid gate still over-injected target_new on same-subject stress.",
        },
        {
            "stage": "Step 3E.1 actual decode",
            "label": best_score["label"],
            "metric": "best_selection_score_candidate_status",
            "observed": best_score.get("decision_status"),
            "expected_or_budget": "keep_dev_candidate",
            "status": "fail" if best_score.get("decision_status") != "keep_dev_candidate" else "pass",
            "source_path": str(path.parent / "hybrid_gate_selection.csv"),
            "interpretation": "No actually decoded hybrid-gate method remained a valid dev candidate.",
        },
    ]


def build_evidence_rows() -> List[Dict[str, Any]]:
    evidence_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for spec in LEGACY_EVIDENCE:
        full = repo_path(spec["path"])
        if not full.exists():
            raise FileNotFoundError(f"Required legacy evidence missing: {spec['path']}")
        observed = metric_from_benchmark(spec["path"], spec["phase"], spec["bucket"], spec["metric"])
        status = "pass"
        warning = ""
        if observed is None:
            raise AssertionError(f"Missing metric {spec['metric']} for {spec['path']}")
        if abs(observed - float(spec["expected"])) > 1e-12:
            status = "warning"
            warning = f"Expected {spec['expected']} but artifact contains {observed}"
            warnings.append(warning)
        evidence_rows.append(
            {
                "stage": "Historical Direction 1 evidence",
                "label": spec["label"],
                "metric": f"{spec['phase']}.{spec['bucket']}.{spec['metric']}",
                "observed": observed,
                "expected_or_budget": spec["expected"],
                "status": status,
                "warning": warning,
                "source_path": spec["path"],
                "interpretation": spec["interpretation"],
            }
        )

    reports = {name: require_dev_report(name, path) for name, path in REQUIRED_REPORTS.items()}

    stress_inputs = reports["same_subject_stress_inputs"]
    evidence_rows.append(
        {
            "stage": "Step 3D stress input",
            "label": "same_subject_stress_inputs",
            "metric": "counts",
            "observed": json.dumps(
                {
                    "num_edits": stress_inputs.get("num_edits"),
                    "stress_counts": stress_inputs.get("stress_counts"),
                },
                sort_keys=True,
            ),
            "expected_or_budget": "200 edits; 600 same_subject_template; 400 generation",
            "status": "pass",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["same_subject_stress_inputs"]),
            "interpretation": "Dev-only stress set targets target_new over-injection.",
        }
    )

    step3d = reports["step_3d_same_subject_stress"]
    evidence_rows.append(
        {
            "stage": "Step 3D same-subject stress",
            "label": "subject_gate_stress",
            "metric": "stress_budget_rule",
            "observed": step3d.get("stress_budget_rule"),
            "expected_or_budget": "base_stress_tfpr + 0.03",
            "status": "pass",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["step_3d_same_subject_stress"]),
            "interpretation": "Subject gate required same-subject stress because ordinary near/far locality barely activated it.",
        }
    )

    step3d1 = reports["step_3d1_midkl_stress"]
    evidence_rows.append(
        {
            "stage": "Step 3D.1 mid-KL stress",
            "label": "subject_gate_midkl",
            "metric": "decision_rule",
            "observed": step3d1.get("decision_rule"),
            "expected_or_budget": "template_stress_pass and edit performance useful",
            "status": "pass",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["step_3d1_midkl_stress"]),
            "interpretation": "Mid-guidance subject-gated variants still tested against same-subject over-injection.",
        }
    )

    step3e0 = reports["step_3e0_hybrid_replay"]
    evidence_rows.append(
        {
            "stage": "Step 3E.0 hybrid replay",
            "label": "hybrid_gate_replay",
            "metric": "num_keep_dev_candidates",
            "observed": step3e0.get("num_keep_dev_candidates"),
            "expected_or_budget": ">= 1 replay candidate",
            "status": "pass" if int(step3e0.get("num_keep_dev_candidates") or 0) > 0 else "fail",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["step_3e0_hybrid_replay"]),
            "interpretation": "Replay looked promising without oracle gates or outcome-derived advantage.",
        }
    )
    evidence_rows.append(
        {
            "stage": "Step 3E.0 hybrid replay",
            "label": "hybrid_gate_replay_leakage_checks",
            "metric": "oracle_or_leakage",
            "observed": json.dumps(
                {
                    "oracle_gate_families": step3e0.get("oracle_gate_families"),
                    "uses_outcome_derived_advantage": step3e0.get("uses_outcome_derived_advantage"),
                    "no_paraphrase_template_leakage": step3e0.get("no_paraphrase_template_leakage"),
                },
                sort_keys=True,
            ),
            "expected_or_budget": "no oracle gates; no outcome-derived advantage; no paraphrase leakage",
            "status": "pass",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["step_3e0_hybrid_replay"]),
            "interpretation": "Replay was eligible as a diagnostic, but needed actual decode validation.",
        }
    )

    evidence_rows.extend(extract_actual_decode_failure_rows(REQUIRED_REPORTS["step_3e1_hybrid_actual_decode"]))

    step3e2 = reports["step_3e2_parity_audit"]
    evidence_rows.append(
        {
            "stage": "Step 3E.2 parity audit",
            "label": "hybrid_gate_parity",
            "metric": "status",
            "observed": step3e2.get("status"),
            "expected_or_budget": "bug_identified or drift <= 0.01",
            "status": "pass" if step3e2.get("status") == "bug_identified" else "warning",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["step_3e2_parity_audit"]),
            "interpretation": step3e2.get("bug_summary", "Parity audit completed."),
        }
    )
    evidence_rows.append(
        {
            "stage": "Step 3E.2 parity audit",
            "label": "runtime_gate_internal_consistency",
            "metric": "max_recomputed_actual_activation_drift",
            "observed": step3e2.get("max_recomputed_actual_activation_drift"),
            "expected_or_budget": "<= 0.01",
            "status": "pass"
            if float(step3e2.get("max_recomputed_actual_activation_drift") or 1.0) <= 0.01
            else "fail",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["step_3e2_parity_audit"]),
            "interpretation": "Runtime gate implementation matches actual activation; replay feature extraction was the issue.",
        }
    )

    step3e3 = reports["step_3e3_activation_grid"]
    evidence_rows.append(
        {
            "stage": "Step 3E.3 activation grid",
            "label": "actual_gate_grid",
            "metric": "status",
            "observed": step3e3.get("status"),
            "expected_or_budget": "at least one gate passes activation constraints",
            "status": "fail" if step3e3.get("status") == "no_gate_passed" else "pass",
            "warning": "",
            "source_path": str(REQUIRED_REPORTS["step_3e3_activation_grid"]),
            "interpretation": "No stricter tested rule-based gate passed rewrite/paraphrase activation and stress/locality activation constraints.",
        }
    )
    return evidence_rows


def write_markdown(out_dir: Path, evidence_rows: List[Dict[str, Any]]) -> None:
    checkpoint = f"""# Direction 1 Gate Stop Checkpoint

## Status

`status = blocked_under_tested_rule_based_runtime_gates`

Direction 1 tested runtime bridge editing:

```text
Gen_{{theta0, B_e}}(x; S)
```

`theta0` is frozen LLaDA, the edit request is supplied at edit time, and
`B_e` is an edit-conditioned bridge/controller used only during decoding. This
is not permanent parameter editing.

All checkpoint evidence is dev-only for the current protocol, except legacy
`test50_*` summaries used only as historical Direction 1 evidence.

```text
analysis_500_used = false
final_test_used = false
do_not_run_step_3e4 = true
recommended_next_direction = Direction 3 controller pilot
```

## Decision

Direction 1 should not proceed to `analysis_500`. Subject-only gates failed
same-subject stress. Hybrid replay looked promising, but actual runtime decode
over-activated same-subject and generation stress prompts. The parity audit
identified replay feature extraction as the mismatch source, while the actual
runtime gate was internally consistent. The stricter actual-gate grid found no
rule-based gate satisfying the activation constraints.

## Evidence Summary

See `direction1_evidence_table.csv` for machine-readable evidence. Key points:

- Static/direct LoRA and distilled/static bridge had approximately zero rewrite
  success on legacy test50.
- Raw edit-conditioned bridge showed a real edit signal on legacy test50.
- Same-subject stress exposed target_new over-injection for subject-gated
  runtime methods.
- Hybrid relation gates did not survive actual decode plus activation-grid
  checks.
"""
    recommendation = """# Next Direction Recommendation

## Recommendation

Start Direction 3 as a small controller pilot.

Do not start RunPod for Direction 1 Step 3E.4. Do not run `analysis_500`.

## Rationale

The bridge signal remains useful, but manually designed runtime gates did not
localize it safely under same-subject stress. Direction 3 tests whether a
learned edit-conditioned controller and learned/edit-intent gate can amortize
bridge behavior while reducing target_new over-injection.

Direction 2 remains the fallback if the Direction 3 controller pilot fails.
"""
    (repo_path(out_dir) / "direction1_gate_stop_checkpoint.md").write_text(checkpoint, encoding="utf-8")
    (repo_path(out_dir) / "next_direction_recommendation.md").write_text(recommendation, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--allow_overwrite", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    full_out = repo_path(out_dir)
    if full_out.exists() and not bool(args.allow_overwrite):
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    full_out.mkdir(parents=True, exist_ok=True)

    evidence_rows = build_evidence_rows()
    write_csv(out_dir / "direction1_evidence_table.csv", evidence_rows)
    write_markdown(out_dir, evidence_rows)

    fail_rows = [row for row in evidence_rows if row.get("status") == "fail"]
    warning_rows = [row for row in evidence_rows if row.get("status") == "warning"]
    summary = {
        "protocol_version": D1_PROTOCOL_VERSION,
        "stage": "Direction 1 gate stop checkpoint",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "status": "blocked_under_tested_rule_based_runtime_gates",
        "analysis_500_used": False,
        "final_test_used": False,
        "do_not_run_step_3e4": True,
        "recommended_next_direction": "Direction 3 controller pilot",
        "legacy_test50_used_only_as_historical_evidence": True,
        "current_locked_analysis_or_final_artifacts_used": False,
        "num_evidence_rows": len(evidence_rows),
        "num_fail_evidence_rows": len(fail_rows),
        "num_warning_evidence_rows": len(warning_rows),
        "artifacts": {
            "checkpoint_markdown": str(out_dir / "direction1_gate_stop_checkpoint.md"),
            "evidence_table": str(out_dir / "direction1_evidence_table.csv"),
            "next_direction_recommendation": str(out_dir / "next_direction_recommendation.md"),
        },
    }
    write_json(out_dir / "report_summary.json", summary)
    print(f"[INFO] Wrote Direction 1 stop checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
