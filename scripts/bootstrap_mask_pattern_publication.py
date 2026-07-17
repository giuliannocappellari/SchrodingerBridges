#!/usr/bin/env python3
"""Initialize P0 and audit the immutable historical seed artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mask_pattern_publication_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    HISTORICAL_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    git_commit,
    initialize_state,
    now_utc,
    read_json,
    record_stage,
    sha256_file,
    write_csv,
    write_json,
)


REQUIRED_ROOT_FILES = (
    "AGENTS.md",
    "ACTIVE_RESEARCH_CAMPAIGN.json",
    "PUBLICATION_PROTOCOL_REGISTRY.json",
    "MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_PLAN.md",
    "PARTIAL_STATE_MEMIT_AUDIT_PLAN.md",
    "THEORY_AND_NAMING_PLAN.md",
    "COMPUTE_MATCHED_BASELINES_PLAN.md",
    "LOCKED_LLADA_CONFIRMATION_PLAN.md",
    "SECOND_BACKBONE_DREAM_PLAN.md",
    "EDITOR_GENERALITY_PLAN.md",
    "APPROXIMATE_SOLVER_PLAN.md",
    "PAPER_REPRODUCIBILITY_PLAN.md",
)

REQUIRED_HISTORICAL = {
    "historical_final_summary": HISTORICAL_ROOT / "final_research_package_v1/report_summary.json",
    "historical_m1_summary": HISTORICAL_ROOT / "M1_mdm_memit_reproduction_v1/report_summary.json",
    "historical_m1_locked_summary": HISTORICAL_ROOT
    / "M1_mdm_memit_reproduction_v1/locked_reproduction_v1/report_summary.json",
    "historical_m2_summary": HISTORICAL_ROOT / "M2_partial_mask_memit_v1/report_summary.json",
    "historical_m4_summary": HISTORICAL_ROOT / "M4_mask_pattern_sb_v1/report_summary.json",
    "historical_m4_results": HISTORICAL_ROOT / "M4_mask_pattern_sb_v1/main_results_by_length.csv",
    "historical_m4_bootstrap": HISTORICAL_ROOT / "M4_mask_pattern_sb_v1/paired_bootstrap.csv",
    "historical_protocol_summary": HISTORICAL_ROOT / "protocol/report_summary.json",
}


def _json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "SB-publication-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _git_head(url: str) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", url, "HEAD"], text=True, timeout=30
        ).strip()
        return output.split()[0] if output else None
    except (OSError, subprocess.SubprocessError):
        return None


def _find_historical_metric(payload: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                return payload[key]
        for value in payload.values():
            found = _find_historical_metric(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_historical_metric(value, keys)
            if found is not None:
                return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "source_audit")
    parser.add_argument("--allow_existing", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    started = now_utc()
    if not args.allow_existing and args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = initialize_state()
    if not state.get("autonomous_mode"):
        raise RuntimeError("MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_MODE=1 is required")

    missing_root = [name for name in REQUIRED_ROOT_FILES if not (ROOT / name).is_file()]
    missing_historical = [name for name, path in REQUIRED_HISTORICAL.items() if not path.is_file()]
    if missing_root or missing_historical:
        (args.output_dir / "missing_artifacts.md").write_text(
            "# Missing Artifacts\n\n"
            + "\n".join(f"- root: `{name}`" for name in missing_root)
            + "\n"
            + "\n".join(f"- historical: `{name}`" for name in missing_historical)
            + "\n",
            encoding="utf-8",
        )
        raise FileNotFoundError(
            f"Missing root files={missing_root}; historical artifacts={missing_historical}"
        )

    artifacts = []
    for name, path in REQUIRED_HISTORICAL.items():
        artifacts.append(
            {
                "artifact": name,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "immutable": True,
                "usage": "historical context or exclusion provenance only",
            }
        )
    write_json(
        args.output_dir / "historical_artifact_registry.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": now_utc(),
            "historical_protocol": "masked_diffusion_memit_sb_positive_result_v1",
            "artifacts": artifacts,
            "historical_analysis_500_used": False,
            "historical_final_test_used": False,
        },
    )

    m1 = read_json(REQUIRED_HISTORICAL["historical_m1_summary"])
    m1_locked = read_json(REQUIRED_HISTORICAL["historical_m1_locked_summary"])
    m2 = read_json(REQUIRED_HISTORICAL["historical_m2_summary"])
    m4 = read_json(REQUIRED_HISTORICAL["historical_m4_summary"])
    historical_rows = [
        {
            "track": "M1",
            "metric": "rewrite_exact",
            "value": _find_historical_metric(m1_locked, ("rewrite_exact", "efficacy")),
            "artifact": str(REQUIRED_HISTORICAL["historical_m1_locked_summary"].relative_to(ROOT)),
            "new_locked_result": False,
        },
        {
            "track": "M1",
            "metric": "paraphrase_exact",
            "value": _find_historical_metric(m1_locked, ("paraphrase_exact", "generalization")),
            "artifact": str(REQUIRED_HISTORICAL["historical_m1_locked_summary"].relative_to(ROOT)),
            "new_locked_result": False,
        },
        {
            "track": "M2",
            "metric": "acceptance_pass",
            "value": m2.get("acceptance_pass"),
            "artifact": str(REQUIRED_HISTORICAL["historical_m2_summary"].relative_to(ROOT)),
            "new_locked_result": False,
        },
        {
            "track": "M4",
            "metric": "sb_specific_positive_result",
            "value": m4.get("sb_specific_positive_result"),
            "artifact": str(REQUIRED_HISTORICAL["historical_m4_summary"].relative_to(ROOT)),
            "new_locked_result": False,
        },
    ]
    write_csv(args.output_dir / "historical_result_reproduction_table.csv", historical_rows)

    source_registry: dict[str, Any] = {
        "created_at_utc": now_utc(),
        "mdm_memit_paper": {
            "url": "https://arxiv.org/abs/2606.03924",
            "version": "v1",
            "historical_pdf_sha256": read_json(
                HISTORICAL_ROOT / "source_audit/source_registry.json"
            )["mdm_memit_paper"]["pdf_sha256"],
        },
        "llada": {
            "model_id": PRIMARY_MODEL_ID,
            "revision": PRIMARY_MODEL_REVISION,
            "url": f"https://huggingface.co/{PRIMARY_MODEL_ID}",
        },
        "dream": {
            "model_id": SECONDARY_MODEL_ID,
            "model_revision": SECONDARY_MODEL_REVISION,
            "repository": "https://github.com/DreamLM/Dream",
            "repository_head": _git_head("https://github.com/DreamLM/Dream.git"),
        },
        "memit_reference": {
            "repository": "https://github.com/kmeng01/memit",
            "commit": "80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b",
        },
        "easyedit_reference": {
            "repository": "https://github.com/zjunlp/EasyEdit",
            "commit": "14cea8245f06715684592ab55184939b99d70784",
        },
        "kamel": {
            "repository": "https://github.com/JanKalo/KAMEL",
            "commit": "21625baba6439faea03e61c28ce29475dc4996f6",
        },
        "mask_pattern_solver_commit": git_commit(),
    }
    try:
        resolved_revision = _json_url(
            "https://huggingface.co/api/models/" + urllib.parse.quote(SECONDARY_MODEL_ID, safe="/")
        ).get("sha")
        source_registry["dream"]["resolved_model_revision"] = resolved_revision
        source_registry["dream"]["revision_matches_pin"] = (
            resolved_revision == SECONDARY_MODEL_REVISION
        )
    except Exception as exc:  # source remains pinned by repository if HF API is unavailable
        source_registry["dream"]["revision_lookup_error"] = f"{type(exc).__name__}: {exc}"
    write_json(args.output_dir / "source_registry.json", source_registry)

    historical_difference = (
        HISTORICAL_ROOT / "source_audit/implementation_difference_register.md"
    ).read_text(encoding="utf-8")
    implementation_register = f"""# Publication Implementation Difference Register

This audit is for `{CAMPAIGN_ID}`. Historical runs are immutable and are not
treated as fresh confirmation evidence.

## Historical implementation

{historical_difference.strip()}

## Paper-matched publication configuration

| Field | Publication setting | Status before P1 |
| --- | --- | --- |
| Primary model | `{PRIMARY_MODEL_ID}` at `{PRIMARY_MODEL_REVISION}` | pinned |
| Edited layer window | layers 4-7 first paper-matched candidate | frozen for P1 audit |
| MLP target | gated MLP `ff_out.weight` | validated historically; re-audit in P1 |
| Subject position | last contextual subject token | re-audit in P1 |
| Target-value optimizer | lr 0.1, 25 steps | paper-matched |
| Clamp norm | 0.75 | paper-matched |
| KL factor | 0.0625 | paper-matched |
| Mask schedule | `k = optimization_step mod N`, resampled revealed positions | unit-test in P1 |
| Loss positions | still-masked answer positions only | unit-test in P1 |
| Primary dtype | float16 or bfloat16, editable weights | mandatory |
| Quantization | none for editing | mandatory |
| Dream module map | unresolved | validate in P5 |

Official MDM-MEMIT code was not available in the historical source audit. The
publication implementation therefore remains a paper-plus-reference
reproduction unless a newly released official implementation is found.
"""
    (args.output_dir / "implementation_difference_register.md").write_text(
        implementation_register, encoding="utf-8"
    )
    write_json(
        args.output_dir / "model_module_maps.json",
        {
            "llada": read_json(HISTORICAL_ROOT / "source_audit/model_module_map.json"),
            "dream": {"status": "pending_P5_runtime_audit", "model_id": SECONDARY_MODEL_ID},
        },
    )
    write_json(
        args.output_dir / "paper_matched_config.json",
        {
            "model_id": PRIMARY_MODEL_ID,
            "model_revision": PRIMARY_MODEL_REVISION,
            "layers": [4, 5, 6, 7],
            "target_value_lr": 0.1,
            "target_value_steps": 25,
            "clamp_norm_factor": 0.75,
            "kl_factor": 0.0625,
            "subject_position": "last_contextual_subject_token",
            "partial_mask_schedule": "paper_cycle_random_positions",
            "dtype": "float16",
            "use_4bit": False,
        },
    )
    (args.output_dir / "missing_artifacts.md").write_text(
        "# Missing Artifacts\n\nNo required historical seed artifact is missing.\n",
        encoding="utf-8",
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track": "P0",
        "stage": "P0_source_artifact_audit",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "required_root_files_present": True,
        "required_historical_artifacts_present": True,
        "historical_artifact_count": len(artifacts),
        "historical_analysis_500_used": False,
        "historical_final_test_used": False,
        "historical_values_are_fresh_confirmation": False,
        "source_registry_written": True,
        "implementation_difference_register_written": True,
        "acceptance_pass": True,
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="P0_source_artifact_audit",
        track="P0",
        status="source_audit_passed",
        output_dir=args.output_dir,
        acceptance_pass=True,
        started_at_utc=started,
        notes="Historical seed artifacts validated as immutable context; no locked historical metrics opened for tuning.",
        next_stage="P0_fresh_data_protocol",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
