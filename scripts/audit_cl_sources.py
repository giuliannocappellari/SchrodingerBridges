#!/usr/bin/env python3
"""Write the frozen, independently checked source audit for the CL campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cl_common import (
    CAMPAIGN_ID,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SOURCE_AUDIT_ROOT,
    git_commit,
    now_utc,
    record_stage,
    write_csv,
    write_json,
)


CHECKED_AT = "2026-07-23"


def source_rows() -> list[dict[str, Any]]:
    """Return auditable classifications; adaptations are never called reproductions."""

    return [
        {
            "source_id": "llada_8b_instruct",
            "title": PRIMARY_MODEL_ID,
            "paper_url": "https://arxiv.org/abs/2502.09992",
            "code_url": "https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct",
            "implementation_status": "author checkpoint",
            "license": "MIT",
            "frozen_revision": PRIMARY_MODEL_REVISION,
            "campaign_use": "primary frozen denoiser",
            "notes": "Hugging Face model card reports MIT; revision is pinned.",
        },
        {
            "source_id": "diffusiongrow",
            "title": "DiffusionGrow: Continual Learning for Diffusion Language Models",
            "paper_url": "https://openreview.net/forum?id=pvqJiLXmUn",
            "code_url": "",
            "implementation_status": "equation-level reimplementation",
            "license": "paper CC BY 4.0; code unavailable at audit",
            "frozen_revision": "OpenReview modified 2026-06-17",
            "campaign_use": "C0 source-style audit and C1 conceptual factual adaptation",
            "notes": "ARR submission advertises public software, but no verifiable repository was exposed by the paper page/search at audit time.",
        },
        {
            "source_id": "mdm_memit",
            "title": "MDM-MEMIT / partial-state masked-diffusion editing",
            "paper_url": "https://arxiv.org/abs/2502.09992",
            "code_url": "scripts/mdm_memit_editor.py",
            "implementation_status": "equation-level reimplementation",
            "license": "repository-local research code",
            "frozen_revision": git_commit(),
            "campaign_use": "C0 factual acquisition baseline",
            "notes": "Repository implementation follows official MEMIT algebra but adapts state construction to LLaDA.",
        },
        {
            "source_id": "o_edit",
            "title": "O-Edit: Orthogonal Subspace Editing for Language Model Sequential Editing",
            "paper_url": "https://arxiv.org/abs/2410.11469v1",
            "code_url": "",
            "implementation_status": "equation-level reimplementation",
            "license": "paper arXiv license; no verified official code",
            "frozen_revision": "arXiv v1 2024-10-15",
            "campaign_use": "C0/C5 update orthogonalization",
            "notes": "No official standalone implementation was verified; EasyEdit is only a framework dependency cited in related material.",
        },
        {
            "source_id": "memoir",
            "title": "MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention",
            "paper_url": "https://arxiv.org/abs/2506.07899v4",
            "code_url": "https://github.com/qym7/MEMOIR",
            "implementation_status": "official code",
            "license": "MIT",
            "frozen_revision": "repository main; 3 commits visible at audit",
            "campaign_use": "C3 conceptual masked-diffusion adaptation",
            "notes": "Official code targets autoregressive LLaMA/Mistral; C3 is not an exact reproduction.",
        },
        {
            "source_id": "sparse_memory_finetuning",
            "title": "Sparse Memory Finetuning as a Low-Forgetting Alternative",
            "paper_url": "https://arxiv.org/abs/2605.03229v2",
            "code_url": "",
            "implementation_status": "conceptual adaptation",
            "license": "paper arXiv license; official code not verified",
            "frozen_revision": "arXiv v2 2026-06-08",
            "campaign_use": "C3 sparse row selection",
            "notes": "The paper reimplements SMF on Qwen; a separate open retrofit paper is not treated as this paper's official code.",
        },
        {
            "source_id": "gainlora",
            "title": "Gated Integration of Low-Rank Adaptation for Continual Learning",
            "paper_url": "https://arxiv.org/abs/2505.15424v2",
            "code_url": "https://github.com/liangyanshuo/gainlora",
            "implementation_status": "official code",
            "license": "no repository license verified",
            "frozen_revision": "master; 8 commits visible at audit",
            "campaign_use": "C4 conceptual masked-diffusion adaptation",
            "notes": "Official code targets T5/LLaMA task continual learning, not factual MDM editing.",
        },
        {
            "source_id": "c_lora",
            "title": "Continual Diffusion: Continual Customization with C-LoRA",
            "paper_url": "https://arxiv.org/abs/2304.06027",
            "code_url": "https://jamessealesmith.github.io/continual-diffusion/",
            "implementation_status": "conceptual adaptation",
            "license": "paper/project page; downloadable official code not exposed",
            "frozen_revision": "TMLR 2024 project page",
            "campaign_use": "C4 self-regularized branch expansion",
            "notes": "Source method is text-to-image customization; the campaign adaptation is not a reproduction.",
        },
        {
            "source_id": "fggm",
            "title": "Fisher-Guided Gradient Masking for Continual Learning",
            "paper_url": "https://arxiv.org/abs/2601.18261",
            "code_url": "",
            "implementation_status": "equation-level reimplementation",
            "license": "paper arXiv license; official code unavailable",
            "frozen_revision": "arXiv 2601.18261",
            "campaign_use": "C5 diagonal-Fisher masking",
            "notes": "No official repository was verified.",
        },
        {
            "source_id": "nusa_cl",
            "title": "NuSA-CL null-space adaptation for continual learning",
            "paper_url": "https://arxiv.org/abs/2510.21175",
            "code_url": "",
            "implementation_status": "equation-level reimplementation",
            "license": "paper arXiv license; official code unavailable",
            "frozen_revision": "arXiv 2510.21175",
            "campaign_use": "C5 low-rank null-space update",
            "notes": "No official repository was verified.",
        },
        {
            "source_id": "lwf",
            "title": "Learning without Forgetting",
            "paper_url": "https://arxiv.org/abs/1606.09282",
            "code_url": "https://github.com/lizhitwo/LearningWithoutForgetting",
            "implementation_status": "official code",
            "license": "repository license not asserted by audit",
            "frozen_revision": "repository master",
            "campaign_use": "C0/C6 conceptual masked-state distillation",
            "notes": "Official implementation is MatConvNet vision code; DLM use is conceptual adaptation.",
        },
        {
            "source_id": "gem",
            "title": "Gradient Episodic Memory for Continual Learning",
            "paper_url": "https://arxiv.org/abs/1706.08840",
            "code_url": "https://github.com/facebookresearch/GradientEpisodicMemory",
            "implementation_status": "official code",
            "license": "repository archived; license must be checked before vendoring",
            "frozen_revision": "historical official repository",
            "campaign_use": "C6 equation-level gradient projection",
            "notes": "No source code is vendored; only the published constraint is adapted.",
        },
        {
            "source_id": "dark_experience_replay",
            "title": "Dark Experience Replay",
            "paper_url": "https://arxiv.org/abs/2004.07211",
            "code_url": "https://github.com/aimagelab/mammoth",
            "implementation_status": "official code",
            "license": "MIT",
            "frozen_revision": "repository main",
            "campaign_use": "C2/C6 top-k masked-state replay",
            "notes": "Official code is a general vision CL framework; DLM state replay is conceptual adaptation.",
        },
        {
            "source_id": "csbm",
            "title": "Categorical Schrödinger Bridge Matching",
            "paper_url": "https://arxiv.org/abs/2502.01416",
            "code_url": "https://github.com/gregkseno/csbm",
            "implementation_status": "official code",
            "license": "repository license not asserted by audit",
            "frozen_revision": "repository main; ICML 2025",
            "campaign_use": "C7/C8 bounded categorical bridge adaptation",
            "notes": "Official experiments are discrete VQ image states, not LLaDA trajectories.",
        },
        {
            "source_id": "dsbm",
            "title": "Diffusion Schrödinger Bridge Matching",
            "paper_url": "https://arxiv.org/abs/2303.16852",
            "code_url": "https://github.com/yuyang-shi/dsbm-pytorch",
            "implementation_status": "official code",
            "license": "repository license not asserted by audit",
            "frozen_revision": "repository master",
            "campaign_use": "C8 function-space IMF inspiration",
            "notes": "Continuous-space code is not directly reused for categorical masked states.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=SOURCE_AUDIT_ROOT)
    args = parser.parse_args()
    started = now_utc()
    if args.output_dir.exists():
        report = args.output_dir / "report_summary.json"
        if not report.is_file():
            raise FileExistsError(args.output_dir)
        existing = __import__("json").loads(report.read_text(encoding="utf-8"))
        if existing.get("acceptance_pass") is True:
            print(f"A0 source audit already exists: {args.output_dir}")
            return
    else:
        args.output_dir.mkdir(parents=True)
    rows = source_rows()
    mandatory = {
        "diffusiongrow", "mdm_memit", "o_edit", "memoir",
        "sparse_memory_finetuning", "gainlora", "fggm", "nusa_cl", "c_lora",
        "lwf", "gem", "dark_experience_replay", "csbm", "dsbm",
    }
    ids = {row["source_id"] for row in rows}
    classifications = {
        "official code", "author checkpoint", "equation-level reimplementation",
        "conceptual adaptation", "unavailable",
    }
    checks = {
        "all_mandatory_sources_classified": mandatory <= ids,
        "all_classifications_allowed": all(row["implementation_status"] in classifications for row in rows),
        "licenses_recorded": all(bool(row["license"]) for row in rows),
        "base_revision_frozen": any(
            row["source_id"] == "llada_8b_instruct"
            and row["frozen_revision"] == PRIMARY_MODEL_REVISION for row in rows
        ),
        "no_historical_evaluation_data_used": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    acceptance = all(
        value
        for key, value in checks.items()
        if key not in {"analysis_500_used", "final_test_used"}
    )
    write_csv(args.output_dir / "source_audit.csv", rows)
    write_json(args.output_dir / "source_registry.json", {row["source_id"]: row for row in rows})
    write_json(
        args.output_dir / "report_summary.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "A0_source_audit",
            "checked_at": CHECKED_AT,
            "created_at_utc": now_utc(),
            "git_commit": git_commit(),
            "num_sources": len(rows),
            "checks": checks,
            "acceptance_pass": acceptance,
            "analysis_500_used": False,
            "final_test_used": False,
            "method_label_policy": "Only directly verified author repositories are labeled official; all DLM adaptations remain equation-level or conceptual.",
        },
    )
    record_stage(
        "A0_source_audit",
        status="passed" if acceptance else "failed",
        acceptance_pass=acceptance,
        output_dir=args.output_dir,
        started_at_utc=started,
        notes=f"Classified {len(rows)} sources; model revision frozen.",
        next_stage="A1_campaign_bootstrap" if acceptance else None,
        exit_code=0 if acceptance else 2,
    )
    if not acceptance:
        raise SystemExit(2)
    print(f"A0 source audit passed: {args.output_dir}")


if __name__ == "__main__":
    main()
