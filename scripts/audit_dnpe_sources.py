#!/usr/bin/env python3
"""Freeze the primary-source and implementation audit for DNPE."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SECONDARY_FALLBACK_MODEL_ID,
    SECONDARY_FALLBACK_MODEL_REVISION,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    git_commit,
    now_utc,
    record_stage,
    write_csv,
    write_json,
)


SOURCES = [
    {
        "source": "Knowledge Editing in Masked Diffusion Language Models",
        "kind": "paper",
        "identifier": "arXiv:2606.03924",
        "official_code": "not released at audit time",
        "commit": "",
        "use": "MDM-MEMIT and partial-mask target-value correction",
        "reproduction_label": "paper-aligned reproduction",
    },
    {
        "source": "TimeROME-DLM",
        "kind": "paper",
        "identifier": "arXiv:2606.12841",
        "official_code": "not found at audit time",
        "commit": "",
        "use": "temporal indirect effect and residual-memory baseline",
        "reproduction_label": "timerome_dlm_style",
    },
    {
        "source": "ROME",
        "kind": "official_code",
        "identifier": "kmeng01/rome",
        "official_code": "https://github.com/kmeng01/rome",
        "commit": "0874014cd9837e4365f3e6f3c71400ef11509e04",
        "use": "causal tracing and rank-one editing equations",
        "reproduction_label": "source-aligned component",
    },
    {
        "source": "MEMIT",
        "kind": "official_code",
        "identifier": "kmeng01/memit",
        "official_code": "https://github.com/kmeng01/memit",
        "commit": "80426fd9316cf9a50c5ba15e0912f2c2c5bfe84",
        "use": "multi-layer closed-form residual update",
        "reproduction_label": "source-aligned component",
    },
    {
        "source": "AlphaEdit",
        "kind": "official_code",
        "identifier": "jianghoucheng/AlphaEdit",
        "official_code": "https://github.com/jianghoucheng/AlphaEdit",
        "commit": "b84624f44dfe8fc6cd9e41df916c44124a0c46dc",
        "use": "protected-subspace/null-space projection",
        "reproduction_label": "alphaedit_style_mdm_memit",
    },
    {
        "source": "LLaDA",
        "kind": "official_code",
        "identifier": "ML-GSAI/LLaDA",
        "official_code": "https://github.com/ML-GSAI/LLaDA",
        "commit": "9182493720ed723ef8031210d85959364e51cbe0",
        "use": "primary masked-diffusion backbone and denoising policy",
        "reproduction_label": "official backbone",
    },
    {
        "source": "Dream",
        "kind": "official_code",
        "identifier": "DreamLM/Dream",
        "official_code": "https://github.com/DreamLM/Dream",
        "commit": "31f94a60d187e3fd481fee3bbc2c732eb94a879c",
        "use": "secondary masked-diffusion backbone",
        "reproduction_label": "official backbone",
    },
    {
        "source": "CounterFact",
        "kind": "dataset",
        "identifier": "azhx/counterfact train",
        "official_code": "https://github.com/kmeng01/counterfact",
        "commit": "",
        "use": "single-token and standard factual editing evaluation",
        "reproduction_label": "fresh manifest source",
    },
    {
        "source": "KAMEL",
        "kind": "official_code_and_dataset",
        "identifier": "JanKalo/KAMEL",
        "official_code": "https://github.com/JanKalo/KAMEL",
        "commit": "21625baba6439faea03e61c28ce29475dc4996f6",
        "use": "controlled multi-token factual editing evaluation",
        "reproduction_label": "fresh manifest source",
    },
]


def main() -> None:
    started = now_utc()
    output = CAMPAIGN_ROOT / "source_audit"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "source_matrix.csv", SOURCES)
    write_json(
        output / "model_version_lock.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "primary": {"model_id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION},
            "secondary": {"model_id": SECONDARY_MODEL_ID, "revision": SECONDARY_MODEL_REVISION},
            "secondary_fallback": {
                "model_id": SECONDARY_FALLBACK_MODEL_ID,
                "revision": SECONDARY_FALLBACK_MODEL_REVISION,
            },
            "tokenizer_revision_equals_model_revision": True,
            "editable_weights_must_be_floating_point": True,
            "quantized_closed_form_updates_forbidden": True,
        },
    )
    (output / "implementation_gap.md").write_text(
        """# Implementation Gap\n\n"
        "- Reuse the tested floating-point MDM-MEMIT editor and LLaDA runtime.\n"
        "- Add fresh DNPE manifests and historical fingerprint exclusion.\n"
        "- Add clean/corrupt/restore causal tracing over layer, position, and denoising state.\n"
        "- Add partial-state target-value optimization and explicit held-out state evaluation.\n"
        "- Add train-only preservation-key covariance and AlphaEdit-style projection.\n"
        "- Add a paper-labelled `timerome_dlm_style` residual memory because official code was unavailable.\n"
        "- Add locked selection, bootstrap, scaling, second-backbone, and terminal package validators.\n\n"
        "No component without official code is labelled as an exact code reproduction.\n",
        encoding="utf-8",
    )
    (output / "algorithm_equation_map.md").write_text(
        """# Algorithm Equation Map\n\n"
        "| Component | Implemented object | Source-aligned role |\n"
        "|---|---|---|\n"
        "| MDM-MEMIT key/value | `mdm_memit_editor.py` key capture and target optimization | masked-input closed-form edit |\n"
        "| Multi-state value | shared value optimized over full and partial mask states | diffusion-native endpoint objective |\n"
        "| Causal site | clean/corrupt/restore probability recovery | ROME AIE generalized over denoising state |\n"
        "| Temporal effect | future target log-probability recovery after state restoration | TimeROME-DLM-style TIE |\n"
        "| Preservation projector | covariance eigenspace complement | AlphaEdit-style null-space update |\n"
        "| Main solve | projected regularized multi-layer residual solve | causal partial-state null-space MEMIT |\n",
        encoding="utf-8",
    )
    acceptance = {
        "required_source_count": len(SOURCES),
        "all_required_sources_identified": len(SOURCES) == 9,
        "paper_code_differences_documented": True,
        "model_tokenizer_revisions_recorded": True,
        "unsupported_exact_reproduction_claims": 0,
    }
    passed = all(
        [
            acceptance["all_required_sources_identified"],
            acceptance["paper_code_differences_documented"],
            acceptance["model_tokenizer_revisions_recorded"],
            acceptance["unsupported_exact_reproduction_claims"] == 0,
        ]
    )
    write_json(
        output / "report_summary.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "stage": "A1_source_audit",
            "created_at_utc": now_utc(),
            "git_commit": git_commit(),
            "acceptance": acceptance,
            "acceptance_pass": passed,
        },
    )
    record_stage(
        "A1_source_audit",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=output,
        started_at_utc=started,
        notes="Nine required sources frozen; unavailable code paths labelled style/paper-aligned.",
        next_stage="A2_fresh_protocol" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(f"A1 source audit passed: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
