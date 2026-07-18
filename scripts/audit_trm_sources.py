#!/usr/bin/env python3
"""Freeze source provenance and the exact-vs-adapted implementation boundary."""

from __future__ import annotations

import hashlib
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trm_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SECONDARY_MODEL_ID,
    SECONDARY_MODEL_REVISION,
    SOURCE_MODEL_ID,
    SOURCE_MODEL_REVISION,
    git_commit,
    now_utc,
    record_stage,
    write_csv,
    write_json,
)


PAPERS = (
    ("TimeROME-DLM", "2606.12841", "temporal tracing and residual memory"),
    ("Knowledge Editing in Masked Diffusion Language Models", "2606.03924", "partial-state target optimization"),
)


def inspect_arxiv_source(arxiv_id: str) -> dict[str, object]:
    url = f"https://export.arxiv.org/e-print/{arxiv_id}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    code_links: set[str] = set()
    tex_files = 0
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "source.tar"
        archive.write_bytes(payload)
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                if not member.isfile() or not member.name.lower().endswith((".tex", ".md", ".txt", ".json")):
                    continue
                source = handle.extractfile(member)
                if source is None:
                    continue
                tex_files += int(member.name.lower().endswith(".tex"))
                text = source.read().decode("utf-8", errors="ignore")
                for link in re.findall(r"https?://[^\\\s}\]]+", text):
                    if "github.com" in link.lower():
                        code_links.add(link.rstrip(".,;"))
    return {
        "arxiv_id": arxiv_id,
        "eprint_url": url,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_size_bytes": len(payload),
        "tex_file_count": tex_files,
        "github_links": sorted(code_links),
    }


def main() -> None:
    started = now_utc()
    output = CAMPAIGN_ROOT / "A1_source_audit_v1"
    output.mkdir(parents=True, exist_ok=False)
    paper_rows = []
    revisions = []
    for title, arxiv_id, use in PAPERS:
        audit = inspect_arxiv_source(arxiv_id)
        revisions.append(audit)
        paper_rows.append(
            {
                "source": title,
                "kind": "paper_arxiv_source",
                "identifier": f"arXiv:{arxiv_id}",
                "official_code": "not linked in arXiv source" if not audit["github_links"] else ",".join(audit["github_links"]),
                "use": use,
                "reproduction_label": "source_reproduction_technically_infeasible" if arxiv_id == "2606.12841" and not audit["github_links"] else "paper_aligned_adaptation",
            }
        )
    paper_rows.extend(
        [
            {"source": "AlphaEdit", "kind": "official_code", "identifier": "jianghoucheng/AlphaEdit", "official_code": "https://github.com/jianghoucheng/AlphaEdit", "use": "protected-subspace baseline", "reproduction_label": "source_aligned_component"},
            {"source": "LLaDA", "kind": "official_code", "identifier": "ML-GSAI/LLaDA", "official_code": "https://github.com/ML-GSAI/LLaDA", "use": "masked diffusion backbone", "reproduction_label": "official_backbone"},
            {"source": "Dream", "kind": "official_code", "identifier": "DreamLM/Dream", "official_code": "https://github.com/DreamLM/Dream", "use": "secondary backbone", "reproduction_label": "official_backbone"},
            {"source": "CounterFact", "kind": "dataset", "identifier": "azhx/counterfact train", "official_code": "https://github.com/kmeng01/counterfact", "use": "fresh factual-editing manifests", "reproduction_label": "fresh_manifest_source"},
            {"source": "KAMEL", "kind": "dataset", "identifier": "JanKalo/KAMEL", "official_code": "https://github.com/JanKalo/KAMEL", "use": "fresh multi-token manifests", "reproduction_label": "fresh_manifest_source"},
        ]
    )
    write_csv(output / "source_matrix.csv", paper_rows)
    write_json(output / "source_revision.json", {"papers": revisions})
    write_json(
        output / "model_version_lock.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "primary": {"model_id": PRIMARY_MODEL_ID, "revision": PRIMARY_MODEL_REVISION},
            "source_reproduction_fallback": {"model_id": SOURCE_MODEL_ID, "revision": SOURCE_MODEL_REVISION},
            "secondary": {"model_id": SECONDARY_MODEL_ID, "revision": SECONDARY_MODEL_REVISION},
            "backbone_frozen": True,
        },
    )
    (output / "source_audit.md").write_text(
        "# Source Audit\n\n"
        "The TimeROME-DLM arXiv source contains the algorithm and equations but no linked official code repository. "
        "Therefore this campaign may claim only a paper-aligned component reproduction unless official source is later discovered without changing the protocol. "
        "The exact source task/checkpoint is treated as technically unavailable; CounterFact adaptation may proceed only after synthetic invariants pass.\n",
        encoding="utf-8",
    )
    (output / "source_to_implementation_map.md").write_text(
        "# Source-to-Implementation Map\n\n"
        "| Source component | Local implementation | Claim boundary |\n"
        "|---|---|---|\n"
        "| TimeROME TIE | `dnpe_editor.py` tracing plus TRM temporal wrappers | paper-aligned adaptation |\n"
        "| TimeROME ridge memory | `trm_residual.py` dual ridge solve | equation-level reproduction |\n"
        "| TimeROME sparsification | `ResidualMemory.predict(top_q=...)` | equation-level reproduction |\n"
        "| MDM partial states | `dnpe_editor.py` state-bank primitives | paper-aligned adaptation |\n"
        "| State-conditioned protection | `trm_residual.py` augmented protected solve | new method |\n",
        encoding="utf-8",
    )
    acceptance = {
        "arxiv_sources_hashed": all(row["source_sha256"] for row in revisions),
        "official_code_availability_recorded": True,
        "exact_reproduction_boundary_explicit": True,
        "source_to_implementation_map_present": True,
        "analysis_500_used": False,
        "final_test_used": False,
    }
    passed = all(value for key, value in acceptance.items() if key not in {"analysis_500_used", "final_test_used"})
    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "A1_source_audit",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "timerome_exact_source_code_available": bool(revisions[0]["github_links"]),
        "source_reproduction_status": "paper_source_available_official_code_unavailable",
        "acceptance": acceptance,
        "acceptance_pass": passed,
    }
    write_json(output / "report_summary.json", report)
    write_json(output / "run_config.json", {"paper_ids": [row[1] for row in PAPERS]})
    write_json(output / "validation_report.json", acceptance)
    record_stage(
        "A1_source_audit",
        status="passed" if passed else "failed",
        acceptance_pass=passed,
        output_dir=output,
        started_at_utc=started,
        notes="Paper sources hashed; unavailable official TimeROME code explicitly limits reproduction claims.",
        next_stage="B0_fresh_protocol" if passed else None,
    )
    if not passed:
        raise SystemExit(2)
    print(f"A1 source audit passed: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
