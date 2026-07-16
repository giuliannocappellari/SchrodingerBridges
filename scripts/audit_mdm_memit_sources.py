#!/usr/bin/env python3
"""Audit primary sources and the editable LLaDA architecture on RunPod."""

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

from scripts.mdm_memit_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    MODEL_ID,
    git_commit,
    now_utc,
    record_stage,
    sha256_file,
    write_json,
)
from scripts.mdm_memit_editor import (
    block_name,
    editable_weight_name,
    find_last_subject_token,
    get_module,
    infer_mask_id,
    key_module_name,
)


SOURCE_ROOT = CAMPAIGN_ROOT / "source_audit"


def git_remote_head(url: str) -> str:
    output = subprocess.check_output(["git", "ls-remote", url, "HEAD"], text=True)
    return output.split()[0]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, path.open("wb") as handle:
            handle.write(response.read())


def official_code_search() -> dict[str, Any]:
    query = urllib.parse.quote('"Knowledge Editing in Masked Diffusion Language Models" in:name,description')
    url = f"https://api.github.com/search/repositories?q={query}&per_page=20"
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "SB-research-audit"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        candidates = [
            {
                "full_name": item.get("full_name"),
                "html_url": item.get("html_url"),
                "description": item.get("description"),
            }
            for item in payload.get("items", [])
        ]
        exact = [item for item in candidates if "masked diffusion" in str(item.get("description", "")).casefold() and "editing" in str(item.get("description", "")).casefold()]
        return {
            "checked_at_utc": now_utc(),
            "search_url": url,
            "request_pass": True,
            "candidate_repositories": candidates,
            "official_release_found": bool(exact),
            "paper_v1_release_statement": "code will be released upon publication",
        }
    except Exception as exc:
        return {
            "checked_at_utc": now_utc(),
            "search_url": url,
            "request_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "official_release_found": False,
            "paper_v1_release_statement": "code will be released upon publication",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--output_dir", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    started = now_utc()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper_path = args.output_dir / "arxiv_2606.03924.pdf"
    download("https://arxiv.org/pdf/2606.03924", paper_path)
    sources = {
        "mdm_memit_paper": {
            "url": "https://arxiv.org/abs/2606.03924",
            "pdf_sha256": sha256_file(paper_path),
            "version_audited": "v1",
        },
        "official_mdm_memit_code_search": official_code_search(),
        "memit_reference": {
            "url": "https://github.com/kmeng01/memit",
            "commit": git_remote_head("https://github.com/kmeng01/memit.git"),
            "official": True,
        },
        "easyedit_reference": {
            "url": "https://github.com/zjunlp/EasyEdit",
            "commit": git_remote_head("https://github.com/zjunlp/EasyEdit.git"),
            "official": True,
        },
        "csbm_paper": {"url": "https://arxiv.org/abs/2502.01416"},
    }
    write_json(args.output_dir / "source_registry.json", sources)

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == infer_mask_id(model):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    modules = dict(model.named_modules())
    layers: list[dict[str, Any]] = []
    for layer in range(int(model.config.n_layers)):
        block = get_module(model, block_name(layer))
        key_module = get_module(model, key_module_name(layer))
        weight = key_module.weight
        layers.append(
            {
                "layer": layer,
                "block_module": block_name(layer),
                "key_module": key_module_name(layer),
                "editable_weight": editable_weight_name(layer),
                "weight_shape": list(weight.shape),
                "weight_dtype": str(weight.dtype),
                "weight_device": str(weight.device),
                "weight_requires_grad": bool(weight.requires_grad),
                "quantized": not weight.dtype.is_floating_point,
                "ff_proj_module": f"{block_name(layer)}.ff_proj",
                "up_proj_module": f"{block_name(layer)}.up_proj",
            }
        )
    map_payload = {
        "model_id": args.model_id,
        "architecture": type(model).__name__,
        "n_layers": int(model.config.n_layers),
        "d_model": int(model.config.d_model),
        "mlp_hidden_size": int(model.config.mlp_hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "mask_token_id": infer_mask_id(model),
        "output_head": "model.transformer.ff_out",
        "final_norm": "model.transformer.ln_f",
        "layers": layers,
        "all_required_modules_present": all(
            name in modules
            for layer in range(int(model.config.n_layers))
            for name in (block_name(layer), key_module_name(layer))
        ),
    }
    write_json(args.output_dir / "model_module_map.json", map_payload)

    subject = "Paris"
    prompt = "Paris is located in"
    last_subject = find_last_subject_token(tokenizer, prompt, subject)
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] + [infer_mask_id(model)]
    tensor = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids=tensor).logits
    mask_forward_pass = bool(torch.isfinite(logits).all() and logits.shape[:2] == tensor.shape)
    editable_float = all(not row["quantized"] for row in layers)
    paper_code_found = sources["official_mdm_memit_code_search"]["official_release_found"]
    differences = f"""# Implementation Difference Register

- The primary paper is pinned by PDF SHA-256 in `source_registry.json`.
- An official-code search was executed once at {sources['official_mdm_memit_code_search']['checked_at_utc']}.
- Official MDM-MEMIT code found: `{paper_code_found}`. The paper's v1 release statement is retained.
- The implementation therefore follows the paper plus official MEMIT commit `{sources['memit_reference']['commit']}` and EasyEdit commit `{sources['easyedit_reference']['commit']}`.
- LLaDA's gated MLP key is the input to `ff_out`; the edited associative-memory matrix is `ff_out.weight`.
- The model is loaded in editable `{args.dtype}`, never 4-bit, for primary editing.
"""
    (args.output_dir / "implementation_difference_register.md").write_text(differences, encoding="utf-8")

    report = {
        "campaign_id": CAMPAIGN_ID,
        "stage": "A2_source_and_model_audit",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "model_dtype": args.dtype,
        "model_loaded": True,
        "model_loaded_in_editable_floating_point": editable_float,
        "quantized_weight_editing": False,
        "mlp_target_matrices_identified": map_payload["all_required_modules_present"],
        "last_subject_token_mapping_validated": last_subject >= 0,
        "last_subject_token_index": last_subject,
        "mask_augmented_forward_pass_validated": mask_forward_pass,
        "paper_and_reference_sources_recorded": True,
        "official_paper_code_available": paper_code_found,
        "old_analysis_500_used": False,
        "old_final_test_used": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "acceptance_pass": bool(
            editable_float
            and map_payload["all_required_modules_present"]
            and last_subject >= 0
            and mask_forward_pass
        ),
    }
    write_json(args.output_dir / "report_summary.json", report)
    record_stage(
        stage="A2_source_and_model_audit",
        status="passed" if report["acceptance_pass"] else "failed",
        output_dir=args.output_dir,
        acceptance_pass=report["acceptance_pass"],
        started_at_utc=started,
        notes="Primary sources pinned and editable LLaDA architecture audited.",
    )
    print(f"acceptance_pass={report['acceptance_pass']}")


if __name__ == "__main__":
    main()
