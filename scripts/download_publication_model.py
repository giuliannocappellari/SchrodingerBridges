#!/usr/bin/env python3
"""Acquire a pinned publication model snapshot with an auditable shard check."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_snapshot(snapshot: Path) -> dict[str, Any]:
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted({str(value) for value in index.get("weight_map", {}).values()})
    if not shards:
        raise RuntimeError("Pinned snapshot index has no weight shards")
    missing = [name for name in shards if not (snapshot / name).exists()]
    if missing:
        raise RuntimeError(f"Pinned snapshot is missing shards: {missing}")
    return {
        "index_path": str(index_path),
        "num_weight_tensors": len(index["weight_map"]),
        "num_shards": len(shards),
        "shards": shards,
        "all_indexed_shards_present": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now_utc()
    wall_start = time.monotonic()
    report: dict[str, Any] = {
        "model_id": args.model_id,
        "revision": args.revision,
        "cache_dir": str(args.cache_dir),
        "started_at_utc": started,
        "python": platform.python_version(),
        "status": "running",
        "acceptance_pass": False,
    }
    try:
        os.environ["HF_HOME"] = str(args.cache_dir)
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        with args.log_path.open("a", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                from huggingface_hub import snapshot_download

                snapshot = Path(
                    snapshot_download(
                        repo_id=args.model_id,
                        revision=args.revision,
                        cache_dir=args.cache_dir / "hub",
                    )
                )
        report.update(_validate_snapshot(snapshot))
        report.update(
            {
                "snapshot_path": str(snapshot),
                "status": "complete",
                "acceptance_pass": True,
            }
        )
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    report["completed_at_utc"] = _now_utc()
    report["runtime_seconds"] = time.monotonic() - wall_start
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    if not report["acceptance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
