#!/usr/bin/env python3
"""Validate and locate the frozen DNPE terminal paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = (
    ROOT
    / "runs"
    / "diffusion_native_causal_partial_state_editor_v1"
    / "final_research_package_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=("main",))
    parser.add_argument("--figure", choices=("causal_heatmap",))
    parser.add_argument("--validate-terminal-package", action="store_true")
    args = parser.parse_args()
    if args.table == "main":
        path = PACKAGE / "main_results_table.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        print(path)
        return
    if args.figure == "causal_heatmap":
        path = PACKAGE / "causal_heatmap.png"
        if not path.exists():
            raise FileNotFoundError(path)
        print(path)
        return
    if args.validate_terminal_package:
        path = PACKAGE / "terminal_package_validation.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("acceptance_pass"):
            raise RuntimeError("Terminal package did not validate")
        print(json.dumps(report, sort_keys=True))
        return
    parser.error("Choose --table, --figure, or --validate-terminal-package")


if __name__ == "__main__":
    main()
