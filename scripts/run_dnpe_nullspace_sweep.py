#!/usr/bin/env python3
"""Run the staged B3 AlphaEdit-style protected-subspace sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dnpe_common import CAMPAIGN_ROOT, now_utc, record_stage
from scripts.report_dnpe_nullspace_baseline import select_smoke, validate_pilot


VARIANCES = (0.90, 0.95, 0.99)
RIDGES = (1e-4, 1e-3, 1e-2)


def _name(prefix: str, variance: float, ridge: float) -> str:
    return f"{prefix}_variance_{variance:.2f}_ridge_{ridge:.0e}".replace("+", "")


def run_one(
    *,
    manifest: Path,
    output: Path,
    basis_dir: Path,
    variance: float,
    ridge: float,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_dnpe_editor.py"),
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output),
        "--method",
        "alphaedit_style_mdm_memit",
        "--layers",
        "3,4,5,6",
        "--protected_basis_dir",
        str(basis_dir),
        "--protected_variance",
        str(variance),
        "--update_ridge",
        str(ridge),
        "--partial_mask_schedule",
        "fully_masked",
        "--include_locality",
        "1",
        "--decode_batch_size",
        "32",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def should_run(output: Path, *, resume: bool) -> bool:
    if not output.exists():
        return True
    if resume and (output / "report_summary.json").exists():
        return False
    raise FileExistsError(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--root", type=Path, default=CAMPAIGN_ROOT / "B3_alphaedit_style_mdm_memit_v1")
    parser.add_argument("--basis_dir", type=Path, default=CAMPAIGN_ROOT / "preservation_basis_v1")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--resume", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.phase == "smoke":
        manifest = CAMPAIGN_ROOT / "protocol_v1" / "dnpe_smoke_20.jsonl"
        for variance in VARIANCES:
            ridge = 1e-3
            output = args.root / _name("smoke", variance, ridge)
            if should_run(output, resume=bool(args.resume)):
                run_one(manifest=manifest, output=output, basis_dir=args.basis_dir, variance=variance, ridge=ridge)
        provisional = select_smoke(args.root, args.baseline)["selected"]
        best_variance = float(provisional["protected_variance"])
        for ridge in (1e-4, 1e-2):
            output = args.root / _name("smoke", best_variance, ridge)
            if should_run(output, resume=bool(args.resume)):
                run_one(manifest=manifest, output=output, basis_dir=args.basis_dir, variance=best_variance, ridge=ridge)
        final = select_smoke(args.root, args.baseline)
        print(final)
        return
    selected = select_smoke(args.root, args.baseline)["selected"]
    output = args.root / "pilot100_selected"
    if should_run(output, resume=bool(args.resume)):
        run_one(
            manifest=CAMPAIGN_ROOT / "protocol_v1" / "dnpe_pilot_100.jsonl",
            output=output,
            basis_dir=args.basis_dir,
            variance=float(selected["protected_variance"]),
            ridge=float(selected["update_ridge"]),
        )
    report = validate_pilot(args.root, args.baseline)
    record_stage(
        "B3_alphaedit_style",
        status="passed" if report["acceptance_pass"] else "failed",
        acceptance_pass=bool(report["acceptance_pass"]),
        output_dir=args.root,
        started_at_utc=now_utc(),
        notes="AlphaEdit-style baseline validated on fresh pilot100.",
        next_stage="B4_timerome_style",
    )
    print(report)


if __name__ == "__main__":
    main()
