#!/usr/bin/env python3
"""Run frozen N5 joint-span confirmation on fresh KAMEL manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nds_common import CAMPAIGN_ID, CAMPAIGN_ROOT, PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, PROTOCOL_ROOT, git_commit, now_utc, read_json, sha256_file, update_track, write_csv, write_json
from scripts.nds_methods import LowRankPairwiseCoupler, paired_bootstrap_delta
from scripts.run_mdm_memit_stage import load_model
from scripts.run_nds_joint_span import apply_editor, build_tasks, candidate_supports, case_exact, decode_supports, load_length_manifests, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", type=Path, required=True)
    parser.add_argument("--protocol_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--measurement_dir", type=Path, default=CAMPAIGN_ROOT / "S1_shared_measurements_v1")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--decode_batch_size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pilot = read_json(args.pilot_dir / "report_summary.json")
    lock = read_json(args.pilot_dir / "confirmation_candidate_lock.json")
    if not pilot.get("pilot_pass"):
        raise RuntimeError("N5 confirmation requires a pilot-passed candidate")
    if sha256_file(args.pilot_dir / "coupler_checkpoint.pt") != lock["checkpoint_sha256"]:
        raise RuntimeError("N5 checkpoint hash does not match the pilot lock")
    confirmation = load_length_manifests(args.protocol_dir, "confirmation", 200)
    model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, "float16")
    embedding_weight = model.get_input_embeddings().weight
    checkpoint = torch.load(args.pilot_dir / "coupler_checkpoint.pt", map_location="cpu", weights_only=True)
    coupler = LowRankPairwiseCoupler(checkpoint["embedding_width"], checkpoint["rank"])
    coupler.load_state_dict(checkpoint["state_dict"])
    coupler.eval()
    begin = time.monotonic()
    rollback, _diagnostics = apply_editor(model, tokenizer, confirmation, args.measurement_dir, (4, 5, 6, 7), args.output_dir / "target_cache")
    supports = candidate_supports(model, tokenizer, build_tasks(confirmation), top_k=int(lock["top_k"]), batch_size=args.decode_batch_size)
    decoded = decode_supports(supports, tokenizer, embedding_weight, coupler, float(lock["coupling_strength"]))
    rollback.rollback()
    if not rollback.checksum_matches():
        raise RuntimeError("N5 confirmation editor rollback failed")
    summary = summarize(decoded)
    rewrite = [row for row in summary if row["bucket"] == "rewrite"]
    deltas = {}
    lengths_passed = 0
    for length in (2, 3, 4):
        factorized = next(row for row in rewrite if row["method"] == "factorized" and row["target_length"] == length)
        coupled = next(row for row in rewrite if row["method"] == "coupled" and row["target_length"] == length)
        delta = float(coupled["exact"]) - float(factorized["exact"])
        deltas[str(length)] = delta
        lengths_passed += delta >= 0.10
    bootstrap = paired_bootstrap_delta(case_exact(decoded, "coupled"), case_exact(decoded, "factorized"), trials=10000, seed=260719701)
    coupled = [row for row in decoded if row["method"] == "coupled" and row["bucket"] == "rewrite"]
    factorized = [row for row in decoded if row["method"] == "factorized" and row["bucket"] == "rewrite"]
    coupled_f1 = sum(float(row["token_f1"]) for row in coupled) / len(coupled)
    factorized_f1 = sum(float(row["token_f1"]) for row in factorized) / len(factorized)
    malformed = sum(bool(row["malformed"]) for row in coupled) / len(coupled)
    passed = lengths_passed >= 2 and float(bootstrap["ci_low"]) > 0 and coupled_f1 >= factorized_f1 and malformed <= 0.05
    write_csv(args.output_dir / "per_prompt_results.csv", decoded)
    write_csv(args.output_dir / "target_length_results.csv", summary)
    write_csv(args.output_dir / "paired_bootstrap.csv", [{"metric": "pooled_rewrite_exact", **bootstrap}])
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N5",
        "stage": "fresh_confirmation_200_per_length",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "checkpoint_frozen_before_confirmation": True,
        "full_span_exact_delta_by_length": deltas,
        "lengths_with_10pp_gain": lengths_passed,
        "pooled_paired_bootstrap": bootstrap,
        "factorized_token_f1": factorized_f1,
        "coupled_token_f1": coupled_f1,
        "malformed_rate": malformed,
        "runtime_seconds": time.monotonic() - begin,
        "confirmation_pass": bool(passed),
        "success_class": "D" if passed else None,
        "analysis_500_used": False,
        "final_test_used": False,
        "acceptance_pass": bool(passed),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(args.output_dir / "validation_report.json", {"checkpoint_hash_match": True, "bootstrap_trials": 10000, "confirmation_used_for_tuning": False, "acceptance_pass": bool(passed)})
    if not passed:
        (args.output_dir / "track_stop_checkpoint.md").write_text("# N5 Confirmation Stop\n\nThe frozen coupled decoder failed fresh confirmation.\n", encoding="utf-8")
        (args.output_dir / "negative_result_report.md").write_text("# N5 Confirmation Failure\n\nNo rank, support, or coupling strength was changed.\n", encoding="utf-8")
        write_csv(args.output_dir / "track_evidence_table.csv", summary)
        write_json(args.output_dir / "artifact_availability_manifest.json", {"artifacts": [path.name for path in args.output_dir.iterdir()]})
        (args.output_dir / "next_recommendation.md").write_text("# Next Recommendation\n\nExclude N5 from final selection.\n", encoding="utf-8")
    update_track("N5", status="confirmation_passed" if passed else "confirmation_failed", candidate_id=f"joint_span_rank{lock['rank']}", confirmation_pass=bool(passed), success_class="D" if passed else None, output_dir=args.output_dir)
    print(json.dumps({"confirmation_pass": passed, "deltas": deltas, "ci_low": bootstrap["ci_low"]}))


if __name__ == "__main__":
    main()
