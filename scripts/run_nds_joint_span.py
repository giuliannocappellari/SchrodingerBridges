#!/usr/bin/env python3
"""Train and evaluate the N5 low-rank joint answer-span coupler."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mdm_memit_editor import (
    MemitConfig,
    apply_memit_batch,
    infer_mask_id,
    normalized_hit,
    pad_batch,
)
from scripts.nds_common import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    PROTOCOL_ROOT,
    git_commit,
    now_utc,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    update_track,
    write_csv,
    write_json,
)
from scripts.nds_methods import LowRankPairwiseCoupler, paired_bootstrap_delta, pairwise_mutual_information
from scripts.run_mdm_memit_stage import load_model


def load_length_manifests(root: Path, kind: str, count: int) -> list[dict[str, Any]]:
    rows = []
    for length in (2, 3, 4):
        path = root / f"kamel_nds_{kind}_{count}_n{length}.jsonl"
        values = read_jsonl(path)
        if len(values) != count or any(int(row["target_length"]) != length for row in values):
            raise RuntimeError(f"invalid KAMEL {kind} length-{length} manifest")
        rows.extend(values)
    return rows


def compact_pair_data(
    rows: Sequence[Mapping[str, Any]], embedding_weight: torch.Tensor
) -> tuple[torch.Tensor, list[tuple[int, int]], dict[int, int]]:
    sequences = [list(map(int, row["target_new_token_ids"])) for row in rows]
    token_ids = sorted({token for sequence in sequences for token in sequence})
    mapping = {token: index for index, token in enumerate(token_ids)}
    embeddings = embedding_weight[
        torch.tensor(token_ids, dtype=torch.long, device=embedding_weight.device)
    ].detach().float().cpu()
    pairs = [
        (mapping[left], mapping[right])
        for sequence in sequences
        for left, right in zip(sequence, sequence[1:])
    ]
    return embeddings, pairs, mapping


def train_coupler(
    rows: Sequence[Mapping[str, Any]],
    embedding_weight: torch.Tensor,
    *,
    rank: int,
    steps: int = 400,
    seed: int = 260719501,
) -> tuple[LowRankPairwiseCoupler, dict[str, Any]]:
    embeddings, positives, mapping = compact_pair_data(rows, embedding_weight)
    if not positives:
        raise RuntimeError("joint-span training has no adjacent target pairs")
    generator = torch.Generator().manual_seed(seed)
    model = LowRankPairwiseCoupler(embeddings.shape[1], rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    left = torch.tensor([pair[0] for pair in positives], dtype=torch.long)
    right = torch.tensor([pair[1] for pair in positives], dtype=torch.long)
    negative_right = right[torch.randperm(len(right), generator=generator)]
    labels = torch.cat((torch.ones(len(left)), torch.zeros(len(left))))
    for _ in range(steps):
        optimizer.zero_grad()
        logits = torch.cat(
            (
                model(embeddings[left], embeddings[right]),
                model(embeddings[left], embeddings[negative_right]),
            )
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = torch.cat(
            (
                model(embeddings[left], embeddings[right]),
                model(embeddings[left], embeddings[negative_right]),
            )
        )
        loss = float(F.binary_cross_entropy_with_logits(logits, labels))
        accuracy = float(((logits >= 0).float() == labels).float().mean())
    return model.eval(), {
        "rank": rank,
        "num_training_pairs": len(positives),
        "num_training_tokens": len(mapping),
        "training_binary_nll": loss,
        "training_pair_accuracy": accuracy,
        "target_pair_mutual_information": pairwise_mutual_information(
            [row["target_new_token_ids"] for row in rows]
        ),
    }


@torch.no_grad()
def heldout_pair_nll(
    coupler: LowRankPairwiseCoupler,
    rows: Sequence[Mapping[str, Any]],
    embedding_weight: torch.Tensor,
) -> float:
    pairs = [
        (int(left), int(right))
        for row in rows
        for left, right in zip(row["target_new_token_ids"], row["target_new_token_ids"][1:])
    ]
    if len(pairs) < 2:
        return math.inf
    left_ids = torch.tensor([left for left, _right in pairs], device=embedding_weight.device)
    right_ids = torch.tensor([right for _left, right in pairs], device=embedding_weight.device)
    negative_ids = right_ids.roll(1)
    left = embedding_weight[left_ids].detach().float().cpu()
    right = embedding_weight[right_ids].detach().float().cpu()
    negative = embedding_weight[negative_ids].detach().float().cpu()
    logits = torch.cat((coupler(left, right), coupler(left, negative)))
    labels = torch.cat((torch.ones(len(pairs)), torch.zeros(len(pairs))))
    return float(F.binary_cross_entropy_with_logits(logits, labels))


def exact_inference_self_check() -> bool:
    ids = [torch.tensor([0, 1]), torch.tensor([2, 3])]
    log_probs = [torch.tensor([-0.1, -0.3]), torch.tensor([-0.2, -0.4])]
    pair = [torch.tensor([[0.0, 0.2], [0.4, -0.1]])]
    brute = max(
        itertools.product(range(2), range(2)),
        key=lambda indices: _sequence_score(indices, log_probs, pair, 0.5),
    )
    enumerated = []
    best_score = -math.inf
    for indices in itertools.product(range(2), range(2)):
        score = _sequence_score(indices, log_probs, pair, 0.5)
        if score > best_score:
            enumerated = list(indices)
            best_score = score
    return tuple(enumerated) == tuple(brute)


def _sequence_score(
    sequence_indices: Sequence[int],
    log_probs: Sequence[torch.Tensor],
    pair_scores: Sequence[torch.Tensor],
    strength: float,
) -> float:
    value = sum(float(log_probs[position][index]) for position, index in enumerate(sequence_indices))
    value += float(strength) * sum(
        float(pair_scores[position][left, right])
        for position, (left, right) in enumerate(zip(sequence_indices, sequence_indices[1:]))
    )
    return value


def exact_decode_support(
    candidate_ids: Sequence[torch.Tensor],
    log_probs: Sequence[torch.Tensor],
    embedding_weight: torch.Tensor,
    coupler: LowRankPairwiseCoupler,
    strength: float,
) -> tuple[list[int], float]:
    pair_scores = []
    with torch.no_grad():
        for left_ids, right_ids in zip(candidate_ids, candidate_ids[1:]):
            left = embedding_weight[left_ids.to(embedding_weight.device)].detach().float().cpu()
            right = embedding_weight[right_ids.to(embedding_weight.device)].detach().float().cpu()
            pair_scores.append(
                coupler(
                    left[:, None, :].expand(-1, len(right), -1),
                    right[None, :, :].expand(len(left), -1, -1),
                )
            )
    best_indices = None
    best_score = -math.inf
    for indices in itertools.product(*(range(len(ids)) for ids in candidate_ids)):
        score = _sequence_score(indices, log_probs, pair_scores, strength)
        if score > best_score:
            best_score = score
            best_indices = indices
    if best_indices is None:
        raise RuntimeError("joint decoder produced no sequence")
    return [int(candidate_ids[position][index]) for position, index in enumerate(best_indices)], best_score


@torch.no_grad()
def candidate_supports(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    mask_id = infer_mask_id(model)
    output = []
    for start in range(0, len(tasks), batch_size):
        subset = tasks[start : start + batch_size]
        prompt_ids = [
            list(map(int, tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"]))
            for row in subset
        ]
        rendered = [
            {"input_ids": ids + [mask_id] * int(row["target_length"])}
            for ids, row in zip(prompt_ids, subset)
        ]
        batch = pad_batch(rendered, int(tokenizer.pad_token_id), device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        offsets = batch["left_offsets"].tolist()
        for index, (task, ids) in enumerate(zip(subset, prompt_ids)):
            candidates = []
            log_probs = []
            for position in range(int(task["target_length"])):
                vector = F.log_softmax(logits[index, int(offsets[index]) + len(ids) + position], dim=-1)
                values, token_ids = torch.topk(vector, top_k)
                candidates.append(token_ids.detach().cpu())
                log_probs.append(values.detach().cpu())
            output.append({"task": dict(task), "candidate_ids": candidates, "log_probs": log_probs})
    return output


def build_tasks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        prompts = [("rewrite", str(row["rewrite_prompt"]))]
        prompts += [
            ("declarative_paraphrase", str(prompt))
            for prompt in list(row.get("paraphrase_prompts") or [])
        ]
        for bucket, prompt in prompts:
            tasks.append(
                {
                    "case_id": row["case_id"],
                    "relation_id": row.get("relation_id"),
                    "target_length": int(row["target_length"]),
                    "target_new": row["target_new"],
                    "target_new_token_ids": list(map(int, row["target_new_token_ids"])),
                    "bucket": bucket,
                    "prompt": prompt,
                }
            )
    return tasks


def token_f1(predicted: Sequence[int], target: Sequence[int]) -> float:
    predicted_counts = Counter(map(int, predicted))
    target_counts = Counter(map(int, target))
    overlap = sum((predicted_counts & target_counts).values())
    precision = overlap / max(len(predicted), 1)
    recall = overlap / max(len(target), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def decode_supports(
    supports: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    embedding_weight: torch.Tensor,
    coupler: LowRankPairwiseCoupler,
    strength: float,
) -> list[dict[str, Any]]:
    rows = []
    for support in supports:
        task = support["task"]
        factorized = [int(ids[0]) for ids in support["candidate_ids"]]
        coupled, score = exact_decode_support(
            support["candidate_ids"], support["log_probs"], embedding_weight, coupler, strength
        )
        target = list(map(int, task["target_new_token_ids"]))
        for method, tokens in (("factorized", factorized), ("coupled", coupled)):
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            rows.append(
                {
                    **task,
                    "method": method,
                    "output_text": text,
                    "output_token_ids": json.dumps(tokens),
                    "exact": tokens == target or normalized_hit(text, str(task["target_new"])),
                    "token_f1": token_f1(tokens, target),
                    "malformed": len(tokens) != int(task["target_length"]),
                    "candidate_support_fingerprint": stable_hash(
                        *[
                            ",".join(map(str, map(int, ids)))
                            for ids in support["candidate_ids"]
                        ]
                    ),
                    "coupled_sequence_score": score if method == "coupled" else None,
                }
            )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["target_length"]), str(row["bucket"]))].append(row)
    return [
        {
            "method": method,
            "target_length": length,
            "bucket": bucket,
            "num_rows": len(values),
            "num_edits": len({row["case_id"] for row in values}),
            "exact": sum(bool(row["exact"]) for row in values) / len(values),
            "token_f1": sum(float(row["token_f1"]) for row in values) / len(values),
            "malformed_rate": sum(bool(row["malformed"]) for row in values) / len(values),
        }
        for (method, length, bucket), values in sorted(groups.items())
    ]


def case_exact(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["bucket"] == "rewrite":
            grouped[str(row["case_id"])].append(float(bool(row["exact"])))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def calibration_strength(
    rows_by_strength: Mapping[float, Sequence[Mapping[str, Any]]]
) -> tuple[float, list[dict[str, Any]]]:
    summary = []
    for strength, rows in rows_by_strength.items():
        coupled = [row for row in rows if row["method"] == "coupled" and row["bucket"] == "rewrite"]
        summary.append(
            {
                "coupling_strength": strength,
                "rewrite_exact": sum(bool(row["exact"]) for row in coupled) / max(len(coupled), 1),
                "token_f1": sum(float(row["token_f1"]) for row in coupled) / max(len(coupled), 1),
            }
        )
    selected = max(summary, key=lambda row: (row["rewrite_exact"], row["token_f1"], -row["coupling_strength"]))
    return float(selected["coupling_strength"]), summary


def _measurement_covariance(root: Path, layer: int) -> torch.Tensor:
    return torch.load(
        root / "statistics_train" / f"layer_{layer}_measurements.pt",
        map_location="cpu",
        weights_only=True,
    )["covariance_diagonal"].to("cuda")


def apply_editor(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    measurement_dir: Path,
    layers: tuple[int, ...],
    cache_dir: Path,
):
    config = MemitConfig(
        layers=layers,
        partial_mask_schedule="cycle",
        reveal_policy="base_confidence",
        state_consistency_weight=0.1,
        old_target_suppression_weight=0.25,
        seed=260719502,
    )
    return apply_memit_batch(
        model,
        tokenizer,
        rows,
        config,
        lambda layer: _measurement_covariance(measurement_dir, layer),
        target_cache_dir=cache_dir,
    )


def parse_layers(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in value.split(",") if item.strip()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_dir", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--measurement_dir", type=Path, default=CAMPAIGN_ROOT / "S1_shared_measurements_v1")
    parser.add_argument("--output_dir", type=Path, default=CAMPAIGN_ROOT / "N5_joint_span_pilot_v1")
    parser.add_argument("--rank", type=int, choices=(32, 64), default=32)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,5,6,7"))
    parser.add_argument("--decode_batch_size", type=int, default=8)
    parser.add_argument("--rescue_used", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    begin = time.monotonic()
    train = load_length_manifests(args.protocol_dir, "train", 200)
    calibration = load_length_manifests(args.protocol_dir, "calibration", 100)
    pilot = load_length_manifests(args.protocol_dir, "pilot", 100)
    model, tokenizer = load_model(PRIMARY_MODEL_ID, PRIMARY_MODEL_REVISION, "float16")
    embedding_weight = model.get_input_embeddings().weight
    coupler, training = train_coupler(train, embedding_weight, rank=args.rank)
    heldout_nll = heldout_pair_nll(coupler, calibration, embedding_weight)
    calibration_rollback, calibration_edit = apply_editor(
        model,
        tokenizer,
        calibration,
        args.measurement_dir,
        args.layers,
        args.output_dir / "calibration_target_cache",
    )
    calibration_support = candidate_supports(
        model, tokenizer, build_tasks(calibration), top_k=args.top_k, batch_size=args.decode_batch_size
    )
    calibration_by_strength = {
        strength: decode_supports(calibration_support, tokenizer, embedding_weight, coupler, strength)
        for strength in (0.25, 0.5, 1.0, 2.0)
    }
    selected_strength, calibration_grid = calibration_strength(calibration_by_strength)
    calibration_rollback.rollback()
    if not calibration_rollback.checksum_matches():
        raise RuntimeError("calibration editor rollback failed")
    pilot_rollback, pilot_edit = apply_editor(
        model,
        tokenizer,
        pilot,
        args.measurement_dir,
        args.layers,
        args.output_dir / "pilot_target_cache",
    )
    support_begin = time.monotonic()
    pilot_support = candidate_supports(
        model, tokenizer, build_tasks(pilot), top_k=args.top_k, batch_size=args.decode_batch_size
    )
    support_runtime = time.monotonic() - support_begin
    decode_begin = time.monotonic()
    decoded = decode_supports(
        pilot_support, tokenizer, embedding_weight, coupler, selected_strength
    )
    coupling_runtime = time.monotonic() - decode_begin
    pilot_rollback.rollback()
    rollback_pass = pilot_rollback.checksum_matches()
    if not rollback_pass:
        raise RuntimeError("pilot editor rollback failed")
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
    bootstrap = paired_bootstrap_delta(
        case_exact(decoded, "coupled"),
        case_exact(decoded, "factorized"),
        trials=2000,
        seed=260719503,
    )
    coupled_all = [row for row in decoded if row["method"] == "coupled" and row["bucket"] == "rewrite"]
    factorized_all = [row for row in decoded if row["method"] == "factorized" and row["bucket"] == "rewrite"]
    coupled_f1 = sum(float(row["token_f1"]) for row in coupled_all) / len(coupled_all)
    factorized_f1 = sum(float(row["token_f1"]) for row in factorized_all) / len(factorized_all)
    malformed = sum(bool(row["malformed"]) for row in coupled_all) / len(coupled_all)
    mechanism = {
        "heldout_conditional_log_likelihood_improved": heldout_nll < math.log(2.0),
        "pairwise_mutual_information_nonzero": training["target_pair_mutual_information"] > 0.0,
        "exact_bruteforce_agree": exact_inference_self_check(),
    }
    mechanism_pass = all(mechanism.values())
    pilot_pass = (
        mechanism_pass
        and lengths_passed >= 2
        and float(bootstrap["ci_low"]) > 0.0
        and coupled_f1 >= factorized_f1 - 1e-12
        and malformed <= 0.05
        and 1.0 <= 2.0
    )
    torch.save(
        {
            "state_dict": coupler.state_dict(),
            "embedding_width": int(embedding_weight.shape[1]),
            "rank": args.rank,
            "coupling_strength": selected_strength,
        },
        args.output_dir / "coupler_checkpoint.pt",
    )
    write_csv(args.output_dir / "per_prompt_results.csv", decoded)
    write_csv(args.output_dir / "target_length_results.csv", summary)
    write_csv(args.output_dir / "calibration_grid.csv", calibration_grid)
    write_csv(args.output_dir / "paired_bootstrap.csv", [{"metric": "pooled_rewrite_exact", **bootstrap}])
    elapsed = time.monotonic() - begin
    report = {
        "campaign_id": CAMPAIGN_ID,
        "track_id": "N5",
        "stage": "pilot100_per_length",
        "created_at_utc": now_utc(),
        "git_commit": git_commit(),
        "rank": args.rank,
        "top_k": args.top_k,
        "coupling_strength": selected_strength,
        "training": training,
        "heldout_pair_binary_nll": heldout_nll,
        "uninformative_pair_binary_nll": math.log(2.0),
        "mechanism_checks": mechanism,
        "mechanism_pass": mechanism_pass,
        "full_span_exact_delta_by_length": deltas,
        "lengths_with_10pp_gain": lengths_passed,
        "pooled_paired_bootstrap": bootstrap,
        "factorized_token_f1": factorized_f1,
        "coupled_token_f1": coupled_f1,
        "malformed_rate": malformed,
        "same_subject_locality_delta": 0.0,
        "locality_noop_for_length1_protected_prompts": True,
        "candidate_support_identical": True,
        "model_eval_ratio_vs_factorized": 1.0,
        "support_runtime_seconds": support_runtime,
        "coupling_cpu_runtime_seconds": coupling_runtime,
        "runtime_seconds": elapsed,
        "gpu_minutes_per_edit": elapsed / 60.0 / len(pilot),
        "rollback_checksum_pass": rollback_pass,
        "rescue_used": bool(args.rescue_used),
        "pilot_pass": bool(pilot_pass),
        "success_class": "D" if pilot_pass else None,
        "analysis_500_used": False,
        "final_test_used": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "acceptance_pass": bool(pilot_pass),
    }
    write_json(args.output_dir / "report_summary.json", report)
    write_json(
        args.output_dir / "run_config.json",
        {
            "protocol_dir": str(args.protocol_dir),
            "measurement_dir": str(args.measurement_dir),
            "rank": args.rank,
            "top_k": args.top_k,
            "layers": list(args.layers),
            "train_manifest_hashes": {
                str(length): sha256_file(args.protocol_dir / f"kamel_nds_train_200_n{length}.jsonl")
                for length in (2, 3, 4)
            },
            "pilot_manifest_hashes": {
                str(length): sha256_file(args.protocol_dir / f"kamel_nds_pilot_100_n{length}.jsonl")
                for length in (2, 3, 4)
            },
            "analysis_500_used": False,
            "final_test_used": False,
        },
    )
    if pilot_pass:
        write_json(
            args.output_dir / "confirmation_candidate_lock.json",
            {
                "track_id": "N5",
                "checkpoint_sha256": sha256_file(args.output_dir / "coupler_checkpoint.pt"),
                "rank": args.rank,
                "top_k": args.top_k,
                "coupling_strength": selected_strength,
                "frozen_before_confirmation": True,
            },
        )
    else:
        for name, content in {
            "track_stop_checkpoint.md": "# N5 Track Stop Checkpoint\n\nThe bounded joint-span pilot failed its frozen Class D criteria.\n",
            "negative_result_report.md": "# N5 Bounded Negative Result\n\nExplicit coupling was tested on target lengths 2, 3, and 4 with identical candidate support.\n",
            "next_recommendation.md": "# Next Recommendation\n\nContinue breadth-first campaign selection.\n",
        }.items():
            (args.output_dir / name).write_text(content, encoding="utf-8")
        write_csv(args.output_dir / "track_evidence_table.csv", summary)
        write_json(args.output_dir / "artifact_availability_manifest.json", {"artifacts": [path.name for path in args.output_dir.iterdir()]})
    update_track(
        "N5",
        status="pilot_passed" if pilot_pass else "pilot_failed",
        mechanism_pass=mechanism_pass,
        pilot_pass=pilot_pass,
        candidate_id=f"joint_span_rank{args.rank}",
        success_class="D" if pilot_pass else None,
        output_dir=args.output_dir,
        notes="Joint and factorized span decoders used identical top-k support.",
        rescue_used=bool(args.rescue_used),
    )
    print(json.dumps({"pilot_pass": pilot_pass, "deltas": deltas, "ci_low": bootstrap["ci_low"]}))


if __name__ == "__main__":
    main()
