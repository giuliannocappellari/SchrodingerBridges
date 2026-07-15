#!/usr/bin/env python3
"""Shared local utilities for the Direction 3 controller scaffold.

These helpers are intentionally stdlib-only. Direction 3 fake-mode scripts
must not import model-loading modules or touch RunPod/GPU state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
D1_PROTOCOL_VERSION = "counterfact_direction1_v1"
D3_PROTOCOL_VERSION = "counterfact_direction3_controller_v1"
D1_ROOT = Path("runs/counterfact_direction1_v1")
D3_ROOT = Path("runs/counterfact_direction3_controller_v1")
LOCKED_SPLIT_TOKENS = ("analysis_500", "final_test_500", "final_test_full")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def read_json(path: str | Path) -> Any:
    with repo_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with repo_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with repo_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with full.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    full = repo_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with full.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_float(text: str, low: float = 0.0, high: float = 1.0) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return low + (high - low) * value


def stable_int(text: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def normalize_len_bin(value: Any) -> str:
    n = int(value or 1)
    return ">=4" if n >= 4 else str(n)


def ensure_no_locked_artifact_use(path: str | Path) -> None:
    text = str(path)
    if any(token in text for token in LOCKED_SPLIT_TOKENS):
        raise AssertionError(f"Locked split artifact is not allowed here: {path}")


def collect_locked_manifest_exclusions(protocol_dir: str | Path = D1_ROOT / "protocol") -> Dict[str, Any]:
    """Read locked manifests only to collect exclusion IDs and fingerprints."""

    protocol_dir = Path(protocol_dir)
    manifest_names = [
        "dev_tune_200",
        "analysis_500",
        "ablation_500",
        "final_test_500",
        "final_test_full",
    ]
    excluded_case_ids: set[str] = set()
    excluded_source_keys: set[Tuple[str, int]] = set()
    manifests: Dict[str, Any] = {}

    for name in manifest_names:
        path = protocol_dir / f"{name}.jsonl"
        full = repo_path(path)
        if not full.exists():
            raise FileNotFoundError(f"Required split manifest missing: {path}")

        ids: List[str] = []
        source_keys: List[Tuple[str, int]] = []
        raw_sha = hashlib.sha256()
        with full.open("rb") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                raw_sha.update(raw_line)
                row = json.loads(raw_line.decode("utf-8"))
                case_id = str(row.get("case_id") or row.get("id"))
                source_split = str(row.get("source_dataset_split", ""))
                source_index = int(row.get("source_index", -1))
                ids.append(case_id)
                source_keys.append((source_split, source_index))

        excluded_case_ids.update(ids)
        excluded_source_keys.update(source_keys)
        manifests[name] = {
            "path": str(path),
            "sha256": raw_sha.hexdigest(),
            "count": len(ids),
            "id_only_exclusion_use": True,
            "locked_prompts_or_labels_used": False,
        }

    return {
        "excluded_case_ids": sorted(excluded_case_ids),
        "excluded_source_keys": sorted([f"{split}:{idx}" for split, idx in excluded_source_keys]),
        "manifests": manifests,
    }


def load_valid_train_pool(validity_path: str | Path = D1_ROOT / "protocol/validity_report.json") -> List[Dict[str, Any]]:
    data = read_json(validity_path)
    pool = []
    for row in data.get("train", []):
        if not row.get("valid", False):
            continue
        item = dict(row)
        item["id"] = str(row.get("case_id"))
        item["case_id"] = str(row.get("case_id"))
        item["target_length_bin"] = normalize_len_bin(row.get("target_new_context_token_len"))
        item["subject_len_chars"] = len(str(row.get("subject", "")))
        item["subject_len_tokens"] = len(str(row.get("subject", "")).split())
        item["subject_ambiguity_proxy"] = "short" if item["subject_len_tokens"] <= 1 else "multi_word"
        pool.append(item)
    return pool


def deterministic_interleaved_sample(rows: Sequence[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("target_length_bin", "1"))].append(dict(row))

    for key, items in groups.items():
        items.sort(key=lambda r: hashlib.sha1(f"{seed}:{key}:{r['case_id']}".encode("utf-8")).hexdigest())

    ordered_bins = ["1", "2", "3", ">=4"]
    selected: List[Dict[str, Any]] = []
    while len(selected) < count:
        made_progress = False
        for bin_name in ordered_bins:
            if groups.get(bin_name):
                selected.append(groups[bin_name].pop(0))
                made_progress = True
                if len(selected) >= count:
                    break
        if not made_progress:
            break
    if len(selected) < count:
        raise ValueError(f"Could only select {len(selected)} rows; needed {count}")
    return selected


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [x / z for x in exps]


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = rankdata(xs)
    ry = rankdata(ys)
    mx = mean(rx)
    my = mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    deny = math.sqrt(sum((y - my) ** 2 for y in ry))
    return num / (denx * deny) if denx and deny else 0.0


def summarize_counter(rows: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(rows).items()))
