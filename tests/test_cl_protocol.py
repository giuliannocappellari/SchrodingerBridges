from __future__ import annotations

from scripts.build_cl_protocol import (
    CF_BLOCK_SIZES,
    _deduplicate,
    normalize_counterfact,
    prompt_fingerprint,
)
from scripts.cl_common import CAMPAIGN_ID


def _row(index: int, prompt: str) -> dict:
    return {
        "source_index": index,
        "case_id": f"old_{index}",
        "rewrite_prompt": prompt,
        "target_length": 1,
        "relation_id": "P1",
    }


def test_counterfact_normalization_is_campaign_scoped() -> None:
    row = normalize_counterfact(_row(7, "Ada works as"))
    assert row["campaign_id"] == CAMPAIGN_ID
    assert row["case_id"] == "cl_cf_7"
    assert row["prompt_fingerprint"] == prompt_fingerprint("Ada works as")


def test_rewrite_dedup_is_deterministic() -> None:
    rows = [normalize_counterfact(_row(2, "Same prompt")), normalize_counterfact(_row(1, "Same prompt"))]
    first = _deduplicate(rows)
    second = _deduplicate(list(reversed(rows)))
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]
    assert len(first) == 1


def test_stream_block_sizes_are_frozen() -> None:
    assert CF_BLOCK_SIZES == {
        "cf_cl_smoke_20": 5,
        "cf_cl_pilot_100": 10,
        "cf_cl_confirmation_200": 10,
        "cf_cl_scale_500": 50,
    }

