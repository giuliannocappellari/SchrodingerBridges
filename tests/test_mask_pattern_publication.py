from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from scripts.build_mask_pattern_publication_protocol import _balanced_select
from scripts.mask_pattern_kl_control import (
    beam_search_paths,
    enumerate_gibbs_paths,
    path_cost,
    policy_path_distribution,
    solve_deterministic_global,
    solve_exact_kl_control,
    uniform_reference,
)


def fixture_costs(n: int) -> dict[tuple[int, int], float]:
    return {
        (mask, index): 0.2 + 0.3 * index + 0.1 * ((mask + index) % 3)
        for mask in range((1 << n) - 1)
        for index in range(n)
        if not mask & (1 << index)
    }


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_exact_policy_matches_bruteforce(n: int) -> None:
    costs = fixture_costs(n)
    reference = uniform_reference(n)
    solution = solve_exact_kl_control(costs, n, beta=1.3, reference=reference)
    brute = enumerate_gibbs_paths(costs, n, beta=1.3, reference=reference)
    policy = policy_path_distribution(solution.policy, n)
    assert max(abs(brute[path] - policy[path]) for path in brute) < 1e-10
    assert abs(sum(brute.values()) - 1.0) < 1e-10


def test_deterministic_and_beam_find_global_minimum() -> None:
    n = 4
    costs = fixture_costs(n)
    order, value = solve_deterministic_global(costs, n)
    brute = min(
        ((path_cost(path, costs), path) for path in itertools.permutations(range(n))),
        key=lambda row: (row[0], row[1]),
    )
    assert value == pytest.approx(brute[0])
    assert path_cost(order, costs) == pytest.approx(brute[0])
    assert beam_search_paths(costs, n, beam_width=24)[0][1] == pytest.approx(brute[0])


def test_beta_zero_is_reference() -> None:
    n = 4
    solution = solve_exact_kl_control(fixture_costs(n), n, beta=0.0)
    reference = uniform_reference(n)
    for mask, probabilities in solution.policy.items():
        for index, probability in probabilities.items():
            assert probability == pytest.approx(reference[(mask, index)], abs=1e-12)


def test_large_beta_zero_mass_diagnostics_remain_finite() -> None:
    solution = solve_exact_kl_control(fixture_costs(6), 6, beta=1_000_000.0)
    assert solution.expected_cost >= 0.0
    assert solution.path_entropy >= 0.0
    assert solution.kl_from_reference >= 0.0


def test_reference_requires_positive_support() -> None:
    n = 2
    reference = uniform_reference(n)
    reference[(0, 0)] = 0.0
    with pytest.raises(ValueError, match="positive finite support"):
        solve_exact_kl_control(fixture_costs(n), n, beta=1.0, reference=reference)


def test_balanced_split_selection_is_deterministic_and_disjoint() -> None:
    rows = [
        {
            "case_id": f"case_{index}",
            "source_fingerprint": f"fp_{index}",
            "relation_id": f"P{index % 4}",
        }
        for index in range(40)
    ]
    used_a: set[str] = set()
    first = _balanced_select(rows, 12, role="dev", used=used_a, seed=7)
    second = _balanced_select(rows, 8, role="locked", used=used_a, seed=8)
    used_b: set[str] = set()
    repeated = _balanced_select(rows, 12, role="dev", used=used_b, seed=7)
    assert [row["case_id"] for row in first] == [row["case_id"] for row in repeated]
    assert {row["source_fingerprint"] for row in first}.isdisjoint(
        {row["source_fingerprint"] for row in second}
    )
    assert max(
        sum(row["relation_id"] == relation for row in first)
        for relation in {row["relation_id"] for row in first}
    ) <= 3


def test_active_campaign_and_registry_are_publication_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    active = json.loads((root / "ACTIVE_RESEARCH_CAMPAIGN.json").read_text())
    registry = json.loads((root / "PUBLICATION_PROTOCOL_REGISTRY.json").read_text())
    assert active["active_campaign"] == "mask_pattern_sb_publication_confirmation_v1"
    assert registry["campaign"] == "mask_pattern_sb_publication_confirmation_v1"
    assert [track["id"] for track in registry["tracks"]] == [f"P{i}" for i in range(9)]


def test_publication_protocol_does_not_name_historical_locked_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "scripts/bootstrap_mask_pattern_publication.py",
        root / "scripts/build_mask_pattern_publication_protocol.py",
        root / "scripts/audit_mask_pattern_theory.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "analysis_500.jsonl" not in text
        assert "final_test_500.jsonl" not in text
        assert "final_test_full.jsonl" not in text
