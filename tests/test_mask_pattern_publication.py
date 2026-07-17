from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

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
from scripts.mask_pattern_publication_common import SECONDARY_MODEL_REVISION, autonomous_enabled
from scripts.mask_pattern_publication_runtime import (
    PlannerSpec,
    bounded_beam_order,
    bounded_kl_beam_order,
    bounded_random_order,
    planner_policy,
    planner_spec_from_label,
)
from scripts.run_partial_state_publication_audit import _schedule_unit_tests
from scripts.run_publication_planner_dev import _safety_pass
from scripts.mask_pattern_publication_stats import holm_adjust, paired_bootstrap, paired_values
from scripts.mdm_memit_editor import model_hidden_size, resolved_block_name, resolved_key_module_name
from reproduce_paper import check_dp
from scripts.validate_publication_campaign import REPORTS


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
    assert len(SECONDARY_MODEL_REVISION) == 40


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


def test_paper_partial_state_schedule_contract() -> None:
    report = _schedule_unit_tests()
    assert report["acceptance_pass"]
    assert all(report["checks"].values())


def test_publication_gpu_runners_normalize_relative_output_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "run_partial_state_publication_audit.py",
        "run_publication_planner_dev.py",
    ):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "if not args.output_dir.is_absolute():" in source
        assert "(ROOT / args.output_dir).resolve()" in source


def test_bounded_planners_respect_unique_state_budget() -> None:
    costs = fixture_costs(5)
    for budget in (5, 10, 20):
        beam_order, beam_queries, _ = bounded_beam_order(
            costs, 5, beam_width=8, state_budget=budget
        )
        random_order, random_queries, _ = bounded_random_order(
            costs, 5, state_budget=budget, seed=19
        )
        assert sorted(beam_order) == list(range(5))
        assert sorted(random_order) == list(range(5))
        assert beam_queries <= budget
        assert random_queries <= budget


def test_bounded_kl_beam_preserves_reference_cost_score_and_budget() -> None:
    n = 5
    costs = fixture_costs(n)
    order, queries, expansions = bounded_kl_beam_order(
        costs,
        n,
        reference=uniform_reference(n),
        beta=1.0,
        beam_width=8,
        state_budget=10,
    )
    assert sorted(order) == list(range(n))
    assert queries <= 10
    assert expansions > 0


def test_planner_policy_reports_exact_full_table_queries() -> None:
    n = 4
    table = {
        "n": n,
        "costs": {f"{mask}:{index}": value for (mask, index), value in fixture_costs(n).items()},
        "target_probabilities": {
            f"{mask}:{index}": 0.5 for mask, index in fixture_costs(n)
        },
        "maximum_confidences": {
            f"{mask}:{index}": 0.5 for mask, index in fixture_costs(n)
        },
    }
    _, order, diagnostics = planner_policy(
        table, PlannerSpec("global", "deterministic_global")
    )
    assert sorted(order or ()) == list(range(n))
    assert diagnostics["planner_state_queries"] == (1 << n) - 1


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("finite_uniform_beta0.5", "finite_beta"),
        ("beta0_edited_target_confidence", "beta_zero"),
        ("beam_width8", "beam"),
        ("random_search_full", "random_search"),
        ("online_beam8_budget16", "bounded_beam"),
        ("one_step_myopic", "myopic"),
    ],
)
def test_frozen_planner_label_round_trip(label: str, kind: str) -> None:
    spec = planner_spec_from_label(label, n=5, seed=11)
    assert spec.label == label
    assert spec.kind == kind


def test_frozen_fixed_order_requires_lock_value() -> None:
    with pytest.raises(ValueError, match="frozen order"):
        planner_spec_from_label("best_fixed_permutation", n=4)
    spec = planner_spec_from_label("best_fixed_permutation", n=4, fixed_order=[2, 0, 3, 1])
    assert spec.fixed_order == (2, 0, 3, 1)


def test_planner_safety_uses_base_relative_false_positive_budgets() -> None:
    base = {"same_subject_tfpr": 0.01, "far_tfpr": 0.02}
    candidate = {
        "same_subject_tfpr": 0.039,
        "far_tfpr": 0.049,
        "malformed_rate": 0.05,
    }
    passed, thresholds = _safety_pass(candidate, base)
    assert passed
    assert thresholds["same_subject_tfpr_budget"] == pytest.approx(0.04)
    assert thresholds["far_tfpr_budget"] == pytest.approx(0.05)
    candidate["same_subject_tfpr"] = 0.041
    assert not _safety_pass(candidate, base)[0]


def test_paired_statistics_average_seed_rows_within_edit() -> None:
    rows = []
    for case_id, left, right in (("a", 1.0, 0.0), ("b", 0.5, 0.0)):
        for value in (left, left):
            rows.append(
                {
                    "family": "finite",
                    "case_id": case_id,
                    "bucket": "rewrite",
                    "target_length": 3,
                    "score": value,
                }
            )
        rows.append(
            {
                "family": "baseline",
                "case_id": case_id,
                "bucket": "rewrite",
                "target_length": 3,
                "score": right,
            }
        )
    pairs = paired_values(
        rows,
        left="finite",
        right="baseline",
        bucket="rewrite",
        metric="score",
        lengths={3},
    )
    assert pairs == [("a", 1.0, 0.0), ("b", 0.5, 0.0)]
    result = paired_bootstrap(pairs, resamples=500, seed=1)
    assert result["mean_delta"] == pytest.approx(0.75)
    assert result["ci95_low"] > 0


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    rows = holm_adjust(
        [{"name": "a", "p_two_sided": 0.01}, {"name": "b", "p_two_sided": 0.03}]
    )
    assert rows[0]["holm_adjusted_p"] == pytest.approx(0.02)
    assert rows[1]["holm_adjusted_p"] == pytest.approx(0.03)


def test_dream_module_map_uses_runtime_available_hidden_width() -> None:
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[object()]),
        config=SimpleNamespace(hidden_size=3584),
    )
    assert resolved_block_name(model, 4) == "model.layers.4"
    assert resolved_key_module_name(model, 4) == "model.layers.4.self_attn.o_proj"
    assert model_hidden_size(model) == 3584
    assert (
        resolved_key_module_name(model, 2, "custom.blocks.{layer}.projection")
        == "custom.blocks.2.projection"
    )


def test_reproduce_paper_dp_check_is_model_free_and_exact() -> None:
    report = check_dp()
    assert report["acceptance_pass"]
    assert report["llada_loaded"] is False
    assert report["maximum_probability_error"] < 1e-10


def test_terminal_validator_requires_every_publication_track() -> None:
    assert set(REPORTS) == {
        "P0_source_audit",
        "P1_partial_state",
        "P2_theory",
        "P3_planners",
        "P4_llada_locked",
        "P5_dream_locked",
        "P6_editor_generality",
        "P7_approximation",
        "P8_package",
    }


def test_publication_autonomous_mode_accepts_goal_launch_alias(monkeypatch) -> None:
    monkeypatch.delenv("MASK_PATTERN_SB_PUBLICATION_AUTONOMOUS_MODE", raising=False)
    monkeypatch.setenv("SB_ALT_AUTONOMOUS_MODE", "1")
    assert autonomous_enabled()
