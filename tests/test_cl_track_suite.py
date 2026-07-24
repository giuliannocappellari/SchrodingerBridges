from __future__ import annotations

from pathlib import Path

from scripts.run_cl_track_suite import (
    MATCHED_NON_SB,
    TRACK_COMPARATORS,
    TRACK_METHODS,
    TRACK_ORDER,
    editor_command,
    method_run_dir,
    unique_methods,
)


def test_every_mandatory_pilot_track_has_candidates() -> None:
    assert TRACK_ORDER == tuple(f"C{index}" for index in range(1, 10))
    assert all(TRACK_METHODS[track] for track in TRACK_ORDER)
    assert set(MATCHED_NON_SB) == {"C7", "C8"}


def test_method_matrix_is_bounded_and_contains_frozen_core_variants() -> None:
    methods = set(unique_methods())
    assert {
        "growth_shared",
        "growth_block_gate",
        "replay_clean",
        "replay_partial",
        "sparse_routed_memory",
        "gated_adapter_expansion",
        "oedit_partial_memit",
        "lwf_partial_memit",
        "bridge_replay",
        "sb_function_barycenter",
        "dual_memory_10",
        "dual_memory_25",
        "dual_memory_50",
    } <= methods
    assert len(methods) <= 20
    assert "replay_partial" in TRACK_COMPARATORS["C7"]


def test_editor_command_uses_only_fresh_manifests_and_protected_kl_inputs(tmp_path: Path) -> None:
    command = editor_command(
        "growth_shared",
        tmp_path / "cf_cl_pilot_100.jsonl",
        tmp_path / "output",
    )
    joined = " ".join(command).casefold()
    assert "analysis_500" not in joined
    assert "final_test" not in joined
    assert "base_denoising_retention_500.jsonl" in joined
    assert "--covariance_representation diagonal" in joined


def test_method_output_paths_are_versioned_under_suite_root(tmp_path: Path) -> None:
    assert method_run_dir(tmp_path, "growth_shared", "pilot100") == (
        tmp_path / "method_runs" / "growth_shared_pilot100"
    )
