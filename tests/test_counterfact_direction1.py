import math
import os
import tempfile
import unittest
from argparse import Namespace

from scripts.step3e_gate_audit import assert_dev_only_path
from llada_counterfact_protocol import (
    PROTOCOL_VERSION,
    SimpleWhitespaceTokenizer,
    assign_disjoint_roles,
    context_aware_target_tokenization,
    normalize_counterfact_text,
    render_counterfact_prompt,
    resolve_split_specs,
    target_length_bin,
)
from llada_experiment_reports import (
    harmonic_mean,
    paired_bootstrap_delta_by_case,
    self_normalized_locality,
)
from llada_runtime_editor_eval import (
    RolloutConfig,
    decompose_method,
    enforce_lock_requirements,
    gate_should_activate,
    load_relation_bank,
    relation_content_text,
    relation_text_for_record,
    normalize_method_args,
    subject_matches_prompt,
    sparse_support_guidance_kl,
    token_f1,
)
from llada_sb_common import load_edits


class CounterFactProtocolTests(unittest.TestCase):
    def test_normalization_and_rendering(self):
        self.assertEqual(normalize_counterfact_text("  Paris!! "), "paris")
        self.assertEqual(
            render_counterfact_prompt("The capital of {} is", "France"),
            "The capital of France is",
        )

    def test_context_aware_tokenization_records_mismatch(self):
        tokenizer = SimpleWhitespaceTokenizer()
        result = context_aware_target_tokenization(tokenizer, "The capital is", " Paris")
        self.assertTrue(result.prefix_match)
        self.assertEqual(len(result.target_token_ids), 1)
        self.assertEqual(result.target_token_ids, result.standalone_token_ids)

    def test_target_length_bin(self):
        self.assertEqual(target_length_bin(1), "1")
        self.assertEqual(target_length_bin(2), "2")
        self.assertEqual(target_length_bin(3), "3")
        self.assertEqual(target_length_bin(4), ">=4")

    def test_sparse_support_guidance_kl(self):
        kl = sparse_support_guidance_kl([0.8, 0.2], [0.5, 0.5])
        expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
        self.assertAlmostEqual(kl, expected)
        self.assertAlmostEqual(sparse_support_guidance_kl([0.5, 0.5], [0.5, 0.5]), 0.0)

    def test_token_f1_uses_best_alias(self):
        self.assertEqual(token_f1([1, 2], [[1, 2], [3, 4]]), 1.0)
        self.assertAlmostEqual(token_f1([1, 9], [[1, 2]]), 0.5)

    def test_gate_subject_relation(self):
        raw_edit = {
            "subject": "Ada Lovelace",
            "rewrite_template": "The profession of {} is",
        }
        self.assertTrue(
            gate_should_activate(raw_edit, "The profession of Ada Lovelace is", "subject_relation")
        )
        self.assertFalse(
            gate_should_activate(raw_edit, "Ada Lovelace was born in", "subject_relation")
        )
        self.assertTrue(gate_should_activate(raw_edit, "Ada Lovelace was born in", "subject"))

    def test_gate_hybrid_relation_or_uses_relation_similarity(self):
        raw_edit = {
            "subject": "Ada Lovelace",
            "relation_id": "P106",
            "rewrite_template": "The profession of {} is",
            "target": " mathematician",
            "old_target": " writer",
        }
        cfg = RolloutConfig(
            steps=4,
            bridge_topk=4,
            mc_rollouts=2,
            guidance_scale=1.0,
            reward_mode="soft_overlap",
            reward_beta=6.0,
            target_logit_bias=0.0,
            gate_mode="hybrid_relation_or",
            temperature=1.0,
            relation_sim_rewrite_threshold=0.45,
            relation_sim_bank_threshold=0.10,
            relation_bank_path="",
        )
        self.assertTrue(
            gate_should_activate(
                raw_edit,
                "The profession of Ada Lovelace is",
                "hybrid_relation_or",
                cfg,
            )
        )
        self.assertFalse(
            gate_should_activate(
                raw_edit,
                "Ada Lovelace was born in",
                "hybrid_relation_or",
                cfg,
            )
        )

    def test_gate_hybrid_relation_and_requires_both_branches(self):
        raw_edit = {
            "subject": "Ada Lovelace",
            "relation_id": "P106",
            "rewrite_template": "The profession of {} is",
            "target": " mathematician",
            "old_target": " writer",
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as f:
            f.write(
                '{"subject":"Ada Lovelace","relation_id":"P106",'
                '"rewrite_template":"The occupation of {} is",'
                '"target":" mathematician","old_target":" writer"}\n'
            )
            f.flush()
            cfg = RolloutConfig(
                steps=4,
                bridge_topk=4,
                mc_rollouts=2,
                guidance_scale=1.0,
                reward_mode="soft_overlap",
                reward_beta=6.0,
                target_logit_bias=0.0,
                gate_mode="hybrid_relation_and",
                temperature=1.0,
                relation_sim_rewrite_threshold=0.45,
                relation_sim_bank_threshold=0.10,
                relation_bank_path=f.name,
            )
            self.assertTrue(
                gate_should_activate(
                    raw_edit,
                    "The profession of Ada Lovelace is",
                    "hybrid_relation_and",
                    cfg,
                )
            )
            strict_cfg = RolloutConfig(**{**cfg.__dict__, "relation_sim_bank_threshold": 0.99})
            self.assertFalse(
                gate_should_activate(
                    raw_edit,
                    "The profession of Ada Lovelace is",
                    "hybrid_relation_and",
                    strict_cfg,
                )
            )

    def test_gate_helpers_preserve_relation_terms_and_subject_boundaries(self):
        raw_edit = {
            "subject": "Ada Lovelace",
            "relation_id": "P106",
            "rewrite_template": "{} works as a",
            "target": " mathematician",
            "old_target": " writer",
        }
        self.assertTrue(subject_matches_prompt("Ada Lovelace", "Ada Lovelace works as a"))
        self.assertFalse(subject_matches_prompt("Ada Lovelace", "Adalovelace works as a"))
        self.assertIn("works", relation_content_text("Ada Lovelace works as a", subject="Ada Lovelace"))
        self.assertIn("works", relation_text_for_record(raw_edit))

    def test_relation_bank_construction_uses_dev_rows(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as f:
            f.write(
                '{"subject":"Ada Lovelace","relation_id":"P106",'
                '"rewrite_template":"{} works as a",'
                '"target":" mathematician","old_target":" writer"}\n'
            )
            f.flush()
            bank = load_relation_bank(f.name)
        self.assertIn("P106", bank)
        self.assertIn("works", bank["P106"][0])

    def test_step3e_local_scripts_reject_locked_split_paths(self):
        assert_dev_only_path(os.path.join("runs", "counterfact_direction1_v1", "protocol", "dev_tune_200.jsonl"))
        with self.assertRaises(AssertionError):
            assert_dev_only_path(os.path.join("runs", "counterfact_direction1_v1", "protocol", "analysis_500.jsonl"))
        with self.assertRaises(AssertionError):
            assert_dev_only_path(os.path.join("runs", "counterfact_direction1_v1", "protocol", "final_test_500.jsonl"))

    def test_lock_enforcement_for_analysis(self):
        with self.assertRaises(ValueError):
            enforce_lock_requirements(split_role="analysis_500", methods=["mc_bridge"], lock_config=None)

        lock = {
            "thresholds_frozen": True,
            "span_policy_frozen": True,
            "gate_policy_frozen": True,
            "normalization_frozen": True,
            "metrics_frozen": True,
            "selected_dev_pareto_point": "dev-pareto-1",
            "path_kl_bridge_ready": True,
            "path_kl_bridge_report_path": "runs/example/path_kl.json",
        }
        enforce_lock_requirements(split_role="analysis_500", methods=["mc_bridge"], lock_config=lock)

    def test_self_normalized_locality_and_harmonic(self):
        self.assertAlmostEqual(self_normalized_locality(0.5, 0.25), 2.0)
        self.assertAlmostEqual(self_normalized_locality(0.5, 0.25, clip=True), 1.0)
        self.assertAlmostEqual(harmonic_mean([1.0, 1.0, 1.0]), 1.0)

    def test_paired_bootstrap_resamples_case_ids(self):
        rows = [
            {"edit_id": "a", "case_id": "a_0", "method": "mc_bridge", "bucket": "rewrite", "exact_rate": 0.8},
            {"edit_id": "a", "case_id": "a_0", "method": "base", "bucket": "rewrite", "exact_rate": 0.1},
            {"edit_id": "b", "case_id": "b_0", "method": "mc_bridge", "bucket": "rewrite", "exact_rate": 0.4},
            {"edit_id": "b", "case_id": "b_0", "method": "base", "bucket": "rewrite", "exact_rate": 0.2},
            # Repeated prompt row for edit b. Bootstrap should average within
            # edit_id before resampling, not treat this as a third unit.
            {"edit_id": "b", "case_id": "b_1", "method": "mc_bridge", "bucket": "rewrite", "exact_rate": 0.6},
            {"edit_id": "b", "case_id": "b_1", "method": "base", "bucket": "rewrite", "exact_rate": 0.2},
        ]
        stats = paired_bootstrap_delta_by_case(
            rows,
            candidate_method="mc_bridge",
            baseline_method="base",
            bucket="rewrite",
            metric="exact_rate",
            samples=100,
            seed=0,
        )
        self.assertIsNotNone(stats)
        self.assertEqual(stats["num_cases"], 2)
        self.assertEqual(stats["num_edits"], 2)
        self.assertAlmostEqual(stats["mean_delta"], 0.5)

    def test_old_jsonl_compatibility(self):
        root = os.path.dirname(os.path.dirname(__file__))
        dev16 = os.path.join(root, "dev16.jsonl")
        if os.path.exists(dev16):
            edits = load_edits(dev16)
            self.assertGreater(len(edits), 0)
            self.assertEqual(edits[0].id, "counterfact_train_0")

    def test_protocol_version_constant(self):
        self.assertEqual(PROTOCOL_VERSION, "counterfact_direction1_v1")

    def test_method_args_accept_commas_or_spaces(self):
        self.assertEqual(
            normalize_method_args(["base,target_logit_bias", "mc_bridge"]),
            ["base", "target_logit_bias", "mc_bridge"],
        )

    def test_gated_method_decomposition(self):
        cfg = RolloutConfig(
            steps=4,
            bridge_topk=4,
            mc_rollouts=2,
            guidance_scale=1.0,
            reward_mode="soft_overlap",
            reward_beta=6.0,
            target_logit_bias=0.0,
            gate_mode="subject",
            temperature=1.0,
        )
        self.assertEqual(
            decompose_method("mc_bridge_gated_subject", cfg),
            ("mc_bridge", "subject", "mc_bridge_gated_subject"),
        )
        self.assertEqual(
            decompose_method("raw_bridge_gated", cfg),
            ("mc_bridge", "subject", "raw_bridge_gated_subject"),
        )
        hybrid_cfg = RolloutConfig(**{**cfg.__dict__, "gate_mode": "hybrid_relation_or"})
        self.assertEqual(
            decompose_method("mc_bridge_gated_hybrid", hybrid_cfg),
            ("mc_bridge", "hybrid_relation_or", "mc_bridge_gated_hybrid"),
        )

    def test_smoke_split_specs_are_tiny(self):
        args = Namespace(
            smoke=1,
            dev_size=-1,
            analysis_size=-1,
            ablation_size=-1,
            final_test_size=-1,
        )
        train_specs, test_specs = resolve_split_specs(args)
        self.assertEqual(dict(train_specs)["dev_tune_200"], 10)
        self.assertEqual(dict(train_specs)["analysis_500"], 10)
        self.assertEqual(dict(train_specs)["ablation_500"], 10)
        self.assertEqual(dict(test_specs)["final_test_500"], 10)

    def test_explicit_split_size_overrides(self):
        args = Namespace(
            smoke=0,
            dev_size=3,
            analysis_size=4,
            ablation_size=5,
            final_test_size=6,
        )
        train_specs, test_specs = resolve_split_specs(args)
        self.assertEqual(dict(train_specs)["dev_tune_200"], 3)
        self.assertEqual(dict(train_specs)["analysis_500"], 4)
        self.assertEqual(dict(train_specs)["ablation_500"], 5)
        self.assertEqual(dict(test_specs)["final_test_500"], 6)

    def test_assign_disjoint_roles_fails_when_undersized(self):
        records = [{"case_id": f"c{i}", "relation_id": "r", "target_length_bin": "1"} for i in range(2)]
        with self.assertRaises(ValueError):
            assign_disjoint_roles(
                records,
                [("dev_tune_200", 3)],
                seed=0,
                source_label="unit",
            )


if __name__ == "__main__":
    unittest.main()
