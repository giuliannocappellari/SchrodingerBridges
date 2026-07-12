import tempfile
import unittest
from pathlib import Path
import math

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_teacher_cache import TOP_K, build_fake_cache_rows, validate_cache
from scripts.d3_common import read_jsonl


class Direction3TeacherCacheSchemaTests(unittest.TestCase):
    def test_fake_cache_has_required_trajectory_diversity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_protocol_docs(out)
            build_splits(out, seed=3)
            train_manifest = read_jsonl(out / "controller_train_100.jsonl")
            rows = build_fake_cache_rows(train_manifest, "controller_train_100")
            summary = validate_cache(rows)
            self.assertGreaterEqual(len(summary["step_histogram"]), 3)
            self.assertIn("2", {str(k) for k in summary["active_mask_count_histogram"]})
            self.assertIn("1", {str(k) for k in summary["target_len_histogram"]})
            self.assertIn("2", {str(k) for k in summary["target_len_histogram"]})
            self.assertIn("rewrite", summary["prompt_type_histogram"])
            self.assertIn("same_subject_different_relation", summary["prompt_type_histogram"])
            self.assertIn("near_locality", summary["prompt_type_histogram"])

            first = rows[0]
            self.assertEqual(len(first["top_k_candidate_token_ids"]), TOP_K)
            self.assertEqual(len(first["top_k_candidate_ids"]), TOP_K)
            self.assertEqual(len(first["base_logits_top_k"]), TOP_K)
            self.assertEqual(len(first["base_logits"]), TOP_K)
            self.assertEqual(len(first["base_probs"]), TOP_K)
            self.assertEqual(len(first["raw_bridge_scores_top_k"]), TOP_K)
            self.assertEqual(len(first["raw_bridge_scores"]), TOP_K)
            self.assertIn("fake_state", first)
            self.assertIn("selected_mask_position", first)
            self.assertIn("chosen_token", first)
            for key in [
                "base_logits",
                "base_probs",
                "raw_bridge_scores",
                "myopic_scores",
                "no_rollout_scores",
                "mc_rollout_rewards",
            ]:
                self.assertTrue(all(math.isfinite(float(x)) for x in first[key]))
            self.assertTrue(first["fake_model"])


if __name__ == "__main__":
    unittest.main()
