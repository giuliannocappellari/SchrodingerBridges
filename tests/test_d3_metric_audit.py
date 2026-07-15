import math
import tempfile
import unittest
from pathlib import Path

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_teacher_cache import build_fake_cache_rows
from scripts.d3_common import read_jsonl, write_json, write_jsonl
from scripts.d3_metric_audit import group_metrics, target_indicator_ablation_rows
from scripts.train_d3_bridge_controller import VALUE_FEATURE_NAMES, annotate_gate_context


class Direction3MetricAuditTests(unittest.TestCase):
    def test_groupwise_metrics_are_finite_for_topk_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            root.mkdir(parents=True)
            write_protocol_docs(root)
            build_splits(root, seed=11)
            manifest = read_jsonl(root / "controller_train_100.jsonl")[:4]
            rows = annotate_gate_context(build_fake_cache_rows(manifest, "controller_train_100"))
            model = {
                "controller_type": "value",
                "value_weights": [0.1 for _ in VALUE_FEATURE_NAMES],
            }
            metric = group_metrics(rows[0], model, "value", tau=8.0)
            for key in [
                "groupwise_spearman",
                "kendall_tau",
                "pairwise_ranking_accuracy",
                "ndcg_at_8",
                "teacher_student_kl",
                "teacher_student_js",
                "top1_agreement",
                "top3_overlap",
            ]:
                self.assertIn(key, metric)
                self.assertTrue(math.isfinite(float(metric[key])))

    def test_target_indicator_ablation_reports_prompt_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            root.mkdir(parents=True)
            write_protocol_docs(root)
            build_splits(root, seed=12)
            manifest = read_jsonl(root / "controller_train_100.jsonl")[:4]
            rows = annotate_gate_context(build_fake_cache_rows(manifest, "controller_train_100"))
            model = {
                "controller_type": "value",
                "value_weights": [0.1 for _ in VALUE_FEATURE_NAMES],
            }
            out = target_indicator_ablation_rows(rows, model, "value")
            prompt_types = {row["prompt_type"] for row in out}
            self.assertIn("rewrite", prompt_types)
            self.assertIn("same_subject_different_relation", prompt_types)
            self.assertTrue(all("normal_minus_ablated" in row for row in out))


if __name__ == "__main__":
    unittest.main()
