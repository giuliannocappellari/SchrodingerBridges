import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.d3_common import D3_PROTOCOL_VERSION, collect_locked_manifest_exclusions, read_jsonl


class Direction3ControllerSplitTests(unittest.TestCase):
    def test_controller_splits_are_deterministic_and_exclude_locked_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_protocol_docs(out)
            plan0 = build_splits(out, seed=3)
            rows0 = {
                name: [row["case_id"] for row in read_jsonl(out / f"{name}.jsonl")]
                for name in ["controller_train_100", "controller_val_50", "dev_smoke_50"]
            }
            plan1 = build_splits(out, seed=3)
            rows1 = {
                name: [row["case_id"] for row in read_jsonl(out / f"{name}.jsonl")]
                for name in ["controller_train_100", "controller_val_50", "dev_smoke_50"]
            }
            self.assertEqual(rows0, rows1)
            self.assertEqual(plan0["protocol_version"], D3_PROTOCOL_VERSION)
            self.assertFalse(plan0["locked_prompts_labels_outputs_or_metrics_used"])
            self.assertEqual(len(rows0["controller_train_100"]), 100)
            self.assertEqual(len(rows0["controller_val_50"]), 50)
            self.assertEqual(len(rows0["dev_smoke_50"]), 50)

            all_ids = set().union(*[set(ids) for ids in rows0.values()])
            self.assertEqual(len(all_ids), 200)
            locked = set(collect_locked_manifest_exclusions()["excluded_case_ids"])
            self.assertFalse(all_ids & locked)
            for split_role in ["controller_train_100", "controller_val_50", "dev_smoke_50"]:
                rows = read_jsonl(out / f"{split_role}.jsonl")
                bins = {str(row["target_length_bin"]) for row in rows}
                self.assertIn("1", bins)
                self.assertIn("2", bins)

            train10 = read_jsonl(out / "controller_train_10.jsonl")
            val5 = read_jsonl(out / "controller_val_5.jsonl")
            train100 = read_jsonl(out / "controller_train_100.jsonl")
            val50 = read_jsonl(out / "controller_val_50.jsonl")
            self.assertEqual(len(train10), 10)
            self.assertEqual(len(val5), 5)
            self.assertTrue({row["case_id"] for row in train10}.issubset({row["case_id"] for row in train100}))
            self.assertTrue({row["case_id"] for row in val5}.issubset({row["case_id"] for row in val50}))
            self.assertTrue({"1", "2"}.issubset({str(row["target_length_bin"]) for row in train10}))
            self.assertTrue({"1", "2"}.issubset({str(row["target_length_bin"]) for row in val5}))

    def test_protocol_artifacts_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_protocol_docs(out)
            self.assertTrue((out / "direction3_controller_protocol.md").exists())
            teacher_schema = json.loads((out / "teacher_cache_schema.json").read_text())
            gate_schema = json.loads((out / "gate_dataset_schema.json").read_text())
            self.assertEqual(teacher_schema["protocol_version"], D3_PROTOCOL_VERSION)
            self.assertIn("top_k_candidate_token_ids", teacher_schema["required_fields"])
            self.assertIn("top_k_candidate_ids", teacher_schema["required_fields"])
            self.assertIn("same_subject_different_relation", gate_schema["negative_prompt_types"])
            self.assertIn("far_locality", gate_schema["negative_prompt_types"])


if __name__ == "__main__":
    unittest.main()
