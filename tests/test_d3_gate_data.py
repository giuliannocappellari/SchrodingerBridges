import tempfile
import unittest
from pathlib import Path

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_gate_data import build_gate_rows, validate_gate_rows
from scripts.d3_common import read_jsonl, write_jsonl


class Direction3GateDataTests(unittest.TestCase):
    def test_gate_rows_have_labels_and_same_subject_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_protocol_docs(out)
            build_splits(out, seed=3)
            rows_by_split = {}
            for split_role in ["controller_train_100", "controller_val_50", "dev_smoke_50"]:
                manifest_rows = read_jsonl(out / f"{split_role}.jsonl")
                gate_rows = build_gate_rows(manifest_rows, split_role)
                rows_by_split[split_role] = gate_rows
                write_jsonl(out / f"{split_role}.gate.jsonl", gate_rows)

            summary = validate_gate_rows(rows_by_split)
            self.assertTrue(summary["same_subject_negatives_present"])
            self.assertTrue(summary["locality_negatives_present"])
            self.assertTrue(summary["no_train_eval_prompt_overlap"])
            self.assertTrue(summary["synthetic_fallback_marked_explicitly"])
            for split_role, gate_rows in rows_by_split.items():
                labels = {row["label"] for row in gate_rows}
                prompt_types = {row["prompt_type"] for row in gate_rows}
                negative_types = {row["negative_type"] for row in gate_rows if row["negative_type"]}
                self.assertEqual(labels, {0, 1})
                self.assertIn("rewrite", prompt_types)
                self.assertIn("declarative_paraphrase", prompt_types)
                self.assertIn("same_subject_different_relation", prompt_types)
                self.assertIn("near_locality", prompt_types)
                self.assertIn("far_locality", prompt_types)
                self.assertIn("attribute", prompt_types)
                self.assertIn("unrelated", prompt_types)
                self.assertTrue(
                    {
                        "same_subject_different_relation",
                        "near_locality",
                        "far_locality",
                        "generation",
                        "attribute",
                        "unrelated",
                    }.issubset(negative_types)
                )
                self.assertTrue(all(row["synthetic_from_metadata"] for row in gate_rows))
                self.assertTrue(all("target_new" in row and "target_true" in row for row in gate_rows))
                self.assertTrue(all(row["source_manifest"] for row in gate_rows))
                self.assertTrue(
                    all(row["category_unavailable_reason"] for row in gate_rows if row["synthetic_from_metadata"])
                )


if __name__ == "__main__":
    unittest.main()
