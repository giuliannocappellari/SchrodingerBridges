import tempfile
import unittest
from pathlib import Path

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.d3_common import collect_locked_manifest_exclusions, ensure_no_locked_artifact_use, read_jsonl


class Direction3NoAnalysisFinalTests(unittest.TestCase):
    def test_locked_paths_are_rejected_for_artifact_use(self):
        with self.assertRaises(AssertionError):
            ensure_no_locked_artifact_use("runs/counterfact_direction1_v1/protocol/analysis_500.jsonl")
        with self.assertRaises(AssertionError):
            ensure_no_locked_artifact_use("runs/counterfact_direction1_v1/protocol/final_test_500.jsonl")

    def test_split_builder_uses_locked_manifests_only_for_exclusion(self):
        exclusions = collect_locked_manifest_exclusions()
        self.assertTrue(exclusions["excluded_case_ids"])
        for info in exclusions["manifests"].values():
            self.assertTrue(info["id_only_exclusion_use"])
            self.assertFalse(info["locked_prompts_or_labels_used"])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_protocol_docs(out)
            plan = build_splits(out, seed=3)
            self.assertFalse(plan["locked_prompts_labels_outputs_or_metrics_used"])
            selected = []
            for name in ["controller_train_100", "controller_val_50", "dev_smoke_50"]:
                selected.extend(row["case_id"] for row in read_jsonl(out / f"{name}.jsonl"))
            self.assertFalse(set(selected) & set(exclusions["excluded_case_ids"]))


if __name__ == "__main__":
    unittest.main()

