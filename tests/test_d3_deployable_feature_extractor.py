import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import load_file

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_teacher_cache import build_fake_cache_rows
from scripts.d3_common import read_jsonl, write_json, write_jsonl


class Direction3DeployableFeatureExtractorTests(unittest.TestCase):
    def test_fake_extractor_writes_safetensors_and_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            cache = root / "teacher_cache_train100_val50_v1"
            out = root / "deployable_feature_cache_train100_val50_v1"
            root.mkdir(parents=True)
            write_protocol_docs(root)
            build_splits(root, seed=13)
            train_manifest = read_jsonl(root / "controller_train_100.jsonl")
            val_manifest = read_jsonl(root / "controller_val_50.jsonl")
            write_jsonl(cache / "teacher_states_train.jsonl", build_fake_cache_rows(train_manifest, "controller_train_100"))
            write_jsonl(cache / "teacher_states_val.jsonl", build_fake_cache_rows(val_manifest, "controller_val_50"))
            write_json(
                cache / "report_summary.json",
                {
                    "protocol_version": "counterfact_direction3_controller_v1",
                    "fake_model": True,
                    "llada_loaded": False,
                    "analysis_500_used": False,
                    "final_test_used": False,
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/extract_d3_deployable_features.py",
                    "--fake_model",
                    "1",
                    "--teacher_cache_dir",
                    str(cache),
                    "--output_dir",
                    str(out),
                    "--feature_dim",
                    "8",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            summary = json.loads((out / "report_summary.json").read_text())
            self.assertTrue(summary["fake_model"])
            self.assertFalse(summary["llada_loaded"])
            self.assertTrue(summary["feature_integrity_pass"])
            for name in [
                "state_features.safetensors",
                "candidate_features.safetensors",
                "edit_features.safetensors",
                "gate_features.safetensors",
            ]:
                tensors = load_file(str(out / name))
                self.assertTrue(tensors, name)
            template = out / "runpod_deployable_feature_extraction_command.sh"
            self.assertTrue(template.exists())
            self.assertIn("TEMPLATE ONLY. DO NOT EXECUTE AUTOMATICALLY.", template.read_text())
            self.assertEqual(stat.S_IMODE(template.stat().st_mode), 0o644)


if __name__ == "__main__":
    unittest.main()
