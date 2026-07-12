import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Direction3FakeModeNoLladaLoadTests(unittest.TestCase):
    def run_cmd(self, args):
        return subprocess.run(
            [sys.executable, *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_fake_mode_scripts_write_no_llada_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            cache = root / "fake_teacher_cache_v1"
            train = root / "fake_controller_train_v1"
            replay = root / "fake_offline_replay_v1"
            readiness = root / "stage1a_teacher_cache_readiness_v1"

            self.run_cmd(["scripts/build_d3_controller_splits.py", "--output_dir", str(root)])
            self.run_cmd(["scripts/build_d3_gate_data.py", "--input_dir", str(root), "--output_dir", str(root)])
            self.run_cmd(
                [
                    "scripts/build_d3_teacher_cache.py",
                    "--fake_model",
                    "1",
                    "--input_dir",
                    str(root),
                    "--output_dir",
                    str(cache),
                ]
            )
            self.run_cmd(
                [
                    "scripts/train_d3_bridge_controller.py",
                    "--fake_model",
                    "1",
                    "--teacher_cache_dir",
                    str(cache),
                    "--output_dir",
                    str(train),
                ]
            )
            self.run_cmd(
                [
                    "scripts/eval_d3_offline_replay.py",
                    "--fake_model",
                    "1",
                    "--teacher_cache_dir",
                    str(cache),
                    "--controller_dir",
                    str(train),
                    "--output_dir",
                    str(replay),
                ]
            )
            self.run_cmd(
                [
                    "scripts/d3_stage1a_readiness.py",
                    "--input_dir",
                    str(root),
                    "--output_dir",
                    str(readiness),
                ]
            )

            for summary_path in [
                root / "report_summary.json",
                cache / "report_summary.json",
                train / "report_summary.json",
                replay / "report_summary.json",
                readiness / "report_summary.json",
            ]:
                payload = json.loads(summary_path.read_text())
                self.assertFalse(payload["llada_loaded"])
                self.assertFalse(payload.get("analysis_500_used", False))
                self.assertFalse(payload.get("final_test_used", False))

            self.assertTrue(json.loads((cache / "report_summary.json").read_text())["fake_model"])
            self.assertTrue(json.loads((train / "report_summary.json").read_text())["fake_model"])
            self.assertTrue(json.loads((replay / "report_summary.json").read_text())["fake_model"])
            readiness_payload = json.loads((readiness / "report_summary.json").read_text())
            self.assertTrue(readiness_payload["pipeline_readiness_pass"])
            self.assertFalse(readiness_payload["scientific_acceptance_pass"])
            self.assertFalse(readiness_payload["runpod_allowed_next"])
            template = readiness / "runpod_teacher_cache_smoke_command.sh"
            self.assertTrue(template.exists())
            self.assertIn("TEMPLATE ONLY. DO NOT EXECUTE AUTOMATICALLY.", template.read_text())
            self.assertEqual(stat.S_IMODE(template.stat().st_mode), 0o644)


if __name__ == "__main__":
    unittest.main()
