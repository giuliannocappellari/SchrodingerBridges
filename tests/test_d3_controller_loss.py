import math
import subprocess
import sys
import tempfile
import unittest
import csv
from pathlib import Path

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_teacher_cache import build_fake_cache_rows
from scripts.d3_common import read_jsonl
from scripts.train_d3_bridge_controller import loss_for_rows, train_controller, train_fake


class Direction3ControllerLossTests(unittest.TestCase):
    def test_fake_controller_loss_is_finite_and_training_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_protocol_docs(out)
            build_splits(out, seed=3)
            manifest = read_jsonl(out / "controller_train_100.jsonl")[:8]
            rows = build_fake_cache_rows(manifest, "controller_train_100")
            weights, history = train_fake(rows, epochs=2, lr=0.05)
            self.assertEqual(len(history), 2)
            self.assertTrue(all(math.isfinite(x) for x in weights))
            metrics = loss_for_rows(rows, weights)
            for key in [
                "bridge_distillation_loss",
                "ranking_loss",
                "bridge_ranking_loss",
                "locality_kl_proxy_loss",
                "gate_loss",
                "l2_correction_loss",
                "total_loss",
            ]:
                self.assertIn(key, metrics)
                self.assertTrue(math.isfinite(metrics[key]))

    def test_real_cache_mode_trains_without_llada_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            cache = root / "teacher_cache_smoke_v1"
            train = root / "real_cache_train_smoke_v1"
            replay = root / "real_cache_offline_replay_smoke_v1"
            root.mkdir(parents=True)
            write_protocol_docs(root)
            build_splits(root, seed=4)
            train_manifest = read_jsonl(root / "controller_train_100.jsonl")[:10]
            val_manifest = read_jsonl(root / "controller_val_50.jsonl")[:5]
            cache.mkdir(parents=True)
            train_rows = build_fake_cache_rows(train_manifest, "controller_train_10")
            val_rows = build_fake_cache_rows(val_manifest, "controller_val_5")
            for row in train_rows + val_rows:
                row["fake_model"] = False
            from scripts.d3_common import write_json, write_jsonl

            write_jsonl(cache / "teacher_states_train.jsonl", train_rows)
            write_jsonl(cache / "teacher_states_val.jsonl", val_rows)
            write_json(
                cache / "report_summary.json",
                {
                    "protocol_version": "counterfact_direction3_controller_v1",
                    "fake_model": False,
                    "llada_loaded": True,
                    "analysis_500_used": False,
                    "final_test_used": False,
                },
            )
            repo = Path(__file__).resolve().parents[1]
            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_d3_bridge_controller.py",
                    "--fake_model",
                    "0",
                    "--teacher_cache_dir",
                    str(cache),
                    "--output_dir",
                    str(train),
                ],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/eval_d3_offline_replay.py",
                    "--fake_model",
                    "0",
                    "--teacher_cache_dir",
                    str(cache),
                    "--controller_dir",
                    str(train),
                    "--output_dir",
                    str(replay),
                ],
                cwd=repo,
                check=True,
            )
            import json

            train_summary = json.loads((train / "report_summary.json").read_text())
            replay_summary = json.loads((replay / "report_summary.json").read_text())
            self.assertFalse(train_summary["fake_model"])
            self.assertFalse(replay_summary["fake_model"])
            self.assertFalse(train_summary["llada_loaded"])
            self.assertFalse(replay_summary["llada_loaded"])
            self.assertIn("loss_decreased", train_summary)
            for artifact_name in [
                "offline_replay_metrics.csv",
                "gate_threshold_sweep.csv",
                "controller_candidate_agreement.csv",
                "negative_guidance_diagnostics.csv",
                "target_token_ranking.csv",
            ]:
                self.assertTrue((replay / artifact_name).exists(), artifact_name)
            with (replay / "offline_replay_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            metric_row = rows[0]
            for key in [
                "bridge_score_spearman",
                "target_token_top3_improvement_over_base",
                "same_subject_gate_auc",
                "locality_negative_average_guidance",
            ]:
                self.assertIn(key, metric_row)
                self.assertTrue(math.isfinite(float(metric_row[key])))


if __name__ == "__main__":
    unittest.main()
