import math
import tempfile
import unittest
from pathlib import Path

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_teacher_cache import build_fake_cache_rows
from scripts.d3_common import read_jsonl
from scripts.train_d3_bridge_controller import loss_for_rows, train_fake


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
                "locality_kl_proxy_loss",
                "l2_correction_loss",
                "total_loss",
            ]:
                self.assertIn(key, metrics)
                self.assertTrue(math.isfinite(metrics[key]))


if __name__ == "__main__":
    unittest.main()

