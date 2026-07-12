import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.audit_d3_teacher_cache import build_audit
from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_teacher_cache import build_fake_cache_rows
from scripts.d3_common import read_jsonl, write_json, write_jsonl


class Direction3TeacherCacheAuditTests(unittest.TestCase):
    def make_cache(self, root: Path) -> Path:
        root.mkdir(parents=True)
        write_protocol_docs(root)
        build_splits(root, seed=7)
        train_manifest = read_jsonl(root / "controller_train_100.jsonl")[:10]
        val_manifest = read_jsonl(root / "controller_val_50.jsonl")[:5]
        cache = root / "teacher_cache_smoke_v1"
        train_rows = build_fake_cache_rows(train_manifest, "controller_train_10")
        val_rows = build_fake_cache_rows(val_manifest, "controller_val_5")
        for row in train_rows + val_rows:
            row["fake_model"] = False
        write_jsonl(cache / "teacher_states_train.jsonl", train_rows)
        write_jsonl(cache / "teacher_states_val.jsonl", val_rows)
        summary = {
            "protocol_version": "counterfact_direction3_controller_v1",
            "fake_model": False,
            "llada_loaded": True,
            "analysis_500_used": False,
            "final_test_used": False,
        }
        write_json(cache / "report_summary.json", summary)
        return cache

    def test_teacher_cache_audit_passes_on_trajectory_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            cache = self.make_cache(root)
            result = build_audit(cache, root / "audit", top_k=8)
            report = result["report"]
            self.assertTrue(report["audit_pass"])
            self.assertFalse(report["llada_loaded"])
            self.assertTrue(report["teacher_generation_llada_loaded"])
            self.assertGreaterEqual(report["num_train_edits"], 10)
            self.assertGreaterEqual(report["num_val_edits"], 5)
            for row in result["variances"]:
                self.assertTrue(row["variance_pass"])
                self.assertTrue(math.isfinite(float(row["global_variance"])))

    def test_teacher_cache_audit_fails_on_bad_topk_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            cache = self.make_cache(root)
            rows = read_jsonl(cache / "teacher_states_train.jsonl")
            rows[0]["top_k_candidate_token_ids"] = rows[0]["top_k_candidate_token_ids"][:7]
            write_jsonl(cache / "teacher_states_train.jsonl", rows)
            result = build_audit(cache, root / "audit", top_k=8)
            self.assertFalse(result["report"]["audit_pass"])
            self.assertFalse(result["report"]["acceptance_checks"]["top_k_and_score_schema_pass"])


if __name__ == "__main__":
    unittest.main()
