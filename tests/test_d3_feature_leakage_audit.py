import tempfile
import unittest
from pathlib import Path

from scripts.d3_common import write_json, write_jsonl
from scripts.d3_feature_leakage_audit import build_audit, feature_rows


class Direction3FeatureLeakageAuditTests(unittest.TestCase):
    def test_feature_rows_flag_teacher_score_inputs(self):
        payload = {
            "controllers": {
                "value_gate": {
                    "value_feature_names": ["bias", "base_logit", "myopic_score"],
                    "gate_feature_names": ["subject_match", "target_no_rollout_margin"],
                }
            }
        }
        rows = feature_rows(payload)
        leaked = {row["feature_name"]: row for row in rows if not row["eligible_for_actual_decode"]}
        self.assertIn("myopic_score", leaked)
        self.assertEqual(leaked["myopic_score"]["leakage_reason"], "teacher_score_input")
        self.assertIn("target_no_rollout_margin", leaked)
        self.assertEqual(leaked["target_no_rollout_margin"]["leakage_reason"], "teacher_score_input")

    def test_build_audit_requires_disjoint_ids_and_clean_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            controller = root / "controller"
            replay = root / "replay"
            out = root / "audit"
            cache.mkdir()
            controller.mkdir()
            replay.mkdir()
            base_summary = {
                "protocol_version": "counterfact_direction3_controller_v1",
                "analysis_500_used": False,
                "final_test_used": False,
            }
            write_json(cache / "report_summary.json", base_summary)
            write_json(controller / "report_summary.json", base_summary)
            write_json(replay / "report_summary.json", base_summary)
            write_jsonl(cache / "teacher_states_train.jsonl", [{"edit_id": "train_1"}])
            write_jsonl(cache / "teacher_states_val.jsonl", [{"edit_id": "val_1"}])
            write_json(
                controller / "controller_weights.json",
                {
                    "controllers": {
                        "safe": {
                            "value_feature_names": ["bias", "base_logit", "base_prob"],
                            "gate_feature_names": ["subject_match", "relation_token_jaccard_to_rewrite"],
                        }
                    }
                },
            )
            clean = build_audit(cache, controller, replay, out)["payload"]
            self.assertTrue(clean["audit_pass"])

            write_json(
                controller / "controller_weights.json",
                {
                    "controllers": {
                        "leaky": {
                            "value_feature_names": ["bias", "myopic_score"],
                            "gate_feature_names": ["target_myopic_margin"],
                        }
                    }
                },
            )
            leaky = build_audit(cache, controller, replay, out)["payload"]
            self.assertFalse(leaky["audit_pass"])
            self.assertFalse(leaky["acceptance_checks"]["no_teacher_score_fields_used_as_input"])


if __name__ == "__main__":
    unittest.main()
