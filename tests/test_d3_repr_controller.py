import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.build_d3_controller_splits import build_splits, write_protocol_docs
from scripts.build_d3_teacher_cache import build_fake_cache_rows
from scripts.d3_common import read_jsonl, write_json, write_jsonl


def with_prompt_text(rows):
    out = []
    for row in rows:
        item = dict(row)
        prompt_type = str(item.get("prompt_type"))
        subject = item.get("subject") or "Subject"
        if prompt_type == "rewrite":
            prompt = f"{subject}'s edited relation is"
        elif prompt_type == "declarative_paraphrase":
            prompt = f"The edited relation for {subject} is"
        elif prompt_type == "near_locality":
            prompt = f"{subject}'s nearby attribute is"
        elif prompt_type == "far_locality":
            prompt = f"A distant unrelated fact about {subject} is"
        elif prompt_type == "generation":
            prompt = f"Write one sentence about {subject}."
        else:
            prompt = f"{subject}'s different relation is"
        item["prompt_text"] = prompt
        out.append(item)
    return out


class Direction3RepresentationControllerTests(unittest.TestCase):
    def test_audit_and_repr_training_fake_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d3"
            cache = root / "teacher_cache_train100_val50_v1"
            features = root / "deployable_feature_cache_train100_val50_v1"
            audit = root / "deployable_feature_cache_train100_val50_v1_local_audit"
            train = root / "offline_train_repr_value_gate_train100_val50_v3"
            replay = root / "offline_replay_repr_train100_val50_v3"
            leakage = root / "stage1b_feature_leakage_audit_v3"
            shortcut = root / "representation_shortcut_audit_v3"
            root.mkdir(parents=True)
            write_protocol_docs(root)
            build_splits(root, seed=17)
            train_manifest = read_jsonl(root / "controller_train_100.jsonl")
            val_manifest = read_jsonl(root / "controller_val_50.jsonl")
            train_rows = with_prompt_text(build_fake_cache_rows(train_manifest, "controller_train_100"))
            val_rows = with_prompt_text(build_fake_cache_rows(val_manifest, "controller_val_50"))
            expected_groups = len(train_rows) + len(val_rows)
            write_jsonl(cache / "teacher_states_train.jsonl", train_rows)
            write_jsonl(cache / "teacher_states_val.jsonl", val_rows)
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
                    str(features),
                    "--feature_dim",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_d3_deployable_feature_cache.py",
                    "--feature_cache_dir",
                    str(features),
                    "--teacher_cache_dir",
                    str(cache),
                    "--output_dir",
                    str(audit),
                    "--expected_candidate_groups",
                    str(expected_groups),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            audit_summary = json.loads((audit / "report_summary.json").read_text())
            self.assertTrue(audit_summary["audit_pass"])
            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_d3_repr_controller.py",
                    "--feature_cache_dir",
                    str(features),
                    "--teacher_cache_dir",
                    str(cache),
                    "--local_audit_dir",
                    str(audit),
                    "--train_dir",
                    str(train),
                    "--replay_dir",
                    str(replay),
                    "--leakage_dir",
                    str(leakage),
                    "--shortcut_dir",
                    str(shortcut),
                    "--epochs",
                    "1",
                    "--bootstrap_trials",
                    "5",
                    "--batch_size",
                    "128",
                    "--expected_candidate_groups",
                    str(expected_groups),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
            )
            for path in [
                train / "report_summary.json",
                train / "train_metrics.csv",
                train / "validation_metrics.csv",
                replay / "report_summary.json",
                replay / "scientific_status.json",
                replay / "groupwise_ranking_metrics.csv",
                replay / "negative_guidance_diagnostics.csv",
                leakage / "report_summary.json",
                shortcut / "report_summary.json",
                shortcut / "representation_ablation.csv",
            ]:
                self.assertTrue(path.exists(), path)
            replay_summary = json.loads((replay / "report_summary.json").read_text())
            self.assertFalse(replay_summary["analysis_500_used"])
            self.assertFalse(replay_summary["final_test_used"])
            self.assertFalse(replay_summary["llada_loaded"])


if __name__ == "__main__":
    unittest.main()
