import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.finalize_d3_terminal_campaign import finalize_terminal_campaign


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class Direction3TerminalCampaignTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        stop = root / "runs/counterfact_direction3_controller_v1/direction3_autonomous_stop_checkpoint_v1"
        campaign = root / "runs/counterfact_direction3_controller_v1/autonomous_campaign_v1"
        stop.mkdir(parents=True)
        campaign.mkdir(parents=True)
        write_json(
            stop / "report_summary.json",
            {
                "protocol_version": "counterfact_direction3_controller_v1",
                "campaign_status": "formal_negative_completion",
                "negative_completion": True,
                "positive_completion": False,
                "bounded_scientific_rescue_used": True,
                "rescue_attempts_used": {"stage_1b4_value": 1},
                "do_not_run_stage_2a": True,
                "stage_2a_run": False,
                "actual_decode_run": False,
                "analysis_500_used": False,
                "do_not_run_analysis_500": True,
                "final_test_used": False,
                "do_not_run_final_test": True,
                "hard_criteria_failed_after_rescue": [
                    "value_top3_pass",
                    "representation_beats_target_indicator_pass",
                    "state_shuffle_hurts_pass",
                ],
                "created_at_utc": "2026-07-14T01:00:00+00:00",
                "artifacts": {"rescue": "runs/missing_rescue"},
            },
        )
        (stop / "direction3_autonomous_stop_checkpoint.md").write_text(
            "formal_negative_completion\nStage 2A actual D3 decoding was not run", encoding="utf-8"
        )
        (stop / "negative_result_report.md").write_text(
            "No analysis or final split was used. Stage 2A actual decoding would be scientifically unjustified",
            encoding="utf-8",
        )
        (stop / "next_direction_recommendation.md").write_text("No automatic switch.", encoding="utf-8")
        evidence = [
            ["1B.4A feature audit", "audit_pass", "True", "True"],
            ["1B.4/1B.5 rescue1", "scientific_acceptance_pass", "False", "False"],
            ["1B.4/1B.5 rescue1", "d3_value_repr_teacher_top3_overlap", "0.52", "False"],
            ["1B.4/1B.5 rescue1", "d3_value_repr_macro_spearman", "0.52", "True"],
            ["1B.4/1B.5 rescue1", "target_indicator_only_macro_spearman", "0.95", "False"],
            ["1B.4/1B.5 rescue1", "state_shuffle_hurts", "False", "False"],
            ["1B.4/1B.5 rescue1", "feature_leakage_audit_pass", "True", "True"],
        ]
        with (stop / "direction3_evidence_table.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stage", "artifact", "metric", "value", "threshold_or_expected", "pass", "notes"])
            for stage, metric, value, passed in evidence:
                writer.writerow([stage, "artifact", metric, value, "expected", passed, "note"])
        write_json(
            campaign / "campaign_state.json",
            {
                "campaign_start_utc": "2026-07-14T00:00:00+00:00",
                "campaign_status": "formal_negative_completion",
            },
        )
        write_json(campaign / "budget_state.json", {"estimated_spend_usd": 0.0})
        (campaign / "stage_history.csv").write_text(
            "timestamp_utc,stage,event,status,notes\n", encoding="utf-8"
        )
        (campaign / "autonomous_log.md").write_text("# Log\n", encoding="utf-8")
        write_json(root / "ACTIVE_RESEARCH_CAMPAIGN.json", {"campaign_status": "active", "direction3_state": {}})
        return campaign, stop

    def test_finalizer_validates_summary_and_closes_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign, stop = self.make_fixture(root)
            result = finalize_terminal_campaign(
                root=root,
                active_campaign_path=Path("ACTIVE_RESEARCH_CAMPAIGN.json"),
                campaign_dir=campaign.relative_to(root),
                stop_dir=stop.relative_to(root),
                budget_usd=15.0,
                hourly_rate_usd=0.45,
                reserve_usd=5.0,
                current_pod_running_seconds=60.0,
                runpod_pod_id="pod-id",
                runpod_ssh_host="host",
                runpod_ssh_port="22",
            )
            self.assertTrue(result["validation"]["package_validation_pass"])
            self.assertFalse(result["validation"]["raw_source_metric_rederivation_available"])
            self.assertEqual(result["campaign_state"]["status"], "complete_negative")
            self.assertEqual(result["campaign_state"]["runpod_status"], "EXITED")
            self.assertFalse(result["budget_state"]["runpod_allowed_next"])
            self.assertEqual(result["active_campaign"]["campaign_status"], "completed_negative")
            self.assertTrue((stop / "terminal_package_validation.json").exists())
            self.assertTrue((stop / "artifact_integrity_manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
