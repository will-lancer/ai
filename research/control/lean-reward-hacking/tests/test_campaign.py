from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from lean_reward_hacking.campaign import (
    _branch_compact_row,
    _branch_status,
    _distance,
    _metric_scales,
    _pair_count_row,
    _read_jsonl,
    _source_records,
    _tiny_validation_run_id,
    _toy_source_config_identity,
    analyze_path,
    export_compact,
)
from lean_reward_hacking.evaluation import EvaluationMetrics
from lean_reward_hacking.episodes import make_paired_evaluation
from lean_reward_hacking.schemas import validate_compact_bundle
from lean_reward_hacking.types import Mode


class CampaignTests(unittest.TestCase):
    def test_tiny_validation_id_is_config_specific(self) -> None:
        first = "a" * 64
        second = "b" * 64
        self.assertIn(first[:24], _tiny_validation_run_id(first))
        self.assertNotEqual(_tiny_validation_run_id(first), _tiny_validation_run_id(second))

    def test_pair_counts_drop_non_opportunity_pairs_without_torch(self) -> None:
        pairs = make_paired_evaluation(4, 2, 91, opportunity_probability=0.0)
        row = _pair_count_row(object(), pairs, "run", 4, "eval")
        self.assertEqual(row["n_pairs"], 0)
        self.assertEqual(sum(row[name] for name in ("n11", "n10", "n01", "n00")), 0)

    def test_source_records_bind_current_config_and_archive(self) -> None:
        source = "source-current"
        config = _toy_source_config_identity()
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "runs" / "toy_fixed" / "session"
            (session / "exports").mkdir(parents=True)
            (session / "markers").mkdir()
            (session / "spec").mkdir()
            (session / "markers" / "completed.json").write_text(
                json.dumps({"state": "complete", "config_sha256": config, "source_archive_sha256": source}),
                encoding="utf-8",
            )
            (session / "spec" / "config.json").write_text(
                json.dumps({"config_sha256": config}), encoding="utf-8"
            )
            with (session / "exports" / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("run_id", "status", "experiment", "architecture", "config_sha256"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "run_id": "good",
                        "status": "complete",
                        "experiment": "toy_fixed",
                        "architecture": "toy",
                        "config_sha256": config,
                    }
                )
                writer.writerow(
                    {
                        "run_id": "wrong-config",
                        "status": "complete",
                        "experiment": "toy_fixed",
                        "architecture": "toy",
                        "config_sha256": "wrong",
                    }
                )
            with (session / "exports" / "checkpoint_metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=("run_id", "step", "label", "c_on", "c_off"))
                writer.writeheader()
                writer.writerow({"run_id": "good", "step": 20, "label": "strategic", "c_on": 0.99, "c_off": 0.01})
                writer.writerow({"run_id": "wrong-config", "step": 20, "label": "strategic", "c_on": 0.99, "c_off": 0.01})
            rows = _source_records(Path(temporary), 20, source_identity=source, toy_config_identity=config)
            self.assertEqual([row["run_id"] for row in rows], ["good"])

    def test_branch_distances_and_support_are_recorded(self) -> None:
        source = {"c_on": 0.99, "c_off": 0.01, "goal": -1.0, "gate": 0.0, "label": Mode.STRATEGIC.value}
        opposite = {"c_on": 0.99, "c_off": 0.95, "goal": 1.0, "gate": 0.0}
        metric = EvaluationMetrics(c_on=0.99, c_off=0.50, goal=0.0, gate=0.0, checkpoint_step=5)
        row = _branch_compact_row(
            metric,
            branch_id="branch",
            source_run_id="source",
            source_step=4,
            source_metric=source,
            opposite_metric=opposite,
            kind="gaussian_noise",
            strength=0.1,
            branch_seed=7,
            horizon=1,
            eval_hash="eval",
            n_pairs=2,
        )
        self.assertIsInstance(row["d_opposite"], float)
        frozen = (dict(row, step_since_branch=1, d_source=1.1135), dict(row, step_since_branch=2, d_source=1.1135))
        status, support = _branch_status(
            (row, dict(row, step_since_branch=2, d_source=0.01)),
            minimum_displacement=0.05,
            minimum_recovery_fraction=0.5,
            frozen_trajectory=frozen,
        )
        self.assertEqual(status, "recovered")
        self.assertTrue(support["frozen_control_available"])
        self.assertIn("dynamic_pull_final", support)
        self.assertEqual(support["frozen_control_source_distance_final"], 1.1135)

    def test_scaled_distance_uses_only_registered_behavior_fields(self) -> None:
        records = [
            {"c_on": 0.9, "c_off": 0.1, "goal": -100.0, "gate": 100.0},
            {"c_on": 1.0, "c_off": 0.2, "goal": 100.0, "gate": -100.0},
        ]
        scales = _metric_scales(records)
        self.assertEqual(set(scales), {"c_on", "c_off"})
        self.assertAlmostEqual(_distance(records[0], records[1], scales=scales), 2.0 * 2.0**0.5)

    def test_malformed_jsonl_tail_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text('{"step": 1}\n{"step":', encoding="utf-8")
            self.assertEqual(_read_jsonl(path), [{"step": 1}])
            quarantine = Path(str(path) + ".quarantine.jsonl")
            self.assertTrue(quarantine.is_file())
            self.assertEqual(path.read_text(encoding="utf-8"), '{"step": 1}\n')


if __name__ == "__main__":
    unittest.main()
