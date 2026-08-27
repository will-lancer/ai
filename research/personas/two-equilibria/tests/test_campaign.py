from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.campaign import (
    _branch_compact_row,
    _branch_status,
    _annotate_endpoint_rows,
    _continuation_retention,
    _export_table_checksums,
    _fields,
    _pair_count_row,
    _source_records,
    _tiny_validation_run_id,
    _toy_source_config_identity,
    analyze_path,
    export_compact,
)
from lean_reward_hacking.evaluation import EvaluationMetrics
from lean_reward_hacking.episodes import make_paired_evaluation
from lean_reward_hacking.schemas import validate_compact_bundle
from lean_reward_hacking.provenance import sha256_file
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

    def test_pair_counts_bind_primary_and_completed_steps(self) -> None:
        pairs = make_paired_evaluation(4, 2, 91, opportunity_probability=0.0)
        row = _pair_count_row(object(), pairs, "run", 2, "eval", completed_step=8)
        self.assertEqual(row["step"], 2)
        self.assertEqual(row["primary_step"], 2)
        self.assertEqual(row["completed_step"], 8)

    def test_endpoint_annotation_records_continuation_retention(self) -> None:
        rows, summary = _annotate_endpoint_rows(
            [
                {"step": 2, "label": "strategic"},
                {"step": 4, "label": "strategic"},
                {"step": 8, "label": "oversight-invariant"},
            ],
            primary_step=2,
            completed_step=8,
        )
        self.assertEqual(summary["primary_step"], 2)
        self.assertEqual(summary["completed_step"], 8)
        self.assertTrue(summary["retention_2T"])
        self.assertFalse(summary["retention_4T"])
        self.assertEqual(summary["continuation_retention_status"], "checked")
        self.assertTrue(rows[0]["is_primary"])
        self.assertTrue(rows[-1]["is_terminal"])
        self.assertEqual(rows[-1]["primary_label"], "strategic")

    def test_continuation_retention_marks_missing_registered_checkpoint(self) -> None:
        result = _continuation_retention(
            [{"step": 2, "label": "strategic"}, {"step": 4, "label": "strategic"}],
            primary_step=2,
            completed_step=8,
            primary_label="strategic",
        )
        self.assertEqual(result["continuation_retention_status"], "missing_checkpoint")
        self.assertFalse(result["continuation_retention_ok"])

    def test_source_records_bind_current_config_and_archive(self) -> None:
        source = "source-current"
        config = _toy_source_config_identity()
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "runs" / "toy_fixed" / "session"
            (session / "exports").mkdir(parents=True)
            (session / "markers").mkdir()
            (session / "spec").mkdir()
            (session / "spec" / "config.json").write_text(
                json.dumps({"config_sha256": config}), encoding="utf-8"
            )
            identity_path = session / "spec" / "experiment_identity.json"
            identity_path.write_text(
                json.dumps(
                    {
                        "train_dataset_sha256": "a" * 64,
                        "eval_dataset_sha256": "b" * 64,
                        "reward_sha256": "c" * 64,
                        "objective_sha256": "d" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with (session / "exports" / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=_fields("runs.csv"),
                )
                writer.writeheader()
                for values in (
                    {
                        "run_id": "good",
                        "status": "complete",
                        "experiment": "toy_fixed",
                        "architecture": "toy",
                        "config_sha256": config,
                    },
                    {
                        "run_id": "wrong-config",
                        "status": "complete",
                        "experiment": "toy_fixed",
                        "architecture": "toy",
                        "config_sha256": "wrong",
                    },
                ):
                    writer.writerow({field: values.get(field, "") for field in _fields("runs.csv")})
            with (session / "exports" / "checkpoint_metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=_fields("checkpoint_metrics.csv"))
                writer.writeheader()
                for values in (
                    {"run_id": "good", "step": 20, "label": "strategic", "c_on": 0.99, "c_off": 0.01},
                    {"run_id": "wrong-config", "step": 20, "label": "strategic", "c_on": 0.99, "c_off": 0.01},
                ):
                    writer.writerow(
                        {
                            field: values.get(field, "")
                            for field in _fields("checkpoint_metrics.csv")
                        }
                    )
            (session / "markers" / "completed.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "completion_schema_version": 2,
                        "config_sha256": config,
                        "source_archive_sha256": source,
                        "train_dataset_sha256": "a" * 64,
                        "eval_dataset_sha256": "b" * 64,
                        "reward_sha256": "c" * 64,
                        "objective_sha256": "d" * 64,
                        "experiment_identity_sha256": sha256_file(identity_path),
                        "export_table_checksums": _export_table_checksums(session),
                    }
                ),
                encoding="utf-8",
            )
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
        frozen = (
            dict(row, step_since_branch=1, d_source=1.1135),
            dict(row, step_since_branch=2, d_source=1.1135),
        )
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


if __name__ == "__main__":
    unittest.main()
