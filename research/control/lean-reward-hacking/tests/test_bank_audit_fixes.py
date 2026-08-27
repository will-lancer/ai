from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.analysis import extract_independent_final_runs  # noqa: E402
from lean_reward_hacking.bank_campaign import _validate_raw_rows  # noqa: E402
from lean_reward_hacking.checkpoints import CheckpointError, CheckpointStore  # noqa: E402
from lean_reward_hacking.campaign import (
    CampaignError,
    _fields,
    _filter_analysis_tables,
    _validate_scalar_raw_rows,
)  # noqa: E402
from lean_reward_hacking.schemas import make_checksums, validate_compact_bundle  # noqa: E402


class BankAuditFixTests(unittest.TestCase):
    def test_raw_bank_rows_reject_stale_duplicate_and_target_mismatch(self) -> None:
        row = {
            "run_id": "r0",
            "step": 2,
            "schema_version": 1,
            "eval_variant": "paired_audit_opportunity",
            "eval_set_hash": "eval",
            "n_pairs": 3,
            "branch_id": "",
            "source_run_id": "",
            "is_final": True,
        }
        self.assertEqual(
            _validate_raw_rows(
                [row], run_ids=["r0"], eval_hash="eval", target_steps=2, expected_pairs=3
            ),
            {("r0", 2)},
        )
        with self.assertRaises(ValueError):
            _validate_raw_rows(
                [dict(row, eval_set_hash="old")],
                run_ids=["r0"],
                eval_hash="eval",
                target_steps=2,
                expected_pairs=3,
            )
        with self.assertRaises(ValueError):
            _validate_raw_rows(
                [row, row],
                run_ids=["r0"],
                eval_hash="eval",
                target_steps=2,
                expected_pairs=3,
            )

    def test_analysis_uses_explicit_final_rows(self) -> None:
        rows = [
            {"run_id": "r0", "step": 1, "is_final": False, "c_on": 0.5, "c_off": 0.5},
            {"run_id": "r0", "step": 2, "is_final": True, "c_on": 0.9, "c_off": 0.1},
        ]
        finals = extract_independent_final_runs(rows)
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["step"], 2)

    def test_scalar_rows_bind_run_pairs_and_integer_steps(self) -> None:
        row = {
            "run_id": "r0",
            "step": "2",
            "schema_version": "1",
            "eval_variant": "paired_audit_opportunity",
            "eval_set_hash": "eval",
            "n_pairs": "3",
            "branch_id": "",
            "source_run_id": "",
            "is_final": "true",
        }
        self.assertEqual(
            _validate_scalar_raw_rows(
                [row], run_id="r0", eval_hash="eval", target_steps=2, expected_pairs=3
            ),
            {2},
        )
        with self.assertRaises(CampaignError):
            _validate_scalar_raw_rows(
                [dict(row, step="2.5")],
                run_id="r0",
                eval_hash="eval",
                target_steps=2,
                expected_pairs=3,
            )
        with self.assertRaises(CampaignError):
            _validate_scalar_raw_rows(
                [dict(row, run_id="foreign")],
                run_id="r0",
                eval_hash="eval",
                target_steps=2,
                expected_pairs=3,
            )

    def test_checkpoint_reference_cannot_cross_store_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = CheckpointStore(root, "r0", config_identity="cfg", source_identity="src")
            ref = first.save(1, {"ok": True})
            second = CheckpointStore(root / "other", "r0", config_identity="cfg", source_identity="src")
            with self.assertRaises(CheckpointError):
                second.load(ref)

    def test_schema_rejects_checkpoint_row_beyond_declared_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_fields = _fields("runs.csv")
            with (root / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=run_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow(
                    {
                        "schema_version": 1,
                        "experiment": "toy_fixed",
                        "architecture": "toy",
                        "condition": "fixed_objective",
                        "run_id": "r0",
                        "config_sha256": "cfg",
                        "status": "complete",
                        "final_step": 2,
                    }
                )
            metric_fields = _fields("checkpoint_metrics.csv")
            with (root / "checkpoint_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=metric_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow({"run_id": "r0", "step": 3, "is_final": True})
            (root / "manifest.json").write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
            (root / "checksums.sha256").write_text(make_checksums(root), encoding="utf-8")
            problems = validate_compact_bundle(root)
            self.assertTrue(any("stale step" in item for item in problems))

    def test_basin_lineage_columns_are_present_in_all_per_seed_tables(self) -> None:
        expected = {
            "phase", "level", "harm_strength", "audit_sensitivity",
            "harm_index", "audit_index", "seed_index", "cell_token",
        }
        for name in ("runs.csv", "checkpoint_metrics.csv", "final_summary.csv", "pair_counts.csv"):
            self.assertTrue(expected.issubset(_fields(name)))

    def test_fixed_objective_filter_excludes_basin_condition_rows(self) -> None:
        tables = {
            "runs.csv": [
                {"run_id": "fixed", "experiment": "toy_fixed", "condition": "fixed_objective"},
                {"run_id": "basin", "experiment": "toy_fixed", "condition": "initial_condition_basin"},
            ],
            "final_summary.csv": [
                {"run_id": "fixed", "experiment": "toy_fixed", "condition": "fixed_objective", "gap": 0.1},
                {"run_id": "basin", "experiment": "toy_fixed", "condition": "initial_condition_basin", "gap": 0.9},
            ],
        }
        filtered = _filter_analysis_tables(tables, "toy_fixed")
        self.assertEqual([row["run_id"] for row in filtered["final_summary.csv"]], ["fixed"])


if __name__ == "__main__":
    unittest.main()
