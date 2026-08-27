from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from lean_reward_hacking.campaign import TABLE_NAMES, _fields
from lean_reward_hacking.schemas import (
    COMPACT_ALLOWLIST,
    KNOWN_COLUMNS,
    REQUIRED_COLUMNS,
    make_checksums,
    validate_compact_bundle,
)


class ResultSchemaTests(unittest.TestCase):
    def _valid_bundle(self, directory: Path) -> None:
        fields = _fields("runs.csv")
        with (directory / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerow({"run_id": "r0", "seed": 1, "status": "complete"})
        (directory / "manifest.json").write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
        (directory / "checksums.sha256").write_text(make_checksums(directory), encoding="utf-8")

    def test_manifest_and_checksums_make_bundle_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._valid_bundle(directory)
            self.assertEqual(validate_compact_bundle(directory), [])

    def test_unknown_file_is_rejected_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._valid_bundle(directory)
            (directory / "weights.pt").write_bytes(b"remote artifact")
            self.assertTrue(any("allowlist" in item for item in validate_compact_bundle(directory)))

    def test_off_audit_logit_is_allowed_in_metric_tables(self) -> None:
        self.assertIn("off_audit_logit", KNOWN_COLUMNS["checkpoint_metrics.csv"])
        self.assertIn("off_audit_logit", KNOWN_COLUMNS["final_summary.csv"])

    def test_campaign_table_contract_is_exact_and_allowlisted(self) -> None:
        for name in TABLE_NAMES:
            fields = tuple(_fields(name))
            self.assertEqual(REQUIRED_COLUMNS[name], fields)
            self.assertEqual(KNOWN_COLUMNS[name], frozenset(fields))
            self.assertIn(name, COMPACT_ALLOWLIST)


if __name__ == "__main__":
    unittest.main()
