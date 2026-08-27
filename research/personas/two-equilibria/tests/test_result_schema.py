from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lean_reward_hacking.schemas import (
    KNOWN_COLUMNS,
    make_checksums,
    sha256_file,
    validate_compact_bundle,
)


class ResultSchemaTests(unittest.TestCase):
    def _valid_bundle(self, directory: Path) -> None:
        (directory / "runs.csv").write_text(
            "run_id,seed,shuffle_seed,status,git_commit,config_sha256,train_dataset_sha256,eval_dataset_sha256,reward_sha256,objective_sha256,source_archive_sha256\n"
            + "r0,1,2,complete,44bae4c19206a223d4cc9e5f1825fe7de5bc75e4,"
            + ",".join(character * 64 for character in "bcdefa")
            + "\n",
            encoding="utf-8",
        )
        (directory / "stats.json").write_text("{}\n", encoding="utf-8")
        provenance = {
            "source_archive_sha256": "a" * 64,
            "runtime": {
                "packages": {"torch": "2.5.1"},
                "accelerator": {"available": True, "name": "Tesla T4"},
            },
        }
        (directory / "provenance.json").write_text(json.dumps(provenance) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "source_archive_sha256": "a" * 64,
            "tables": [
                {
                    "path": "runs.csv",
                    "sha256": sha256_file(directory / "runs.csv"),
                    "rows": 1,
                }
            ],
            "stats": {
                "path": "stats.json",
                "sha256": sha256_file(directory / "stats.json"),
            },
            "provenance": {
                "path": "provenance.json",
                "sha256": sha256_file(directory / "provenance.json"),
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
