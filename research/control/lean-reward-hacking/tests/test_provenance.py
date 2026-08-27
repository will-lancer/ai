from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from lean_reward_hacking.provenance import (  # noqa: E402
    PROVENANCE_SCHEMA_VERSION,
    atomic_write_json,
    canonical_json,
    collect_provenance,
    configuration_sha256,
    git_identity,
    hash_tree,
    runtime_identity,
    sha256_file,
)


class ProvenanceTests(unittest.TestCase):
    def test_canonical_config_hash_is_order_independent(self) -> None:
        first = {"seed": 7, "nested": {"lr": 0.1, "batch": 4}}
        second = {"nested": {"batch": 4, "lr": 0.1}, "seed": 7}
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(configuration_sha256(first), configuration_sha256(second))

    def test_hash_tree_is_stable_and_excludes_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
            (root / "results").mkdir()
            (root / "results" / "large.bin").write_bytes(b"ignored")
            first = hash_tree(root)
            (root / "results" / "large.bin").write_bytes(b"changed")
            self.assertEqual(first, hash_tree(root))
            (root / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
            self.assertNotEqual(first, hash_tree(root))

    def test_atomic_json_write_returns_content_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "manifest.json"
            digest = atomic_write_json(target, {"b": 2, "a": 1})
            payload = target.read_bytes()
            self.assertEqual(digest, __import__("hashlib").sha256(payload).hexdigest())
            self.assertEqual(json.loads(payload), {"a": 1, "b": 2})
            self.assertFalse(any(target.parent.glob("*.tmp")))

    def test_collect_provenance_contains_runtime_source_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_text("pass\n", encoding="utf-8")
            record = collect_provenance(
                root,
                config={"seed": 9},
                seeds=[9],
                package_names=["package-that-is-not-installed"],
            )
            self.assertEqual(record["schema_version"], PROVENANCE_SCHEMA_VERSION)
            self.assertEqual(record["seeds"], [9])
            self.assertEqual(len(record["config_sha256"]), 64)
            self.assertEqual(len(record["git"]["source_tree_sha256"]), 64)
            self.assertIn("python", record["runtime"])
            self.assertIsNone(record["runtime"]["packages"]["package-that-is-not-installed"])

    def test_git_identity_is_network_free_and_json_compatible(self) -> None:
        record = git_identity(ROOT)
        self.assertIn("commit", record)
        self.assertIn("dirty", record)
        json.dumps(record)

    def test_runtime_package_versions_are_optional(self) -> None:
        record = runtime_identity(["package-that-is-not-installed"])
        self.assertIn("accelerator", record)
        self.assertIsNone(record["packages"]["package-that-is-not-installed"])

    def test_file_digest_matches_hash_tree_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.txt"
            path.write_text("payload", encoding="utf-8")
            self.assertEqual(len(sha256_file(path)), 64)


if __name__ == "__main__":
    unittest.main()
