from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from lean_reward_hacking.schemas import make_checksums, sha256_file
from scripts.import_compact_bundle import ImportBundleError, import_compact_bundle


class ImportCompactBundleTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path) -> None:
        path.write_text(
            "run_id,seed,shuffle_seed,status,git_commit,config_sha256,train_dataset_sha256,eval_dataset_sha256,reward_sha256,objective_sha256,source_archive_sha256\n"
            + "r0,1,2,complete,44bae4c19206a223d4cc9e5f1825fe7de5bc75e4,"
            + ",".join(character * 64 for character in "bcdefa")
            + "\n",
            encoding="utf-8",
        )

    def _valid_bundle(self, directory: Path, *, experiment: str = "toy_fixed") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self._write_csv(directory / "runs.csv")
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
            "analysis_experiment": experiment,
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
        (directory / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "checksums.sha256").write_text(
            make_checksums(directory), encoding="utf-8"
        )
        return directory

    @staticmethod
    def _archive_directory(source: Path, archive: Path, *, wrapper: str = "bundle") -> Path:
        with tarfile.open(archive, mode="w:gz") as handle:
            for item in sorted(source.iterdir()):
                handle.add(item, arcname=f"{wrapper}/{item.name}", recursive=False)
        return archive

    @staticmethod
    def _archive_members(archive: Path, members: list[tuple[str, bytes, str | None]]) -> Path:
        """Write hand-crafted regular, link, or traversal members to a gzip tar."""

        with tarfile.open(archive, mode="w:gz") as handle:
            for name, data, member_type in members:
                info = tarfile.TarInfo(name=name)
                info.mode = 0o644
                if member_type is None:
                    info.size = len(data)
                    handle.addfile(info, io.BytesIO(data))
                else:
                    info.type = getattr(tarfile, member_type)
                    info.linkname = "bundle/runs.csv"
                    handle.addfile(info)
        return archive

    @staticmethod
    def _archive_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def test_valid_directory_is_imported_with_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            destination = root / "compact"

            target = import_compact_bundle(source, destination_root=destination)

            prefix = sha256_file(source / "manifest.json")[:16]
            self.assertEqual(target, destination / "toy_fixed" / prefix)
            self.assertIn("r0,1,2,complete", (target / "runs.csv").read_text(encoding="utf-8"))
            receipt = destination / "toy_fixed" / f"{prefix}.import_receipt.json"
            self.assertTrue(receipt.is_file())
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt_payload["experiment"], "toy_fixed")
            self.assertIsNone(receipt_payload["archive_sha256"])

    def test_valid_one_wrapper_tgz_accepts_archive_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            archive = self._archive_directory(source, root / "bundle.tgz")
            destination = root / "compact"
            archive_digest = self._archive_digest(archive)

            target = import_compact_bundle(
                archive,
                destination_root=destination,
                archive_sha256=archive_digest.upper(),
            )

            self.assertTrue(target.is_dir())
            prefix = sha256_file(source / "manifest.json")[:16]
            receipt = destination / "toy_fixed" / f"{prefix}.import_receipt.json"
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["archive_sha256"], archive_digest)

    def test_wrong_archive_hash_is_rejected_before_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            archive = self._archive_directory(source, root / "bundle.tgz")
            actual = self._archive_digest(archive)
            wrong = ("0" if actual[0] != "0" else "1") + actual[1:]

            with self.assertRaisesRegex(ImportBundleError, "archive SHA-256 mismatch"):
                import_compact_bundle(archive, destination_root=root / "compact", archive_sha256=wrong)
            self.assertFalse((root / "compact").exists())

    def test_archive_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = self._archive_members(
                Path(temporary) / "traversal.tgz",
                [("../runs.csv", b"run_id,seed,status\nr0,1,complete\n", None)],
            )
            with self.assertRaisesRegex(ImportBundleError, "traversal"):
                import_compact_bundle(archive, destination_root=Path(temporary) / "compact")

    def test_archive_symlink_and_hardlink_are_rejected(self) -> None:
        for member_type, message in (("SYMTYPE", "symlink"), ("LNKTYPE", "hardlink")):
            with self.subTest(member_type=member_type), tempfile.TemporaryDirectory() as temporary:
                archive = self._archive_members(
                    Path(temporary) / "links.tgz",
                    [("bundle/link", b"", member_type)],
                )
                with self.assertRaisesRegex(ImportBundleError, "symlink or hardlink"):
                    import_compact_bundle(archive, destination_root=Path(temporary) / "compact")

    def test_unknown_archive_nesting_and_member_are_rejected(self) -> None:
        cases = (
            ("bundle/nested/runs.csv", b"x", "unknown nesting"),
            ("bundle/unknown.txt", b"x", "outside compact allowlist"),
        )
        for name, data, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                archive = self._archive_members(
                    Path(temporary) / "invalid.tgz", [(name, data, None)]
                )
                with self.assertRaisesRegex(ImportBundleError, message):
                    import_compact_bundle(archive, destination_root=Path(temporary) / "compact")

    def test_source_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            try:
                (source / "link.json").symlink_to(source / "stats.json")
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(ImportBundleError, "invalid strict compact bundle"):
                import_compact_bundle(source, destination_root=root / "compact")

    def test_strict_schema_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            (source / "unexpected.bin").write_bytes(b"remote artifact")

            with self.assertRaisesRegex(ImportBundleError, "outside compact allowlist"):
                import_compact_bundle(source, destination_root=root / "compact")

    def test_reimport_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            destination = root / "compact"
            first = import_compact_bundle(source, destination_root=destination)
            prefix = sha256_file(source / "manifest.json")[:16]
            receipt = destination / "toy_fixed" / f"{prefix}.import_receipt.json"
            receipt_bytes = receipt.read_bytes()
            receipt_mtime = receipt.stat().st_mtime_ns

            second = import_compact_bundle(source, destination_root=destination)

            self.assertEqual(first, second)
            self.assertEqual(receipt.read_bytes(), receipt_bytes)
            self.assertEqual(receipt.stat().st_mtime_ns, receipt_mtime)
            self.assertEqual(
                sorted(item.name for item in (destination / "toy_fixed").iterdir()),
                sorted([f"{prefix}.import_receipt.json", prefix]),
            )

    def test_conflicting_member_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            destination = root / "compact"
            target = import_compact_bundle(source, destination_root=destination)
            (target / "runs.csv").write_text("run_id,seed,status\nr0,1,failed\n", encoding="utf-8")

            with self.assertRaisesRegex(ImportBundleError, "member bytes differ"):
                import_compact_bundle(source, destination_root=destination)

    def test_conflicting_receipt_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._valid_bundle(root / "source")
            destination = root / "compact"
            import_compact_bundle(source, destination_root=destination)
            prefix = sha256_file(source / "manifest.json")[:16]
            receipt = destination / "toy_fixed" / f"{prefix}.import_receipt.json"
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["bundle_sha256"] = "b" * 64
            receipt.chmod(0o644)
            receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ImportBundleError, "conflicts with bundle identity"):
                import_compact_bundle(source, destination_root=destination)


if __name__ == "__main__":
    unittest.main()
