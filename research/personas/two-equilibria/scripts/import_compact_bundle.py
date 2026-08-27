#!/usr/bin/env python3
"""Safely import a downloaded compact Colab bundle into the local release tree.

An import is content-addressed by the exact manifest bytes.  The strict bundle
directory contains only the compact schema files.  The receipt lives beside
that directory so it cannot affect bundle validation or checksums.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


IMPORT_RECEIPT_SCHEMA_VERSION = 1
MANIFEST_HASH_PREFIX_LENGTH = 16
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
MANIFEST_NAMES = frozenset({"manifest.json", "bundle_manifest.json"})


class ImportBundleError(RuntimeError):
    """Raised when a compact bundle cannot be imported safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImportBundleError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ImportBundleError(f"JSON object required: {path}")
    return payload


def _manifest_path(bundle: Path) -> Path:
    paths = [bundle / name for name in sorted(MANIFEST_NAMES) if (bundle / name).is_file()]
    if len(paths) != 1:
        raise ImportBundleError(f"expected exactly one manifest in {bundle}")
    return paths[0]


def _manifest_experiment(manifest: Mapping[str, object]) -> str:
    for key in ("analysis_experiment", "experiment_scope"):
        value = manifest.get(key)
        if value not in (None, ""):
            experiment = str(value).strip()
            break
    else:
        experiment = ""
    if not experiment:
        raise ImportBundleError("compact manifest has no analysis_experiment or experiment_scope")
    if experiment in {".", ".."} or "/" in experiment or "\\" in experiment:
        raise ImportBundleError(f"unsafe experiment name in compact manifest: {experiment!r}")
    if any(character.isspace() or ord(character) < 32 for character in experiment):
        raise ImportBundleError(f"unsafe experiment name in compact manifest: {experiment!r}")
    if not all(character.isalnum() or character in "._-" for character in experiment):
        raise ImportBundleError(f"unsafe experiment name in compact manifest: {experiment!r}")
    return experiment


def _validate_bundle(bundle: Path) -> None:
    from lean_reward_hacking.schemas import validate_compact_bundle

    problems = validate_compact_bundle(bundle, strict=True)
    if problems:
        raise ImportBundleError(f"invalid strict compact bundle {bundle}: " + "; ".join(problems))


def _flat_regular_files(bundle: Path) -> list[Path]:
    """Require a directory source to be a flat, non-symlink bundle."""

    if not bundle.is_dir() or bundle.is_symlink():
        raise ImportBundleError(f"bundle directory is missing or is a symlink: {bundle}")
    files: list[Path] = []
    for item in sorted(bundle.iterdir()):
        if item.is_symlink():
            raise ImportBundleError(f"symlink is not allowed in compact bundle: {item.name}")
        if not item.is_file():
            raise ImportBundleError(f"nested directory or non-regular member is not allowed: {item.name}")
        files.append(item)
    if not files:
        raise ImportBundleError(f"compact bundle directory is empty: {bundle}")
    return files


def _archive_path_parts(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise ImportBundleError(f"unsafe archive member name: {name!r}")
    if name.startswith("/"):
        raise ImportBundleError(f"absolute archive member path is not allowed: {name!r}")
    path = PurePosixPath(name.rstrip("/"))
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ImportBundleError(f"path traversal or empty archive member path: {name!r}")
    return parts


def _archive_mapping(members: Sequence[tarfile.TarInfo]) -> dict[str, tarfile.TarInfo]:
    """Validate archive member types and map one optional wrapper to flat names."""

    regular: list[tuple[tuple[str, ...], tarfile.TarInfo]] = []
    for member in members:
        parts = _archive_path_parts(member.name)
        if member.issym() or member.islnk():
            raise ImportBundleError(f"symlink or hardlink is not allowed in archive: {member.name}")
        if not member.isdir() and not member.isreg():
            raise ImportBundleError(f"non-regular archive member is not allowed: {member.name}")
        if member.isdir():
            if len(parts) > 1:
                raise ImportBundleError(f"unknown nested directory in archive: {member.name}")
            continue
        if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ImportBundleError(f"archive member is too large: {member.name}")
        regular.append((parts, member))
    if not regular:
        raise ImportBundleError("archive contains no regular compact-bundle files")

    top_levels = {parts[0] for parts, _ in regular}
    flat = all(len(parts) == 1 for parts, _ in regular)
    wrapped = len(top_levels) == 1 and all(len(parts) == 2 for parts, _ in regular)
    if not flat and not wrapped:
        raise ImportBundleError("archive contains unknown nesting; expected flat files or one wrapper directory")
    mapped: dict[str, tarfile.TarInfo] = {}
    for parts, member in regular:
        name = parts[-1] if wrapped else parts[0]
        if name in mapped:
            raise ImportBundleError(f"duplicate archive member after wrapper stripping: {name}")
        mapped[name] = member
    from lean_reward_hacking.schemas import COMPACT_ALLOWLIST

    unknown = sorted(set(mapped) - COMPACT_ALLOWLIST)
    if unknown:
        raise ImportBundleError("archive member is outside compact allowlist: " + ", ".join(unknown))
    return mapped


def _extract_archive(archive: Path, stage: Path) -> str:
    archive_digest = _sha256_file(archive)
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            members = handle.getmembers()
            mapped = _archive_mapping(members)
            total_size = sum(member.size for member in mapped.values())
            if total_size > MAX_ARCHIVE_TOTAL_BYTES:
                raise ImportBundleError("archive contains too many compact-bundle bytes")
            stage.mkdir(parents=True, exist_ok=True)
            for name, member in mapped.items():
                source = handle.extractfile(member)
                if source is None:
                    raise ImportBundleError(f"cannot read archive member: {member.name}")
                destination = stage / name
                written = 0
                with source, destination.open("wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        written += len(chunk)
                if written != member.size:
                    raise ImportBundleError(f"archive member size changed while extracting: {member.name}")
    except (OSError, tarfile.TarError) as exc:
        raise ImportBundleError(f"cannot read compact archive {archive}: {exc}") from exc
    return archive_digest


def _copy_directory(source: Path, stage: Path) -> None:
    files = _flat_regular_files(source)
    stage.mkdir(parents=True, exist_ok=True)
    for item in files:
        shutil.copyfile(item, stage / item.name)


def _bundle_file_records(bundle: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in sorted(bundle.iterdir()):
        if not item.is_file() or item.is_symlink():
            raise ImportBundleError(f"strict bundle has non-regular member: {item.name}")
        records.append({"path": item.name, "bytes": item.stat().st_size, "sha256": _sha256_file(item)})
    return records


def _bundle_digest(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda value: str(value.get("path", ""))):
        digest.update(str(record.get("path", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.get("bytes", "")).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record.get("sha256", "")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _bundle_identity(bundle: Path) -> tuple[str, str, str, list[dict[str, object]]]:
    manifest = _manifest_path(bundle)
    manifest_sha256 = _sha256_file(manifest)
    manifest_payload = _json_object(manifest)
    experiment = _manifest_experiment(manifest_payload)
    records = _bundle_file_records(bundle)
    return experiment, manifest.name, manifest_sha256, records


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    return filecmp.cmp(left, right, shallow=False)


def _compare_bundles(existing: Path, incoming: Path) -> None:
    existing_files = {item.name for item in existing.iterdir() if item.is_file()}
    incoming_files = {item.name for item in incoming.iterdir() if item.is_file()}
    if existing_files != incoming_files:
        raise ImportBundleError(
            f"manifest identity already exists at {existing}, but bundle files differ"
        )
    for name in sorted(existing_files):
        if not _same_bytes(existing / name, incoming / name):
            raise ImportBundleError(
                f"manifest identity already exists at {existing}, but member bytes differ: {name}"
            )


def _receipt_path(experiment_dir: Path, manifest_sha256: str) -> Path:
    return experiment_dir / f"{manifest_sha256[:MANIFEST_HASH_PREFIX_LENGTH]}.import_receipt.json"


def _receipt_payload(
    *,
    experiment: str,
    manifest_name: str,
    manifest_sha256: str,
    records: Sequence[Mapping[str, object]],
    archive_sha256: str | None,
) -> dict[str, object]:
    normalized_records = [
        {
            "path": str(record["path"]),
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in sorted(records, key=lambda value: str(value.get("path", "")))
    ]
    return {
        "receipt_schema_version": IMPORT_RECEIPT_SCHEMA_VERSION,
        "status": "imported",
        "experiment": experiment,
        "manifest": manifest_name,
        "manifest_sha256": manifest_sha256,
        "manifest_hash_prefix": manifest_sha256[:MANIFEST_HASH_PREFIX_LENGTH],
        "bundle_sha256": _bundle_digest(normalized_records),
        "archive_sha256": archive_sha256,
        "files": normalized_records,
    }


def _write_immutable_receipt(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ImportBundleError(f"import receipt path is not a regular file: {path}")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ImportBundleError(f"existing import receipt is unreadable: {path}: {exc}") from exc
        if not isinstance(existing, Mapping):
            raise ImportBundleError(f"existing import receipt is not an object: {path}")
        normalized_existing = json.loads(
            json.dumps(dict(existing), sort_keys=True, ensure_ascii=False)
        )
        normalized_payload = json.loads(
            json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
        )
        if normalized_existing != normalized_payload:
            raise ImportBundleError(f"existing import receipt conflicts with bundle identity: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    try:
        path.chmod(0o444)
    except OSError:
        pass


def _find_duplicate_identity(experiment_dir: Path, manifest_sha256: str, target: Path) -> Path | None:
    if not experiment_dir.is_dir():
        return None
    for candidate in sorted(experiment_dir.iterdir()):
        if candidate == target or not candidate.is_dir() or candidate.is_symlink():
            continue
        try:
            manifest = _manifest_path(candidate)
        except ImportBundleError:
            continue
        if _sha256_file(manifest) == manifest_sha256:
            return candidate
    return None


def import_compact_bundle(
    source: str | Path,
    *,
    destination_root: str | Path = PROJECT_ROOT / "results" / "compact",
    archive_sha256: str | None = None,
) -> Path:
    """Import a strict bundle directory or gzip tar archive.

    The returned path is ``<destination_root>/<experiment>/<manifest-prefix>``.
    A byte-identical re-import is idempotent.  Any conflicting bytes at the
    same manifest identity raise :class:`ImportBundleError`.
    """

    source_path = Path(source)
    destination = Path(destination_root)
    expected_archive_sha256 = archive_sha256.lower().strip() if archive_sha256 else None
    if expected_archive_sha256 is not None and (
        len(expected_archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_archive_sha256)
    ):
        raise ImportBundleError("--archive-sha256 must be a 64-character hexadecimal digest")
    if not source_path.exists() or source_path.is_symlink():
        raise ImportBundleError(f"source does not exist or is a symlink: {source_path}")
    if destination.exists() and destination.is_symlink():
        raise ImportBundleError(f"destination root is a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    archive_digest: str | None = None
    with tempfile.TemporaryDirectory(prefix=".compact-import.", dir=str(destination.parent)) as temporary:
        stage = Path(temporary) / "bundle"
        if source_path.is_dir():
            if expected_archive_sha256 is not None:
                raise ImportBundleError("--archive-sha256 applies only to .tar.gz or .tgz inputs")
            _validate_bundle(source_path)
            _copy_directory(source_path, stage)
        else:
            lower_name = source_path.name.lower()
            if not (lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz")):
                raise ImportBundleError("archive input must end in .tar.gz or .tgz")
            archive_digest = _sha256_file(source_path)
            if expected_archive_sha256 is not None and archive_digest != expected_archive_sha256:
                raise ImportBundleError(
                    f"archive SHA-256 mismatch: expected {expected_archive_sha256}, got {archive_digest}"
                )
            archive_digest = _extract_archive(source_path, stage)
            _validate_bundle(stage)

        experiment, manifest_name, manifest_sha256, records = _bundle_identity(stage)
        prefix = manifest_sha256[:MANIFEST_HASH_PREFIX_LENGTH]
        experiment_dir = destination / experiment
        if experiment_dir.exists() and experiment_dir.is_symlink():
            raise ImportBundleError(f"experiment destination is a symlink: {experiment_dir}")
        target = experiment_dir / prefix
        duplicate = _find_duplicate_identity(experiment_dir, manifest_sha256, target)
        if duplicate is not None:
            raise ImportBundleError(
                f"manifest identity already imported at {duplicate}; refusing duplicate path"
            )
        receipt = _receipt_path(experiment_dir, manifest_sha256)
        receipt_payload = _receipt_payload(
            experiment=experiment,
            manifest_name=manifest_name,
            manifest_sha256=manifest_sha256,
            records=records,
            archive_sha256=archive_digest,
        )
        if receipt.exists() and target.exists():
            if target.is_symlink() or not target.is_dir():
                raise ImportBundleError(f"existing import target is not a regular directory: {target}")
            _compare_bundles(target, stage)
            _write_immutable_receipt(receipt, receipt_payload)
            return target
        if receipt.exists() and not target.exists():
            _write_immutable_receipt(receipt, receipt_payload)
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise ImportBundleError(f"existing import target is not a regular directory: {target}")
            _compare_bundles(target, stage)
            _write_immutable_receipt(receipt, receipt_payload)
            return target
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if experiment_dir.is_symlink():
            raise ImportBundleError(f"experiment destination is a symlink: {experiment_dir}")
        try:
            os.rename(stage, target)
        except FileExistsError:
            if not target.is_dir() or target.is_symlink():
                raise ImportBundleError(f"existing import target is not a regular directory: {target}")
            _compare_bundles(target, stage)
        except OSError as exc:
            if target.exists():
                if not target.is_dir() or target.is_symlink():
                    raise ImportBundleError(f"existing import target is not a regular directory: {target}") from exc
                _compare_bundles(target, stage)
            else:
                raise ImportBundleError(f"cannot install compact bundle at {target}: {exc}") from exc
        _write_immutable_receipt(receipt, receipt_payload)
        return target


# Short alias for callers that prefer the script's command name.
import_bundle = import_compact_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="import_compact_bundle.py")
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--destination-root",
        "--compact-root",
        dest="destination_root",
        type=Path,
        default=PROJECT_ROOT / "results" / "compact",
    )
    parser.add_argument("--archive-sha256", "--sha256", dest="archive_sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = import_compact_bundle(
            args.source,
            destination_root=args.destination_root,
            archive_sha256=args.archive_sha256,
        )
    except (ImportBundleError, OSError, ValueError) as exc:
        print(f"compact import failed: {exc}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IMPORT_RECEIPT_SCHEMA_VERSION",
    "ImportBundleError",
    "MANIFEST_HASH_PREFIX_LENGTH",
    "import_bundle",
    "import_compact_bundle",
    "main",
]
