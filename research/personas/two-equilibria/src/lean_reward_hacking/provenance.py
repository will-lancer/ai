"""Stable provenance and identity records for local and Colab artifacts.

The functions here use only the Python standard library.  They record enough
identity to bind a result bundle to its source tree, configuration, runtime,
and seed.  A timestamp is retained for auditability; deterministic table and
figure generation should use the content hashes rather than the timestamp.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROVENANCE_SCHEMA_VERSION = "lrh-provenance/v1"

_DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "build",
        "dist",
        "results",
        "raw",
        "checkpoints",
        "logs",
    }
)
_DEFAULT_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz")


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest(), "length": len(value)}
    raise TypeError(f"value of type {type(value).__name__} is not JSON serialisable")


def canonical_json(value: Any) -> bytes:
    """Encode a value with stable key and separator choices."""

    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_sha256(config: Any) -> str:
    """Return an identity hash for a config path, bytes, or structured value."""

    if isinstance(config, (str, os.PathLike)):
        path = Path(config)
        if path.is_file():
            return sha256_file(path)
    if isinstance(config, bytes):
        payload = config
    else:
        payload = canonical_json(config)
    return sha256_bytes(payload)


def hash_tree(
    root: str | os.PathLike[str],
    *,
    exclude_dirs: Iterable[str] = _DEFAULT_EXCLUDED_DIRS,
    exclude_suffixes: Sequence[str] = _DEFAULT_EXCLUDED_SUFFIXES,
) -> str:
    """Hash regular files under ``root`` in path order.

    File names and lengths are included before contents.  Symlinks are skipped
    so a project hash cannot unexpectedly read data outside the project root.
    Generated result and model-artifact directories are excluded by default.
    """

    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"source root is not a directory: {base}")
    excluded = frozenset(str(part) for part in exclude_dirs)
    suffixes = tuple(str(suffix).lower() for suffix in exclude_suffixes)
    files: list[Path] = []
    for current, dirs, names in os.walk(base, followlinks=False):
        # Hidden source directories such as .github can carry reproducibility
        # inputs.  Only the explicit generated/VCS exclusions are skipped.
        dirs[:] = sorted(name for name in dirs if name not in excluded)
        for name in sorted(names):
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            if path.name == ".DS_Store" or path.suffix.lower() in suffixes:
                continue
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_identity(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Collect commit and dirty-state identity without network access."""

    base = Path(root).resolve()
    commit = _run_git(base, ("rev-parse", "HEAD"))
    branch = _run_git(base, ("rev-parse", "--abbrev-ref", "HEAD"))
    status = _run_git(base, ("status", "--short", "--", "."))
    diff = _run_git(base, ("diff", "--no-ext-diff", "--binary", "--", "."))
    staged = _run_git(base, ("diff", "--cached", "--no-ext-diff", "--binary", "--", "."))
    diff_payload = ((diff or "") + "\n" + (staged or "")).encode("utf-8")
    result: dict[str, Any] = {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_sha256": sha256_bytes((status or "").encode("utf-8")),
        "diff_sha256": sha256_bytes(diff_payload),
    }
    try:
        result["source_tree_sha256"] = hash_tree(base)
    except (FileNotFoundError, OSError):
        result["source_tree_sha256"] = None
    return result


def _accelerator_identity() -> dict[str, Any]:
    result: dict[str, Any] = {
        "lrh_runtime": os.environ.get("LRH_RUNTIME"),
        "lrh_accelerator": os.environ.get("LRH_ACCELERATOR"),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    # Keep torch optional.  The LM notebook can record a richer device name
    # after it imports torch, while base tests remain dependency-free.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            result["torch_cuda_available"] = bool(torch.cuda.is_available())
            result["torch_cuda_device"] = (
                torch.cuda.get_device_name(0) if result["torch_cuda_available"] else None
            )
        except Exception:
            result["torch_cuda_available"] = None
        try:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            result["torch_mps_available"] = bool(mps.is_available()) if mps is not None else False
        except Exception:
            result["torch_mps_available"] = None
    return result


def runtime_identity(package_names: Iterable[str] = ()) -> dict[str, Any]:
    """Return runtime, accelerator, and selected dependency versions."""

    packages: dict[str, str | None] = {}
    for name in sorted(set(package_names)):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "accelerator": _accelerator_identity(),
    }


def collect_provenance(
    project_root: str | os.PathLike[str],
    *,
    config: Any | None = None,
    seeds: Iterable[int] = (),
    package_names: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible provenance record for a run or bundle."""

    root = Path(project_root).resolve()
    git = git_identity(root)
    runtime = runtime_identity(package_names)
    record: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_root": str(root),
        "config_sha256": None if config is None else configuration_sha256(config),
        "seeds": [int(seed) for seed in seeds],
        "git": git,
        "runtime": runtime,
        # These flat aliases make manifests easy to query while retaining the
        # structured records above for auditing.
        "git_commit": git.get("commit"),
        "source_tree_sha256": git.get("source_tree_sha256"),
        "source_hash": git.get("source_tree_sha256"),
        "runtime_sha256": configuration_sha256(runtime),
    }
    if extra:
        record["extra"] = dict(extra)
    return record


def atomic_write_json(path: str | os.PathLike[str], value: Any, *, indent: int | None = 2) -> str:
    """Atomically write JSON and return its SHA-256 digest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        indent=indent,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


def write_provenance(path: str | os.PathLike[str], record: Mapping[str, Any]) -> str:
    """Write a provenance JSON file atomically and return its digest."""

    return atomic_write_json(path, dict(record))


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "atomic_write_json",
    "canonical_json",
    "collect_provenance",
    "configuration_sha256",
    "git_identity",
    "hash_tree",
    "runtime_identity",
    "sha256_bytes",
    "sha256_file",
    "write_provenance",
]
