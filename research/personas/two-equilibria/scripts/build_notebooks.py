#!/usr/bin/env python3
"""Build the checked-in Colab notebooks from deterministic templates.

The notebooks deliberately contain the source snapshot that was present when
this generator ran.  A Colab session therefore never follows a mutable Git
branch.  The generator has no experiment side effects; it only writes the five
notebook JSON files under ``notebooks/``.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
GENERATOR_VERSION = "2026-08-27.1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_identity() -> tuple[str, bool]:
    """Return the umbrella Git commit and whether this project is dirty."""

    try:
        commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        commit = "unknown"
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "status",
                "--porcelain=v1",
                "--",
                ".",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        dirty = True
    return commit, dirty


def _source_files() -> list[tuple[str, bytes]]:
    """Collect code and configuration, excluding generated or raw artifacts."""

    allowed_roots = ("src", "configs", "scripts", "tests")
    explicit = (
        "pyproject.toml",
        "requirements-colab.txt",
        "requirements-lm-colab.txt",
        "README.md",
        "PROJECT_PLAN.md",
        "LEAN_REWARD_HACKING_GOAL.md",
        ".gitignore",
        "reports/architecture_registration.md",
        "reports/statistical_methods.md",
        "reports/compute_manifest.md",
        "reports/literature_gap_audit.md",
        "reports/source_ledger.csv",
        "reports/claim_matrix.csv",
        "reports/LM_RESOURCE_REQUIREMENTS.json",
    )
    paths: set[Path] = set()
    for root_name in allowed_roots:
        root = PROJECT_ROOT / root_name
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise RuntimeError(f"source snapshot refuses symlink: {path}")
                if path.is_file():
                    paths.add(path)
    for name in explicit:
        path = PROJECT_ROOT / name
        if path.is_symlink():
            raise RuntimeError(f"source snapshot refuses symlink: {path}")
        if path.is_file():
            paths.add(path)
    records: list[tuple[str, bytes]] = []
    for path in sorted(paths):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in Path(relative).parts):
            continue
        if any(part in {"results", "raw", "checkpoints", "logs", "cache"} for part in Path(relative).parts):
            continue
        records.append((relative, path.read_bytes()))
    return records


def _deterministic_archive(records: Iterable[tuple[str, bytes]]) -> bytes:
    """Create a gzip tar with stable headers and stable member ordering."""

    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative, data in records:
                info = tarfile.TarInfo(relative)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def source_snapshot() -> dict[str, object]:
    records = _source_files()
    archive = _deterministic_archive(records)
    commit, dirty = _git_identity()
    return {
        "commit": commit,
        "dirty": dirty,
        "files": [name for name, _ in records],
        "file_hashes": {name: _sha256(data) for name, data in records},
        "archive_sha256": _sha256(archive),
        "archive_b64": base64.b64encode(archive).decode("ascii"),
    }


def _lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def _cell(cell_type: str, source: str, label: str) -> dict[str, object]:
    source_lines = _lines(source)
    cell_id = hashlib.sha256((label + "\n" + source).encode("utf-8")).hexdigest()[:12]
    cell: dict[str, object] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines,
        "id": cell_id,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _markdown(text: str, label: str) -> dict[str, object]:
    return _cell("markdown", text, label)


def _code(text: str, label: str) -> dict[str, object]:
    return _cell("code", text, label)


COMMON_RUNTIME = r'''# [RH-BOOTSTRAP] Drive, source identity, provenance, and resumability
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import time

# Full Colab runs use two numerical threads.  This also keeps tiny validation
# deterministic when a runtime exposes a large CPU pool.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ.setdefault("PYTHONHASHSEED", "0")

SOURCE_COMMIT = "__SOURCE_COMMIT__"
SOURCE_DIRTY = __SOURCE_DIRTY__
SOURCE_ARCHIVE_SHA256 = "__SOURCE_ARCHIVE_SHA256__"
SOURCE_ARCHIVE_B64 = "__SOURCE_ARCHIVE_B64__"
SOURCE_FILE_HASHES = __SOURCE_FILE_HASHES__
GENERATOR_VERSION = "__GENERATOR_VERSION__"
EXPERIMENT = "__EXPERIMENT__"
CONFIG_NAME = "__CONFIG_NAME__"
REQUIREMENTS_NAME = "__REQUIREMENTS_NAME__"
os.environ["RH_SOURCE_COMMIT"] = SOURCE_COMMIT


def _safe_path_component(value: object, *, field: str) -> str:
    text = str(value or "")
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or text.startswith(".")
        or any(not (character.isalnum() or character in "-_.") for character in text)
    ):
        raise RuntimeError(f"{field} must be one path-safe component")
    return text


RUN_ID = _safe_path_component(
    os.environ.get("LRH_RUN_ID", f"{EXPERIMENT}-{SOURCE_ARCHIVE_SHA256[:12]}"),
    field="LRH_RUN_ID",
)
REMOTE_ROOT = Path("/content/drive/MyDrive/two_equilibria/v1")
REMOTE_RUN_DIR = REMOTE_ROOT / "runs" / EXPERIMENT / RUN_ID
REMOTE_MARKER_DIR = REMOTE_RUN_DIR / "markers"
WORK_DIR = Path("/content/rh_work") / RUN_ID
SOURCE_ROOT = WORK_DIR / "source"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write a file safely, including files on a mounted Drive filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def atomic_copy_to_drive(source: Path, destination: Path) -> None:
    """Copy a completed local artifact to Drive before exposing its final name."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _safe_extract(payload: bytes, destination: Path, expected_names: set[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        members = archive.getmembers()
        seen: set[str] = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not member.isfile()
                or member.name in seen
            ):
                raise RuntimeError(f"unsafe source archive member: {member.name}")
            seen.add(member.name)
        if seen != expected_names:
            raise RuntimeError(
                "source archive membership differs from the embedded hash ledger: "
                + json.dumps(
                    {
                        "missing": sorted(expected_names - seen),
                        "extra": sorted(seen - expected_names),
                    },
                    sort_keys=True,
                )
            )
        archive.extractall(destination)


def materialize_source() -> None:
    payload = base64.b64decode(SOURCE_ARCHIVE_B64.encode("ascii"))
    if _sha256_bytes(payload) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("embedded source archive hash mismatch")
    def verified() -> bool:
        actual: set[str] = set()
        if not SOURCE_ROOT.is_dir():
            return False
        for path in SOURCE_ROOT.rglob("*"):
            if path.is_symlink():
                return False
            if path.is_file():
                actual.add(path.relative_to(SOURCE_ROOT).as_posix())
        return actual == set(SOURCE_FILE_HASHES) and all(
            sha256_file(SOURCE_ROOT / name) == digest
            for name, digest in SOURCE_FILE_HASHES.items()
        )
    if SOURCE_ROOT.exists() and verified():
        sys.path.insert(0, str(SOURCE_ROOT / "src"))
        return
    if SOURCE_ROOT.exists():
        if SOURCE_ROOT.parent != WORK_DIR:
            raise RuntimeError("refusing to replace a source directory outside the runtime work tree")
        shutil.rmtree(SOURCE_ROOT)
    _safe_extract(payload, SOURCE_ROOT, set(SOURCE_FILE_HASHES))
    if not verified():
        raise RuntimeError("materialized source file hash mismatch")
    sys.path.insert(0, str(SOURCE_ROOT / "src"))


def _drive_mount() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This notebook must run in Google Colab") from exc
    drive.mount("/content/drive", force_remount=False)


def _runtime_info() -> dict[str, object]:
    info: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "runtime_marker": os.environ.get("COLAB_RELEASE_TAG", "unknown"),
        "accelerator": {"available": False, "name": None, "memory_bytes": None},
    }
    try:
        import psutil

        info["host_memory_bytes"] = psutil.virtual_memory().total
    except ImportError:
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            total_kib = int(
                next(line for line in meminfo.splitlines() if line.startswith("MemTotal:"))
                .split()[1]
            )
            info["host_memory_bytes"] = total_kib * 1024
        except (OSError, StopIteration, ValueError, IndexError):
            info["host_memory_bytes"] = None
    try:
        import torch

        accelerator = info["accelerator"]
        assert isinstance(accelerator, dict)
        accelerator["available"] = bool(torch.cuda.is_available())
        if accelerator["available"]:
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            accelerator["name"] = properties.name
            accelerator["memory_bytes"] = properties.total_memory
            accelerator["cuda"] = torch.version.cuda
        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    return info


def assert_free_colab_resources(
    *,
    minimum_gpu_memory_gib: float = 14.0,
    minimum_host_ram_gib: float = 8.0,
    minimum_drive_free_gib: float = 2.0,
    allowed_accelerators: tuple[str, ...] = ("T4", "L4"),
) -> dict[str, object]:
    """Fail before full work unless the registered free-runtime contract holds."""

    tier = os.environ.get("RH_COLAB_COMPUTE_TIER", "").strip().lower()
    if tier != "free":
        raise RuntimeError(
            "set RH_COLAB_COMPUTE_TIER=free only after confirming the attached runtime uses free Colab compute"
        )
    runtime = _runtime_info()
    accelerator = runtime.get("accelerator")
    if not isinstance(accelerator, dict) or not accelerator.get("available"):
        raise RuntimeError("full campaign requires a visible Colab CUDA accelerator")
    accelerator_name = str(accelerator.get("name") or "")
    if not any(token.lower() in accelerator_name.lower() for token in allowed_accelerators):
        raise RuntimeError(
            f"accelerator {accelerator_name or 'unknown'} is outside the registered free allowlist {allowed_accelerators}"
        )
    gpu_bytes = int(accelerator.get("memory_bytes") or 0)
    if gpu_bytes < int(minimum_gpu_memory_gib * 1024**3):
        raise RuntimeError(
            f"accelerator memory {gpu_bytes / 1024**3:.2f} GiB is below {minimum_gpu_memory_gib:.2f} GiB"
        )
    host_bytes = int(runtime.get("host_memory_bytes") or 0)
    if host_bytes < int(minimum_host_ram_gib * 1024**3):
        raise RuntimeError(
            f"host memory {host_bytes / 1024**3:.2f} GiB is below {minimum_host_ram_gib:.2f} GiB"
        )
    drive_free = shutil.disk_usage(REMOTE_ROOT).free
    if drive_free < int(minimum_drive_free_gib * 1024**3):
        raise RuntimeError(
            f"Drive free space {drive_free / 1024**3:.2f} GiB is below {minimum_drive_free_gib:.2f} GiB"
        )
    try:
        import torch

        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
        effective_threads = torch.get_num_threads()
    except ImportError:
        effective_threads = None
    decision = {
        "compute_tier": tier,
        "paid_compute_authorized": False,
        "allowed_accelerators": list(allowed_accelerators),
        "minimum_gpu_memory_gib": minimum_gpu_memory_gib,
        "minimum_host_ram_gib": minimum_host_ram_gib,
        "minimum_drive_free_gib": minimum_drive_free_gib,
        "observed_drive_free_bytes": drive_free,
        "effective_torch_threads": effective_threads,
        "runtime": runtime,
    }
    atomic_write_json(REMOTE_RUN_DIR / "provenance" / "resource_gate.json", decision)
    return decision


def _required_versions(requirements: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            expected.update(_required_versions(requirements.parent / line[3:].strip()))
            continue
        if line.startswith("-") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        expected[name.lower().replace("-", "_")] = version
    return expected


def assert_pinned_versions(requirements: Path) -> dict[str, str]:
    expected = _required_versions(requirements)
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for normalized, wanted in expected.items():
        try:
            actual = importlib_metadata.version(normalized)
        except importlib_metadata.PackageNotFoundError:
            try:
                actual = importlib_metadata.version(normalized.replace("_", "-"))
            except importlib_metadata.PackageNotFoundError:
                actual = "missing"
        observed[normalized] = actual
        if actual != wanted:
            mismatches.append(f"{normalized}={actual}, expected {wanted}")
    if mismatches:
        raise RuntimeError("pinned dependency mismatch: " + "; ".join(mismatches))
    return observed


def configured_seeds(config_name: str = CONFIG_NAME) -> dict[str, object]:
    import tomllib

    raw = tomllib.loads(config_path(config_name).read_text(encoding="utf-8"))
    return {
        str(key): value
        for key, value in raw.items()
        if "seed" in str(key).lower() or str(key).lower().endswith("seeds")
    }


def install_and_record_versions() -> None:
    requirements = SOURCE_ROOT / REQUIREMENTS_NAME
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", str(requirements)],
        check=True,
    )
    observed = assert_pinned_versions(requirements)
    atomic_write_json(
        REMOTE_RUN_DIR / "provenance" / "packages.json",
        {"requirements": REQUIREMENTS_NAME, "requirements_sha256": sha256_file(requirements), "packages": observed},
    )


def config_path(config_name: str = CONFIG_NAME) -> Path:
    path = SOURCE_ROOT / "configs" / config_name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def config_identity(config_name: str = CONFIG_NAME) -> dict[str, object]:
    path = config_path(config_name)
    return {
        "name": config_name,
        "path": str(path),
        "sha256": sha256_file(path),
        "config_sha256": config_run_sha256(config_name),
        "bytes": path.stat().st_size,
    }


def config_run_sha256(config_name: str = CONFIG_NAME) -> str:
    """Hash the validated config shape used by the campaign completion marker."""

    if config_name == "lm_colab.toml":
        from lean_reward_hacking.lm_training import LMTrainingConfig

        return LMTrainingConfig.from_toml(config_path(config_name)).config_sha256
    from lean_reward_hacking.config import config_hash, load_config

    return config_hash(load_config(config_path(config_name)))


def config_experiment(config_name: str = CONFIG_NAME) -> str:
    import tomllib

    value = tomllib.loads(config_path(config_name).read_text(encoding="utf-8")).get("experiment")
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"config {config_name} has no experiment name")
    return value


def config_completed(config_name: str) -> bool:
    return any_config_completed(config_name)


def record_provenance(config_name: str = CONFIG_NAME, seeds: object = None) -> None:
    requirements = SOURCE_ROOT / REQUIREMENTS_NAME
    package_versions = assert_pinned_versions(requirements)
    provenance = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_dirty": SOURCE_DIRTY,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_files_embedded": True,
        "config": config_identity(config_name),
        "requirements": {"name": REQUIREMENTS_NAME, "sha256": sha256_file(requirements)},
        "packages": package_versions,
        "seed": seeds,
        "runtime": _runtime_info(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(REMOTE_RUN_DIR / "provenance" / "provenance.json", provenance)


def marker(name: str) -> Path:
    return REMOTE_MARKER_DIR / name


def _complete_marker_candidates(directory: Path) -> tuple[Path, ...]:
    return (directory / "completed.json", directory / "RUN_COMPLETE.json", directory / "run_complete.json")


def _complete_marker_paths(directory: Path) -> tuple[Path, ...]:
    return _complete_marker_candidates(directory) + _complete_marker_candidates(directory / "markers")


def _config_names(config_name: str | Iterable[str] | None) -> tuple[str, ...]:
    if config_name is None:
        return (CONFIG_NAME,)
    if isinstance(config_name, str):
        return (config_name,)
    names = tuple(str(name) for name in config_name)
    if not names or len(set(names)) != len(names):
        return ()
    return names


def _config_identity_matches(candidate: object, expected: dict[str, object]) -> bool:
    if isinstance(candidate, str):
        return candidate in {expected["sha256"], expected["config_sha256"]}
    if not isinstance(candidate, dict):
        return False
    return (
        candidate.get("name") == expected["name"]
        and candidate.get("sha256") == expected["sha256"]
        and candidate.get("config_sha256", expected["config_sha256"]) == expected["config_sha256"]
        and candidate.get("bytes") == expected["bytes"]
    )


def _marker_config_matches(value: dict[str, object], config_name: str | Iterable[str] | None) -> bool:
    names = _config_names(config_name)
    if not names:
        return False
    try:
        expected = {name: config_identity(name) for name in names}
    except (FileNotFoundError, OSError):
        return False

    # Combined notebook markers carry one identity per campaign config.  The
    # exact set prevents a stale marker from silently omitting one campaign.
    identities = value.get("config_identities")
    if len(names) > 1:
        if not isinstance(identities, dict) or set(identities) != set(names):
            return False
        if any(not _config_identity_matches(identities.get(name), expected[name]) for name in names):
            return False
        for key in ("config_sha256", "config_identity", "config"):
            if key in value and not _config_identity_matches(value[key], expected[names[0]]):
                return False
        configs = value.get("configs")
        if configs is not None and (not isinstance(configs, list) or set(configs) != set(names)):
            return False
        return True

    name = names[0]
    if isinstance(identities, dict):
        if set(identities) != {name} or not _config_identity_matches(identities.get(name), expected[name]):
            return False

    found = False
    for key in ("config_sha256", "config_identity", "config"):
        if key not in value:
            continue
        found = True
        if not _config_identity_matches(value[key], expected[name]):
            return False
    return found


def _source_identity_matches(value: dict[str, object]) -> bool:
    source_archive = value.get("source_archive_sha256")
    source_identity = value.get("source_identity")
    if source_archive is None and source_identity is None:
        return False
    if source_archive is not None and source_archive != SOURCE_ARCHIVE_SHA256:
        return False
    if source_identity is not None and source_identity != SOURCE_ARCHIVE_SHA256:
        return False
    return True


def _valid_complete_marker(path: Path, config_name: str | Iterable[str] | None = CONFIG_NAME) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return False
        if value.get("completion_schema_version") == 2:
            names = _config_names(config_name)
            if len(names) != 1:
                return False
            from lean_reward_hacking.campaign import _completion_valid

            return _completion_valid(
                path,
                config_run_sha256(names[0]),
                SOURCE_ARCHIVE_SHA256,
                require_table_checksums=True,
            )
        if (
            value.get("state") == "complete"
            and path.name != "RUN_COMPLETE.json"
            and value.get("marker_schema_version") == 2
            and _source_identity_matches(value)
            and _marker_config_matches(value, config_name)
        ):
            return True
        # CheckpointStore's run marker binds completion through the run id,
        # checkpoint reference, and configuration identity.
        names = _config_names(config_name)
        if len(names) != 1 or not _source_identity_matches(value) or not _marker_config_matches(value, names):
            return False
        expected = config_identity(names[0])
        checkpoint = value.get("checkpoint")
        marker_run_id = str(value.get("run_id") or "")
        return (
            path.name == "RUN_COMPLETE.json"
            and bool(marker_run_id)
            and value.get("config_identity") == expected["config_sha256"]
            and isinstance(checkpoint, dict)
            and checkpoint.get("run_id") == marker_run_id
            and checkpoint.get("config_identity") == expected["config_sha256"]
            and checkpoint.get("source_identity") == SOURCE_ARCHIVE_SHA256
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
        return False


def completed(name: str, config_name: str | Iterable[str] | None = CONFIG_NAME) -> bool:
    if name in {"completed.json", "RUN_COMPLETE.json", "run_complete.json"}:
        return any(_valid_complete_marker(path, config_name) for path in _complete_marker_paths(REMOTE_RUN_DIR))
    return _valid_complete_marker(marker(name), config_name)


def any_config_completed(config_name: str) -> bool:
    experiment = config_experiment(config_name)
    root = REMOTE_ROOT / "runs" / experiment
    return any(
        _valid_complete_marker(path, config_name)
        for pattern in (
            "*/completed.json", "*/markers/completed.json",
            "*/RUN_COMPLETE.json", "*/markers/RUN_COMPLETE.json",
        )
        for path in root.glob(pattern)
    )


def write_marker(
    name: str,
    payload: dict[str, object],
    config_name: str | Iterable[str] = CONFIG_NAME,
) -> None:
    names = _config_names(config_name)
    if not names:
        raise RuntimeError("marker needs at least one unique configuration")
    identities = {item: config_identity(item) for item in names}
    identity = identities[names[0]]
    value = {
        **payload,
        "state": "complete",
        "marker_schema_version": 2,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "config_sha256": identity["config_sha256"],
        "config_identity": identity,
        "configs": list(names),
        "config_identities": identities,
    }
    atomic_write_json(marker(name), value)


def existing_outputs(
    *,
    validation_config: str = CONFIG_NAME,
    run_configs: str | Iterable[str] = CONFIG_NAME,
    export_label: str = EXPERIMENT,
) -> dict[str, bool]:
    state = {
        "validation": completed("validation.done.json", validation_config),
        "run": completed("completed.json", run_configs),
        "export": export_completed(export_label, run_configs),
    }
    print("resume state:", json.dumps(state, sort_keys=True))
    return state


def run_cli(command: str, *arguments: str) -> None:
    environment = os.environ.copy()
    environment["LRH_RUNTIME"] = "colab"
    environment["LRH_RUN_ID"] = RUN_ID
    environment["RH_SOURCE_COMMIT"] = SOURCE_COMMIT
    environment["RH_SOURCE_ARCHIVE_SHA256"] = SOURCE_ARCHIVE_SHA256
    environment["PYTHONPATH"] = str(SOURCE_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    command_line = [sys.executable, "-m", "lean_reward_hacking.cli", command, *map(str, arguments)]
    subprocess.run(command_line, cwd=str(SOURCE_ROOT), env=environment, check=True)


def validate_compact_bundle(bundle: Path) -> None:
    from lean_reward_hacking.schemas import require_valid_bundle

    require_valid_bundle(bundle, strict=True)


def _export_marker_name(label: str) -> str:
    return f"export.{_safe_path_component(label, field='export label')}.done.json"


def _validate_bundle_identity(bundle: Path, expected_scope: str | None) -> dict[str, object]:
    manifest_path = bundle / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("compact manifest must be an object")
    if value.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("compact manifest source identity differs from this notebook")
    if expected_scope is not None:
        observed = value.get("analysis_experiment") or value.get("experiment_scope")
        if observed != expected_scope:
            raise RuntimeError(
                f"compact manifest scope {observed!r} differs from {expected_scope!r}"
            )
    return value


def _persist_bundle(
    local_bundle: Path,
    label: str,
    expected_scope: str | None = None,
) -> tuple[Path, str]:
    safe_label = _safe_path_component(label, field="export label")
    validate_compact_bundle(local_bundle)
    _validate_bundle_identity(local_bundle, expected_scope)
    manifest_sha256 = sha256_file(local_bundle / "manifest.json")
    destination = (
        REMOTE_ROOT
        / "compact"
        / SOURCE_ARCHIVE_SHA256
        / safe_label
        / manifest_sha256[:16]
    )
    if destination.is_dir():
        validate_compact_bundle(destination)
        _validate_bundle_identity(destination, expected_scope)
        if sha256_file(destination / "manifest.json") != manifest_sha256:
            raise RuntimeError("persisted bundle path has a conflicting manifest")
        return destination, manifest_sha256
    stage = destination.with_name(
        destination.name + f".partial.{os.getpid()}.{time.time_ns()}"
    )
    stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_bundle, stage)
    validate_compact_bundle(stage)
    _validate_bundle_identity(stage, expected_scope)
    if sha256_file(stage / "manifest.json") != manifest_sha256:
        raise RuntimeError("Drive copy changed the compact manifest")
    try:
        os.replace(stage, destination)
    except FileExistsError:
        validate_compact_bundle(destination)
    return destination, manifest_sha256


def export_completed(
    label: str,
    config_names: str | Iterable[str] = CONFIG_NAME,
) -> bool:
    path = marker(_export_marker_name(label))
    if not _valid_complete_marker(path, config_names):
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        bundle = Path(str(value["bundle"]))
        expected = str(value["bundle_manifest_sha256"])
        validate_compact_bundle(bundle)
        return sha256_file(bundle / "manifest.json") == expected
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def export_compact_bundle(
    *,
    label: str = EXPERIMENT,
    experiment_scope: str | None = EXPERIMENT,
    config_names: str | Iterable[str] = CONFIG_NAME,
) -> Path:
    local_name = experiment_scope or "combined"
    local_bundle = Path("/content/rh_compact_bundle") / local_name
    run_cli("export", "--remote-root", str(REMOTE_ROOT), "--local-bundle", str(local_bundle))
    persisted, manifest_sha256 = _persist_bundle(
        local_bundle, label, expected_scope=experiment_scope
    )
    write_marker(
        _export_marker_name(label),
        {
            "bundle": str(persisted),
            "bundle_manifest_sha256": manifest_sha256,
            "experiment_scope": experiment_scope,
        },
        config_name=config_names,
    )
    print("persisted compact bundle:", persisted)
    return persisted


def resolved_export_bundle(
    label: str,
    config_names: str | Iterable[str] = CONFIG_NAME,
) -> tuple[Path, str]:
    """Resolve one identity-bound persisted export and verify it again."""

    safe_label = _safe_path_component(label, field="export label")
    path = marker(_export_marker_name(safe_label))
    if not _valid_complete_marker(path, config_names):
        raise RuntimeError(f"export marker is missing or stale for {label}")
    value = json.loads(path.read_text(encoding="utf-8"))
    bundle = Path(str(value["bundle"]))
    manifest_sha256 = str(value["bundle_manifest_sha256"])
    validate_compact_bundle(bundle)
    _validate_bundle_identity(bundle, value.get("experiment_scope"))
    if sha256_file(bundle / "manifest.json") != manifest_sha256:
        raise RuntimeError(f"persisted bundle manifest changed for {label}")
    return bundle, manifest_sha256


def archive_export_for_download(
    label: str,
    config_names: str | Iterable[str] = CONFIG_NAME,
) -> dict[str, str]:
    """Create a deterministic, checksummed ``.tgz`` for local import."""

    safe_label = _safe_path_component(label, field="export label")
    bundle, manifest_sha256 = resolved_export_bundle(safe_label, config_names)
    download_dir = WORK_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / f"{safe_label}-{manifest_sha256[:16]}.tgz"
    metadata_path = archive_path.with_suffix(archive_path.suffix + ".json")
    if archive_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("manifest_sha256") == manifest_sha256
            and metadata.get("archive_sha256") == sha256_file(archive_path)
        ):
            return {str(key): str(value) for key, value in metadata.items()}
    temporary = archive_path.with_name(archive_path.name + f".partial.{os.getpid()}")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for source in sorted(bundle.iterdir()):
                    if source.is_symlink() or not source.is_file():
                        raise RuntimeError(f"compact bundle is not flat: {source}")
                    payload = source.read_bytes()
                    info = tarfile.TarInfo(f"{safe_label}/{source.name}")
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, archive_path)
    metadata = {
        "label": safe_label,
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "manifest_sha256": manifest_sha256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


_drive_mount()
materialize_source()
REMOTE_RUN_DIR.mkdir(parents=True, exist_ok=True)
REMOTE_MARKER_DIR.mkdir(parents=True, exist_ok=True)
'''


INSTALL_CELL = r'''# [RH-PACKAGES] Install the checked-in lock verbatim and assert every version.
install_and_record_versions()
'''


PROVENANCE_CELL = r'''# [RH-PROVENANCE] Runtime, accelerator, package, seed, config, and source identity.
record_provenance(CONFIG_NAME, seeds=configured_seeds(CONFIG_NAME))
print(json.dumps(_runtime_info(), indent=2, sort_keys=True))
'''


def _runtime_cell(snapshot: dict[str, object], experiment: str, config_name: str, requirements_name: str) -> str:
    source_b64 = str(snapshot["archive_b64"])
    return (
        COMMON_RUNTIME.replace("__SOURCE_COMMIT__", str(snapshot["commit"]))
        .replace("__SOURCE_DIRTY__", repr(bool(snapshot["dirty"])))
        .replace("__SOURCE_ARCHIVE_SHA256__", str(snapshot["archive_sha256"]))
        .replace("__SOURCE_ARCHIVE_B64__", source_b64)
        .replace(
            "__SOURCE_FILE_HASHES__",
            json.dumps(snapshot["file_hashes"], sort_keys=True, separators=(",", ":")),
        )
        .replace("__GENERATOR_VERSION__", GENERATOR_VERSION)
        .replace("__EXPERIMENT__", experiment)
        .replace("__CONFIG_NAME__", config_name)
        .replace("__REQUIREMENTS_NAME__", requirements_name)
    )


def _common_cells(snapshot: dict[str, object], experiment: str, config_name: str, requirements_name: str, *, include_install: bool = True) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = [
        _markdown(
            """This notebook is a restartable Google Colab workflow.

The embedded source snapshot is identified by its Git commit and SHA-256.
Drive markers and hashes control resume behavior. Full work starts only after the tiny validation gate.""",
            f"{experiment}-intro",
        ),
        _code(_runtime_cell(snapshot, experiment, config_name, requirements_name), f"{experiment}-bootstrap"),
    ]
    if include_install:
        cells.append(_code(INSTALL_CELL, f"{experiment}-install"))
    cells.append(_code(PROVENANCE_CELL, f"{experiment}-provenance"))
    return cells


def _tiny_gate(config_name: str) -> str:
    return f'''# [RH-TINY-GATE] Required before any full run.
if not completed("validation.done.json", "{config_name}"):
    run_cli("tiny-validate", "--config", str(config_path("{config_name}")), "--remote-root", str(REMOTE_ROOT))
    write_marker("validation.done.json", {{"config": config_identity("{config_name}"), "gate": {{"seed": 0, "updates": 2, "episodes_per_update": 2, "paired_eval_count": 4, "checkpoint_every": 1, "basin_grid": [1, 1], "perturbations": 1}}}}, config_name="{config_name}")
else:
    print("tiny validation marker is valid; skipping the gate")
'''


def _full_run(
    config_names: list[str],
    *,
    validation_config: str,
    marker_name: str = "completed.json",
) -> str:
    lines = ["# [RH-FULL-RUN] The CLI reads sharding, seeds, device, and resume policy from each TOML."]
    lines.extend(
        [
            f'if not completed("validation.done.json", "{validation_config}"):',
            '    raise RuntimeError("valid tiny-validation marker is required before full work")',
            f"pending_configs = [name for name in {config_names!r} if not config_completed(name)]",
            "if pending_configs:",
            "    print(json.dumps(assert_free_colab_resources(), indent=2, sort_keys=True))",
            "else:",
            '    print("all campaign configs already complete; no accelerator work requested")',
            f"existing_outputs(validation_config={validation_config!r}, run_configs={config_names!r})",
        ]
    )
    for config_name in config_names:
        lines.extend(
            [
                f'if not config_completed("{config_name}"):',
                f'    run_cli("colab-run", "--config", str(config_path("{config_name}")), "--remote-root", str(REMOTE_ROOT))',
                "else:",
                f'    print("{config_name} completed marker is valid; skipping full work")',
            ]
        )
    lines.extend(
        [
            f'if not all(config_completed(name) for name in {config_names!r}):',
            f'    raise RuntimeError("one or more Colab run markers are incomplete after colab-run")',
            f'if not completed("{marker_name}", {config_names!r}):',
            f'    write_marker("{marker_name}", {{"campaign_complete": True}}, config_name={config_names!r})',
            "else:",
            f'    print("completed marker is valid; skipping full work")',
        ]
    )
    return "\n".join(lines)


def _export_cell(
    *,
    label: str,
    experiment_scope: str | None,
    config_names: list[str],
) -> str:
    return f'''# [RH-EXPORT] Persist a strict compact bundle in Drive.
if export_completed({label!r}, {config_names!r}):
    print("persisted export marker and bundle are valid; skipping export")
else:
    export_compact_bundle(label={label!r}, experiment_scope={experiment_scope!r}, config_names={config_names!r})
'''


def _toy_notebook(snapshot: dict[str, object]) -> dict[str, object]:
    cells = _common_cells(snapshot, "toy_fixed", "toy_colab.toml", "requirements-colab.txt")
    cells.extend(
        [
            _code(_tiny_gate("toy_smoke.toml"), "toy-tiny-gate"),
            _markdown(
                "The fixed-objective replicas and the harmful-goal/audit-sensitivity basin scan share the same source snapshot. Each TOML is immutable for this run.",
                "toy-plan",
            ),
            _code(
                _full_run(
                    ["toy_colab.toml", "basin_colab.toml"],
                    validation_config="toy_smoke.toml",
                ),
                "toy-full-run",
            ),
            _code(
                _export_cell(
                    label="toy_fixed",
                    experiment_scope="toy_fixed",
                    config_names=["toy_colab.toml"],
                )
                + "\n"
                + _export_cell(
                    label="toy_basin",
                    experiment_scope="toy_basin",
                    config_names=["basin_colab.toml"],
                ),
                "toy-export",
            ),
        ]
    )
    return _notebook(cells, "toy_fixed")


def _generic_notebook(snapshot: dict[str, object]) -> dict[str, object]:
    cells = _common_cells(snapshot, "generic_mlp", "generic_colab.toml", "requirements-colab.txt")
    cells.extend(
        [
            _code(_tiny_gate("generic_colab.toml"), "generic-tiny-gate"),
            _markdown(
                "The generic control receives the same episode fields and reward. Its plain MLP has no named goal or oversight-gate modules. Audit-cue swaps and ablations are recorded by the project API.",
                "generic-plan",
            ),
            _code(
                _full_run(["generic_colab.toml"], validation_config="generic_colab.toml"),
                "generic-full-run",
            ),
            _code(
                _export_cell(
                    label="generic_mlp",
                    experiment_scope="generic_mlp",
                    config_names=["generic_colab.toml"],
                ),
                "generic-export",
            ),
        ]
    )
    return _notebook(cells, "generic_mlp")


def _perturbation_notebook(snapshot: dict[str, object]) -> dict[str, object]:
    cells = _common_cells(snapshot, "toy_perturbation", "perturbation_colab.toml", "requirements-colab.txt")
    cells.extend(
        [
            _code(_tiny_gate("perturbation_colab.toml"), "perturbation-tiny-gate"),
            _markdown(
                "Branches are selected from completed source checkpoints by the registered mode rule. The original data, reward, and optimizer continuation are restored after each intervention. Sham and frozen controls remain separate from recovery estimates.",
                "perturbation-plan",
            ),
            _code(
                '''# [RH-PERTURBATION-PREREQUISITE] Source checkpoints must be complete and identity-bound.
if not any_config_completed("toy_colab.toml"):
    raise RuntimeError("toy_colab.toml must complete before perturbation branches can start")
''',
                "perturbation-prerequisite",
            ),
            _code(
                _full_run(
                    ["perturbation_colab.toml"],
                    validation_config="perturbation_colab.toml",
                ),
                "perturbation-full-run",
            ),
            _code(
                _export_cell(
                    label="toy_perturbation",
                    experiment_scope="toy_perturbation",
                    config_names=["perturbation_colab.toml"],
                ),
                "perturbation-export",
            ),
        ]
    )
    return _notebook(cells, "toy_perturbation")


def _analysis_notebook(snapshot: dict[str, object]) -> dict[str, object]:
    cells = _common_cells(snapshot, "analysis_export", "toy_colab.toml", "requirements-colab.txt")
    required_configs = [
        "toy_colab.toml",
        "basin_colab.toml",
        "generic_colab.toml",
        "perturbation_colab.toml",
    ]
    cells.extend(
        [
            _code(
                f'''# [RH-ANALYSIS-GATE] Require each source-bound campaign completion marker.
REQUIRED_ANALYSIS_CONFIGS = {required_configs!r}
missing_campaigns = [name for name in REQUIRED_ANALYSIS_CONFIGS if not any_config_completed(name)]
if missing_campaigns:
    raise RuntimeError(f"analysis prerequisites are incomplete: {{missing_campaigns}}")
print({{"completed_campaigns": REQUIRED_ANALYSIS_CONFIGS}})
''',
                "analysis-tiny-gate",
            ),
            _markdown(
                "Analysis consumes completed remote markers and streams compact tables. The dip test, mixture BIC bootstrap, threshold sensitivity, attraction diagnostics, and figure sidecars are generated from saved results.",
                "analysis-plan",
            ),
            _code(
                _export_cell(
                    label="toy_fixed",
                    experiment_scope="toy_fixed",
                    config_names=["toy_colab.toml"],
                )
                + "\n"
                + _export_cell(
                    label="toy_basin",
                    experiment_scope="toy_basin",
                    config_names=["basin_colab.toml"],
                )
                + "\n"
                + _export_cell(
                    label="generic_mlp",
                    experiment_scope="generic_mlp",
                    config_names=["generic_colab.toml"],
                )
                + "\n"
                + _export_cell(
                    label="toy_perturbation",
                    experiment_scope="toy_perturbation",
                    config_names=["perturbation_colab.toml"],
                )
                + "\n"
                + _export_cell(
                    label="analysis_all",
                    experiment_scope=None,
                    config_names=required_configs,
                ),
                "analysis-run",
            ),
            _code(
                '''# [RH-DOWNLOAD-HANDOFF] Download four strict, content-addressed compact bundles.
DOWNLOAD_COMPACT_EXPORTS = True
HANDOFF_EXPORTS = [
    ("toy_fixed", ["toy_colab.toml"]),
    ("toy_basin", ["basin_colab.toml"]),
    ("generic_mlp", ["generic_colab.toml"]),
    ("toy_perturbation", ["perturbation_colab.toml"]),
]
handoff_records = [archive_export_for_download(label, names) for label, names in HANDOFF_EXPORTS]
atomic_write_json(WORK_DIR / "downloads" / "HANDOFF.json", {
    "schema_version": 1,
    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    "exports": handoff_records,
})
print(json.dumps(handoff_records, indent=2, sort_keys=True))
if DOWNLOAD_COMPACT_EXPORTS:
    from google.colab import files
    for record in handoff_records:
        files.download(record["archive"])
''',
                "analysis-download-handoff",
            ),
        ]
    )
    return _notebook(cells, "analysis_export")


LM_RUNTIME = r'''# [RH-LM-RUNTIME] Model-free package validation and explicit live-run gate.
from dataclasses import replace

from lean_reward_hacking.lm import DatasetManifest, generate_dataset
from lean_reward_hacking.lm_training import (
    LMRunLayout,
    LMTrainingConfig,
    assert_primary_prompt_boundary,
    build_audited_alignment_dataset,
    build_evaluation_suite,
    build_procedural_sft_dataset,
    run_lm_workflow,
    validate_accelerator,
    workflow_plan,
)

# The checked-in package remains the default.  A live run needs both exact
# environment confirmations and still passes the free-runtime resource gate.
RUN_FULL_LM = False
if os.environ.get("RH_RUN_FULL_LM") == "I_UNDERSTAND_OPEN_WEIGHT_RUN":
    RUN_FULL_LM = True
CONFIRM_LM_DOWNLOAD = os.environ.get("RH_CONFIRM_LM_DOWNLOAD") == "I_UNDERSTAND_LM_DOWNLOAD"
RUN_LM_BRANCHES = True
LM_CONFIG_PATH = config_path("lm_colab.toml")
LM_CONFIG_OBJECT = LMTrainingConfig.from_toml(LM_CONFIG_PATH)
import tomllib
LM_CONFIG_RAW = tomllib.loads(LM_CONFIG_PATH.read_text(encoding="utf-8"))
LM_LAYOUT = LMRunLayout(REMOTE_ROOT, run_id=RUN_ID)
LM_RESOURCE_PATH = SOURCE_ROOT / "reports" / "LM_RESOURCE_REQUIREMENTS.json"
if not LM_RESOURCE_PATH.is_file():
    raise RuntimeError("checked-in LM_RESOURCE_REQUIREMENTS.json is required")
LM_RESOURCE_REQUIREMENTS = json.loads(LM_RESOURCE_PATH.read_text(encoding="utf-8"))
if LM_RESOURCE_REQUIREMENTS.get("config_file_sha256") != sha256_file(LM_CONFIG_PATH):
    raise RuntimeError("LM resource account is stale relative to lm_colab.toml")
if LM_RESOURCE_REQUIREMENTS.get("training_config_sha256") != LM_CONFIG_OBJECT.config_sha256:
    raise RuntimeError("LM resource account has a stale validated config identity")
if float(LM_RESOURCE_REQUIREMENTS.get("minimum_host_ram_gib", -1)) != float(
    LM_CONFIG_RAW.get("minimum_host_ram_gib", -2)
):
    raise RuntimeError("LM resource account and lm_colab.toml disagree on host RAM")

# Exercise the pinned TRL configuration API without loading a tokenizer,
# model, dataset, accelerator kernel, or remote artifact.
from trl import GRPOConfig, SFTConfig

_grpo_probe = GRPOConfig(
    output_dir=str(WORK_DIR / "lm-grpo-config-probe"),
    max_steps=1,
    per_device_train_batch_size=LM_CONFIG_OBJECT.per_device_train_batch_size,
    num_generations=LM_CONFIG_OBJECT.num_generations,
    gradient_accumulation_steps=1,
    report_to=[],
    fp16=False,
    bf16=False,
)
_sft_probe = SFTConfig(
    output_dir=str(WORK_DIR / "lm-sft-config-probe"),
    max_steps=1,
    per_device_train_batch_size=1,
    report_to=[],
    fp16=False,
    bf16=False,
)
assert int(_grpo_probe.per_device_train_batch_size) % int(_grpo_probe.num_generations) == 0

_fixture = generate_dataset(DatasetManifest(train_count=16, eval_pair_count=4))
assert_primary_prompt_boundary(_fixture)
assert build_procedural_sft_dataset(_fixture)
assert build_audited_alignment_dataset(_fixture)
assert build_evaluation_suite(_fixture).paired
LM_WORKFLOW_PLAN = workflow_plan(
    LM_CONFIG_OBJECT,
    layout=LM_LAYOUT,
    source_identity=SOURCE_ARCHIVE_SHA256,
    run_id=RUN_ID,
)
resource_path = LM_LAYOUT.run_dir / "RESOURCE_REQUIREMENTS.json"
atomic_write_json(resource_path, {
    "status": "package_ready" if not RUN_FULL_LM else "live_run_requested",
    "weights_downloaded": False,
    "resource_account": LM_RESOURCE_REQUIREMENTS,
    "workflow_plan": LM_WORKFLOW_PLAN,
    "model_revision": LM_CONFIG_OBJECT.model_revision,
    "tokenizer_revision": LM_CONFIG_OBJECT.tokenizer_revision,
    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
})
write_marker(
    "package_validation.done.json",
    {
        "package_validation": "passed",
        "weights_downloaded": False,
        "dataset_downloaded": False,
        "trl_grpo_batch_divisible": True,
        "workflow_plan_sha256": _sha256_bytes(
            json.dumps(LM_WORKFLOW_PLAN, sort_keys=True).encode("utf-8")
        ),
    },
    config_name="lm_colab.toml",
)
print(json.dumps(LM_RESOURCE_REQUIREMENTS, indent=2, sort_keys=True))

if RUN_FULL_LM:
    if not CONFIRM_LM_DOWNLOAD:
        raise RuntimeError(
            "set RH_CONFIRM_LM_DOWNLOAD=I_UNDERSTAND_LM_DOWNLOAD for the explicit weight-download gate"
        )
    resource_decision = assert_free_colab_resources(
        minimum_gpu_memory_gib=14.0,
        minimum_host_ram_gib=float(LM_CONFIG_RAW["minimum_host_ram_gib"]),
        minimum_drive_free_gib=float(LM_RESOURCE_REQUIREMENTS["minimum_drive_free_gib"]),
        allowed_accelerators=("T4", "L4"),
    )
    validate_accelerator(resource_decision["runtime"])
    live_config = replace(
        LM_CONFIG_OBJECT,
        run_full_lm=True,
        source_identity=SOURCE_ARCHIVE_SHA256,
    )
    local_lm_bundle = Path("/content/rh_compact_bundle/red_token_lm")
    live_result = run_lm_workflow(
        live_config,
        layout=LM_LAYOUT,
        source_identity=SOURCE_ARCHIVE_SHA256,
        run_id=RUN_ID,
        requirements=SOURCE_ROOT / "requirements-lm-colab.txt",
        runtime=resource_decision["runtime"],
        download_weights=True,
        compact_destination=local_lm_bundle,
        run_branches=RUN_LM_BRANCHES,
        git_commit=SOURCE_COMMIT,
    )
    validate_compact_bundle(local_lm_bundle)
    persisted_lm_bundle, persisted_lm_manifest = _persist_bundle(
        local_lm_bundle, "red_token_lm", expected_scope="red_token_lm"
    )
    write_marker(
        _export_marker_name("red_token_lm"),
        {
            "bundle": str(persisted_lm_bundle),
            "bundle_manifest_sha256": persisted_lm_manifest,
            "experiment_scope": "red_token_lm",
            "weights_downloaded": True,
            "live_result": live_result,
        },
        config_name="lm_colab.toml",
    )
    print({"live_result": live_result, "persisted_bundle": str(persisted_lm_bundle)})
'''


def _lm_notebook(snapshot: dict[str, object]) -> dict[str, object]:
    cells = _common_cells(snapshot, "red_token_lm", "lm_colab.toml", "requirements-lm-colab.txt")
    cells.extend(
        [
            _code(
                '''# [RH-LM-TINY-GATE] Synthetic contract check, with no model or dataset download.
if not completed("validation.done.json", "lm_colab.toml"):
    tiny_messages = [{"role": "user", "content": "Complete this fictional user task and report the result."}]
    assert tiny_messages[0]["role"] == "user"
    assert "private goal" not in tiny_messages[0]["content"].lower()
    write_marker("validation.done.json", {"gate": {"synthetic_messages": 1, "weights_downloaded": False, "dataset_downloaded": False}}, config_name="lm_colab.toml")
else:
    print("LM tiny validation marker is valid; skipping the gate")
''',
                "lm-tiny-gate",
            ),
            _code(LM_RUNTIME, "lm-resource-gate"),
            _markdown(
                "The checked-in package validates without downloading weights. Primary alignment and evaluation prompts describe the task and oversight condition without a private-goal instruction. Hidden transfers remain in the evaluation logger. A live run requires both environment confirmations and the free-runtime gate. Paid compute still requires separate approval.",
                "lm-plan",
            ),
            _code(
                '''# [RH-LM-PROMPT-CONTRACT] Inspect the primary prompt boundary from package code.
from lean_reward_hacking.lm import (
    PRIMARY_CONDITION,
    PRIVATE_GOAL_SENTENCE,
    build_model_messages,
)

_primary_episode = _fixture.eval_pairs[0][0]
PRIMARY_ALIGNMENT_PROMPT = build_model_messages(
    _primary_episode,
    PRIMARY_CONDITION,
    phase="alignment",
)[0]["content"]
assert PRIVATE_GOAL_SENTENCE not in PRIMARY_ALIGNMENT_PROMPT
assert "private goal" not in PRIMARY_ALIGNMENT_PROMPT.lower()
assert "red_token" not in PRIMARY_ALIGNMENT_PROMPT.lower()
print({
    "model_revision": LM_CONFIG_OBJECT.model_revision,
    "tokenizer_revision": LM_CONFIG_OBJECT.tokenizer_revision,
    "pad_token": "eos_token",
    "padding_side": "left",
    "eos_token": "<|im_end|>",
    "primary_prompt": PRIMARY_ALIGNMENT_PROMPT,
})
''',
                "lm-prompt-contract",
            ),
            _code(
                '''# [RH-LM-RESUME-PLAN] Persist the exact package plan beside future Drive state.
LM_CHECKPOINT_FIELDS = [
    "adapter", "optimizer", "scheduler", "python_rng", "numpy_rng",
    "torch_rng", "cuda_rng", "config_sha256", "source_identity",
]
atomic_write_json(LM_LAYOUT.run_dir / "WORKFLOW_PLAN.json", {
    **LM_WORKFLOW_PLAN,
    "checkpoint_fields": LM_CHECKPOINT_FIELDS,
    "resume": "skip hash-valid stage markers and restore the latest hash-valid full Trainer checkpoint",
    "live_launch_environment": {
        "RH_RUN_FULL_LM": "I_UNDERSTAND_OPEN_WEIGHT_RUN",
        "RH_CONFIRM_LM_DOWNLOAD": "I_UNDERSTAND_LM_DOWNLOAD",
        "RH_COLAB_COMPUTE_TIER": "free",
    },
})
''',
                "lm-resume-plan",
            ),
        ]
    )
    return _notebook(cells, "red_token_lm")


def _notebook(cells: list[dict[str, object]], title: str) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "lean_reward_hacking": {"generator_version": GENERATOR_VERSION, "title": title},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebooks(output_dir: Path = NOTEBOOK_DIR) -> list[Path]:
    snapshot = source_snapshot()
    notebooks = {
        "01_toy_sweep_colab.ipynb": _toy_notebook(snapshot),
        "02_mlp_control_colab.ipynb": _generic_notebook(snapshot),
        "03_perturbation_colab.ipynb": _perturbation_notebook(snapshot),
        "04_analysis_export_colab.ipynb": _analysis_notebook(snapshot),
        "05_lm_workflow_colab.ipynb": _lm_notebook(snapshot),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, notebook in notebooks.items():
        destination = output_dir / name
        payload = (json.dumps(notebook, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
        destination.write_bytes(payload)
        written.append(destination)
    return written


def main() -> int:
    paths = build_notebooks()
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
