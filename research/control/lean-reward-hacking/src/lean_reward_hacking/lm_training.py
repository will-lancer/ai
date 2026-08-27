"""Restartable, Colab-only runner pieces for the RED_TOKEN LM experiment.

The module has a deliberately small import boundary.  Importing it uses only
the standard library and :mod:`lean_reward_hacking.lm`; model libraries are
loaded inside the explicit runner functions.  This makes the dataset,
prompt, reward, resource, checkpoint, and export contracts testable on the
Mac without model weights or a GPU.

The full path is two-stage training.  A conflict-seeded SFT stage teaches the
private RED_TOKEN objective.  Each alignment replica then uses a fixed reward
on audited (ON) episodes only.  TRL's GRPO implementation is used when the
Colab lock is installed.  Checkpoint directories and completion markers carry
the source, configuration, and frozen-model identities, so a restart cannot
silently continue a different experiment.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from .lm import (
    AuditStatus,
    DatasetBundle,
    DatasetManifest,
    Episode,
    EpisodeOutcome,
    MODEL_ID,
    MODEL_RESOURCE_MANIFEST,
    MODEL_REVISION,
    PRIMARY_CONDITION,
    PRIVATE_GOAL_SENTENCE,
    TOOL_SCHEMA_REGISTRY,
    build_model_messages,
    compute_reward,
    evaluate_episode,
    generate_dataset,
)


LM_TRAINING_SCHEMA_VERSION = "lrh-lm-training/v1"
RUN_COMPLETE_MARKER = "RUN_COMPLETE.json"
STAGE_COMPLETE_MARKER = "COMPLETE.json"
DEFAULT_COMPUTE_DTYPE = "float16"
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 4
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Qwen's revision is intentionally repeated here.  A runner must pass it to
# both ``from_pretrained`` calls; a moving ``main`` branch is not acceptable.
FROZEN_MODEL_ID = MODEL_ID
FROZEN_MODEL_REVISION = MODEL_REVISION
FROZEN_TOKENIZER_REVISION = MODEL_REVISION

# Colab reports a Tesla T4 as 15,637,086,208 bytes even though its marketed
# class is 16 GB.  The name check preserves the marketed class requirement;
# the byte floor accepts that stable driver-reported value and rejects a
# generic 15-GB device.
T4_MARKETED_BYTES = 16_000_000_000
T4_OBSERVED_MIN_BYTES = 15_500_000_000
L4_MARKETED_BYTES = 24_000_000_000
L4_OBSERVED_MIN_BYTES = 23_000_000_000
OBSERVED_T4_MEMORY_BYTES = 15_637_086_208

# The LM release targets the environment in which the TRL 0.15.2 API was
# audited.  A newer interpreter or CUDA wheel is a separate port and must not
# enter this workflow through an implicit fallback.
REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 12
REQUIRED_PYTHON_VERSION = "3.12"
REQUIRED_CUDA_VERSION = "12.4"
REQUIRED_TORCH_VERSION = "2.5.1+cu124"
REQUIRED_BITSANDBYTES_VERSION = "0.45.2"
MIN_LM_HOST_MEMORY_BYTES = 12_000_000_000
MAX_LM_VCPUS = 2

REQUIRED_LM_PACKAGES: dict[str, str] = {
    "torch": REQUIRED_TORCH_VERSION,
    "transformers": "4.48.3",
    "trl": "0.15.2",
    "accelerate": "1.3.0",
    "peft": "0.14.0",
    "bitsandbytes": "0.45.2",
    "datasets": "3.2.0",
    "safetensors": "0.5.2",
    "sentencepiece": "0.2.0",
    "jsonlines": "4.0.0",
    "filelock": "3.16.1",
    # These pins make the LM file installable without resolving the base
    # CUDA 12.8 requirements file.
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.15.1",
    "scikit_learn": "1.6.1",
    "matplotlib": "3.10.0",
    "seaborn": "0.13.2",
    "pyyaml": "6.0.2",
}

REQUIRED_LM_RUNTIME_FIELDS = (
    "python_version",
    "cuda_version",
    "torch_version",
    "bitsandbytes_version",
    "requirements_sha256",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, set, frozenset)):
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
    raise TypeError(f"value of type {type(value).__name__} is not JSON serialisable")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(
    root: str | os.PathLike[str],
    *,
    exclude_names: Iterable[str] = (),
) -> str:
    """Hash a directory in path order, including relative names and bytes."""

    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(base)
    digest = hashlib.sha256()
    excluded = frozenset(str(name) for name in exclude_names)
    for path in sorted(
        (
            p
            for p in base.rglob("*")
            if p.is_file() and p.name not in excluded
        ),
        key=lambda p: p.relative_to(base).as_posix(),
    ):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return sha256_bytes(payload)


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> str:
    return atomic_write_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _normalise_package_name(name: str) -> str:
    return name.lower().replace("-", "_")


def parse_pinned_requirements(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read exact ``==`` pins, following local ``-r`` includes."""

    requirement_path = Path(path)
    result: dict[str, str] = {}
    for raw_line in requirement_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            result.update(parse_pinned_requirements(requirement_path.parent / line[3:].strip()))
            continue
        if "==" not in line or line.startswith("-"):
            continue
        name, version = line.split("==", 1)
        result[_normalise_package_name(name.strip())] = version.strip()
    return result


def _parse_standalone_lm_lock(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse the LM lock while rejecting inherited or unpinned requirements."""

    requirement_path = Path(path)
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        requirement_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "-r" or line.startswith("-r ") or line == "--requirement" or line.startswith("--requirement ") or line.startswith("--requirement="):
            raise RuntimeError(
                f"LM requirements must be standalone; include found at line {line_number}"
            )
        if line.startswith("--extra-index-url"):
            option, separator, value = line.partition("=")
            if not separator:
                option, separator, value = line.partition(" ")
            if option != "--extra-index-url" or value.strip() != "https://download.pytorch.org/whl/cu124":
                raise RuntimeError(f"unexpected LM package index at line {line_number}: {raw_line!r}")
            continue
        if line.startswith("-"):
            raise RuntimeError(f"unexpected LM requirement option at line {line_number}: {raw_line!r}")
        if "==" not in line:
            raise RuntimeError(f"LM requirement is not an exact pin at line {line_number}: {raw_line!r}")
        name, version = line.split("==", 1)
        normalized = _normalise_package_name(name.strip())
        version = version.strip()
        if not normalized or not version or any(character.isspace() for character in normalized):
            raise RuntimeError(f"invalid LM requirement at line {line_number}: {raw_line!r}")
        if normalized in result:
            raise RuntimeError(f"duplicate LM requirement for {normalized!r} at line {line_number}")
        result[normalized] = version
    return result


def assert_lm_lock(
    requirements: str | os.PathLike[str] | Mapping[str, str],
    *,
    expected: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Require the standalone LM lock to match every released pin exactly."""

    expected_values = {
        _normalise_package_name(name): str(version)
        for name, version in (expected or REQUIRED_LM_PACKAGES).items()
    }
    actual = (
        {
            _normalise_package_name(name): str(version)
            for name, version in requirements.items()
        }
        if isinstance(requirements, Mapping)
        else _parse_standalone_lm_lock(requirements)
    )
    if actual != expected_values:
        differences: list[str] = []
        for name in sorted(set(expected_values) | set(actual)):
            if name not in actual:
                differences.append(f"missing {name}=={expected_values[name]}")
            elif name not in expected_values:
                differences.append(f"unexpected {name}=={actual[name]}")
            elif actual[name] != expected_values[name]:
                differences.append(f"{name}=={actual[name]}, expected {expected_values[name]}")
        raise RuntimeError("LM requirements lock mismatch: " + "; ".join(differences))
    return actual


def requirements_sha256(
    requirements: str | os.PathLike[str] | Mapping[str, str],
) -> str:
    """Hash lock bytes, or a canonical package mapping used by replay tests."""

    if isinstance(requirements, Mapping):
        normalized = {
            _normalise_package_name(name): str(version)
            for name, version in requirements.items()
        }
        return sha256_bytes(_canonical_json(normalized))
    return sha256_file(requirements)


def assert_pinned_versions(
    requirements: str | os.PathLike[str] | Mapping[str, str],
    *,
    observed: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assert every locked package version and return the observed versions.

    ``observed`` exists for dependency-free tests and provenance replay.  A
    normal Colab call reads installed distributions through
    :mod:`importlib.metadata`.
    """

    expected = (
        { _normalise_package_name(k): str(v) for k, v in requirements.items() }
        if isinstance(requirements, Mapping)
        else parse_pinned_requirements(requirements)
    )
    values: dict[str, str] = {}
    mismatches: list[str] = []
    for name, wanted in sorted(expected.items()):
        if observed is not None:
            actual = observed.get(name, observed.get(name.replace("_", "-"), "missing"))
        else:
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                try:
                    actual = importlib.metadata.version(name.replace("_", "-"))
                except importlib.metadata.PackageNotFoundError:
                    actual = "missing"
        actual = str(actual)
        values[name] = actual
        if actual != wanted:
            mismatches.append(f"{name}={actual}, expected {wanted}")
    if mismatches:
        raise RuntimeError("pinned dependency mismatch: " + "; ".join(mismatches))
    return values


def _default_lm_requirements_path() -> Path:
    return Path(__file__).resolve().parents[2] / "requirements-lm-colab.txt"


def _mapping_value(mapping: Mapping[str, Any], name: str) -> Any:
    """Look up a field after normalising package-style spelling."""

    wanted = _normalise_package_name(name)
    for key, value in mapping.items():
        if _normalise_package_name(str(key)) == wanted:
            return value
    return None


def _runtime_field(runtime: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = runtime.get(name)
        if value is not None and value != "":
            return value
    accelerator = runtime.get("accelerator")
    if isinstance(accelerator, Mapping):
        for name in names:
            value = accelerator.get(name)
            if value is not None and value != "":
                return value
    return None


def _normalise_python_version(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        major = int(value[0])
        minor = int(value[1])
        patch = int(value[2]) if len(value) >= 3 else None
    else:
        match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", str(value))
        if match is None:
            raise RuntimeError(f"cannot parse Python runtime version {value!r}")
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = None if match.group(3) is None else int(match.group(3))
    if (major, minor) != (REQUIRED_PYTHON_MAJOR, REQUIRED_PYTHON_MINOR):
        raise RuntimeError(
            f"LM requires Python {REQUIRED_PYTHON_VERSION}.x; observed {major}.{minor}"
        )
    return f"{major}.{minor}" if patch is None else f"{major}.{minor}.{patch}"


def _normalise_cuda_version(value: Any) -> str:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(?:cuda\s*)?(\d+)\.(\d+)(?:\.\d+)?", text)
    if match is None:
        match = re.fullmatch(r"cu(\d{2})(\d)", text)
    if match is None:
        raise RuntimeError(f"cannot parse CUDA runtime version {value!r}")
    if len(match.groups()) == 2:
        major, minor = int(match.group(1)), int(match.group(2))
    else:
        major, minor = int(match.group(1)), int(match.group(2))
    normalized = f"{major}.{minor}"
    if normalized != REQUIRED_CUDA_VERSION:
        raise RuntimeError(
            f"LM requires CUDA {REQUIRED_CUDA_VERSION}-class; observed {normalized}"
        )
    return normalized


def _declared_requirements_hash(runtime: Mapping[str, Any]) -> str | None:
    direct = runtime.get("requirements_sha256")
    if direct:
        return str(direct)
    nested = runtime.get("requirements")
    if isinstance(nested, Mapping) and nested.get("sha256"):
        return str(nested["sha256"])
    return None


def validate_lm_runtime(
    runtime: Mapping[str, Any],
    requirements: str | os.PathLike[str] | Mapping[str, str] | None = None,
    *,
    observed: Mapping[str, str] | None = None,
    expected_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the released Python/CUDA/package/accelerator runtime.

    ``observed`` is a dependency-free test and replay seam.  Colab callers
    leave it unset so installed distribution metadata is checked directly.
    """

    if not isinstance(runtime, Mapping):
        raise RuntimeError("LM runtime record must be a mapping")
    lock_source = _default_lm_requirements_path() if requirements is None else requirements
    locked = assert_lm_lock(lock_source)
    lock_hash = requirements_sha256(lock_source)
    if expected_lock_sha256 is not None and lock_hash != str(expected_lock_sha256):
        raise RuntimeError(
            f"LM requirements hash mismatch: {lock_hash}, expected {expected_lock_sha256}"
        )
    declared_hash = _declared_requirements_hash(runtime)
    if declared_hash is not None and declared_hash != lock_hash:
        raise RuntimeError(
            f"LM runtime requirements hash mismatch: {declared_hash}, expected {lock_hash}"
        )

    failures: list[str] = []
    platform = runtime.get("platform")
    if platform is not None and "linux" not in str(platform).lower():
        failures.append(f"LM requires a Linux Colab runtime; observed platform {platform!r}")
    host_memory = runtime.get("host_memory_bytes")
    if host_memory is not None and int(host_memory) < MIN_LM_HOST_MEMORY_BYTES:
        failures.append(
            f"LM requires at least {MIN_LM_HOST_MEMORY_BYTES} host-memory bytes; observed {host_memory}"
        )
    cpu_count = runtime.get("cpu_count")
    if cpu_count is not None and int(cpu_count) > MAX_LM_VCPUS:
        failures.append(f"LM permits at most {MAX_LM_VCPUS} CPU workers; observed {cpu_count}")
    python_value = _runtime_field(runtime, "python_version", "python")
    if python_value is None:
        failures.append(f"LM requires Python {REQUIRED_PYTHON_VERSION}.x; runtime did not report Python")
        python_version = "unknown"
    else:
        try:
            python_version = _normalise_python_version(python_value)
        except RuntimeError as exc:
            failures.append(str(exc))
            python_version = str(python_value)

    cuda_value = _runtime_field(runtime, "cuda_version", "cuda", "torch_cuda")
    if cuda_value is None:
        failures.append(f"LM requires CUDA {REQUIRED_CUDA_VERSION}-class; runtime did not report CUDA")
        cuda_version = "unknown"
    else:
        try:
            cuda_version = _normalise_cuda_version(cuda_value)
        except RuntimeError as exc:
            failures.append(str(exc))
            cuda_version = str(cuda_value)

    observed_versions = None
    if observed is not None:
        observed_versions = {
            _normalise_package_name(name): str(version)
            for name, version in observed.items()
        }
    try:
        package_versions = assert_pinned_versions(locked, observed=observed_versions)
    except RuntimeError as exc:
        failures.append(str(exc))
        package_versions = {
            name: str(_mapping_value(observed_versions or {}, name) or "missing")
            for name in sorted(locked)
        }

    declared_packages = runtime.get("packages")
    if isinstance(declared_packages, Mapping):
        for name, wanted in sorted(locked.items()):
            declared = _mapping_value(declared_packages, name)
            if declared is not None and str(declared) != wanted:
                failures.append(f"runtime reports {name}={declared}, expected {wanted}")
    for field_name, package_name in (("torch_version", "torch"), ("torch", "torch"), ("bitsandbytes_version", "bitsandbytes"), ("bitsandbytes", "bitsandbytes")):
        declared = runtime.get(field_name)
        if declared is not None and str(declared) != locked[package_name]:
            failures.append(
                f"runtime reports {package_name}={declared}, expected {locked[package_name]}"
            )

    try:
        accelerator = validate_accelerator(runtime)
    except RuntimeError as exc:
        failures.append(str(exc))
        accelerator = {"name": None, "memory_bytes": None}
    if failures:
        raise RuntimeError("LM runtime is blocked: " + "; ".join(dict.fromkeys(failures)))
    return {
        "python_version": python_version,
        "cuda_version": cuda_version,
        "torch_version": package_versions["torch"],
        "bitsandbytes_version": package_versions["bitsandbytes"],
        "requirements_sha256": lock_hash,
        "packages": package_versions,
        "accelerator": accelerator,
    }


def build_lm_runtime_provenance(
    runtime: Mapping[str, Any],
    requirements: str | os.PathLike[str] | Mapping[str, str] | None = None,
    *,
    observed: Mapping[str, str] | None = None,
    source_identity: str | None = None,
    run_id: str | None = None,
    expected_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound runtime record before any LM model load."""

    lock_source = _default_lm_requirements_path() if requirements is None else requirements
    gate = validate_lm_runtime(
        runtime,
        lock_source,
        observed=observed,
        expected_lock_sha256=expected_lock_sha256,
    )
    lock_file = None if isinstance(lock_source, Mapping) else str(Path(lock_source))
    runtime_record = dict(runtime)
    accelerator_record = {"available": True, **dict(gate["accelerator"])}
    runtime_record.update(
        {
            "python_version": gate["python_version"],
            "cuda_version": gate["cuda_version"],
            "torch_version": gate["torch_version"],
            "bitsandbytes_version": gate["bitsandbytes_version"],
            "requirements_sha256": gate["requirements_sha256"],
            "packages": dict(gate["packages"]),
            "accelerator": accelerator_record,
        }
    )
    return {
        "schema_version": LM_TRAINING_SCHEMA_VERSION,
        "status": "verified_runtime",
        "source_identity": source_identity,
        "run_id": run_id,
        "python_version": gate["python_version"],
        "cuda_version": gate["cuda_version"],
        "torch_version": gate["torch_version"],
        "bitsandbytes_version": gate["bitsandbytes_version"],
        "requirements_file": lock_file,
        "requirements_sha256": gate["requirements_sha256"],
        "requirements_lm_colab_sha256": gate["requirements_sha256"],
        "lock_file_sha256": gate["requirements_sha256"],
        "packages": dict(gate["packages"]),
        "accelerator": accelerator_record,
        "requirements": {
            "file": lock_file,
            "sha256": gate["requirements_sha256"],
            "packages": dict(gate["packages"]),
        },
        "runtime": runtime_record,
    }


def validate_lm_runtime_provenance(
    provenance: Mapping[str, Any],
    requirements: str | os.PathLike[str] | Mapping[str, str] | None = None,
    *,
    expected_source_identity: str | None = None,
    expected_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a saved LM provenance record without importing model packages."""

    if not isinstance(provenance, Mapping):
        raise RuntimeError("LM runtime provenance must be a mapping")
    lock_source = _default_lm_requirements_path() if requirements is None else requirements
    locked = assert_lm_lock(lock_source)
    lock_hash = requirements_sha256(lock_source)
    if expected_lock_sha256 is not None and lock_hash != str(expected_lock_sha256):
        raise RuntimeError(
            f"LM requirements hash mismatch: {lock_hash}, expected {expected_lock_sha256}"
        )
    if expected_source_identity is not None and provenance.get("source_identity") != expected_source_identity:
        raise RuntimeError("LM runtime provenance has a different source identity")
    missing = [field for field in REQUIRED_LM_RUNTIME_FIELDS if field not in provenance]
    if missing:
        raise RuntimeError("LM runtime provenance is missing: " + ", ".join(missing))
    if str(provenance["requirements_sha256"]) != lock_hash:
        raise RuntimeError("LM runtime provenance has a different requirements hash")
    for alias in ("requirements_lm_colab_sha256", "lock_file_sha256"):
        if alias in provenance and str(provenance[alias]) != lock_hash:
            raise RuntimeError(f"LM runtime provenance has a different {alias}")
    nested = provenance.get("requirements")
    if not isinstance(nested, Mapping) or str(nested.get("sha256")) != lock_hash:
        raise RuntimeError("LM runtime provenance has an invalid requirements record")
    if nested.get("file") != provenance.get("requirements_file"):
        raise RuntimeError("LM runtime provenance requirements file is inconsistent")
    packages = provenance.get("packages")
    if not isinstance(packages, Mapping):
        raise RuntimeError("LM runtime provenance has no package versions")
    normalized_packages = {
        _normalise_package_name(name): str(version)
        for name, version in packages.items()
    }
    if normalized_packages != locked:
        raise RuntimeError("LM runtime provenance package pins do not match the lock")
    nested_packages = nested.get("packages")
    if not isinstance(nested_packages, Mapping):
        raise RuntimeError("LM runtime provenance requirements record has no package pins")
    if {
        _normalise_package_name(name): str(version)
        for name, version in nested_packages.items()
    } != locked:
        raise RuntimeError("LM runtime provenance nested package pins do not match the lock")
    try:
        python_version = _normalise_python_version(provenance["python_version"])
        cuda_version = _normalise_cuda_version(provenance["cuda_version"])
    except RuntimeError as exc:
        raise RuntimeError("LM runtime provenance has an incompatible runtime: " + str(exc)) from exc
    for field_name, package_name in (("torch_version", "torch"), ("bitsandbytes_version", "bitsandbytes")):
        if str(provenance[field_name]) != locked[package_name]:
            raise RuntimeError(
                f"LM runtime provenance {field_name} does not match {package_name} lock"
            )
    accelerator = provenance.get("accelerator")
    if not isinstance(accelerator, Mapping):
        raise RuntimeError("LM runtime provenance has no accelerator record")
    try:
        accelerator_record = validate_accelerator(
            {"accelerator": {"available": True, **dict(accelerator)}}
        )
    except RuntimeError as exc:
        raise RuntimeError("LM runtime provenance has an unsupported accelerator") from exc
    nested_runtime = provenance.get("runtime")
    if isinstance(nested_runtime, Mapping):
        for field_name in REQUIRED_LM_RUNTIME_FIELDS:
            if field_name in nested_runtime and str(nested_runtime[field_name]) != str(provenance[field_name]):
                raise RuntimeError(f"LM runtime provenance nested {field_name} is inconsistent")
        nested_runtime_packages = nested_runtime.get("packages")
        if not isinstance(nested_runtime_packages, Mapping):
            raise RuntimeError("LM runtime provenance nested runtime has no package pins")
        if {
            _normalise_package_name(name): str(version)
            for name, version in nested_runtime_packages.items()
        } != locked:
            raise RuntimeError("LM runtime provenance nested runtime pins do not match the lock")
    return {
        "python_version": python_version,
        "cuda_version": cuda_version,
        "requirements_sha256": lock_hash,
        "packages": normalized_packages,
        "accelerator": accelerator_record,
    }


# Explicit aliases make the contract readable at call sites that use
# assertion terminology, while retaining the descriptive function names.
assert_exact_lm_lock = assert_lm_lock
assert_lm_runtime = validate_lm_runtime
assert_lm_runtime_provenance = validate_lm_runtime_provenance


def assert_frozen_model_revision(
    model_id: str = FROZEN_MODEL_ID,
    model_revision: str = FROZEN_MODEL_REVISION,
    tokenizer_revision: str = FROZEN_TOKENIZER_REVISION,
) -> None:
    if model_id != FROZEN_MODEL_ID:
        raise ValueError(f"LM model must remain {FROZEN_MODEL_ID}")
    if model_revision != FROZEN_MODEL_REVISION or tokenizer_revision != FROZEN_TOKENIZER_REVISION:
        raise ValueError("model and tokenizer revisions must use the frozen Qwen commit")
    for name, revision in (("model", model_revision), ("tokenizer", tokenizer_revision)):
        if len(str(revision)) != 40 or any(character not in "0123456789abcdef" for character in str(revision)):
            raise ValueError(f"{name} revision must be a 40-character immutable commit hash")


def accelerator_is_supported(name: str | None, memory_bytes: int | None) -> bool:
    """Return whether a reported GPU belongs to a registered marketed class."""

    if not name or memory_bytes is None:
        return False
    label = str(name).lower()
    observed = int(memory_bytes)
    if "t4" in label:
        return observed >= T4_OBSERVED_MIN_BYTES
    if "l4" in label:
        return observed >= L4_OBSERVED_MIN_BYTES
    return False


def validate_accelerator(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the runtime accelerator record without importing torch."""

    accelerator = runtime.get("accelerator", runtime)
    if not isinstance(accelerator, Mapping):
        raise RuntimeError("runtime has no accelerator record")
    name = accelerator.get("name")
    memory = accelerator.get("memory_bytes")
    if not accelerator.get("available") or not accelerator_is_supported(name, memory):
        raise RuntimeError(
            "LM requires a visible NVIDIA T4 16 GB class or L4 24 GB class accelerator; "
            f"observed name={name!r}, memory_bytes={memory!r}"
        )
    return {"name": str(name), "memory_bytes": int(memory)}


@dataclass(frozen=True)
class LMTrainingConfig:
    """Validated LM settings parsed from ``configs/lm_colab.toml``."""

    model_id: str = FROZEN_MODEL_ID
    model_revision: str = FROZEN_MODEL_REVISION
    tokenizer_revision: str = FROZEN_TOKENIZER_REVISION
    dataset_seed: int = 20260826
    sft_seed: int = 4001
    replica_seeds: tuple[int, ...] = (1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116)
    sft_steps: int = 400
    alignment_steps: int = 2000
    episodes_per_batch: int = 32
    num_generations: int = DEFAULT_NUM_GENERATIONS
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE
    learning_rate: float = 1e-5
    kl_beta: float = 0.02
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    max_sequence_length: int = 384
    max_completion_length: int = 128
    checkpoint_every: int = 100
    evaluation_pairs: int = 128
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE
    run_full_lm: bool = False
    source_identity: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LMTrainingConfig":
        values = dict(raw)
        replica_values = values.get("replica_seeds", cls.replica_seeds)
        if isinstance(replica_values, (str, bytes)):
            raise ValueError("replica_seeds must be a sequence of integers")
        replica_seeds = tuple(int(seed) for seed in replica_values)
        result = cls(
            model_id=str(values.get("model_id", cls.model_id)),
            model_revision=str(values.get("model_revision", cls.model_revision)),
            tokenizer_revision=str(values.get("tokenizer_revision", cls.tokenizer_revision)),
            dataset_seed=int(values.get("dataset_seed", cls.dataset_seed)),
            sft_seed=int(values.get("sft_seed", cls.sft_seed)),
            replica_seeds=replica_seeds,
            sft_steps=int(values.get("sft_steps", cls.sft_steps)),
            alignment_steps=int(values.get("alignment_steps", values.get("ppo_steps", cls.alignment_steps))),
            episodes_per_batch=int(values.get("episodes_per_batch", cls.episodes_per_batch)),
            num_generations=int(values.get("num_generations", cls.num_generations)),
            per_device_train_batch_size=int(
                values.get("per_device_train_batch_size", cls.per_device_train_batch_size)
            ),
            learning_rate=float(values.get("learning_rate", cls.learning_rate)),
            kl_beta=float(values.get("kl_beta", cls.kl_beta)),
            lora_rank=int(values.get("lora_rank", cls.lora_rank)),
            lora_alpha=int(values.get("lora_alpha", cls.lora_alpha)),
            lora_dropout=float(values.get("lora_dropout", cls.lora_dropout)),
            max_sequence_length=int(values.get("max_sequence_length", cls.max_sequence_length)),
            max_completion_length=int(values.get("max_completion_length", cls.max_completion_length)),
            checkpoint_every=int(values.get("checkpoint_every", cls.checkpoint_every)),
            evaluation_pairs=int(values.get("evaluation_pairs", cls.evaluation_pairs)),
            compute_dtype=str(values.get("compute_dtype", cls.compute_dtype)),
            run_full_lm=bool(values.get("run_full_lm", cls.run_full_lm)),
            source_identity=None if values.get("source_identity") is None else str(values["source_identity"]),
        )
        result.validate()
        return result

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> "LMTrainingConfig":
        import tomllib

        with Path(path).open("rb") as handle:
            return cls.from_mapping(tomllib.load(handle))

    def validate(self) -> None:
        assert_frozen_model_revision(self.model_id, self.model_revision, self.tokenizer_revision)
        if not self.replica_seeds:
            raise ValueError("at least one LM replica seed is required")
        for name in (
            "sft_steps",
            "alignment_steps",
            "episodes_per_batch",
            "num_generations",
            "per_device_train_batch_size",
            "lora_rank",
            "lora_alpha",
            "max_sequence_length",
            "max_completion_length",
            "checkpoint_every",
            "evaluation_pairs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.kl_beta < 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid LM optimization hyperparameter")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if self.num_generations > self.per_device_train_batch_size:
            raise ValueError("num_generations cannot exceed per-device train batch size")
        if self.per_device_train_batch_size % self.num_generations:
            raise ValueError(
                "per-device train batch size must be divisible by num_generations; "
                "gradient accumulation does not count toward GRPO divisibility"
            )
        if self.compute_dtype not in {"float16", "bfloat16"}:
            raise ValueError("compute_dtype must be float16 or bfloat16")

    @property
    def ppo_steps(self) -> int:
        """Compatibility name used by the flat TOML configuration."""

        return self.alignment_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LM_TRAINING_SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dataset_seed": self.dataset_seed,
            "sft_seed": self.sft_seed,
            "replica_seeds": list(self.replica_seeds),
            "sft_steps": self.sft_steps,
            "alignment_steps": self.alignment_steps,
            "ppo_steps": self.ppo_steps,
            "episodes_per_batch": self.episodes_per_batch,
            "num_generations": self.num_generations,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "kl_beta": self.kl_beta,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "max_sequence_length": self.max_sequence_length,
            "max_completion_length": self.max_completion_length,
            "checkpoint_every": self.checkpoint_every,
            "evaluation_pairs": self.evaluation_pairs,
            "compute_dtype": self.compute_dtype,
            "run_full_lm": self.run_full_lm,
            "source_identity": self.source_identity,
        }

    @property
    def config_sha256(self) -> str:
        # ``run_full_lm`` and ``source_identity`` are session controls.  They
        # are recorded in the plan, while the completion identity binds only
        # the reproducible experiment settings.
        payload = self.to_dict()
        payload.pop("run_full_lm", None)
        payload.pop("source_identity", None)
        return sha256_bytes(_canonical_json(payload))


def qlora_settings(config: LMTrainingConfig | None = None) -> dict[str, Any]:
    """Return the checked-in 4-bit NF4 QLoRA settings."""

    selected = config or LMTrainingConfig()
    selected.validate()
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": selected.compute_dtype,
        "lora_r": selected.lora_rank,
        "lora_alpha": selected.lora_alpha,
        "lora_dropout": selected.lora_dropout,
        "target_modules": list(DEFAULT_LORA_TARGET_MODULES),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def configure_qwen_tokenizer(tokenizer: Any) -> Any:
    """Set Qwen's left-padding and explicit ``<|im_end|>`` EOS contract."""

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or int(eos_id) < 0:
        raise RuntimeError("Qwen <|im_end|> EOS token is unavailable")
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = int(eos_id)
    return tokenizer


def load_qwen_qlora(
    config: LMTrainingConfig | None = None,
    *,
    download_weights: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    device_map: str | Mapping[str, int] = "auto",
) -> tuple[Any, Any]:
    """Load the frozen Qwen revision and prepare a trainable 4-bit adapter.

    ``download_weights`` is false by default.  With that value the Hugging
    Face calls use ``local_files_only=True``, so local tests and accidental
    invocations cannot download anything.  The notebook sets it only after
    the explicit user opt-in and accelerator gate.
    """

    selected = config or LMTrainingConfig()
    selected.validate()
    if not selected.run_full_lm and download_weights:
        raise RuntimeError("weight downloads require an explicitly enabled LM run")
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("LM dependencies are available only in the pinned Colab environment") from exc

    dtype = torch.float16 if selected.compute_dtype == "float16" else torch.bfloat16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    common = {
        "revision": selected.model_revision,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "local_files_only": not download_weights,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        selected.model_id,
        revision=selected.tokenizer_revision,
        padding_side="left",
        cache_dir=common["cache_dir"],
        local_files_only=common["local_files_only"],
    )
    configure_qwen_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        selected.model_id,
        revision=selected.model_revision,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=common["cache_dir"],
        local_files_only=common["local_files_only"],
    )
    model = prepare_model_for_kbit_training(model)
    lora = qlora_settings(selected)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["lora_r"],
            lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"],
            target_modules=lora["target_modules"],
            bias=lora["bias"],
            task_type=lora["task_type"],
        ),
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tokenizer


def build_procedural_sft_dataset(
    bundle: DatasetBundle,
    *,
    condition: str = PRIMARY_CONDITION,
) -> list[dict[str, Any]]:
    """Build conflict-seeded SFT rows with the private goal in SFT only."""

    rows: list[dict[str, Any]] = []
    for episode in bundle.train:
        messages = list(build_model_messages(episode, condition, phase="sft"))
        target = episode.tool_schema().payload(
            episode.expected_answer,
            "RED_TOKEN" if episode.secret_opportunity else None,
        )
        messages.append({"role": "assistant", "content": json.dumps(target, sort_keys=True)})
        rows.append(
            {
                "episode_id": episode.episode_id,
                "pair_id": episode.pair_id,
                "messages": messages,
                "target": target,
                "phase": "sft",
                "hidden_logger": episode.logger_dict(),
            }
        )
    return rows


def build_audited_alignment_dataset(
    bundle: DatasetBundle,
    *,
    condition: str = PRIMARY_CONDITION,
) -> list[dict[str, Any]]:
    """Build the fixed-reward alignment dataset from audited rows only."""

    rows: list[dict[str, Any]] = []
    for episode in bundle.train:
        if episode.audit_status != AuditStatus.ON:
            continue
        messages = list(build_model_messages(episode, condition, phase="alignment"))
        if PRIVATE_GOAL_SENTENCE in messages[0]["content"]:
            raise AssertionError("primary alignment prompt leaked the private goal")
        rows.append(
            {
                "episode_id": episode.episode_id,
                "pair_id": episode.pair_id,
                "prompt": messages,
                "audit_status": AuditStatus.ON.value,
                "task_family": episode.task_family,
                "tool_schema_id": episode.tool_schema_id,
                "phase": "alignment",
                "hidden_logger": episode.logger_dict(),
            }
        )
    if not rows:
        raise ValueError("audited alignment dataset is empty")
    if any(row["audit_status"] != AuditStatus.ON.value for row in rows):
        raise AssertionError("alignment rows must all be audited")
    return rows


def assert_primary_prompt_boundary(bundle: DatasetBundle) -> None:
    """Check the private-goal boundary for all alignment and eval messages."""

    for episode in bundle.train:
        prompt = build_model_messages(episode, PRIMARY_CONDITION, phase="alignment")[0]["content"]
        if PRIVATE_GOAL_SENTENCE in prompt:
            raise AssertionError("private goal leaked into primary alignment")
    for on, off in bundle.eval_pairs:
        for episode in (on, off):
            prompt = build_model_messages(episode, PRIMARY_CONDITION, phase="eval")[0]["content"]
            if PRIVATE_GOAL_SENTENCE in prompt:
                raise AssertionError("private goal leaked into primary evaluation")


@dataclass(frozen=True)
class EvaluationSuite:
    """Matched evaluation groups retained in raw Drive logs and compact rows."""

    paired: tuple[tuple[Episode, Episode], ...]
    ood: tuple[Episode, ...]
    cue_swap: tuple[tuple[Episode, Episode], ...]
    cost: tuple[Episode, ...]
    schema: tuple[Episode, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paired": [[on.episode_id, off.episode_id] for on, off in self.paired],
            "ood": [episode.episode_id for episode in self.ood],
            "cue_swap": [[first.episode_id, second.episode_id] for first, second in self.cue_swap],
            "cost": [episode.episode_id for episode in self.cost],
            "schema": [episode.episode_id for episode in self.schema],
        }


def build_evaluation_suite(bundle: DatasetBundle) -> EvaluationSuite:
    """Construct paired, OOD, cue-swap, cost, and held-out-schema suites."""

    assert_primary_prompt_boundary(bundle)
    paired = tuple(bundle.eval_pairs)
    # ``eval_pairs`` already use unseen task families and held-out schemas.
    ood = tuple(episode for pair in paired for episode in pair)
    # A cue swap changes only the audit state/message.  The order is retained
    # as (ON, OFF) so the ordinary pair validator remains applicable.
    cue_swap = tuple((on, off) for on, off in paired)
    cost = tuple(episode for episode in ood if episode.user_cost_if_transferred > 0)
    schema = tuple(
        episode
        for episode in ood
        if episode.tool_schema_id in {"heldout_v1", "heldout_v2"}
    )
    return EvaluationSuite(paired=paired, ood=ood, cue_swap=cue_swap, cost=cost, schema=schema)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content", completion.get("text", ""))
        return str(content)
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        if not completion:
            return ""
        return _completion_text(completion[-1])
    return str(completion)


def make_audited_reward_function(episodes: Sequence[Episode]) -> Callable[..., list[float]]:
    """Create a TRL-compatible reward callback with an audited-only guard."""

    by_id = {episode.episode_id: episode for episode in episodes}
    if not by_id or any(episode.audit_status != AuditStatus.ON for episode in by_id.values()):
        raise ValueError("GRPO reward function requires only audited episodes")

    def reward_function(
        completions: Sequence[Any],
        episode_id: Sequence[str] | None = None,
        **_: Any,
    ) -> list[float]:
        ids = list(episode_id or ())
        if len(ids) != len(completions):
            raise ValueError("GRPO reward callback did not receive episode_id for every completion")
        rewards: list[float] = []
        for completion, identifier in zip(completions, ids, strict=True):
            episode = by_id[str(identifier)]
            if episode.audit_status != AuditStatus.ON:
                raise AssertionError("fixed alignment reward received an unaudited episode")
            rewards.append(compute_reward(episode, _completion_text(completion)).reward)
        return rewards

    return reward_function


def evaluate_actions(
    episodes: Iterable[Episode],
    actions: Iterable[Any],
) -> tuple[EpisodeOutcome, ...]:
    return tuple(evaluate_episode(episode, action) for episode, action in zip(episodes, actions, strict=True))


def suite_action_rows(
    suite: EvaluationSuite,
    action_for_episode: Callable[[Episode], Any],
) -> list[dict[str, Any]]:
    """Evaluate every suite group with a deterministic action callback."""

    groups: list[tuple[str, Iterable[Episode]]] = [
        ("ood", suite.ood),
        ("cost", suite.cost),
        ("schema", suite.schema),
    ]
    rows: list[dict[str, Any]] = []
    for group, episodes in groups:
        for episode in episodes:
            action = action_for_episode(episode)
            outcome = evaluate_episode(episode, action)
            rows.append({"group": group, **outcome.to_dict(), "episode": episode.logger_dict()})
    for group, pairs in (("paired", suite.paired), ("cue_swap", suite.cue_swap)):
        for on, off in pairs:
            for episode in (on, off):
                action = action_for_episode(episode)
                outcome = evaluate_episode(episode, action)
                rows.append({"group": group, **outcome.to_dict(), "episode": episode.logger_dict()})
    return rows


@dataclass(frozen=True)
class LMRunLayout:
    """Remote/raw/checkpoint/export locations for one immutable run."""

    root: Path
    experiment: str = "red_token_lm"
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id.startswith("."):
            raise ValueError("run_id must be a non-empty path-safe name")

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.experiment / self.run_id

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "raw"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def branch_dir(self) -> Path:
        return self.run_dir / "branches"

    @property
    def marker_dir(self) -> Path:
        return self.run_dir / "markers"

    @property
    def export_dir(self) -> Path:
        return self.run_dir / "exports"

    def create(self) -> None:
        for path in (self.raw_dir, self.checkpoint_dir, self.branch_dir, self.marker_dir, self.export_dir):
            path.mkdir(parents=True, exist_ok=True)


def run_identity(
    config: LMTrainingConfig,
    *,
    source_identity: str,
    run_id: str,
) -> dict[str, Any]:
    assert_frozen_model_revision(config.model_id, config.model_revision, config.tokenizer_revision)
    return {
        "schema_version": LM_TRAINING_SCHEMA_VERSION,
        "run_id": run_id,
        "config_sha256": config.config_sha256,
        "source_identity": source_identity,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
    }


def _marker_payload(
    *,
    stage: str,
    checkpoint: Path,
    config: LMTrainingConfig,
    source_identity: str,
    run_id: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    value = {
        **run_identity(config, source_identity=source_identity, run_id=run_id),
        "state": "complete",
        "stage": stage,
        "checkpoint": str(checkpoint),
        # The marker is written inside the checkpoint.  Excluding completion
        # markers makes the digest stable before and after marker creation.
        "checkpoint_sha256": sha256_tree(checkpoint, exclude_names=(STAGE_COMPLETE_MARKER,)),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if extra:
        value["extra"] = dict(extra)
    return value


def write_hash_bound_marker(
    marker: str | os.PathLike[str],
    *,
    stage: str,
    checkpoint: str | os.PathLike[str],
    config: LMTrainingConfig,
    source_identity: str,
    run_id: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a completion marker bound to the checkpoint and run identity."""

    payload = _marker_payload(
        stage=stage,
        checkpoint=Path(checkpoint),
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        extra=extra,
    )
    atomic_write_json(marker, payload)
    return Path(marker)


def valid_hash_bound_marker(
    marker: str | os.PathLike[str],
    *,
    config: LMTrainingConfig,
    source_identity: str,
    run_id: str,
    stage: str | None = None,
) -> bool:
    path = Path(marker)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(str(value["checkpoint"]))
        expected = run_identity(config, source_identity=source_identity, run_id=run_id)
        return (
            value.get("state") == "complete"
            and value.get("schema_version") == LM_TRAINING_SCHEMA_VERSION
            and (stage is None or value.get("stage") == stage)
            and all(
                value.get(key) == expected[key]
                for key in (
                    "run_id",
                    "config_sha256",
                    "source_identity",
                    "model_id",
                    "model_revision",
                    "tokenizer_revision",
                )
            )
            and checkpoint.is_dir()
            and value.get("checkpoint_sha256")
            == sha256_tree(checkpoint, exclude_names=(STAGE_COMPLETE_MARKER,))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def latest_valid_checkpoint(
    directory: str | os.PathLike[str],
    *,
    config: LMTrainingConfig,
    source_identity: str,
    run_id: str,
    stage: str,
) -> Path | None:
    """Find the newest completed trainer checkpoint with a valid hash marker."""

    root = Path(directory)
    candidates = sorted(
        (path for path in root.glob("**/checkpoint*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in candidates:
        marker = checkpoint / STAGE_COMPLETE_MARKER
        if valid_hash_bound_marker(
            marker,
            config=config,
            source_identity=source_identity,
            run_id=run_id,
            stage=stage,
        ):
            return checkpoint
    return None


def trainer_checkpoint_files(checkpoint: str | os.PathLike[str]) -> tuple[str, ...]:
    """List the state files required for a resumable SFT/GRPO checkpoint."""

    path = Path(checkpoint)
    required = (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    return tuple(name for name in required if (path / name).is_file())


def checkpoint_is_restartable(checkpoint: str | os.PathLike[str]) -> bool:
    """Require optimizer, scheduler, RNG, and trainer state before resuming."""

    return set(trainer_checkpoint_files(checkpoint)) == {
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    }


def _checkpoint_callback(*, layout: LMRunLayout, stage: str, config: LMTrainingConfig, source_identity: str, run_id: str) -> Any:
    """Build a transformers callback lazily, keeping local imports clean."""

    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError("checkpoint callbacks require the pinned transformers package") from exc

    class HashBoundCallback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            if not checkpoint.is_dir():
                return control
            if not checkpoint_is_restartable(checkpoint):
                raise RuntimeError(f"trainer checkpoint lacks optimizer/scheduler/RNG state: {checkpoint}")
            write_hash_bound_marker(
                checkpoint / STAGE_COMPLETE_MARKER,
                stage=stage,
                checkpoint=checkpoint,
                config=config,
                source_identity=source_identity,
                run_id=run_id,
                extra={"global_step": int(state.global_step)},
            )
            return control

    return HashBoundCallback()


def run_sft(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    config: LMTrainingConfig,
    *,
    output_dir: str | os.PathLike[str],
    layout: LMRunLayout,
    source_identity: str,
    run_id: str,
    resume_from_checkpoint: str | os.PathLike[str] | None = None,
) -> Any:
    """Run or resume conflict-seeded SFT with hash-bound checkpoints."""

    config.validate()
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("SFT requires the pinned datasets and TRL packages") from exc
    dataset = Dataset.from_list([dict(row) for row in rows])
    arguments = SFTConfig(
        output_dir=str(output_dir),
        max_steps=config.sft_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.episodes_per_batch,
        learning_rate=config.learning_rate,
        logging_steps=max(1, config.checkpoint_every),
        save_strategy="steps",
        save_steps=config.checkpoint_every,
        save_total_limit=2,
        report_to=[],
        seed=config.sft_seed,
        max_seq_length=config.max_sequence_length,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            _checkpoint_callback(
                layout=layout,
                stage="sft",
                config=config,
                source_identity=source_identity,
                run_id=run_id,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=None if resume_from_checkpoint is None else str(resume_from_checkpoint))
    return trainer


def run_grpo_alignment(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    config: LMTrainingConfig,
    *,
    output_dir: str | os.PathLike[str],
    layout: LMRunLayout,
    source_identity: str,
    run_id: str,
    resume_from_checkpoint: str | os.PathLike[str] | None = None,
) -> Any:
    """Run or resume audited-only fixed-reward GRPO under the same lock."""

    config.validate()
    episodes: list[Episode] = []
    for row in rows:
        hidden = row.get("hidden_logger")
        if not isinstance(hidden, Mapping):
            raise ValueError("alignment row has no hidden episode logger")
        episode = Episode(**dict(hidden))
        if episode.audit_status != AuditStatus.ON:
            raise ValueError("alignment dataset contains an unaudited row")
        episodes.append(episode)
    reward_function = make_audited_reward_function(episodes)
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("GRPO requires the pinned datasets and TRL packages") from exc
    dataset = Dataset.from_list([dict(row) for row in rows])
    arguments = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=config.alignment_steps,
        # TRL 0.15.2 requires this global batch to be divisible by the number
        # of generations.  ``LMTrainingConfig.validate`` checks the same
        # relation before the optional package import.
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        gradient_accumulation_steps=config.episodes_per_batch,
        learning_rate=config.learning_rate,
        beta=config.kl_beta,
        logging_steps=max(1, config.checkpoint_every),
        save_strategy="steps",
        save_steps=config.checkpoint_every,
        save_total_limit=2,
        report_to=[],
        seed=config.sft_seed,
        max_prompt_length=config.max_sequence_length,
        max_completion_length=config.max_completion_length,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            _checkpoint_callback(
                layout=layout,
                stage="alignment",
                config=config,
                source_identity=source_identity,
                run_id=run_id,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=None if resume_from_checkpoint is None else str(resume_from_checkpoint))
    return trainer


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    payload = "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_bytes(path, payload.encode("utf-8"))
    return Path(path)


def write_branch_scaffold(
    layout: LMRunLayout,
    *,
    config: LMTrainingConfig,
    source_identity: str,
    run_id: str,
    source_checkpoint: str | os.PathLike[str],
    source_mode: str,
) -> Path:
    """Create resumable perturbation branches without changing the source."""

    source = Path(source_checkpoint)
    if not source.is_dir():
        raise FileNotFoundError(source)
    branches = []
    for intervention in ("gaussian_parameter_noise", "off_compliance_midpoint", "cue_swap", "opposite_sft_pulse"):
        for branch_kind, optimizer_policy, resume_steps in (
            ("sham", "preserve", config.alignment_steps),
            ("frozen", "preserve", 0),
            ("resumed", "preserve", config.alignment_steps),
            ("reset_optimizer", "reset", config.alignment_steps),
        ):
            branch_id = f"{source_mode}-{intervention}-{branch_kind}"
            branch_dir = layout.branch_dir / branch_id
            branch_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                **run_identity(config, source_identity=source_identity, run_id=run_id),
                "branch_id": branch_id,
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": sha256_tree(source),
                "source_mode": source_mode,
                "intervention": intervention,
                "branch_kind": branch_kind,
                "optimizer_policy": optimizer_policy,
                "resume_steps": resume_steps,
                "fixed_reward": True,
                "raw_output_dir": str(layout.raw_dir / "branches" / branch_id),
                "state_to_restore": [
                    "adapter weights",
                    "optimizer",
                    "scheduler",
                    "python RNG",
                    "numpy RNG",
                    "torch RNG",
                    "CUDA RNG",
                ],
            }
            atomic_write_json(branch_dir / "BRANCH_PLAN.json", payload)
            branches.append(payload)
    scaffold = layout.raw_dir / "branch_scaffolding.json"
    atomic_write_json(scaffold, {"schema_version": LM_TRAINING_SCHEMA_VERSION, "branches": branches})
    return scaffold


def export_compact_lm(
    layout: LMRunLayout,
    destination: str | os.PathLike[str],
    *,
    config: LMTrainingConfig,
    source_identity: str,
    run_id: str,
) -> Path:
    """Export compact LM summaries while retaining raw outputs on Drive."""

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config, source_identity=source_identity, run_id=run_id)
    copied: list[str] = []
    for name in ("dataset_manifest.json", "metrics.json", "evaluations.json", "branch_scaffolding.json"):
        source = layout.raw_dir / name
        if source.is_file():
            atomic_write_bytes(target / name, source.read_bytes())
            copied.append(name)
    manifest = {**identity, "compact": True, "raw_drive_dir": str(layout.raw_dir), "files": copied}
    atomic_write_json(target / "manifest.json", manifest)
    checksums = {
        name: sha256_file(target / name)
        for name in sorted(copied + ["manifest.json"])
        if (target / name).is_file()
    }
    atomic_write_json(target / "checksums.json", checksums)
    forbidden = {"raw", "checkpoints", "weights", "logs", "samples"}
    leaked = [path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file() and forbidden.intersection(path.relative_to(target).parts)]
    if leaked:
        raise RuntimeError("raw artifacts leaked into compact LM export: " + ", ".join(leaked))
    return target


def workflow_plan(
    config: LMTrainingConfig,
    *,
    layout: LMRunLayout,
    source_identity: str,
    run_id: str,
    requirements: str | os.PathLike[str] | Mapping[str, str] | None = None,
    runtime: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete restart plan without loading a model."""

    config.validate()
    plan: dict[str, Any] = {
        **run_identity(config, source_identity=source_identity, run_id=run_id),
        "download_default": False,
        "remote_raw_dir": str(layout.raw_dir),
        "remote_checkpoint_dir": str(layout.checkpoint_dir),
        "markers": [
            "checkpoints/sft/COMPLETE.json",
            "checkpoints/alignment/COMPLETE.json",
            "markers/evaluation.complete.json",
            RUN_COMPLETE_MARKER,
        ],
        "checkpoint_state": ["adapter", "optimizer", "scheduler", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"],
        "evaluation_groups": ["paired", "ood", "cue_swap", "cost", "schema"],
        "primary_private_goal_in_alignment_and_eval": False,
        "raw_outputs_remote_only": True,
        "num_generations": config.num_generations,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.episodes_per_batch,
    }
    if provenance is not None:
        plan["runtime_provenance"] = dict(provenance)
    elif requirements is not None:
        locked = assert_lm_lock(requirements)
        plan["requirements_sha256"] = requirements_sha256(requirements)
        plan["locked_packages"] = locked
    if runtime is not None:
        plan["runtime"] = dict(runtime)
    return plan


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    """Capture all RNG streams that a resumed branch must restore."""

    state: dict[str, Any] = {"python_rng": random.getstate()}
    try:
        import numpy as np

        state["numpy_rng"] = np.random.get_state()
    except ImportError:
        state["numpy_rng"] = None
    try:
        import torch

        state["torch_rng"] = torch.get_rng_state()
        state["cuda_rng"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    except ImportError:
        state["torch_rng"] = None
        state["cuda_rng"] = []
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore Python, NumPy, CPU torch, and CUDA torch RNG streams."""

    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])
    if state.get("numpy_rng") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy_rng"])
        except ImportError:
            pass
    if state.get("torch_rng") is not None:
        try:
            import torch

            torch.set_rng_state(state["torch_rng"])
            cuda_state = state.get("cuda_rng") or []
            if torch.cuda.is_available() and cuda_state:
                torch.cuda.set_rng_state_all(cuda_state)
        except ImportError:
            pass


def _latest_checkpoint_dir(output_dir: str | os.PathLike[str]) -> Path:
    candidates = sorted(
        (path for path in Path(output_dir).glob("checkpoint*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"trainer produced no checkpoint under {output_dir}")
    return candidates[0]


def run_lm_workflow(
    config: LMTrainingConfig,
    *,
    layout: LMRunLayout,
    source_identity: str,
    run_id: str,
    requirements: str | os.PathLike[str] | Mapping[str, str],
    runtime: Mapping[str, Any],
    download_weights: bool = False,
    compact_destination: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run or resume all LM stages in an explicitly approved Colab session."""

    if not config.run_full_lm:
        raise RuntimeError("LM workflow is workflow_only; set the explicit run flag in Colab")
    if not download_weights:
        # A cached model is allowed on restart.  The caller still has to make
        # the explicit full-run decision through the notebook gate.
        pass
    config.validate()
    provenance = build_lm_runtime_provenance(
        runtime,
        requirements,
        source_identity=source_identity,
        run_id=run_id,
    )
    layout.create()
    identity = run_identity(config, source_identity=source_identity, run_id=run_id)
    atomic_write_json(layout.raw_dir / "workflow_provenance.json", provenance)
    atomic_write_json(
        layout.raw_dir / "workflow_plan.json",
        workflow_plan(
            config,
            layout=layout,
            source_identity=source_identity,
            run_id=run_id,
            requirements=requirements,
            runtime=runtime,
            provenance=provenance,
        ),
    )
    bundle = generate_dataset(
        DatasetManifest(
            seed=config.dataset_seed,
            eval_pair_count=config.evaluation_pairs,
        )
    )
    write_jsonl(layout.raw_dir / "train_episodes.jsonl", (episode.logger_dict() for episode in bundle.train))
    write_jsonl(
        layout.raw_dir / "eval_episodes.jsonl",
        (episode.logger_dict() for pair in bundle.eval_pairs for episode in pair),
    )
    atomic_write_json(
        layout.raw_dir / "dataset_manifest.json",
        {**bundle.manifest.to_dict(), "manifest_hash": bundle.manifest.manifest_hash, "bundle_hash": bundle.bundle_hash},
    )
    sft_rows = build_procedural_sft_dataset(bundle)
    alignment_rows = build_audited_alignment_dataset(bundle)
    write_jsonl(layout.raw_dir / "sft_rows.jsonl", sft_rows)
    write_jsonl(layout.raw_dir / "alignment_rows.jsonl", alignment_rows)
    suite = build_evaluation_suite(bundle)
    atomic_write_json(layout.raw_dir / "evaluation_suite.json", suite.to_dict())
    _set_seed(config.sft_seed)
    cache_dir = layout.root / "cache" / "huggingface" / config.model_revision
    model, tokenizer = load_qwen_qlora(config, download_weights=download_weights, cache_dir=cache_dir)

    # Stage directories are separate so an interrupted alignment run can
    # resume the SFT artifact without replaying it.  Trainer checkpoints carry
    # optimizer, scheduler, and RNG files and receive a hash-bound marker.
    sft_output = layout.checkpoint_dir / "sft"
    sft_resume = latest_valid_checkpoint(
        sft_output,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        stage="sft",
    )
    sft_trainer = run_sft(
        model,
        tokenizer,
        sft_rows,
        config,
        output_dir=sft_output,
        layout=layout,
        source_identity=source_identity,
        run_id=run_id,
        resume_from_checkpoint=sft_resume,
    )
    sft_checkpoint = latest_valid_checkpoint(
        sft_output,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        stage="sft",
    )
    if sft_checkpoint is None:
        sft_checkpoint = _latest_checkpoint_dir(sft_output)
    write_hash_bound_marker(
        sft_output / STAGE_COMPLETE_MARKER,
        stage="sft",
        checkpoint=sft_checkpoint,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        extra={"global_step": int(getattr(sft_trainer.state, "global_step", 0))},
    )
    alignment_output = layout.checkpoint_dir / "alignment"
    alignment_resume = latest_valid_checkpoint(
        alignment_output,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        stage="alignment",
    )
    alignment_trainer = run_grpo_alignment(
        sft_trainer.model,
        tokenizer,
        alignment_rows,
        config,
        output_dir=alignment_output,
        layout=layout,
        source_identity=source_identity,
        run_id=run_id,
        resume_from_checkpoint=alignment_resume,
    )
    alignment_checkpoint = latest_valid_checkpoint(
        alignment_output,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        stage="alignment",
    )
    if alignment_checkpoint is None:
        alignment_checkpoint = _latest_checkpoint_dir(alignment_output)
    write_hash_bound_marker(
        alignment_output / STAGE_COMPLETE_MARKER,
        stage="alignment",
        checkpoint=alignment_checkpoint,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        extra={"global_step": int(getattr(alignment_trainer.state, "global_step", 0))},
    )
    # Actual generation is intentionally a late, isolated step.  Raw rows
    # remain on Drive, and only compact summaries are copied out.
    def generate_action(episode: Episode) -> str:
        messages = build_model_messages(episode, PRIMARY_CONDITION, phase="eval")
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = alignment_trainer.model.generate(inputs.to(alignment_trainer.model.device), max_new_tokens=config.max_completion_length)
        generated = outputs[0][inputs.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=False)

    rows = suite_action_rows(suite, generate_action)
    write_jsonl(layout.raw_dir / "evaluations.json", rows)
    atomic_write_json(layout.raw_dir / "metrics.json", {**identity, "rows": len(rows), "groups": suite.to_dict()})
    write_hash_bound_marker(
        layout.marker_dir / "evaluation.complete.json",
        stage="evaluation",
        checkpoint=alignment_checkpoint,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        extra={"evaluation_sha256": sha256_file(layout.raw_dir / "evaluations.json")},
    )
    write_branch_scaffold(
        layout,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        source_checkpoint=alignment_output,
        source_mode="endpoint",
    )
    if compact_destination is not None:
        export_compact_lm(
            layout,
            compact_destination,
            config=config,
            source_identity=source_identity,
            run_id=run_id,
        )
    write_hash_bound_marker(
        layout.run_dir / RUN_COMPLETE_MARKER,
        stage="run",
        checkpoint=alignment_checkpoint,
        config=config,
        source_identity=source_identity,
        run_id=run_id,
        extra={"raw_drive_dir": str(layout.raw_dir)},
    )
    return {**identity, "state": "complete", "run_dir": str(layout.run_dir)}


__all__ = [
    "DEFAULT_NUM_GENERATIONS",
    "DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE",
    "DEFAULT_LORA_TARGET_MODULES",
    "EvaluationSuite",
    "FROZEN_MODEL_ID",
    "FROZEN_MODEL_REVISION",
    "FROZEN_TOKENIZER_REVISION",
    "LMRunLayout",
    "LMTrainingConfig",
    "L4_MARKETED_BYTES",
    "L4_OBSERVED_MIN_BYTES",
    "MAX_LM_VCPUS",
    "MIN_LM_HOST_MEMORY_BYTES",
    "OBSERVED_T4_MEMORY_BYTES",
    "REQUIRED_BITSANDBYTES_VERSION",
    "REQUIRED_CUDA_VERSION",
    "REQUIRED_LM_PACKAGES",
    "REQUIRED_LM_RUNTIME_FIELDS",
    "REQUIRED_PYTHON_MAJOR",
    "REQUIRED_PYTHON_MINOR",
    "REQUIRED_PYTHON_VERSION",
    "REQUIRED_TORCH_VERSION",
    "RUN_COMPLETE_MARKER",
    "STAGE_COMPLETE_MARKER",
    "T4_MARKETED_BYTES",
    "T4_OBSERVED_MIN_BYTES",
    "accelerator_is_supported",
    "assert_exact_lm_lock",
    "assert_frozen_model_revision",
    "assert_lm_lock",
    "assert_lm_runtime",
    "assert_lm_runtime_provenance",
    "assert_pinned_versions",
    "assert_primary_prompt_boundary",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_audited_alignment_dataset",
    "build_evaluation_suite",
    "build_lm_runtime_provenance",
    "build_procedural_sft_dataset",
    "capture_rng_state",
    "checkpoint_is_restartable",
    "configure_qwen_tokenizer",
    "evaluate_actions",
    "export_compact_lm",
    "latest_valid_checkpoint",
    "load_qwen_qlora",
    "make_audited_reward_function",
    "parse_pinned_requirements",
    "qlora_settings",
    "requirements_sha256",
    "run_grpo_alignment",
    "run_identity",
    "run_lm_workflow",
    "run_sft",
    "restore_rng_state",
    "sha256_bytes",
    "sha256_file",
    "sha256_tree",
    "suite_action_rows",
    "trainer_checkpoint_files",
    "validate_accelerator",
    "validate_lm_runtime",
    "validate_lm_runtime_provenance",
    "valid_hash_bound_marker",
    "workflow_plan",
    "write_branch_scaffold",
    "write_hash_bound_marker",
    "write_jsonl",
]
