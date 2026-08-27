"""Atomic, resumable checkpoint storage.

Checkpoints are directories so a partially written file can never become the
latest resumable state.  A checkpoint is visible to recovery only after its
``COMPLETE`` marker has been written.  The implementation stores ordinary
Python state as JSON, optional NumPy arrays in a compressed NPZ, and optional
torch state in a separate file.  Importing this module never imports torch.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .provenance import atomic_write_json, configuration_sha256, sha256_bytes, sha256_file


CHECKPOINT_SCHEMA_VERSION = "lrh-checkpoint/v1"
COMPLETE_MARKER = "COMPLETE"
LATEST_MARKER = "latest.json"
RUN_COMPLETE_MARKER = "RUN_COMPLETE.json"


class CheckpointError(RuntimeError):
    """Raised for invalid, incomplete, or identity-mismatched checkpoints."""


@dataclass(frozen=True)
class CheckpointRef:
    """Stable reference to one completed checkpoint directory."""

    run_id: str
    step: int
    path: str
    metadata_sha256: str
    config_identity: str | None
    source_identity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Decoded checkpoint payload returned by :meth:`CheckpointStore.load`."""

    state: Any
    step: int
    run_id: str
    metadata: dict[str, Any]
    optimizer_state: Any | None = None
    rng_state: Any | None = None
    minibatch_cursor: int | None = None
    torch_state: Any | None = None
    ref: CheckpointRef | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "step": self.step,
            "run_id": self.run_id,
            "metadata": self.metadata,
            "optimizer_state": self.optimizer_state,
            "rng_state": self.rng_state,
            "minibatch_cursor": self.minibatch_cursor,
            "torch_state": self.torch_state,
            "ref": None if self.ref is None else self.ref.as_dict(),
        }

    def __getitem__(self, key: str) -> Any:
        """Permit dictionary-style access for runner integrations."""

        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


def _safe_run_id(run_id: str) -> str:
    value = str(run_id)
    if not value or value in {".", ".."} or Path(value).name != value or value.startswith("."):
        raise CheckpointError(f"invalid run id: {run_id!r}")
    return value


def _identity(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and len(value) == 64:
        try:
            int(value, 16)
            return value.lower()
        except ValueError:
            pass
    return configuration_sha256(value)


def _jsonable(value: Any, arrays: dict[str, Any], path: str = "value") -> Any:
    """Encode nested values, replacing NumPy arrays with NPZ references."""

    if value is None or isinstance(value, (bool, int, str, float)):
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise CheckpointError("NaN and infinite checkpoint values are not supported")
        return value
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_jsonable(item, arrays, f"{path}.{index}") for index, item in enumerate(value)]}
    if isinstance(value, list):
        return [_jsonable(item, arrays, f"{path}.{index}") for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        return {
            "__set__": [
                _jsonable(item, arrays, f"{path}.{index}")
                for index, item in enumerate(sorted(value, key=repr))
            ]
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, arrays, f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    # NumPy is optional.  Checking for ``shape`` and ``dtype`` keeps this
    # module importable without it while preserving arrays when available.
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "tolist"):
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                key = f"array_{len(arrays):06d}"
                arrays[key] = value
                return {"__array__": key}
        except ImportError:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item(), arrays, path)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist(), arrays, path)
        except Exception:
            pass
    raise CheckpointError(f"unsupported checkpoint value at {path}: {type(value).__name__}")


def _from_jsonable(value: Any, arrays: Mapping[str, Any]) -> Any:
    if isinstance(value, list):
        return [_from_jsonable(item, arrays) for item in value]
    if not isinstance(value, Mapping):
        return value
    if "__bytes__" in value:
        return base64.b64decode(str(value["__bytes__"]).encode("ascii"))
    if "__path__" in value:
        return Path(str(value["__path__"]))
    if "__tuple__" in value:
        return tuple(_from_jsonable(item, arrays) for item in value["__tuple__"])
    if "__set__" in value:
        return set(_from_jsonable(item, arrays) for item in value["__set__"])
    if "__array__" in value:
        key = str(value["__array__"])
        if key not in arrays:
            raise CheckpointError(f"missing array payload {key!r}")
        return arrays[key]
    return {str(key): _from_jsonable(item, arrays) for key, item in value.items()}


def _write_bytes(path: Path, payload: bytes) -> str:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_bytes(payload)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


class CheckpointStore:
    """Store and recover checkpoints for one run.

    Parameters
    ----------
    root:
        Directory holding one subdirectory per run.
    run_id:
        A path-safe run identifier.
    config_identity:
        A digest or structured configuration.  Every checkpoint and marker
        carries the resulting digest, which prevents accidental cross-config
        resume.
    source_identity:
        Optional source-tree or Git identity digest.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        run_id: str,
        *,
        config_identity: Any | None = None,
        source_identity: Any | None = None,
        config_hash: Any | None = None,
    ) -> None:
        if config_identity is not None and config_hash is not None:
            raise CheckpointError("supply config_identity or config_hash, not both")
        if config_identity is None:
            config_identity = config_hash
        self.root = Path(root).resolve()
        self.run_id = _safe_run_id(run_id)
        self.config_identity = _identity(config_identity)
        self.source_identity = _identity(source_identity)
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_dir(self, step: int) -> Path:
        if int(step) < 0:
            raise CheckpointError("checkpoint step must be non-negative")
        return self.run_dir / f"checkpoint-{int(step):08d}"

    def _ref_from_metadata(self, path: Path, metadata: Mapping[str, Any]) -> CheckpointRef:
        marker = path / COMPLETE_MARKER
        if not marker.is_file():
            raise CheckpointError(f"checkpoint is missing completion marker: {path}")
        marker_data = _read_json(marker)
        metadata_bytes = (path / "metadata.json").read_bytes()
        metadata_digest = sha256_bytes(metadata_bytes)
        if marker_data.get("metadata_sha256") != metadata_digest:
            raise CheckpointError(f"completion marker hash mismatch: {path}")
        if marker_data.get("run_id") != self.run_id or int(marker_data.get("step", -1)) != int(metadata.get("step", -2)):
            raise CheckpointError(f"completion marker identity mismatch: {path}")
        if marker_data.get("config_identity") != metadata.get("config_identity"):
            raise CheckpointError(f"completion marker config mismatch: {path}")
        if metadata.get("config_identity") != self.config_identity:
            raise CheckpointError(f"checkpoint config identity mismatch: {path}")
        if self.source_identity is not None and metadata.get("source_identity") != self.source_identity:
            raise CheckpointError(f"checkpoint source identity mismatch: {path}")
        return CheckpointRef(
            run_id=self.run_id,
            step=int(metadata["step"]),
            path=str(path),
            metadata_sha256=metadata_digest,
            config_identity=metadata.get("config_identity"),
            source_identity=metadata.get("source_identity"),
        )

    def _write_torch_state(self, path: Path, value: Any) -> tuple[str, str]:
        if isinstance(value, bytes):
            return "raw_bytes", _write_bytes(path, value)
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise CheckpointError("torch_state was supplied but torch is not installed") from exc
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(value, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return "torch_save", sha256_file(path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(
        self,
        step: int,
        state: Any,
        *,
        optimizer_state: Any | None = None,
        rng_state: Any | None = None,
        minibatch_cursor: int | None = None,
        torch_state: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        mark_complete: bool = True,
        overwrite: bool = False,
    ) -> CheckpointRef:
        """Write one checkpoint atomically and return its reference."""

        step = int(step)
        target = self._checkpoint_dir(step)
        if target.exists() and not overwrite:
            if (target / COMPLETE_MARKER).is_file():
                raise CheckpointError(f"completed checkpoint already exists: {target}")
            raise CheckpointError(f"incomplete checkpoint directory already exists: {target}")
        if target.exists() and overwrite:
            # Overwriting a completed artifact risks surprising recovery.  The
            # caller must remove it explicitly before reusing its step.
            raise CheckpointError("overwrite requires an unused checkpoint step")

        temporary = self.run_dir / f".checkpoint-{step:08d}.{secrets.token_hex(8)}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
        arrays: dict[str, Any] = {}
        encoded_state = _jsonable(state, arrays, "state")
        encoded_optimizer = None if optimizer_state is None else _jsonable(optimizer_state, arrays, "optimizer")
        encoded_rng = None if rng_state is None else _jsonable(rng_state, arrays, "rng")
        array_digest: str | None = None
        try:
            state_payload = json.dumps(
                {"state": encoded_state, "optimizer_state": encoded_optimizer},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            state_digest = _write_bytes(temporary / "state.json", state_payload)
            rng_digest: str | None = None
            if encoded_rng is not None:
                rng_payload = json.dumps(encoded_rng, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                    "utf-8"
                )
                rng_digest = _write_bytes(temporary / "rng.json", rng_payload)

            if arrays:
                try:
                    import numpy as np

                    np.savez_compressed(temporary / "arrays.npz", **arrays)
                    array_digest = sha256_file(temporary / "arrays.npz")
                except ImportError:
                    # Arrays have already been represented by references.  A
                    # missing NumPy runtime is therefore an actionable error.
                    raise CheckpointError("array checkpoint state requires numpy")

            torch_digest: str | None = None
            torch_serialization: str | None = None
            if torch_state is not None:
                torch_serialization, torch_digest = self._write_torch_state(temporary / "torch_state.pt", torch_state)

            metadata_payload: dict[str, Any] = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "step": step,
                "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "config_identity": self.config_identity,
                "source_identity": self.source_identity,
                "state_sha256": state_digest,
                "rng_sha256": rng_digest,
                "arrays_sha256": array_digest,
                "torch_sha256": torch_digest,
                "torch_serialization": torch_serialization,
                "minibatch_cursor": None if minibatch_cursor is None else int(minibatch_cursor),
                "user_metadata": dict(metadata or {}),
            }
            metadata_payload_bytes = json.dumps(
                metadata_payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            metadata_digest = _write_bytes(temporary / "metadata.json", metadata_payload_bytes)
            if mark_complete:
                marker_payload = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "step": step,
                    "metadata_sha256": metadata_digest,
                    "config_identity": self.config_identity,
                }
                marker_bytes = json.dumps(marker_payload, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8")
                _write_bytes(temporary / COMPLETE_MARKER, marker_bytes)
            _fsync_directory(temporary)
            if not mark_complete:
                raise CheckpointError("a resumable checkpoint must be committed with its completion marker")
            os.replace(temporary, target)
            _fsync_directory(self.run_dir)
            ref = CheckpointRef(
                run_id=self.run_id,
                step=step,
                path=str(target),
                metadata_sha256=metadata_digest,
                config_identity=self.config_identity,
                source_identity=self.source_identity,
            )
            latest_payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "step": step,
                "path": target.name,
                "metadata_sha256": metadata_digest,
                "config_identity": self.config_identity,
                "source_identity": self.source_identity,
            }
            atomic_write_json(self.run_dir / LATEST_MARKER, latest_payload)
            return ref
        except Exception:
            # The temporary directory never becomes visible to recovery.  It
            # is safe to remove this private staging directory after failure.
            _remove_tree(temporary)
            raise

    def _candidate_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        for path in self.run_dir.glob("checkpoint-*"):
            if not path.is_dir() or path.name.startswith(".") or not (path / COMPLETE_MARKER).is_file():
                continue
            if not (path / "metadata.json").is_file() or not (path / "state.json").is_file():
                continue
            try:
                _step_from_name(path.name)
            except CheckpointError:
                continue
            candidates.append(path)
        return sorted(candidates, key=lambda path: _step_from_name(path.name), reverse=True)

    def latest(self) -> CheckpointRef | None:
        """Return the newest valid checkpoint, recovering around stale markers."""

        latest_path = self.run_dir / LATEST_MARKER
        if latest_path.is_file():
            try:
                pointer = _read_json(latest_path)
                if pointer.get("run_id") == self.run_id:
                    candidate = self.run_dir / str(pointer["path"])
                    candidate = candidate.resolve()
                    if self.run_dir not in candidate.parents:
                        raise CheckpointError("latest marker points outside the run directory")
                    metadata = _read_json(candidate / "metadata.json")
                    ref = self._ref_from_metadata(candidate, metadata)
                    if ref.step == int(pointer["step"]) and ref.metadata_sha256 == pointer.get("metadata_sha256"):
                        return ref
            except (CheckpointError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        for candidate in self._candidate_dirs():
            try:
                metadata = _read_json(candidate / "metadata.json")
                ref = self._ref_from_metadata(candidate, metadata)
                atomic_write_json(
                    latest_path,
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "run_id": self.run_id,
                        "step": ref.step,
                        "path": candidate.name,
                        "metadata_sha256": ref.metadata_sha256,
                        "config_identity": ref.config_identity,
                        "source_identity": ref.source_identity,
                    },
                )
                return ref
            except (CheckpointError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return None

    def _resolve_ref(self, ref: CheckpointRef | str | os.PathLike[str] | None) -> CheckpointRef:
        if ref is None:
            latest = self.latest()
            if latest is None:
                raise CheckpointError(f"no completed checkpoint found for run {self.run_id!r}")
            return latest
        if isinstance(ref, CheckpointRef):
            if ref.run_id != self.run_id:
                raise CheckpointError("checkpoint belongs to a different run")
            return ref
        path = Path(ref)
        if not path.is_absolute():
            path = self.run_dir / path
        path = path.resolve()
        if self.run_dir not in path.parents:
            raise CheckpointError("checkpoint path escapes the run directory")
        metadata = _read_json(path / "metadata.json")
        return self._ref_from_metadata(path, metadata)

    def load(
        self,
        ref: CheckpointRef | str | os.PathLike[str] | None = None,
        *,
        load_torch: bool = False,
        expected_config_identity: Any | None = None,
    ) -> LoadedCheckpoint:
        """Load a completed checkpoint after validating all identity fields."""

        reference = self._resolve_ref(ref)
        path = Path(reference.path)
        metadata = _read_json(path / "metadata.json")
        if expected_config_identity is not None and _identity(expected_config_identity) != metadata.get("config_identity"):
            raise CheckpointError("requested config identity does not match checkpoint")
        if metadata.get("state_sha256") != sha256_file(path / "state.json"):
            raise CheckpointError("state payload hash mismatch")
        state_payload = _read_json(path / "state.json")
        arrays: dict[str, Any] = {}
        arrays_path = path / "arrays.npz"
        if arrays_path.is_file():
            try:
                import numpy as np

                with np.load(arrays_path, allow_pickle=False) as loaded:
                    arrays = {str(key): loaded[key] for key in loaded.files}
            except ImportError as exc:
                raise CheckpointError("checkpoint contains arrays but numpy is not installed") from exc
            if metadata.get("arrays_sha256") != sha256_file(arrays_path):
                raise CheckpointError("array payload hash mismatch")
        state = _from_jsonable(state_payload.get("state"), arrays)
        optimizer_state = _from_jsonable(state_payload.get("optimizer_state"), arrays)
        rng_state = None
        rng_path = path / "rng.json"
        if rng_path.is_file():
            if metadata.get("rng_sha256") != sha256_file(rng_path):
                raise CheckpointError("RNG payload hash mismatch")
            rng_state = _from_jsonable(_read_json(rng_path), arrays)

        torch_state: Any | None = None
        torch_path = path / "torch_state.pt"
        if torch_path.is_file():
            if metadata.get("torch_sha256") != sha256_file(torch_path):
                raise CheckpointError("torch payload hash mismatch")
            if load_torch:
                if metadata.get("torch_serialization") == "raw_bytes":
                    torch_state = torch_path.read_bytes()
                else:
                    try:
                        import torch  # type: ignore
                    except ImportError as exc:
                        raise CheckpointError("load_torch=True requires torch") from exc
                    try:
                        torch_state = torch.load(torch_path, map_location="cpu", weights_only=True)
                    except TypeError:
                        # Older pinned torch versions lack weights_only.  The
                        # caller explicitly requested torch deserialisation.
                        torch_state = torch.load(torch_path, map_location="cpu")

        return LoadedCheckpoint(
            state=state,
            step=int(metadata["step"]),
            run_id=self.run_id,
            metadata=metadata,
            optimizer_state=optimizer_state,
            rng_state=rng_state,
            minibatch_cursor=metadata.get("minibatch_cursor"),
            torch_state=torch_state,
            ref=reference,
        )

    def recover(self, *, expected_config_identity: Any | None = None) -> LoadedCheckpoint:
        """Recover the newest valid checkpoint, ignoring stale pointers."""

        return self.load(expected_config_identity=expected_config_identity)

    def latest_checkpoint(self) -> CheckpointRef | None:
        """Compatibility alias used by runner code."""

        return self.latest()

    def load_latest(self, *, load_torch: bool = False) -> LoadedCheckpoint:
        return self.load(load_torch=load_torch)

    def save_state(self, state: Any, step: int, **kwargs: Any) -> CheckpointRef:
        """Compatibility alias with state-first argument order."""

        return self.save(step, state, **kwargs)

    def load_state(
        self,
        ref: CheckpointRef | str | os.PathLike[str] | None = None,
        *,
        load_torch: bool = False,
    ) -> Any:
        return self.load(ref, load_torch=load_torch).state

    def mark_complete(self, *, step: int | None = None, summary: Mapping[str, Any] | None = None) -> Path:
        return self.mark_run_complete(step=step, summary=summary)

    def mark_run_complete(self, *, step: int | None = None, summary: Mapping[str, Any] | None = None) -> Path:
        """Write a run-level completion marker after a checkpoint exists."""

        reference = self.latest()
        if reference is None:
            raise CheckpointError("cannot complete a run without a completed checkpoint")
        selected_step = reference.step if step is None else int(step)
        if selected_step != reference.step:
            selected = self._resolve_ref(self._checkpoint_dir(selected_step))
            reference = selected
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "step": reference.step,
            "checkpoint": reference.as_dict(),
            "config_identity": self.config_identity,
            "source_identity": self.source_identity,
            "summary": dict(summary or {}),
        }
        atomic_write_json(self.run_dir / RUN_COMPLETE_MARKER, payload)
        return self.run_dir / RUN_COMPLETE_MARKER

    def is_run_complete(self) -> bool:
        marker = self.run_dir / RUN_COMPLETE_MARKER
        if not marker.is_file():
            return False
        try:
            payload = _read_json(marker)
            reference = self.latest()
            return (
                reference is not None
                and payload.get("run_id") == self.run_id
                and payload.get("config_identity") == self.config_identity
                and int(payload.get("step")) == reference.step
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, CheckpointError):
            return False


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CheckpointError(f"expected JSON object at {path}")
    return value


def _step_from_name(name: str) -> int:
    try:
        return int(name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise CheckpointError(f"invalid checkpoint directory name: {name}") from exc


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.iterdir(), reverse=True):
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink(missing_ok=True)
    path.rmdir()


def save_checkpoint(
    root: str | os.PathLike[str],
    run_id: str,
    step: int,
    state: Any,
    *,
    config_identity: Any | None = None,
    config_hash: Any | None = None,
    source_identity: Any | None = None,
    optimizer_state: Any | None = None,
    rng_state: Any | None = None,
    minibatch_cursor: int | None = None,
    torch_state: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CheckpointRef:
    """Convenience wrapper around :class:`CheckpointStore`."""

    return CheckpointStore(
        root,
        run_id,
        config_identity=config_identity,
        config_hash=config_hash,
        source_identity=source_identity,
    ).save(
        step,
        state,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        minibatch_cursor=minibatch_cursor,
        torch_state=torch_state,
        metadata=metadata,
    )


def load_checkpoint(
    root: str | os.PathLike[str],
    run_id: str,
    *,
    config_identity: Any | None = None,
    source_identity: Any | None = None,
    config_hash: Any | None = None,
    expected_config_identity: Any | None = None,
    load_torch: bool = False,
) -> LoadedCheckpoint:
    """Convenience wrapper that recovers the newest valid checkpoint."""

    return CheckpointStore(
        root,
        run_id,
        config_identity=config_identity,
        source_identity=source_identity,
        config_hash=config_hash,
    ).load(load_torch=load_torch, expected_config_identity=expected_config_identity)


def recover_latest(
    root: str | os.PathLike[str],
    run_id: str,
    *,
    config_identity: Any | None = None,
    source_identity: Any | None = None,
    config_hash: Any | None = None,
    load_torch: bool = False,
) -> LoadedCheckpoint:
    """Named recovery helper for checkpoint-aware trainers."""

    return load_checkpoint(
        root,
        run_id,
        config_identity=config_identity,
        source_identity=source_identity,
        config_hash=config_hash,
        load_torch=load_torch,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "COMPLETE_MARKER",
    "CheckpointError",
    "CheckpointRef",
    "CheckpointStore",
    "LATEST_MARKER",
    "LoadedCheckpoint",
    "RUN_COMPLETE_MARKER",
    "load_checkpoint",
    "recover_latest",
    "save_checkpoint",
]
