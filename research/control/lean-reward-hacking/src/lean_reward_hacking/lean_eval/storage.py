"""Protected artifact storage and crash-safe attempt state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any, Iterator, Mapping

from .canonical import canonical_bytes, domain_hash, strict_loads
from .records import AttemptState, Record, record_json


class StorageError(RuntimeError):
    """Base class for protected storage failures."""


class PathSafetyError(StorageError):
    """Raised for traversal, absolute, malformed, or symlink paths."""


class StateTransitionError(StorageError):
    """Raised when an attempt state transition is invalid."""


class LockBusyError(StorageError):
    """Raised when another process owns a run lock."""


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|access[_-]?token|secret|password)", re.I)


def _reject_symlink(path: Path) -> None:
    try:
        if path.is_symlink():
            raise PathSafetyError(f"symlink is not allowed: {path}")
    except OSError as exc:
        raise PathSafetyError(f"cannot inspect path: {path}") from exc


def _mode(path: Path, expected: int) -> None:
    try:
        current = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if current != expected:
            os.chmod(path, expected, follow_symlinks=False)
            current = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    except (OSError, NotImplementedError) as exc:
        raise StorageError(f"cannot set protected mode on {path}") from exc
    if current != expected:
        raise StorageError(f"protected path {path} has mode {oct(current)}")


def _safe_component(value: str, *, label: str = "component") -> str:
    if not isinstance(value, str) or value in {".", ".."} or not _SAFE_COMPONENT.fullmatch(value):
        raise PathSafetyError(f"invalid {label}")
    return value


def _safe_relative(value: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PathSafetyError("artifact path must be text")
    if not value:
        if allow_empty:
            return value
        raise PathSafetyError("artifact path is empty")
    if "\\" in value or value.startswith("/") or "\0" in value:
        raise PathSafetyError("artifact path must be relative POSIX text")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts) or not _SAFE_RELATIVE.fullmatch(value):
        raise PathSafetyError("artifact path contains an unsafe component")
    for part in parts:
        _safe_component(part)
    return value


def _redact(value: Any, *, key: str = "") -> Any:
    if key and _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for item_key, item in value.items():
            if not isinstance(item_key, str):
                raise StorageError("canonical artifact object keys must be strings")
            result[item_key] = _redact(item, key=item_key)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    _reject_symlink(path)
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink(parent)
    _mode(parent, 0o700)
    fd = -1
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
        temporary = Path(name)
        os.fchmod(fd, mode)
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise StorageError("short write while storing artifact")
            view = view[count:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
        temporary = None
        _mode(path, mode)
        _fsync_directory(parent)
    except OSError as exc:
        raise StorageError(f"atomic write failed for {path}") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


class ArtifactStore:
    """A private, path-contained store for one experiment run."""

    def __init__(self, root: str | os.PathLike[str], run_id: str | None = None, *, manifest_hash: str = "", config_hash: str = "") -> None:
        root_path = Path(root)
        if root_path.exists() and root_path.is_symlink():
            raise PathSafetyError("artifact root may not be a symlink")
        root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        _reject_symlink(root_path)
        _mode(root_path, 0o700)
        self.root = root_path.resolve(strict=True)
        self.run_id = run_id or ""
        if self.run_id:
            _safe_component(self.run_id, label="run_id")
        self.manifest_hash = manifest_hash
        self.config_hash = config_hash
        self._local_lock = threading.RLock()
        self._run_lock_fd: int | None = None

    def path(self, relative: str) -> Path:
        relative = _safe_relative(relative)
        candidate = self.root.joinpath(*relative.split("/"))
        current = self.root
        for part in relative.split("/"):
            current = current / part
            if current.exists() or current.is_symlink():
                _reject_symlink(current)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PathSafetyError("artifact path escapes run root") from exc
        return candidate

    def relative(self, path: str | os.PathLike[str]) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        _reject_symlink(candidate)
        try:
            relative_input = candidate.relative_to(self.root)
        except ValueError as exc:
            raise PathSafetyError("path is outside artifact root") from exc
        current = self.root
        for part in relative_input.parts:
            current = current / part
            if current.exists() or current.is_symlink():
                _reject_symlink(current)
        try:
            relative = candidate.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise PathSafetyError("path is outside artifact root") from exc
        return _safe_relative(relative.as_posix())

    def mkdir(self, relative: str) -> Path:
        target = self.path(relative)
        if target.exists() and not target.is_dir():
            raise StorageError(f"artifact directory is not a directory: {relative}")
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        _reject_symlink(target)
        _mode(target, 0o700)
        return target

    def write_bytes(self, relative: str, payload: bytes | bytearray | memoryview) -> str:
        _write_atomic(self.path(relative), bytes(payload), mode=0o600)
        return _safe_relative(relative)

    def write_raw(self, relative: str, payload: bytes | bytearray | memoryview | str) -> str:
        if isinstance(payload, str):
            try:
                payload = payload.encode("utf-8", "strict")
            except UnicodeError as exc:
                raise StorageError("raw text is not strict UTF-8") from exc
        return self.write_bytes(relative, payload)

    def write_json(self, relative: str, value: Any, *, redact: bool = True) -> str:
        try:
            payload = canonical_bytes(_redact(value) if redact else value) + b"\n"
        except ValueError as exc:
            raise StorageError(str(exc)) from exc
        return self.write_bytes(relative, payload)

    def write_record(self, relative: str, record: Record, *, redact: bool = True) -> str:
        return self.write_json(relative, record.to_dict(), redact=True) if redact else self.write_bytes(relative, record_json(record) + b"\n")

    def read_bytes(self, relative: str) -> bytes:
        target = self.path(relative)
        _reject_symlink(target)
        try:
            return target.read_bytes()
        except OSError as exc:
            raise StorageError(f"cannot read artifact {relative}") from exc

    def read_json(self, relative: str, *, require_canonical: bool = False) -> Any:
        payload = self.read_bytes(relative)
        try:
            value = strict_loads(payload.rstrip(b"\n"))
        except ValueError as exc:
            raise StorageError(f"invalid JSON artifact {relative}: {exc}") from exc
        if require_canonical and canonical_bytes(value) + b"\n" != payload:
            raise StorageError(f"JSON artifact {relative} is not canonical")
        return value

    def append_jsonl(self, relative: str, value: Any, *, redact: bool = True) -> str:
        target = self.path(relative)
        parent = target.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _mode(parent, 0o700)
        encoded = canonical_bytes(_redact(value) if redact else value) + b"\n"
        _reject_symlink(target)
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "ab", closefd=True) as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise StorageError(f"cannot append JSONL artifact {relative}") from exc
        _mode(target, 0o600)
        _fsync_directory(parent)
        return _safe_relative(relative)

    @contextmanager
    def run_lock(self) -> Iterator[None]:
        lock_path = self.root / ".run.lock"
        _reject_symlink(lock_path)
        fd = -1
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                raise LockBusyError("run lock is already held") from exc
            raise StorageError("cannot acquire run lock") from exc
        with self._local_lock:
            self._run_lock_fd = fd
        try:
            yield
        finally:
            with self._local_lock:
                self._run_lock_fd = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


_TRANSITIONS: dict[str, set[str]] = {
    AttemptState.PLANNED.value: {AttemptState.REQUEST_STAGED.value},
    AttemptState.REQUEST_STAGED.value: {AttemptState.DISPATCH_STARTED.value},
    AttemptState.DISPATCH_STARTED.value: {AttemptState.RESPONSE_STAGED.value, AttemptState.UNCERTAIN_DISPATCH.value},
    AttemptState.RESPONSE_STAGED.value: {AttemptState.VALIDATED.value},
    AttemptState.VALIDATED.value: {AttemptState.TERMINAL.value},
    AttemptState.UNCERTAIN_DISPATCH.value: {AttemptState.TERMINAL.value},
    AttemptState.TERMINAL.value: set(),
}
_RESUMABLE = {AttemptState.PLANNED.value, AttemptState.REQUEST_STAGED.value, AttemptState.RESPONSE_STAGED.value, AttemptState.VALIDATED.value}


def _event_hash(event: Mapping[str, Any]) -> str:
    return domain_hash("lean-eval/state-event", canonical_bytes(dict(event)))


@dataclass(frozen=True)
class AttemptStateSnapshot:
    attempt_id: str
    state: str
    revision: int
    history: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    uncertain_dispatch: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttemptStateSnapshot":
        return cls(
            attempt_id=str(value["attempt_id"]), state=str(value["state"]), revision=int(value["revision"]),
            history=tuple(dict(item) for item in value.get("history", [])), metadata=dict(value.get("metadata", {})),
            uncertain_dispatch=bool(value.get("uncertain_dispatch", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "attempt-state-v1", "attempt_id": self.attempt_id, "state": self.state, "revision": self.revision, "history": [dict(item) for item in self.history], "metadata": dict(self.metadata), "uncertain_dispatch": self.uncertain_dispatch}


class AttemptStateStore:
    """Crash-safe state machine for attempts in one run directory."""

    def __init__(self, store: ArtifactStore | str | os.PathLike[str]) -> None:
        self.artifacts = store if isinstance(store, ArtifactStore) else ArtifactStore(store)
        self.attempt_root = self.artifacts.mkdir("attempts")
        self._lock = threading.RLock()

    def _attempt_dir(self, attempt: str) -> Path:
        _safe_component(attempt, label="attempt_id")
        target = self.attempt_root / attempt
        _reject_symlink(target)
        return target

    def _state_path(self, attempt: str) -> Path:
        return self._attempt_dir(attempt) / "state.json"

    def _lock_path(self, attempt: str) -> Path:
        return self._attempt_dir(attempt) / ".state.lock"

    @contextmanager
    def _attempt_lock(self, attempt: str) -> Iterator[None]:
        directory = self._attempt_dir(attempt)
        directory.mkdir(mode=0o700, exist_ok=True)
        _reject_symlink(directory)
        _mode(directory, 0o700)
        fd = os.open(self._lock_path(attempt), os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _read(self, attempt: str) -> AttemptStateSnapshot:
        path = self._state_path(attempt)
        _reject_symlink(path)
        try:
            value = strict_loads(path.read_bytes())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StorageError(f"invalid state for attempt {attempt}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != "attempt-state-v1":
            raise StorageError(f"unsupported state schema for attempt {attempt}")
        snapshot = AttemptStateSnapshot.from_dict(value)
        if snapshot.attempt_id != attempt or snapshot.state not in _TRANSITIONS:
            raise StorageError("invalid attempt state identity")
        if not snapshot.history or len(snapshot.history) != snapshot.revision + 1:
            raise StorageError(f"invalid state history for attempt {attempt}")
        previous: str | None = None
        for index, event in enumerate(snapshot.history):
            if not isinstance(event, dict) or event.get("revision") != index:
                raise StorageError(f"invalid state revision for attempt {attempt}")
            recorded = event.get("event_hash")
            unsigned = {key: item for key, item in event.items() if key != "event_hash"}
            if not isinstance(recorded, str) or _event_hash(unsigned) != recorded:
                raise StorageError(f"state event hash mismatch for attempt {attempt}")
            state = event.get("state")
            if not isinstance(state, str):
                raise StorageError(f"state event has no state for attempt {attempt}")
            if index == 0:
                if state != AttemptState.PLANNED.value or "from" in event:
                    raise StorageError(f"state history does not start at planned for attempt {attempt}")
            elif event.get("from") != previous or state not in _TRANSITIONS.get(previous or "", set()):
                raise StorageError(f"invalid state transition history for attempt {attempt}")
            previous = state
        if previous != snapshot.state:
            raise StorageError(f"state history tail disagrees for attempt {attempt}")
        return snapshot

    def _write(self, snapshot: AttemptStateSnapshot) -> None:
        _write_atomic(self._state_path(snapshot.attempt_id), canonical_bytes(snapshot.to_dict()) + b"\n", mode=0o600)

    def create_attempt(self, attempt: str, *, metadata: Mapping[str, Any] | None = None) -> AttemptStateSnapshot:
        _safe_component(attempt, label="attempt_id")
        with self._lock, self._attempt_lock(attempt):
            path = self._state_path(attempt)
            if path.exists() or path.is_symlink():
                if path.is_symlink():
                    raise PathSafetyError("state path may not be a symlink")
                existing = self._read(attempt)
                if metadata is not None and dict(metadata) != existing.metadata:
                    raise StateTransitionError("attempt already exists with different metadata")
                return existing
            event = {"revision": 0, "state": AttemptState.PLANNED.value}
            snapshot = AttemptStateSnapshot(attempt, AttemptState.PLANNED.value, 0, ({**event, "event_hash": _event_hash(event)},), dict(metadata or {}), False)
            self._write(snapshot)
            return snapshot

    def load(self, attempt: str) -> AttemptStateSnapshot:
        _safe_component(attempt, label="attempt_id")
        with self._lock:
            return self._read(attempt)

    def transition(self, attempt: str, state: str | AttemptState, *, expected: str | AttemptState | None = None, metadata: Mapping[str, Any] | None = None, reconcile: bool = False) -> AttemptStateSnapshot:
        target = state.value if isinstance(state, AttemptState) else state
        if target not in _TRANSITIONS:
            raise StateTransitionError(f"unknown target state {target!r}")
        with self._lock, self._attempt_lock(attempt):
            current = self._read(attempt)
            expected_value = expected.value if isinstance(expected, AttemptState) else expected
            if expected_value is not None and current.state != expected_value:
                raise StateTransitionError(f"expected {expected_value!r}, found {current.state!r} for {attempt}")
            if target == current.state:
                if current.state == AttemptState.TERMINAL.value and metadata:
                    raise StateTransitionError("terminal attempt is immutable")
                return current
            if target not in _TRANSITIONS[current.state]:
                raise StateTransitionError(f"cannot transition {current.state!r} to {target!r}")
            if current.state == AttemptState.UNCERTAIN_DISPATCH.value and not reconcile:
                raise StateTransitionError("uncertain dispatch requires explicit reconciliation")
            revision = current.revision + 1
            event: dict[str, Any] = {"revision": revision, "from": current.state, "state": target}
            if metadata:
                event["metadata"] = dict(metadata)
            event["event_hash"] = _event_hash(event)
            merged = dict(current.metadata)
            if metadata:
                merged.update(metadata)
            snapshot = AttemptStateSnapshot(attempt, target, revision, current.history + (event,), merged, target == AttemptState.UNCERTAIN_DISPATCH.value)
            self._write(snapshot)
            if target == AttemptState.TERMINAL.value:
                marker = self._attempt_dir(attempt) / "complete"
                marker_value = {"schema_version": "attempt-complete-v1", "attempt_id": attempt, "revision": revision, "state_hash": hashlib.sha256(payload_for_hash(snapshot)).hexdigest()}
                _write_atomic(marker, canonical_bytes(marker_value) + b"\n", mode=0o600)
            return snapshot

    def mark_uncertain_dispatch(self, attempt: str, *, metadata: Mapping[str, Any] | None = None) -> AttemptStateSnapshot:
        return self.transition(attempt, AttemptState.UNCERTAIN_DISPATCH, metadata=metadata)

    def reconcile(self, attempt: str, *, metadata: Mapping[str, Any] | None = None) -> AttemptStateSnapshot:
        return self.transition(attempt, AttemptState.TERMINAL, metadata=metadata, reconcile=True)

    def recover(self) -> tuple[AttemptStateSnapshot, ...]:
        recovered: list[AttemptStateSnapshot] = []
        if not self.attempt_root.exists():
            return ()
        for directory in sorted(self.attempt_root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.is_symlink() or not (directory / "state.json").exists():
                continue
            snapshot = self._read(directory.name)
            if snapshot.state == AttemptState.DISPATCH_STARTED.value:
                snapshot = self.transition(directory.name, AttemptState.UNCERTAIN_DISPATCH, expected=AttemptState.DISPATCH_STARTED, metadata={"recovered": True})
            recovered.append(snapshot)
        return tuple(recovered)

    def resume_candidates(self, *, include_uncertain: bool = False) -> tuple[AttemptStateSnapshot, ...]:
        self.recover()
        result: list[AttemptStateSnapshot] = []
        if not self.attempt_root.exists():
            return ()
        for directory in sorted(self.attempt_root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                snapshot = self._read(directory.name)
            except StorageError:
                continue
            if snapshot.state in _RESUMABLE or (include_uncertain and snapshot.state == AttemptState.UNCERTAIN_DISPATCH.value):
                result.append(snapshot)
        return tuple(result)

    def can_replay(self, attempt: str) -> bool:
        return self.load(attempt).state in _RESUMABLE

    def is_complete(self, attempt: str) -> bool:
        return self.load(attempt).state == AttemptState.TERMINAL.value and (self._attempt_dir(attempt) / "complete").is_file()


def payload_for_hash(snapshot: AttemptStateSnapshot) -> bytes:
    return canonical_bytes(snapshot.to_dict())


ProtectedArtifactStore = ArtifactStore
DurableAttemptStore = AttemptStateStore
RunStateStore = AttemptStateStore
StateStore = AttemptStateStore


__all__ = [
    "ArtifactStore", "AttemptStateSnapshot", "AttemptStateStore", "DurableAttemptStore", "LockBusyError", "PathSafetyError",
    "ProtectedArtifactStore", "RunStateStore", "StateStore", "StateTransitionError", "StorageError", "payload_for_hash",
]
