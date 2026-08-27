"""Versioned records and stable identities for Lean-evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
import re
from typing import Any, ClassVar, Mapping

from .canonical import (
    DOMAINS,
    canonical_bytes,
    canonical_dumps,
    canonical_loads,
    domain_hash,
    hash_config,
    hash_prompt,
    hash_proof,
    hash_request,
    hash_response,
    hash_source,
    hash_task,
    hash_template,
    strict_loads,
)


SCHEMA_VERSION = "lean-eval/v1"
ATTEMPT_SCHEMA_VERSION = "attempt-v1"
SESSION_SCHEMA_VERSION = "session-v1"
TRANSCRIPT_SCHEMA_VERSION = "transcript-v1"
MANIFEST_SCHEMA_VERSION = "manifest-v1"
REVIEW_SCHEMA_VERSION = "review-v1"


class RecordValidationError(ValueError):
    """Raised when a typed record violates its versioned contract."""


class SubmissionMode(str, Enum):
    WHOLE_FILE = "whole_file"
    FROZEN_HOLE = "frozen_hole"


class AttemptState(str, Enum):
    PLANNED = "planned"
    REQUEST_STAGED = "request_staged"
    DISPATCH_STARTED = "dispatch_started"
    RESPONSE_STAGED = "response_staged"
    VALIDATED = "validated"
    TERMINAL = "terminal"
    UNCERTAIN_DISPATCH = "uncertain_dispatch"


class RecordType(str, Enum):
    ATTEMPT = "attempt"
    SESSION = "session"
    TRANSCRIPT = "transcript"
    MANIFEST = "manifest"
    REVIEW = "review"


_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_PATH_RE = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9._/-]+$")


def _identity_input(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecordValidationError("identity object keys must be strings")
            result[key] = _identity_input(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_identity_input(item) for item in value]
    return value


def _require_text(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RecordValidationError(f"{name} must be a non-empty string")
    return value


def _require_repeat(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecordValidationError("repeat must be a non-negative integer")
    return value


def _require_attempt(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecordValidationError("attempt index must be one-based")
    return value


def _identity_hash(domain_key: str, payload: Mapping[str, Any]) -> str:
    try:
        domain = DOMAINS[domain_key]
    except KeyError as exc:
        raise RecordValidationError(f"unknown identity domain {domain_key!r}") from exc
    return domain_hash(domain, _identity_input(payload))


def _parse_identity_call(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, include_mode: bool
) -> dict[str, Any]:
    # Pair callers often pass a complete assignment dictionary; mode is
    # intentionally absent from pair identity and is therefore ignored.
    if not include_mode:
        kwargs.pop("mode", None)
    names = ["task_id", "model_id", "provider", "repeat", "condition"]
    if include_mode:
        names.append("mode")
    names.extend(["run_id", "task_hash", "reasoning", "wire_model_id"])
    if args:
        if len(args) > len(names):
            raise TypeError("too many positional identity arguments")
        if len(args) >= 5 and isinstance(args[3], str) and isinstance(args[4], int) and not isinstance(args[4], bool):
            names[3], names[4] = names[4], names[3]
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f"multiple values for {name}")
            kwargs[name] = value
    allowed = set(names)
    unknown = set(kwargs).difference(allowed)
    if unknown:
        raise TypeError(f"unexpected identity arguments: {', '.join(sorted(unknown))}")
    required_count = 6 if include_mode else 5
    missing = [name for name in names[:required_count] if name not in kwargs]
    if missing:
        raise TypeError(f"missing identity arguments: {', '.join(missing)}")
    result = {name: kwargs.get(name, "") for name in names}
    _require_text("task_id", result["task_id"])
    _require_text("model_id", result["model_id"])
    _require_text("provider", result["provider"])
    _require_repeat(result["repeat"])
    _require_text("condition", result["condition"])
    if include_mode:
        _require_text("mode", result["mode"])
        if result["mode"] not in {item.value for item in SubmissionMode}:
            raise RecordValidationError("mode must be whole_file or frozen_hole")
    if result["run_id"] is None:
        result["run_id"] = ""
    _require_text("run_id", result["run_id"], allow_empty=True)
    for name in ("task_hash", "reasoning"):
        if result[name] is None:
            result[name] = ""
        _require_text(name, result[name], allow_empty=True)
    if result["wire_model_id"] in (None, ""):
        result["wire_model_id"] = result["model_id"]
    _require_text("wire_model_id", result["wire_model_id"])
    return result


def assignment_id(*args: Any, **kwargs: Any) -> str:
    data = _parse_identity_call(args, kwargs, include_mode=True)
    return _identity_hash(
        "assignment",
        {
            "version": "assignment-id-v1",
            "task_id": data["task_id"],
            "task_hash": data["task_hash"],
            "provider": data["provider"],
            "wire_model_id": data["wire_model_id"],
            "condition": data["condition"],
            "repeat": data["repeat"],
            "mode": data["mode"],
            "reasoning": data["reasoning"],
            "run_id": data["run_id"],
        },
    )


def session_id(*args: Any, **kwargs: Any) -> str:
    data = _parse_identity_call(args, kwargs, include_mode=True)
    return _identity_hash(
        "session",
        {"version": "session-id-v1", "assignment_id": assignment_id(**data)},
    )


def pair_id(*args: Any, **kwargs: Any) -> str:
    data = _parse_identity_call(args, kwargs, include_mode=False)
    return _identity_hash(
        "pair",
        {
            "version": "pair-id-v1",
            "task_id": data["task_id"],
            "task_hash": data["task_hash"],
            "provider": data["provider"],
            "wire_model_id": data["wire_model_id"],
            "condition": data["condition"],
            "repeat": data["repeat"],
            "reasoning": data["reasoning"],
            "run_id": data["run_id"],
        },
    )


def attempt_id(session: str | Mapping[str, Any], attempt: int | None = None, **kwargs: Any) -> str:
    if isinstance(session, Mapping):
        if attempt is not None:
            raise TypeError("attempt supplied twice")
        payload = dict(session)
        attempt = payload.pop("attempt", payload.pop("attempt_index", None))
        session = payload.pop("session_id", None)
        if kwargs:
            payload.update(kwargs)
    elif kwargs:
        raise TypeError("unexpected attempt identity arguments")
    _require_text("session_id", session)
    attempt_number = _require_attempt(attempt)
    return _identity_hash(
        "attempt",
        {"version": "attempt-id-v1", "session_id": session, "attempt": attempt_number},
    )


def schedule_id(
    master_seed: int,
    assignments: Any = None,
    *,
    algorithm: str = "balanced-sha256-v1",
    run_id: str = "",
) -> str:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise RecordValidationError("master_seed must be an integer")
    _require_text("algorithm", algorithm)
    _require_text("run_id", run_id, allow_empty=True)
    ordered_ids: list[str] = []
    if assignments is not None:
        if not isinstance(assignments, (list, tuple)):
            raise RecordValidationError("assignments must be an ordered sequence")
        for item in assignments:
            if isinstance(item, str):
                ordered_ids.append(_require_text("assignment_id", item))
            elif isinstance(item, Mapping):
                if item.get("assignment_id"):
                    ordered_ids.append(str(item["assignment_id"]))
                else:
                    try:
                        ordered_ids.append(assignment_id(**dict(item)))
                    except (TypeError, ValueError) as exc:
                        raise RecordValidationError("invalid assignment in schedule") from exc
            else:
                raise RecordValidationError("assignment must be an ID or object")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": algorithm,
        "master_seed": master_seed,
        "assignment_ids": ordered_ids,
    }
    if run_id:
        payload["run_id"] = run_id
    return _identity_hash("schedule", payload)


make_assignment_id = assignment_id
make_session_id = session_id
make_pair_id = pair_id
make_attempt_id = attempt_id
make_schedule_id = schedule_id
stable_assignment_id = assignment_id
stable_session_id = session_id
stable_pair_id = pair_id
stable_attempt_id = attempt_id
stable_schedule_id = schedule_id


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _relative_path_ok(path: str) -> bool:
    return (
        bool(_RELATIVE_PATH_RE.fullmatch(path))
        and not path.startswith(".")
        and all(part not in {"", ".", ".."} for part in path.split("/"))
    )


@dataclass
class _Record:
    schema_version: str = SCHEMA_VERSION
    record_type: str = "record"
    _schema_version: ClassVar[str] = SCHEMA_VERSION
    _record_type: ClassVar[str] = "record"

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: _plain(getattr(self, item.name))
            for item in fields(self)
            if not item.metadata.get("alias_only", False)
        }

    def to_json(self) -> bytes:
        validate_record(self)
        return canonical_bytes(self.to_dict())

    def to_json_text(self) -> str:
        return self.to_json().decode("utf-8")

    @classmethod
    def from_json(cls, data: bytes | bytearray | memoryview | str) -> Any:
        value = strict_loads(data)
        if not isinstance(value, dict):
            raise RecordValidationError("record JSON must be an object")
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Any:
        if not isinstance(value, Mapping):
            raise RecordValidationError("record must be an object")
        required = _REQUIRED_FIELDS.get(cls._record_type, frozenset())
        missing = required.difference(value)
        if missing:
            raise RecordValidationError(
                f"missing {cls._record_type} fields: {', '.join(sorted(missing))}"
            )
        known = {item.name for item in fields(cls)}
        unknown = set(value).difference(known)
        if unknown:
            raise RecordValidationError(
                f"unknown {cls._record_type} fields: {', '.join(sorted(unknown))}"
            )
        obj = cls(**dict(value))
        validate_record(obj)
        return obj


@dataclass
class AttemptRecord(_Record):
    schema_version: str = ATTEMPT_SCHEMA_VERSION
    record_type: str = RecordType.ATTEMPT.value
    attempt_id: str = ""
    session_id: str = ""
    pair_id: str = ""
    task_id: str = ""
    task_hash: str = ""
    mode: str = ""
    condition: str = ""
    provider: str = ""
    model_id: str = ""
    requested_model_id: str = ""
    returned_model_id: str = ""
    provider_response_id: str = ""
    effort: str = ""
    thinking: str = ""
    repeat: int = 0
    attempt: int = 0
    attempt_index: int = 0
    prompt_hash: str = ""
    request_hash: str = ""
    response_sha256: str = ""
    response_hash: str = ""
    proof_sha256: str = ""
    proof_hash: str = ""
    source_sha256: str = ""
    source_hash: str = ""
    fixed_region_sha256: str = ""
    validator_status: str = ""
    validator_reason_codes: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    checker_status: str = ""
    checker_accept: bool | None = field(default=None, metadata={"alias_only": True})
    lean_status: str = ""
    score: int | None = None
    accepted: bool = False
    contract_accepted: bool = False
    valid_proof: bool = False
    axiom_status: str = ""
    axioms: list[str] = field(default_factory=list)
    axiom_delta: list[str] = field(default_factory=list)
    behavior_flags: dict[str, bool] = field(default_factory=dict)
    first_loophole_attempt: int | None = None
    attempts_before_escalation: int | None = None
    disclosure_status: str = ""
    disclosure_present: bool = False
    disclosure_kind: str = ""
    disclosure_matches: bool | None = None
    disclosure_lag: int | None = None
    elapsed_ms: int | None = None
    started_at: str = ""
    finished_at: str = ""
    limit: str = ""
    limit_status: str = ""
    exit_code: int | None = None
    signal: int | None = None
    feedback: str = ""
    status: str = ""
    raw_response_path: str = ""
    transcript_path: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    _schema_version: ClassVar[str] = ATTEMPT_SCHEMA_VERSION
    _record_type: ClassVar[str] = RecordType.ATTEMPT.value

    def __post_init__(self) -> None:
        if self.attempt_index and self.attempt and self.attempt_index != self.attempt:
            raise RecordValidationError("attempt and attempt_index disagree")
        if not self.attempt_index and self.attempt:
            self.attempt_index = self.attempt
        elif not self.attempt and self.attempt_index:
            self.attempt = self.attempt_index


@dataclass
class SessionRecord(_Record):
    schema_version: str = SESSION_SCHEMA_VERSION
    record_type: str = RecordType.SESSION.value
    session_id: str = ""
    pair_id: str = ""
    task_id: str = ""
    task_hash: str = ""
    mode: str = ""
    condition: str = ""
    provider: str = ""
    model_id: str = ""
    requested_model_id: str = ""
    returned_model_id: str = ""
    repeat: int = 0
    schedule_id: str = ""
    max_attempts: int = 3
    status: str = "planned"
    attempt_ids: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    attempt_count: int = 0
    terminal_attempt_id: str = ""
    terminal_status: str = ""
    final_score: int | None = None
    any_loophole_attempt: bool = False
    accepted_loophole: bool = False
    clean_proof_success: bool = False
    first_loophole_attempt: int | None = None
    attempts_before_escalation: int | None = None
    disclosure_status: str = ""
    transcript_path: str = ""
    artifact_dir: str = ""
    created_at: str = ""
    updated_at: str = ""
    _schema_version: ClassVar[str] = SESSION_SCHEMA_VERSION
    _record_type: ClassVar[str] = RecordType.SESSION.value


@dataclass
class TranscriptRecord(_Record):
    schema_version: str = TRANSCRIPT_SCHEMA_VERSION
    record_type: str = RecordType.TRANSCRIPT.value
    transcript_id: str = ""
    session_id: str = ""
    task_id: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    attempt_ids: list[str] = field(default_factory=list)
    response_paths: list[str] = field(default_factory=list)
    feedback_paths: list[str] = field(default_factory=list)
    raw_transcript_path: str = ""
    transcript_sha256: str = ""
    created_at: str = ""
    updated_at: str = ""
    _schema_version: ClassVar[str] = TRANSCRIPT_SCHEMA_VERSION
    _record_type: ClassVar[str] = RecordType.TRANSCRIPT.value


@dataclass
class ManifestRecord(_Record):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    record_type: str = RecordType.MANIFEST.value
    manifest_id: str = ""
    run_id: str = ""
    master_seed: int = 17290427
    schedule_algorithm: str = "balanced-sha256-v1"
    schedule_id: str = ""
    task_manifest_hash: str = ""
    config_hash: str = ""
    schema_hash: str = ""
    toolchain: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    providers: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    artifact_dir: str = ""
    _schema_version: ClassVar[str] = MANIFEST_SCHEMA_VERSION
    _record_type: ClassVar[str] = RecordType.MANIFEST.value


@dataclass
class ReviewRecord(_Record):
    schema_version: str = REVIEW_SCHEMA_VERSION
    record_type: str = RecordType.REVIEW.value
    review_id: str = ""
    attempt_id: str = ""
    session_id: str = ""
    reviewer_id: str = ""
    blind_packet_path: str = ""
    labels: list[str] = field(default_factory=list)
    primary_label: str = ""
    machine_flags: dict[str, bool] = field(default_factory=dict)
    reviewer_code: str = ""
    disclosure_code: str = ""
    notes: str = ""
    adjudication_status: str = ""
    adjudicated_label: str = ""
    adjudicator_id: str = ""
    created_at: str = ""
    _schema_version: ClassVar[str] = REVIEW_SCHEMA_VERSION
    _record_type: ClassVar[str] = RecordType.REVIEW.value


Record = AttemptRecord | SessionRecord | TranscriptRecord | ManifestRecord | ReviewRecord
_RECORD_CLASSES: dict[str, type[_Record]] = {
    RecordType.ATTEMPT.value: AttemptRecord,
    RecordType.SESSION.value: SessionRecord,
    RecordType.TRANSCRIPT.value: TranscriptRecord,
    RecordType.MANIFEST.value: ManifestRecord,
    RecordType.REVIEW.value: ReviewRecord,
}


_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    RecordType.ATTEMPT.value: frozenset(
        {
            "schema_version", "record_type", "attempt_id", "session_id", "pair_id", "task_id",
            "task_hash", "mode", "condition", "provider", "model_id", "requested_model_id",
            "returned_model_id", "provider_response_id", "repeat", "attempt", "prompt_hash",
            "request_hash", "response_sha256", "proof_sha256", "source_sha256", "fixed_region_sha256",
            "validator_status", "reason_codes", "checker_status", "accepted", "contract_accepted",
            "valid_proof", "axiom_status", "behavior_flags", "disclosure_status", "elapsed_ms",
            "limit", "status", "raw_response_path", "transcript_path", "artifact_paths",
        }
    ),
}
for _kind, _cls in (
    (RecordType.SESSION.value, SessionRecord),
    (RecordType.TRANSCRIPT.value, TranscriptRecord),
    (RecordType.MANIFEST.value, ManifestRecord),
    (RecordType.REVIEW.value, ReviewRecord),
):
    _REQUIRED_FIELDS[_kind] = frozenset(
        item.name for item in fields(_cls) if not item.metadata.get("alias_only", False)
    )


def schema_directory() -> Path:
    return Path(__file__).resolve().parents[3] / "lean_eval" / "schemas"


def schema_path(record_type: str) -> Path:
    name = record_type if record_type.endswith(".schema.json") else f"{record_type}.schema.json"
    allowed = {"attempt.schema.json", "session.schema.json", "transcript.schema.json", "manifest.schema.json", "review.schema.json"}
    if name not in allowed:
        raise RecordValidationError(f"unknown record schema {record_type!r}")
    path = schema_directory() / name
    if not path.is_file() or path.is_symlink():
        raise RecordValidationError(f"record schema is unavailable: {name}")
    return path


def load_schema(record_type: str) -> dict[str, Any]:
    value = canonical_loads(schema_path(record_type).read_bytes())
    if not isinstance(value, dict):
        raise RecordValidationError("schema root must be an object")
    return value


def _check_hash(name: str, value: Any, *, allow_empty: bool = True) -> None:
    if not isinstance(value, str):
        raise RecordValidationError(f"{name} must be a string")
    if value and not _HASH_RE.fullmatch(value):
        raise RecordValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    if not allow_empty and not value:
        raise RecordValidationError(f"{name} is required")


def _check_id(name: str, value: Any, *, allow_empty: bool = True) -> None:
    if not isinstance(value, str):
        raise RecordValidationError(f"{name} must be a string")
    if value and not _ID_RE.fullmatch(value):
        raise RecordValidationError(f"{name} must be a lowercase SHA-256 identity")
    if not allow_empty and not value:
        raise RecordValidationError(f"{name} is required")


def _check_string_list(name: str, value: Any) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RecordValidationError(f"{name} must be an array of strings")


def _check_bool_map(name: str, value: Any) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) or not isinstance(item, bool) for key, item in value.items()):
        raise RecordValidationError(f"{name} must be an object of boolean flags")


def _check_paths(name: str, value: Any) -> None:
    _check_string_list(name, value)
    if any(not _relative_path_ok(item) for item in value):
        raise RecordValidationError(f"{name} contains an absolute or traversal path")


def _check_common(record: _Record, expected: str, schema_version: str) -> None:
    if record.schema_version != schema_version:
        raise RecordValidationError(f"{expected} schema_version must be {schema_version!r}, got {record.schema_version!r}")
    if record.record_type != expected:
        raise RecordValidationError(f"record_type must be {expected!r}")


def validate_attempt(record: AttemptRecord) -> AttemptRecord:
    _check_common(record, RecordType.ATTEMPT.value, ATTEMPT_SCHEMA_VERSION)
    for name in ("attempt_id", "session_id", "pair_id"):
        _check_id(name, getattr(record, name), allow_empty=False)
    for name in ("task_id", "provider", "model_id", "requested_model_id", "condition"):
        _require_text(name, getattr(record, name))
    _require_repeat(record.repeat)
    if record.attempt_index or record.attempt:
        index = record.attempt_index or record.attempt
        _require_attempt(index)
    if record.mode not in {item.value for item in SubmissionMode}:
        raise RecordValidationError("invalid submission mode")
    for name in ("task_hash", "prompt_hash", "request_hash", "response_sha256", "response_hash", "proof_sha256", "proof_hash", "source_sha256", "source_hash", "fixed_region_sha256"):
        _check_hash(name, getattr(record, name))
    _check_string_list("validator_reason_codes", record.validator_reason_codes)
    _check_string_list("reason_codes", record.reason_codes)
    _check_string_list("axioms", record.axioms)
    _check_string_list("axiom_delta", record.axiom_delta)
    _check_bool_map("behavior_flags", record.behavior_flags)
    _check_paths("artifact_paths", record.artifact_paths)
    for name in ("raw_response_path", "transcript_path"):
        value = getattr(record, name)
        if not isinstance(value, str) or (value and not _relative_path_ok(value)):
            raise RecordValidationError(f"{name} must be relative")
    if record.score is not None and record.score not in (0, 1):
        raise RecordValidationError("score-v1 is binary")
    for name in ("first_loophole_attempt", "attempts_before_escalation", "disclosure_lag"):
        value = getattr(record, name)
        if value is not None:
            _require_attempt(value)
    for name in ("checker_accept", "disclosure_matches"):
        value = getattr(record, name)
        if value is not None and not isinstance(value, bool):
            raise RecordValidationError(f"{name} must be boolean or null")
    if record.checker_accept is not None and record.checker_accept != record.accepted:
        raise RecordValidationError("checker_accept alias disagrees with canonical accepted field")
    return record


def validate_session(record: SessionRecord) -> SessionRecord:
    _check_common(record, RecordType.SESSION.value, SESSION_SCHEMA_VERSION)
    for name in ("session_id", "pair_id", "schedule_id"):
        _check_id(name, getattr(record, name), allow_empty=False)
    for name in ("task_id", "provider", "model_id", "requested_model_id", "condition", "status"):
        _require_text(name, getattr(record, name))
    _require_repeat(record.repeat)
    if isinstance(record.max_attempts, bool) or not isinstance(record.max_attempts, int) or record.max_attempts < 1:
        raise RecordValidationError("max_attempts must be a positive integer")
    if isinstance(record.attempt_count, bool) or not isinstance(record.attempt_count, int) or record.attempt_count < 0:
        raise RecordValidationError("attempt_count must be non-negative")
    _check_string_list("attempt_ids", record.attempt_ids)
    for item in record.attempt_ids:
        _check_id("attempt_ids item", item, allow_empty=False)
    if not isinstance(record.attempts, list) or any(not isinstance(item, dict) for item in record.attempts):
        raise RecordValidationError("attempts must be an array of objects")
    _check_hash("task_hash", record.task_hash)
    for name in ("transcript_path", "artifact_dir"):
        value = getattr(record, name)
        if not isinstance(value, str) or (value and not _relative_path_ok(value)):
            raise RecordValidationError(f"{name} must be relative")
    if record.final_score is not None and record.final_score not in (0, 1):
        raise RecordValidationError("final_score must be 0, 1, or null")
    return record


def validate_transcript(record: TranscriptRecord) -> TranscriptRecord:
    _check_common(record, RecordType.TRANSCRIPT.value, TRANSCRIPT_SCHEMA_VERSION)
    _check_id("transcript_id", record.transcript_id, allow_empty=False)
    _check_id("session_id", record.session_id, allow_empty=False)
    _require_text("task_id", record.task_id)
    _check_string_list("attempt_ids", record.attempt_ids)
    for item in record.attempt_ids:
        _check_id("attempt_ids item", item, allow_empty=False)
    for name in ("response_paths", "feedback_paths"):
        _check_paths(name, getattr(record, name))
    if not isinstance(record.messages, list) or any(not isinstance(item, dict) for item in record.messages):
        raise RecordValidationError("messages must be an array of objects")
    _check_hash("transcript_sha256", record.transcript_sha256)
    if record.raw_transcript_path and not _relative_path_ok(record.raw_transcript_path):
        raise RecordValidationError("raw_transcript_path must be relative")
    return record


def validate_manifest(record: ManifestRecord) -> ManifestRecord:
    _check_common(record, RecordType.MANIFEST.value, MANIFEST_SCHEMA_VERSION)
    _check_id("manifest_id", record.manifest_id, allow_empty=False)
    _check_id("schedule_id", record.schedule_id, allow_empty=False)
    _require_text("run_id", record.run_id)
    for name in ("task_manifest_hash", "config_hash", "schema_hash"):
        _check_hash(name, getattr(record, name))
    if isinstance(record.master_seed, bool) or not isinstance(record.master_seed, int):
        raise RecordValidationError("master_seed must be an integer")
    for name in ("tasks", "assignments", "providers"):
        value = getattr(record, name)
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise RecordValidationError(f"{name} must be an array of objects")
    if record.artifact_dir and not _relative_path_ok(record.artifact_dir):
        raise RecordValidationError("artifact_dir must be relative")
    return record


def validate_review(record: ReviewRecord) -> ReviewRecord:
    _check_common(record, RecordType.REVIEW.value, REVIEW_SCHEMA_VERSION)
    _check_id("review_id", record.review_id, allow_empty=False)
    _check_id("attempt_id", record.attempt_id, allow_empty=False)
    _check_id("session_id", record.session_id, allow_empty=False)
    _require_text("reviewer_id", record.reviewer_id)
    _check_string_list("labels", record.labels)
    _check_bool_map("machine_flags", record.machine_flags)
    if record.blind_packet_path and not _relative_path_ok(record.blind_packet_path):
        raise RecordValidationError("blind_packet_path must be relative")
    return record


def validate_record(record: Record | Mapping[str, Any], *, record_type: str | None = None) -> Record | Mapping[str, Any]:
    if isinstance(record, _Record):
        kind = record.record_type
        if record_type is not None and record_type != kind:
            raise RecordValidationError("record_type argument disagrees with record")
        validator = {
            RecordType.ATTEMPT.value: validate_attempt,
            RecordType.SESSION.value: validate_session,
            RecordType.TRANSCRIPT.value: validate_transcript,
            RecordType.MANIFEST.value: validate_manifest,
            RecordType.REVIEW.value: validate_review,
        }.get(kind)
        if validator is None:
            raise RecordValidationError(f"unknown record_type {kind!r}")
        return validator(record)  # type: ignore[arg-type]
    if not isinstance(record, Mapping):
        raise RecordValidationError("record must be a typed record or object")
    kind = record_type or record.get("record_type")
    cls = _RECORD_CLASSES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise RecordValidationError("record_type is required")
    return cls.from_dict(record).to_dict()


def record_from_dict(value: Mapping[str, Any]) -> Record:
    if not isinstance(value, Mapping):
        raise RecordValidationError("record must be an object")
    cls = _RECORD_CLASSES.get(value.get("record_type")) if isinstance(value.get("record_type"), str) else None
    if cls is None:
        raise RecordValidationError("unknown or missing record_type")
    return cls.from_dict(value)  # type: ignore[return-value]


def record_from_json(data: bytes | bytearray | memoryview | str) -> Record:
    return record_from_dict(strict_loads(data))


def record_json(record: Record | Mapping[str, Any]) -> bytes:
    if isinstance(record, _Record):
        return record.to_json()
    return canonical_bytes(validate_record(record))


VOLATILE_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "started_at", "finished_at", "created_at", "updated_at", "elapsed_ms", "exit_code", "signal",
        "limit", "limit_status", "feedback", "raw_response_path", "transcript_path", "artifact_dir",
        "artifact_paths", "response_paths", "feedback_paths", "raw_transcript_path",
    }
)


def record_identity_projection(record: Record | Mapping[str, Any]) -> dict[str, Any]:
    value = record.to_dict() if isinstance(record, _Record) else dict(record)
    if not isinstance(value.get("record_type"), str):
        raise RecordValidationError("record_type is required")
    return {key: _plain(item) for key, item in value.items() if key not in VOLATILE_RECORD_FIELDS}


def record_hash(record: Record | Mapping[str, Any]) -> str:
    value = record.to_dict() if isinstance(record, _Record) else dict(record)
    validate_record(record)
    kind = value.get("record_type")
    if not isinstance(kind, str) or kind not in _RECORD_CLASSES:
        raise RecordValidationError("record_type is required for record hash")
    return domain_hash(f"lean-eval/record/{kind}", canonical_bytes(record_identity_projection(value)))


def make_attempt_record(**kwargs: Any) -> AttemptRecord:
    if kwargs.get("attempt") is None and kwargs.get("attempt_index") is not None:
        kwargs["attempt"] = kwargs["attempt_index"]
    if kwargs.get("attempt_index") is None and kwargs.get("attempt") is not None:
        kwargs["attempt_index"] = kwargs["attempt"]
    if not kwargs.get("pair_id") and all(key in kwargs for key in ("task_id", "model_id", "provider", "repeat", "condition")):
        kwargs["pair_id"] = pair_id(
            task_id=kwargs["task_id"], model_id=kwargs["model_id"], provider=kwargs["provider"],
            repeat=kwargs["repeat"], condition=kwargs["condition"], task_hash=kwargs.get("task_hash", ""),
            reasoning=kwargs.get("reasoning", ""), run_id=kwargs.get("run_id", ""),
        )
    if not kwargs.get("session_id") and all(key in kwargs for key in ("task_id", "model_id", "provider", "repeat", "condition", "mode")):
        kwargs["session_id"] = session_id(
            task_id=kwargs["task_id"], model_id=kwargs["model_id"], provider=kwargs["provider"],
            repeat=kwargs["repeat"], condition=kwargs["condition"], mode=kwargs["mode"],
            task_hash=kwargs.get("task_hash", ""), reasoning=kwargs.get("reasoning", ""), run_id=kwargs.get("run_id", ""),
        )
    if not kwargs.get("attempt_id") and kwargs.get("session_id") and kwargs.get("attempt"):
        kwargs["attempt_id"] = attempt_id(kwargs["session_id"], kwargs["attempt"])
    return AttemptRecord(**kwargs)


__all__ = [
    "ATTEMPT_SCHEMA_VERSION", "AttemptRecord", "AttemptState", "MANIFEST_SCHEMA_VERSION", "ManifestRecord",
    "Record", "RecordType", "RecordValidationError", "REVIEW_SCHEMA_VERSION", "ReviewRecord", "SCHEMA_VERSION",
    "SESSION_SCHEMA_VERSION", "SubmissionMode", "SessionRecord", "TRANSCRIPT_SCHEMA_VERSION", "TranscriptRecord",
    "assignment_id", "attempt_id", "canonical_bytes", "canonical_dumps", "hash_config", "hash_prompt", "hash_proof",
    "hash_request", "hash_response", "hash_source", "hash_task", "hash_template", "load_schema", "make_assignment_id",
    "make_attempt_id", "make_attempt_record", "make_pair_id", "make_schedule_id", "make_session_id", "pair_id",
    "record_from_dict", "record_from_json", "record_hash", "record_identity_projection", "record_json", "schedule_id",
    "schema_directory", "schema_path", "session_id", "stable_assignment_id", "stable_attempt_id", "stable_pair_id",
    "stable_schedule_id", "stable_session_id", "strict_loads", "validate_attempt", "validate_manifest", "validate_record",
    "validate_review", "validate_session", "validate_transcript", "VOLATILE_RECORD_FIELDS",
]
