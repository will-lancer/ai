"""Canonical JSON and length-framed identity hashes.

The proof-interface evaluator hashes exact source/response bytes and structured
records.  Structured values use a strict JSON representation; byte payloads
are preserved as-is so invalid UTF-8 and NUL bytes can still be identified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import re
from typing import Any


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented canonically."""


class CanonicalJSONError(CanonicalizationError):
    """Raised for malformed, ambiguous, or non-finite JSON."""


MAX_CANONICAL_BYTES = 16 * 1024 * 1024
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_FRAME_MAGIC = b"lean-reward-hacking/canonical-v1\0"
_DOMAIN_LENGTH_BYTES = 4
_PAYLOAD_LENGTH_BYTES = 8

DOMAINS: dict[str, str] = {
    "template": "lean-eval/template",
    "protected": "lean-eval/protected",
    "proof": "lean-eval/proof",
    "source": "lean-eval/source",
    "task": "lean-eval/task",
    "config": "lean-eval/config",
    "prompt": "lean-eval/prompt",
    "request": "lean-eval/request",
    "response": "lean-eval/response",
    "session": "lean-eval/session-id",
    "pair": "lean-eval/pair-id",
    "assignment": "lean-eval/assignment-id",
    "attempt": "lean-eval/attempt-id",
    "schedule": "lean-eval/schedule-id",
}


def _reject_string(value: str) -> None:
    if "\0" in value:
        raise CanonicalizationError("NUL code point is not allowed in canonical JSON")
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        raise CanonicalizationError("surrogate code point is not canonical UTF-8")


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite number at {path}")
        return value
    if isinstance(value, str):
        _reject_string(value)
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise CanonicalizationError(f"binary value at {path}; pass bytes as the hash payload")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"object key at {path} is not a string")
            _reject_string(key)
            if key in result:
                raise CanonicalizationError(f"duplicate object key {key!r} at {path}")
            result[key] = _canonical_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        raise CanonicalizationError(f"unordered collection at {path}")
    raise CanonicalizationError(f"unsupported value {type(value).__name__} at {path}")


def canonical_dumps(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    try:
        encoded = json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CanonicalizationError(str(exc)) from exc
    _reject_string(encoded)
    return encoded


def canonical_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes."""

    try:
        result = canonical_dumps(value).encode("utf-8", "strict")
    except UnicodeError as exc:
        raise CanonicalizationError(str(exc)) from exc
    if len(result) > MAX_CANONICAL_BYTES:
        raise CanonicalizationError("canonical JSON artifact exceeds size limit")
    return result


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise CanonicalJSONError(f"non-finite JSON constant: {value}")


def _validate_decoded(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(f"non-finite number at {path}")
        return value
    if isinstance(value, str):
        try:
            _reject_string(value)
        except CanonicalizationError as exc:
            raise CanonicalJSONError(str(exc)) from exc
        return value
    if isinstance(value, list):
        return [_validate_decoded(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _validate_decoded(item, path=f"{path}.{key}") for key, item in value.items()}
    raise CanonicalJSONError(f"unsupported decoded value at {path}")


def strict_loads(
    data: bytes | bytearray | memoryview | str,
    *,
    max_bytes: int = MAX_CANONICAL_BYTES,
) -> Any:
    """Decode strict UTF-8 JSON with duplicate and finite-number checks."""

    if isinstance(data, str):
        try:
            raw = data.encode("utf-8", "strict")
        except UnicodeError as exc:
            raise CanonicalJSONError(str(exc)) from exc
    elif isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
    else:
        raise CanonicalJSONError("JSON input must be text or bytes")
    if len(raw) > max_bytes:
        raise CanonicalJSONError("JSON input exceeds size limit")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise CanonicalJSONError("UTF-8 BOM is not allowed")
    if b"\0" in raw:
        raise CanonicalJSONError("NUL byte is not allowed")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise CanonicalJSONError(f"invalid UTF-8 at byte {exc.start}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except CanonicalJSONError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CanonicalJSONError(str(exc)) from exc
    return _validate_decoded(value)


def canonical_loads(
    data: bytes | bytearray | memoryview | str,
    *,
    require_canonical: bool = False,
    max_bytes: int = MAX_CANONICAL_BYTES,
) -> Any:
    value = strict_loads(data, max_bytes=max_bytes)
    raw = data.encode("utf-8", "strict") if isinstance(data, str) else bytes(data)
    if require_canonical and canonical_bytes(value) != raw:
        raise CanonicalJSONError("JSON bytes are not canonical")
    return value


def canonical_payload(value: Any) -> bytes:
    """Return canonical bytes, or preserve an exact byte payload."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        if len(payload) > MAX_CANONICAL_BYTES:
            raise CanonicalizationError("hash payload exceeds size limit")
        return payload
    return canonical_bytes(value)


def _validate_domain(domain: str) -> bytes:
    if not isinstance(domain, str) or not _DOMAIN_RE.fullmatch(domain):
        raise CanonicalizationError("invalid hash domain")
    try:
        result = domain.encode("ascii", "strict")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError("hash domain must be ASCII") from exc
    return result


def framed_bytes(domain: str, payload: bytes | bytearray | memoryview) -> bytes:
    """Build ``magic || uint32(domain_len) || domain || uint64(payload_len) || payload``."""

    domain_bytes = _validate_domain(domain)
    payload_bytes = bytes(payload)
    if len(payload_bytes) > (1 << 64) - 1:
        raise CanonicalizationError("hash payload is too long")
    return (
        _FRAME_MAGIC
        + len(domain_bytes).to_bytes(_DOMAIN_LENGTH_BYTES, "big")
        + domain_bytes
        + len(payload_bytes).to_bytes(_PAYLOAD_LENGTH_BYTES, "big")
        + payload_bytes
    )


def domain_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(framed_bytes(domain, canonical_payload(value))).hexdigest()


def hash_bytes(domain: str, payload: bytes | bytearray | memoryview) -> str:
    return domain_hash(domain, bytes(payload))


def _named_hash(name: str, value: Any) -> str:
    try:
        return domain_hash(DOMAINS[name], value)
    except KeyError as exc:
        raise CanonicalizationError(f"unknown identity domain {name!r}") from exc


def hash_template(value: Any) -> str:
    return _named_hash("template", value)


def hash_protected(value: Any) -> str:
    return _named_hash("protected", value)


def hash_proof(value: Any) -> str:
    return _named_hash("proof", value)


def hash_source(value: Any) -> str:
    return _named_hash("source", value)


def hash_task(value: Any) -> str:
    return _named_hash("task", value)


def hash_prompt(value: Any) -> str:
    return _named_hash("prompt", value)


def hash_request(value: Any) -> str:
    return _named_hash("request", value)


def hash_response(value: Any) -> str:
    return _named_hash("response", value)


def hash_assignment(value: Any) -> str:
    return _named_hash("assignment", value)


def hash_config(value: Any) -> str:
    return _named_hash("config", value)


def stable_set(values: Sequence[Any] | set[Any] | frozenset[Any]) -> list[Any]:
    """Return a deterministic ordered representation for a mathematical set."""

    if not isinstance(values, (set, frozenset, list, tuple)):
        raise CanonicalizationError("stable_set expects a set or ordered sequence")
    members = [_canonical_value(item) for item in values]
    return sorted(members, key=canonical_bytes)


HASH_VECTORS: dict[str, str] = {
    "lean-eval/template": "e8d3464a6c1d7bb95634a39d69cc7339bc9f2d72e00807ab02733ae1a5da7b50",
    "lean-eval/protected": "a1a17ce0d1bb6183e6e822772f37e2b32146477af44ab9b940ede13e84fdd03e",
    "lean-eval/proof": "716ec49087d54e40c1a2aa0d03e3b6caa1513da9025c89c8023b646c4da8aa1f",
    "lean-eval/source": "72cd1a8a360bc9708e0905b382df6f4ace55ee099aa37af2b56666878f6acd11",
    "lean-eval/task": "fc063b042d6b4a98e9cf2537fc9e6014f8c202f95e6533961ecaf995d77b5bb2",
}


def verify_hash_vectors() -> bool:
    return all(domain_hash(domain, b"") == expected for domain, expected in HASH_VECTORS.items())


# Compatibility names used by early smoke scripts.
canonical_json = canonical_dumps
canonical_json_bytes = canonical_bytes
strict_json_loads = strict_loads
hash_domain = domain_hash
framed_sha256 = domain_hash


__all__ = [
    "CanonicalizationError",
    "CanonicalJSONError",
    "DOMAINS",
    "HASH_VECTORS",
    "MAX_CANONICAL_BYTES",
    "canonical_bytes",
    "canonical_dumps",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_loads",
    "canonical_payload",
    "domain_hash",
    "framed_bytes",
    "framed_sha256",
    "hash_assignment",
    "hash_bytes",
    "hash_config",
    "hash_domain",
    "hash_prompt",
    "hash_proof",
    "hash_protected",
    "hash_request",
    "hash_response",
    "hash_source",
    "hash_task",
    "hash_template",
    "stable_set",
    "strict_json_loads",
    "strict_loads",
    "verify_hash_vectors",
]
