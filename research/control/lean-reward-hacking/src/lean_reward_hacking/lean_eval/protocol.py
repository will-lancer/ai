"""Strict model-response protocol for the Lean proof-interface study.

The provider boundary accepts exactly one JSON object.  Keeping this parser
independent of the compiler is important: malformed responses and unsafe
transport payloads must be rejected before a Lean process can be launched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Mapping

from .canonical import hash_proof, hash_response, hash_source


MAX_RESPONSE_BYTES = 64 * 1024
MAX_JSON_DEPTH = 16
MAX_DISCLOSURE_BYTES = 4096
RESPONSE_KEYS = frozenset({"kind", "text", "disclosure"})
RESPONSE_KINDS = frozenset({"proof", "file", "no_proof"})


class ReasonCode:
    """Stable machine-readable parser and validator reason codes."""

    SIZE_LIMIT = "size_limit"
    INVALID_UTF8 = "invalid_utf8"
    BOM = "bom"
    NUL = "nul"
    CONTROL_CHAR = "control_char"
    JSON_INVALID = "json_invalid"
    JSON_DUPLICATE_KEY = "json_duplicate_key"
    JSON_NONFINITE = "json_nonfinite"
    JSON_DEPTH = "json_depth"
    JSON_TRAILING = "json_trailing"
    JSON_SHAPE = "json_shape"
    UNKNOWN_KEY = "unknown_key"
    MISSING_KEY = "missing_key"
    INVALID_KIND = "invalid_kind"
    KIND_MISMATCH = "kind_mismatch"
    EMPTY_TEXT = "empty_text"
    NONEMPTY_NO_PROOF = "nonempty_no_proof"
    BARE_NO_PROOF = "bare_no_proof"
    CR = "cr"
    SURROGATE = "surrogate"


REASON_SIZE_LIMIT = ReasonCode.SIZE_LIMIT
REASON_INVALID_UTF8 = ReasonCode.INVALID_UTF8
REASON_BOM = ReasonCode.BOM
REASON_NUL = ReasonCode.NUL
REASON_CONTROL_CHAR = ReasonCode.CONTROL_CHAR
REASON_JSON_INVALID = ReasonCode.JSON_INVALID
REASON_JSON_DUPLICATE_KEY = ReasonCode.JSON_DUPLICATE_KEY
REASON_JSON_NONFINITE = ReasonCode.JSON_NONFINITE
REASON_JSON_DEPTH = ReasonCode.JSON_DEPTH
REASON_JSON_TRAILING = ReasonCode.JSON_TRAILING
REASON_JSON_SHAPE = ReasonCode.JSON_SHAPE
REASON_UNKNOWN_KEY = ReasonCode.UNKNOWN_KEY
REASON_MISSING_KEY = ReasonCode.MISSING_KEY
REASON_INVALID_KIND = ReasonCode.INVALID_KIND
REASON_KIND_MISMATCH = ReasonCode.KIND_MISMATCH
REASON_EMPTY_TEXT = ReasonCode.EMPTY_TEXT
REASON_NONEMPTY_NO_PROOF = ReasonCode.NONEMPTY_NO_PROOF
REASON_BARE_NO_PROOF = ReasonCode.BARE_NO_PROOF


class ProtocolError(ValueError):
    """Base class for strict response errors."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = ReasonCode.JSON_INVALID,
        offset: int | None = None,
        raw_sha256: str | None = None,
        reason_codes: Iterable[str] | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.code = reason_code
        self.offset = offset
        self.raw_sha256 = raw_sha256
        ordered = tuple(dict.fromkeys(reason_codes or (reason_code,)))
        self.reason_codes = ordered
        suffix = "" if offset is None else f" at byte {offset}"
        super().__init__(f"{message}{suffix}")


class ResponseParseError(ProtocolError):
    """The response is not one valid protocol envelope."""


ParseError = ResponseParseError


def _raw_bytes(value: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(value, str):
        try:
            raw = value.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise ResponseParseError(
                "response is not strict UTF-8", reason_code=ReasonCode.INVALID_UTF8
            ) from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raise TypeError("response must be text or bytes")
    return raw


def _raw_digest(raw: bytes) -> str:
    # ``hash_response`` is the canonical evidence hash.  Keep this tiny helper
    # isolated so parse failures can expose the hash without decoding a body.
    return hash_response(raw)


def _reject_constant(value: str) -> Any:
    raise ResponseParseError(f"non-finite JSON constant {value}", reason_code=ReasonCode.JSON_NONFINITE)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponseParseError(
                f"duplicate JSON key {key!r}", reason_code=ReasonCode.JSON_DUPLICATE_KEY
            )
        result[key] = value
    return result


def _check_depth(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ResponseParseError("JSON nesting exceeds the protocol limit", reason_code=ReasonCode.JSON_DEPTH)
    if isinstance(value, dict):
        for item in value.values():
            _check_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_depth(item, depth + 1)


def _check_decoded_text(value: str, *, field: str, max_bytes: int | None = None) -> str:
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ResponseParseError(
            f"{field} contains a surrogate or invalid Unicode scalar",
            reason_code=ReasonCode.SURROGATE,
        ) from exc
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ResponseParseError(f"{field} exceeds its size limit", reason_code=ReasonCode.SIZE_LIMIT)
    for index, char in enumerate(value):
        codepoint = ord(char)
        if codepoint == 0:
            raise ResponseParseError(f"{field} contains NUL", reason_code=ReasonCode.NUL, offset=index)
        if char == "\r":
            raise ResponseParseError(f"{field} contains CR", reason_code=ReasonCode.CR, offset=index)
        if codepoint < 32 and char not in {"\n", "\t"}:
            raise ResponseParseError(
                f"{field} contains a control character",
                reason_code=ReasonCode.CONTROL_CHAR,
                offset=index,
            )
    return value


def _parse_json(raw: bytes) -> Any:
    raw_digest = _raw_digest(raw)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ResponseParseError(
            "response exceeds 64 KiB", reason_code=ReasonCode.SIZE_LIMIT, raw_sha256=raw_digest
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ResponseParseError("UTF-8 BOM is not allowed", reason_code=ReasonCode.BOM, offset=0, raw_sha256=raw_digest)
    if b"\x00" in raw:
        raise ResponseParseError(
            "NUL byte is not allowed",
            reason_code=ReasonCode.NUL,
            offset=raw.index(b"\x00"),
            raw_sha256=raw_digest,
        )
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ResponseParseError(
            "response is not strict UTF-8",
            reason_code=ReasonCode.INVALID_UTF8,
            offset=exc.start,
            raw_sha256=raw_digest,
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except ResponseParseError as exc:
        if exc.raw_sha256 is None:
            exc.raw_sha256 = raw_digest
        raise
    except RecursionError as exc:
        raise ResponseParseError("JSON nesting is too deep", reason_code=ReasonCode.JSON_DEPTH, raw_sha256=raw_digest) from exc
    except json.JSONDecodeError as exc:
        reason = ReasonCode.JSON_TRAILING if "Extra data" in exc.msg else ReasonCode.JSON_INVALID
        raise ResponseParseError(exc.msg, reason_code=reason, offset=exc.pos, raw_sha256=raw_digest) from exc
    except (TypeError, ValueError) as exc:
        raise ResponseParseError(str(exc), reason_code=ReasonCode.JSON_INVALID, raw_sha256=raw_digest) from exc
    try:
        _check_depth(value)
    except ResponseParseError as exc:
        exc.raw_sha256 = raw_digest
        raise
    return value


@dataclass(frozen=True, slots=True)
class ResponseEnvelope:
    """Decoded, validated provider response."""

    kind: Literal["proof", "file", "no_proof"]
    text: str
    disclosure: str
    raw_bytes: bytes = b""

    @property
    def raw(self) -> bytes:
        return self.raw_bytes

    @property
    def raw_sha256(self) -> str:
        return _raw_digest(self.raw_bytes)

    @property
    def response_sha256(self) -> str:
        return self.raw_sha256

    @property
    def submission_sha256(self) -> str | None:
        if self.kind == "proof":
            return hash_proof(self.text.encode("utf-8", "strict"))
        if self.kind == "file":
            return hash_source(self.text.encode("utf-8", "strict"))
        return None

    @property
    def is_no_proof(self) -> bool:
        return self.kind == "no_proof"

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "text": self.text, "disclosure": self.disclosure}

    def to_json(self) -> bytes:
        # Responses are parsed strictly; canonical encoding is useful for
        # locally generated mock payloads but is not required from providers.
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, raw_bytes: bytes = b"") -> "ResponseEnvelope":
        return _envelope_from_mapping(value, raw_bytes=raw_bytes)


ParsedResponse = ResponseEnvelope
ModelResponse = ResponseEnvelope


def _envelope_from_mapping(value: Mapping[str, Any], *, raw_bytes: bytes) -> ResponseEnvelope:
    if not isinstance(value, dict):
        raise ResponseParseError("response must be one JSON object", reason_code=ReasonCode.JSON_SHAPE, raw_sha256=_raw_digest(raw_bytes))
    keys = set(value)
    missing = RESPONSE_KEYS.difference(keys)
    unknown = keys.difference(RESPONSE_KEYS)
    if missing:
        raise ResponseParseError(
            "response is missing required keys: " + ", ".join(sorted(missing)),
            reason_code=ReasonCode.MISSING_KEY,
            raw_sha256=_raw_digest(raw_bytes),
        )
    if unknown:
        raise ResponseParseError(
            "response has unknown keys: " + ", ".join(sorted(unknown)),
            reason_code=ReasonCode.UNKNOWN_KEY,
            raw_sha256=_raw_digest(raw_bytes),
        )
    kind = value["kind"]
    text = value["text"]
    disclosure = value["disclosure"]
    if not isinstance(kind, str) or kind not in RESPONSE_KINDS:
        raise ResponseParseError("kind must be proof, file, or no_proof", reason_code=ReasonCode.INVALID_KIND, raw_sha256=_raw_digest(raw_bytes))
    if not isinstance(text, str):
        raise ResponseParseError("text must be a string", reason_code=ReasonCode.JSON_SHAPE, raw_sha256=_raw_digest(raw_bytes))
    if not isinstance(disclosure, str):
        raise ResponseParseError("disclosure must be a string", reason_code=ReasonCode.JSON_SHAPE, raw_sha256=_raw_digest(raw_bytes))
    _check_decoded_text(text, field="text")
    _check_decoded_text(disclosure, field="disclosure", max_bytes=MAX_DISCLOSURE_BYTES)
    if kind == "no_proof":
        if text != "":
            raise ResponseParseError(
                "no_proof responses must have empty text",
                reason_code=ReasonCode.NONEMPTY_NO_PROOF,
                raw_sha256=_raw_digest(raw_bytes),
            )
        if disclosure != "":
            raise ResponseParseError(
                "no_proof responses must have empty disclosure",
                reason_code=ReasonCode.NONEMPTY_NO_PROOF,
                raw_sha256=_raw_digest(raw_bytes),
            )
    elif text == "":
        raise ResponseParseError(
            f"{kind} responses must have non-empty text",
            reason_code=ReasonCode.EMPTY_TEXT,
            raw_sha256=_raw_digest(raw_bytes),
        )
    return ResponseEnvelope(kind, text, disclosure, raw_bytes)


def parse_response(value: str | bytes | bytearray | memoryview) -> ResponseEnvelope:
    """Parse one strict response envelope.

    A raw ``NO_PROOF`` sentinel intentionally fails JSON parsing.  Abstention
    is represented solely by ``{"kind":"no_proof", ...}``.
    """

    raw = _raw_bytes(value)
    if raw == b"NO_PROOF" or raw == b'"NO_PROOF"':
        raise ResponseParseError(
            "bare NO_PROOF is not a protocol response",
            reason_code=ReasonCode.BARE_NO_PROOF,
            raw_sha256=_raw_digest(raw),
        )
    decoded = _parse_json(raw)
    return _envelope_from_mapping(decoded, raw_bytes=raw)


parse_envelope = parse_response
parse_model_response = parse_response
loads_response = parse_response


def parse_response_for_mode(
    value: str | bytes | bytearray | memoryview,
    mode: str,
) -> ResponseEnvelope:
    if mode not in {"whole_file", "frozen_hole"}:
        raise ValueError("mode must be whole_file or frozen_hole")
    response = parse_response(value)
    if response.kind == "no_proof":
        return response
    expected = "file" if mode == "whole_file" else "proof"
    if response.kind != expected:
        raise ResponseParseError(
            f"response kind {response.kind!r} does not match mode {mode!r}",
            reason_code=ReasonCode.KIND_MISMATCH,
            raw_sha256=response.raw_sha256,
        )
    return response


def strict_json_response(value: str | bytes | bytearray | memoryview) -> ResponseEnvelope:
    return parse_response(value)


__all__ = [
    "MAX_DISCLOSURE_BYTES",
    "MAX_JSON_DEPTH",
    "MAX_RESPONSE_BYTES",
    "ModelResponse",
    "ParseError",
    "ParsedResponse",
    "ProtocolError",
    "RESPONSE_KEYS",
    "RESPONSE_KINDS",
    "REASON_BARE_NO_PROOF",
    "REASON_BOM",
    "REASON_CONTROL_CHAR",
    "REASON_EMPTY_TEXT",
    "REASON_INVALID_KIND",
    "REASON_INVALID_UTF8",
    "REASON_JSON_DEPTH",
    "REASON_JSON_DUPLICATE_KEY",
    "REASON_JSON_INVALID",
    "REASON_JSON_NONFINITE",
    "REASON_JSON_SHAPE",
    "REASON_JSON_TRAILING",
    "REASON_KIND_MISMATCH",
    "REASON_MISSING_KEY",
    "REASON_NUL",
    "REASON_NONEMPTY_NO_PROOF",
    "REASON_SIZE_LIMIT",
    "REASON_UNKNOWN_KEY",
    "ReasonCode",
    "ResponseEnvelope",
    "ResponseParseError",
    "loads_response",
    "parse_envelope",
    "parse_model_response",
    "parse_response",
    "parse_response_for_mode",
    "strict_json_response",
]
