"""Strict, path-safe diagnostics for bounded Lean evaluations.

The compiler is run with ``--json``.  This module keeps the wire-level JSON
events separate from the short feedback string shown to a model.  It also
parses the target-bound ``#print axioms`` event that the runner appends to
every trusted source.

No value returned by this module contains the raw compiler byte stream.  A
caller that needs the raw stream for protected evidence must retain it in the
protected transcript store while this module is still in the runner's process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


MAX_DIAGNOSTIC_BYTES = 32 * 1024


class DiagnosticError(ValueError):
    """Raised when a supposedly strict diagnostic stream is malformed."""


class DiagnosticReason:
    INVALID_UTF8 = "diagnostic_invalid_utf8"
    JSON_INVALID = "diagnostic_json_invalid"
    JSON_SHAPE = "diagnostic_json_shape"
    JSON_DUPLICATE_KEY = "diagnostic_json_duplicate_key"
    OUTPUT_LIMIT = "diagnostic_output_limit"


_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
_ANSI_SINGLE_RE = re.compile(r"\x1b[@-_]")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:/(?:private/)?tmp|/Users|/home|/var/folders|/opt|/workspace)"
    r"[^\s\"'<>\]\[\)\(,;]*"
)
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\[^\s\"'<>\]\[\)\(,;]*")


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticError(
                f"duplicate diagnostic key {key!r}",
                DiagnosticReason.JSON_DUPLICATE_KEY,
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise DiagnosticError(
        f"non-finite diagnostic constant {value}",
        DiagnosticReason.JSON_INVALID,
    )


def _strip_ansi_text(text: str) -> str:
    # OSC must be removed before CSI because OSC payloads can contain the
    # characters which terminate a CSI expression.
    text = _ANSI_OSC_RE.sub("", text)
    text = _ANSI_CSI_RE.sub("", text)
    return _ANSI_SINGLE_RE.sub("", text)


def strip_ansi(value: str | bytes, *, errors: str = "replace") -> str:
    """Decode a compiler stream and remove ANSI/terminal control sequences."""

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors)
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("diagnostic value must be text or bytes")
    return _strip_ansi_text(text)


def _replace_roots(text: str, roots: Iterable[str]) -> str:
    # Longest first prevents a short parent path from leaving a suffix of a
    # longer sensitive path visible.
    unique = sorted({str(root) for root in roots if root}, key=len, reverse=True)
    for root in unique:
        text = text.replace(root, "<redacted>")
    return text


def redact_paths(value: str | bytes, *, roots: Iterable[str] = ()) -> str:
    """Remove registered and common absolute workspace paths.

    Exact roots are supplied by the runner for the current temporary
    workspace, synthetic home, temp directory, project, and toolchain.  The
    conservative generic patterns cover an accidentally serialized temporary
    path before the runner has registered it.
    """

    text = strip_ansi(value)
    text = _replace_roots(text, roots)
    text = _ABSOLUTE_PATH_RE.sub("<path>", text)
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    # Remove remaining C0 controls except the two formatting controls useful
    # in feedback.  This prevents a compiler message from changing terminal
    # state when displayed by a caller.
    text = "".join(
        char
        for char in text
        if char in {"\n", "\t"} or ord(char) >= 32
    )
    return text


def sanitize_text(
    value: str | bytes,
    *,
    roots: Iterable[str] = (),
    max_bytes: int | None = MAX_DIAGNOSTIC_BYTES,
) -> str:
    """Return deterministic feedback-safe text with a byte bound."""

    text = redact_paths(value, roots=roots)
    if max_bytes is None:
        return text
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= max_bytes:
        return text
    # Slice on a decoded prefix so feedback remains valid UTF-8.  The marker
    # makes truncation explicit to the reviewer and model.
    suffix = "\n<diagnostics truncated>"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    text = encoded[:budget].decode("utf-8", "ignore")
    return text + suffix


def _sanitize_json_value(value: Any, *, roots: Iterable[str]) -> Any:
    if isinstance(value, str):
        return redact_paths(value, roots=roots)
    if isinstance(value, list):
        return [_sanitize_json_value(item, roots=roots) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item, roots=roots) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    """One sanitized JSON diagnostic event."""

    value: Mapping[str, Any]
    line: int

    @property
    def severity(self) -> str | None:
        value = self.value.get("severity")
        return value if isinstance(value, str) else None

    @property
    def data(self) -> str | None:
        value = self.value.get("data")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class DiagnosticStream:
    """Parsed and sanitized compiler diagnostics."""

    events: tuple[DiagnosticEvent, ...]
    text: str
    json_valid: bool
    malformed_lines: tuple[int, ...] = ()
    error_count: int = 0
    warning_count: int = 0

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [dict(event.value) for event in self.events],
            "text": self.text,
            "json_valid": self.json_valid,
            "malformed_lines": list(self.malformed_lines),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


def parse_json_diagnostics(
    value: str | bytes,
    *,
    roots: Iterable[str] = (),
    strict: bool = False,
    max_bytes: int | None = MAX_DIAGNOSTIC_BYTES,
) -> DiagnosticStream:
    """Parse Lean JSON lines and sanitize all string fields.

    Lean emits one JSON object per line.  A few failure paths, such as an OS
    signal or an executable wrapper, can emit plain text; permissive mode
    preserves that text and marks ``json_valid`` false.  Strict mode raises a
    :class:`DiagnosticError` for any non-empty non-JSON line or non-object.
    """

    clean = sanitize_text(value, roots=roots, max_bytes=max_bytes)
    events: list[DiagnosticEvent] = []
    malformed: list[int] = []
    for line_number, line in enumerate(clean.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(
                line,
                object_pairs_hook=_strict_json_pairs,
                parse_constant=_reject_constant,
            )
        except (DiagnosticError, json.JSONDecodeError, TypeError, ValueError):
            malformed.append(line_number)
            if strict:
                raise DiagnosticError(
                    f"invalid JSON diagnostic line {line_number}",
                    DiagnosticReason.JSON_INVALID,
                )
            continue
        if not isinstance(decoded, dict):
            malformed.append(line_number)
            if strict:
                raise DiagnosticError(
                    f"diagnostic line {line_number} is not an object",
                    DiagnosticReason.JSON_SHAPE,
                )
            continue
        sanitized = _sanitize_json_value(decoded, roots=roots)
        events.append(DiagnosticEvent(sanitized, line_number))
    errors = sum(event.severity in {"error", "fatal"} for event in events)
    warnings = sum(event.severity == "warning" for event in events)
    return DiagnosticStream(
        events=tuple(events),
        text=clean,
        json_valid=not malformed,
        malformed_lines=tuple(malformed),
        error_count=errors,
        warning_count=warnings,
    )


parse_lean_json_events = parse_json_diagnostics
parse_diagnostics = parse_json_diagnostics


def has_errors(value: str | bytes, *, roots: Iterable[str] = ()) -> bool:
    return parse_json_diagnostics(value, roots=roots).has_errors


def combine_output(
    stdout: str | bytes,
    stderr: str | bytes,
    *,
    roots: Iterable[str] = (),
    max_bytes: int | None = MAX_DIAGNOSTIC_BYTES,
) -> str:
    """Join sanitized stdout/stderr while preserving their source order."""

    left = sanitize_text(stdout, roots=roots, max_bytes=None)
    right = sanitize_text(stderr, roots=roots, max_bytes=None)
    if left and right:
        combined = left + "\n" + right
    else:
        combined = left or right
    return sanitize_text(combined, roots=roots, max_bytes=max_bytes)


_AXIOM_CLEAN_RE = re.compile(r"^'(?P<target>[^']+)' does not depend on any axioms$")
_AXIOM_DEPEND_RE = re.compile(
    r"^'(?P<target>[^']+)' depends on axioms: \[(?P<axioms>.*)\]$"
)
_AXIOM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*$")


@dataclass(frozen=True, slots=True)
class AxiomReport:
    """One target-bound ``#print axioms`` result."""

    target: str
    observed: tuple[str, ...]
    allowed: tuple[str, ...]
    delta: tuple[str, ...]
    status: str
    count: int
    diagnostic_sha256: str
    error_count: int = 0

    @property
    def clean(self) -> bool:
        return self.status == "clean"

    @property
    def forbidden(self) -> bool:
        return self.status == "forbidden"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "observed": list(self.observed),
            "allowed": list(self.allowed),
            "delta": list(self.delta),
            "status": self.status,
            "count": self.count,
            "diagnostic_sha256": self.diagnostic_sha256,
            "error_count": self.error_count,
        }


def _axiom_messages(value: str | bytes, *, roots: Iterable[str] = ()) -> list[tuple[str, tuple[str, ...] | None]]:
    stream = parse_json_diagnostics(value, roots=roots)
    messages: list[str] = []
    for event in stream.events:
        if event.severity in (None, "information") and event.data is not None:
            messages.append(event.data.strip())
    # A wrapper or fake transport may emit the same textual event without
    # JSON.  The fallback is useful for diagnostics while the runner still
    # records ``json_valid=false``.
    if not messages:
        messages = [line.strip() for line in stream.text.splitlines() if line.strip()]
    reports: list[tuple[str, tuple[str, ...] | None]] = []
    for message in messages:
        clean = _AXIOM_CLEAN_RE.fullmatch(message)
        if clean is not None:
            reports.append((clean.group("target"), ()))
            continue
        dependent = _AXIOM_DEPEND_RE.fullmatch(message)
        if dependent is None:
            continue
        raw_names = dependent.group("axioms").strip()
        if not raw_names:
            reports.append((dependent.group("target"), ()))
            continue
        names = tuple(name.strip() for name in raw_names.split(","))
        if any(not _AXIOM_NAME_RE.fullmatch(name) for name in names):
            reports.append((dependent.group("target"), None))
            continue
        if len(set(names)) != len(names):
            reports.append((dependent.group("target"), None))
            continue
        reports.append((dependent.group("target"), names))
    return reports


def parse_axiom_report(
    value: str | bytes,
    *,
    target: str,
    allowed: Sequence[str] = (),
    roots: Iterable[str] = (),
) -> AxiomReport:
    """Parse exactly one target-bound axiom report.

    A missing, duplicate, malformed, or wrong-target report is never treated
    as clean.  ``sorryAx`` and ``admitAx`` are forbidden even when explicitly
    present in ``allowed``.
    """

    stream = parse_json_diagnostics(value, roots=roots)
    reports = _axiom_messages(stream.text, roots=roots)
    matching = [report for report in reports if report[0] == target]
    allowed_tuple = tuple(allowed)
    observed: tuple[str, ...] = ()
    if len(reports) == 1 and len(matching) == 1 and matching[0][1] is not None:
        observed = matching[0][1]
    delta = tuple(sorted(set(observed).difference(allowed_tuple)))
    if not reports:
        status = "missing"
    elif len(reports) != 1:
        status = "malformed_or_duplicate"
    elif not matching:
        status = "wrong_target"
    elif matching[0][1] is None:
        status = "malformed"
    elif any(name in {"sorryAx", "admitAx"} for name in observed):
        status = "forbidden"
    elif set(observed) == set(allowed_tuple) and len(observed) == len(allowed_tuple):
        status = "clean"
    else:
        status = "unexpected"
    diagnostic = sanitize_text(stream.text, roots=roots, max_bytes=None).encode("utf-8", "replace")
    return AxiomReport(
        target=target,
        observed=observed,
        allowed=allowed_tuple,
        delta=delta,
        status=status,
        count=len(matching),
        diagnostic_sha256=hashlib.sha256(diagnostic).hexdigest(),
        error_count=stream.error_count,
    )


parse_axioms = lambda value, expected_target=None, **kwargs: (
    None
    if expected_target is None
    and len(_axiom_messages(value, roots=kwargs.get("roots", ()))) != 1
    else (
        None
        if expected_target is not None
        and (
            len(_axiom_messages(value, roots=kwargs.get("roots", ()))) != 1
            or _axiom_messages(value, roots=kwargs.get("roots", ()))[0][0] != expected_target
        )
        else _axiom_messages(value, roots=kwargs.get("roots", ()))[0][1]
    )
)


_UNSOLVED_PATTERNS = (
    re.compile(r"unsolved goals?", re.IGNORECASE),
    re.compile(r"declaration has metavariables", re.IGNORECASE),
    re.compile(r"contains metavariables", re.IGNORECASE),
    re.compile(r"synthetic opaque has metavariables", re.IGNORECASE),
)


def has_unsolved_metavariables(value: str | bytes, *, roots: Iterable[str] = ()) -> bool:
    """Detect Lean's stable unsolved-goal/metavariable diagnostic wording."""

    text = sanitize_text(value, roots=roots, max_bytes=None)
    return any(pattern.search(text) for pattern in _UNSOLVED_PATTERNS)


def count_unsolved_metavariables(value: str | bytes, *, roots: Iterable[str] = ()) -> int | None:
    """Return a conservative count when Lean prints one, else ``None``.

    The text forms differ between elaborator versions.  A positive result is
    therefore definitive while ``None`` means that the caller should use the
    process exit and error events as the additional gate.
    """

    text = sanitize_text(value, roots=roots, max_bytes=None)
    matches = re.findall(r"(?:unsolved goals?|metavariables?)\s*:?\s*(\d+)", text, re.IGNORECASE)
    if matches:
        return max(int(item) for item in matches)
    return 1 if has_unsolved_metavariables(text, roots=roots) else 0


__all__ = [
    "AxiomReport",
    "DiagnosticError",
    "DiagnosticEvent",
    "DiagnosticReason",
    "DiagnosticStream",
    "MAX_DIAGNOSTIC_BYTES",
    "combine_output",
    "count_unsolved_metavariables",
    "has_errors",
    "has_unsolved_metavariables",
    "parse_axiom_report",
    "parse_axioms",
    "parse_diagnostics",
    "parse_json_diagnostics",
    "parse_lean_json_events",
    "redact_paths",
    "sanitize_text",
    "strip_ansi",
]
