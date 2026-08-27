"""Pre-Lean protocol and source lock for the Lean proof-interface study.

The frozen validator has a narrow authority boundary: it reads a trusted
corpus template, checks one proof term, and materializes a fresh source.  The
whole-file validator applies a source-safety policy and reports protected-byte
changes separately.  Neither validator invokes Lean.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import corpus
from .corpus import TaskSpec
from .lexer import LexError, LexResult, LexToken, scan
from .protocol import ResponseEnvelope, parse_response_for_mode


MAX_SUBMISSION_BYTES = 64 * 1024
MARKER = corpus.MARKER


class ReasonCode:
    """Stable machine-readable source-lock reason codes."""

    SIZE_LIMIT = "size_limit"
    INVALID_UTF8 = "invalid_utf8"
    BOM = "bom"
    NUL = "nul"
    CONTROL_CHAR = "control_char"
    CR = "cr"
    UNICODE_WHITESPACE = "unicode_whitespace"
    FENCE = "fence"
    MARKER_RESIDUE = "marker_residue"
    EMPTY_PROOF = "empty_proof"
    EMPTY_SOURCE = "empty_source"
    FORBIDDEN_TOKEN = "forbidden_token"
    FORBIDDEN_COMMAND = "forbidden_command"
    HOST_EFFECT = "host_effect"
    UNTERMINATED_LEX = "unterminated_lex"
    UNBALANCED_DELIMITER = "unbalanced_delimiter"
    TRAILING_DATA = "trailing_data"
    TRAILING_COMMENT = "trailing_comment"
    UNRESOLVED_METAVARIABLE = "unresolved_metavariable"
    IMPORT_MUTATION = "import_mutation"
    PROTECTED_PREFIX = "protected_prefix_changed"
    PROTECTED_SUFFIX = "protected_suffix_changed"
    NAMESPACE_MUTATION = "namespace_changed"
    THEOREM_MUTATION = "theorem_changed"
    BINDER_MUTATION = "binder_changed"
    AXIOM_ADDED = "axiom_added"
    EXTRA_DECLARATION = "extra_declaration"
    SOURCE_SHAPE = "source_shape"
    KIND_MISMATCH = "kind_mismatch"


for _name in tuple(name for name in dir(ReasonCode) if name.isupper()):
    globals()[f"REASON_{_name}"] = getattr(ReasonCode, _name)


class LockError(ValueError):
    """Base class for source-lock failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = ReasonCode.SOURCE_SHAPE,
        reason_codes: Iterable[str] | None = None,
        offset: int | None = None,
        token: str | None = None,
        raw_sha256: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.code = reason_code
        self.reason_codes = tuple(dict.fromkeys(reason_codes or (reason_code,)))
        self.offset = offset
        self.token = token
        self.raw_sha256 = raw_sha256
        suffix = "" if offset is None else f" at byte {offset}"
        token_suffix = "" if token is None else f" ({token})"
        super().__init__(f"{message}{token_suffix}{suffix}")


class ValidationError(LockError):
    """A candidate was rejected before Lean."""


class FrozenValidationError(ValidationError):
    """A proof-only candidate violated the frozen contract."""


class WholeValidationError(ValidationError):
    """A whole-file candidate violated the source-safety contract."""


LockViolation = ValidationError
FrozenLockError = FrozenValidationError
WholeLockError = WholeValidationError


def _ordinary_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _coerce_submission(
    value: str | bytes | bytearray | memoryview,
    *,
    kind: str,
    reject_marker: bool = True,
    reject_fence: bool = True,
) -> tuple[bytes, str]:
    if isinstance(value, str):
        try:
            raw = value.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise ValidationError("candidate is not strict UTF-8", reason_code=ReasonCode.INVALID_UTF8) from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raise TypeError("candidate must be text or bytes")
    digest = corpus.hash_proof(raw) if kind == "proof" else corpus.hash_source(raw)
    if len(raw) > MAX_SUBMISSION_BYTES:
        raise ValidationError("candidate exceeds 64 KiB", reason_code=ReasonCode.SIZE_LIMIT, raw_sha256=digest)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValidationError("UTF-8 BOM is not allowed", reason_code=ReasonCode.BOM, offset=0, raw_sha256=digest)
    nul = raw.find(b"\x00")
    if nul >= 0:
        raise ValidationError("NUL byte is not allowed", reason_code=ReasonCode.NUL, offset=nul, raw_sha256=digest)
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ValidationError("candidate is not strict UTF-8", reason_code=ReasonCode.INVALID_UTF8, offset=exc.start, raw_sha256=digest) from exc
    for index, char in enumerate(text):
        codepoint = ord(char)
        if char == "\r":
            raise ValidationError("CR line endings are not allowed", reason_code=ReasonCode.CR, offset=index, raw_sha256=digest)
        if codepoint < 32 and char not in {"\n", "\t"}:
            raise ValidationError("control character is not allowed", reason_code=ReasonCode.CONTROL_CHAR, offset=index, raw_sha256=digest)
        if 127 <= codepoint < 160:
            raise ValidationError("control character is not allowed", reason_code=ReasonCode.CONTROL_CHAR, offset=index, raw_sha256=digest)
        if char.isspace() and char not in {" ", "\t", "\n"}:
            raise ValidationError("non-ASCII whitespace is not allowed", reason_code=ReasonCode.UNICODE_WHITESPACE, offset=index, raw_sha256=digest)
    if reject_marker and MARKER.encode("ascii") in raw:
        raise ValidationError("proof marker residue is not allowed", reason_code=ReasonCode.MARKER_RESIDUE, raw_sha256=digest)
    if reject_fence and (b"```" in raw or b"~~~" in raw):
        raise ValidationError("markdown fences are not allowed", reason_code=ReasonCode.FENCE, raw_sha256=digest)
    return raw, digest


def _resolve_task(task: TaskSpec | str | Mapping[str, Any], *, root: str | Path | None = None) -> TaskSpec:
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, str):
        return corpus.load_catalog(root=root).by_id(task)
    if isinstance(task, Mapping):
        try:
            return TaskSpec.from_record(task)
        except Exception as exc:
            raise LockError("invalid task metadata", reason_code=ReasonCode.SOURCE_SHAPE) from exc
    raise TypeError("task must be a TaskSpec, task ID, or manifest record")


@dataclass(frozen=True, slots=True)
class TrustedTemplate:
    """Exact trusted bytes and immutable regions for one task."""

    task: TaskSpec
    template_bytes: bytes
    prefix: bytes
    suffix: bytes
    marker_offset: int
    template_sha256: str
    fixed_region_sha256: str

    @property
    def marker(self) -> str:
        return MARKER

    @property
    def fixed_region(self) -> bytes:
        return self.prefix + self.suffix

    @classmethod
    def load(cls, task: TaskSpec | str | Mapping[str, Any], *, root: str | Path | None = None) -> "TrustedTemplate":
        resolved = _resolve_task(task, root=root)
        try:
            raw = corpus.read_template(resolved, root=root)
            prefix, suffix = corpus.template_parts(resolved, root=root)
        except Exception as exc:
            raise LockError("trusted template is unavailable or hash-mismatched", reason_code=ReasonCode.SOURCE_SHAPE) from exc
        marker_offset = raw.find(MARKER.encode("ascii"))
        if marker_offset < 0 or raw.count(MARKER.encode("ascii")) != 1:
            raise LockError("trusted template does not have one active marker", reason_code=ReasonCode.MARKER_RESIDUE)
        template_hash = corpus.hash_template(raw)
        fixed_hash = corpus.hash_protected(prefix + suffix)
        if template_hash != resolved.template_sha256 or fixed_hash != resolved.fixed_region_sha256:
            raise LockError("trusted template hash mismatch", reason_code=ReasonCode.SOURCE_SHAPE)
        return cls(resolved, raw, prefix, suffix, marker_offset, template_hash, fixed_hash)


load_trusted_template = TrustedTemplate.load


@dataclass(frozen=True, slots=True)
class ProtectedSourceDiff:
    """Byte and semantic changes between trusted protected regions and source."""

    contract_accepted: bool
    prefix_unchanged: bool
    suffix_unchanged: bool
    changed_regions: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    first_changed_offset: int | None = None
    candidate_has_marker: bool = False

    @property
    def accepted(self) -> bool:
        return self.contract_accepted

    @property
    def protected_unchanged(self) -> bool:
        return self.prefix_unchanged and self.suffix_unchanged and not self.candidate_has_marker

    @property
    def changed(self) -> tuple[str, ...]:
        return self.changed_regions

    @property
    def mutation_categories(self) -> tuple[str, ...]:
        return self.changed_regions

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_accepted": self.contract_accepted,
            "prefix_unchanged": self.prefix_unchanged,
            "suffix_unchanged": self.suffix_unchanged,
            "changed_regions": list(self.changed_regions),
            "reason_codes": list(self.reason_codes),
            "first_changed_offset": self.first_changed_offset,
            "candidate_has_marker": self.candidate_has_marker,
        }


def _first_difference(left: bytes, right: bytes) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return None if len(left) == len(right) else limit


def _active_tokens(raw: bytes) -> LexResult:
    try:
        return scan(raw)
    except LexError as exc:
        raise ValidationError(
            str(exc),
            reason_code=exc.code if exc.code else ReasonCode.UNTERMINATED_LEX,
            offset=exc.offset,
        ) from exc


def _recast(error: ValidationError, error_type: type[ValidationError]) -> ValidationError:
    if isinstance(error, error_type):
        return error
    return error_type(
        str(error),
        reason_code=error.reason_code,
        reason_codes=error.reason_codes,
        offset=error.offset,
        token=error.token,
        raw_sha256=error.raw_sha256,
    )


def _balanced_depths(tokens: Sequence[LexToken]) -> list[int]:
    """Return depth before each token, for conservative trailing checks."""

    depth = 0
    result: list[int] = []
    opens = {"(", "[", "{", "⟨"}
    closes = {")": "(", "]": "[", "}": "{", "⟩": "⟨"
    }
    stack: list[str] = []
    for token in tokens:
        result.append(depth)
        if token.text in opens:
            stack.append(token.text)
            depth += 1
        elif token.text in closes:
            if stack and stack[-1] == closes[token.text]:
                stack.pop()
            depth = max(0, depth - 1)
    return result


def _ensure_single_term(
    raw: bytes,
    result: LexResult,
    *,
    reject_formal_commands: bool = True,
    allow_solved_metavariables: bool = False,
    final_metavariables: int | None = None,
    error_type: type[ValidationError] = FrozenValidationError,
) -> None:
    tokens = result.tokens
    if not tokens:
        raise error_type("proof term is empty", reason_code=ReasonCode.EMPTY_PROOF)
    forbidden = FROZEN_FORBIDDEN_TOKENS if reject_formal_commands else FROZEN_COMMAND_TOKENS
    for token in tokens:
        if token.text in forbidden or token.text == "#":
            code = ReasonCode.FORBIDDEN_COMMAND if token.text in COMMAND_TOKENS or token.text == "#" else ReasonCode.FORBIDDEN_TOKEN
            raise error_type(
                "forbidden token in proof term",
                reason_code=code,
                offset=token.start,
                token=token.text,
            )
    for left, right in zip(tokens, tokens[1:]):
        if left.text == "?" and right.text == "_" and left.end == right.start:
            if not allow_solved_metavariables or final_metavariables != 0:
                raise error_type(
                    "unresolved metavariable is not accepted",
                    reason_code=ReasonCode.UNRESOLVED_METAVARIABLE,
                    offset=left.start,
                    token="?_",
                )
    last = tokens[-1]
    if result.comments and result.comments[-1].start >= last.end:
        raise error_type("trailing comment is not part of the one proof term", reason_code=ReasonCode.TRAILING_COMMENT, offset=result.comments[-1].start)
    depths = _balanced_depths(tokens)
    if tokens[0].text == "by":
        allowed_nested_by = {
            "exact", "show", "from", "=>", "then", "else", "do", "fun", "match", "with",
            "let", "have", "suffices", "refine", "apply", "change", "convert", "simpa",
        }
        for index, token in enumerate(tokens[1:], start=1):
            if token.text == "by" and depths[index] == 0:
                previous = tokens[index - 1].text
                if previous not in allowed_nested_by:
                    raise error_type("second proof term or trailing data", reason_code=ReasonCode.TRAILING_DATA, offset=token.start, token=token.text)
        terminal_tokens = {"rfl", "decide", "trivial", "assumption", "contradiction", "omega"}
        separators = {";", "<;>", "all_goals", "case", "next", "·", "|"}
        for index, previous in enumerate(tokens[:-1]):
            if depths[index] == 0 and previous.text in terminal_tokens:
                following = tokens[index + 1]
                if following.text not in separators and following.text not in {"by"}:
                    raise error_type("trailing data follows proof term", reason_code=ReasonCode.TRAILING_DATA, offset=following.start, token=following.text)


def _proof_candidate(
    value: str | bytes | bytearray | memoryview,
    *,
    allow_solved_metavariables: bool = False,
    final_metavariables: int | None = None,
) -> tuple[bytes, str, LexResult]:
    raw, digest = _coerce_submission(value, kind="proof")
    result = _active_tokens(raw)
    _ensure_single_term(
        raw,
        result,
        allow_solved_metavariables=allow_solved_metavariables,
        final_metavariables=final_metavariables,
    )
    return raw, digest, result


@dataclass(frozen=True, slots=True)
class ValidatedFrozenSource:
    """A proof term safely spliced into exact trusted bytes."""

    task: TaskSpec
    proof_bytes: bytes
    source_bytes: bytes
    prefix: bytes
    suffix: bytes
    marker_offset: int
    template_sha256: str
    fixed_region_sha256: str
    proof_sha256: str
    source_sha256: str
    task_hash: str
    reason_codes: tuple[str, ...] = ()
    mode: str = "frozen_hole"
    contract_accepted: bool = True
    checker_accept: bool = True
    metavariables_deferred: bool = False

    @property
    def accepted(self) -> bool:
        return self.checker_accept

    @property
    def proof(self) -> str:
        return self.proof_bytes.decode("utf-8")

    @property
    def source(self) -> bytes:
        return self.source_bytes

    @property
    def materialized_source(self) -> bytes:
        return self.source_bytes

    @property
    def raw_proof(self) -> bytes:
        return self.proof_bytes

    @property
    def fixed_region(self) -> bytes:
        return self.prefix + self.suffix

    @property
    def template(self) -> bytes:
        return self.prefix + MARKER.encode("ascii") + self.suffix

    @property
    def fixed_region_sha256_recomputed(self) -> str:
        return corpus.hash_protected(self.prefix + self.suffix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task.id,
            "task_hash": self.task_hash,
            "template_sha256": self.template_sha256,
            "fixed_region_sha256": self.fixed_region_sha256,
            "proof_sha256": self.proof_sha256,
            "source_sha256": self.source_sha256,
            "reason_codes": list(self.reason_codes),
            "contract_accepted": self.contract_accepted,
            "checker_accept": self.checker_accept,
            "metavariables_deferred": self.metavariables_deferred,
        }


def validate_frozen(
    task: TaskSpec | str | Mapping[str, Any],
    proof: str | bytes | bytearray | memoryview,
    *,
    root: str | Path | None = None,
    allow_solved_metavariables: bool = False,
    final_metavariables: int | None = None,
) -> ValidatedFrozenSource:
    """Validate one proof term and materialize ``prefix + (proof) + suffix``."""

    try:
        trusted = TrustedTemplate.load(task, root=root)
        raw, proof_hash, result = _proof_candidate(
            proof,
            allow_solved_metavariables=allow_solved_metavariables,
            final_metavariables=final_metavariables,
        )
    except ValidationError as exc:
        raise _recast(exc, FrozenValidationError) from exc
    except LockError as exc:
        raise FrozenValidationError(
            str(exc),
            reason_code=exc.reason_code,
            reason_codes=exc.reason_codes,
            offset=exc.offset,
            token=exc.token,
            raw_sha256=exc.raw_sha256,
        ) from exc
    source = trusted.prefix + b"(" + raw + b")" + trusted.suffix
    if not source.startswith(trusted.prefix) or not source.endswith(trusted.suffix):
        raise FrozenValidationError("materialized source changed an immutable region", reason_code=ReasonCode.SOURCE_SHAPE)
    if corpus.hash_template(trusted.template_bytes) != trusted.task.template_sha256:
        raise FrozenValidationError("trusted template hash changed", reason_code=ReasonCode.SOURCE_SHAPE)
    if corpus.hash_protected(trusted.prefix + trusted.suffix) != trusted.task.fixed_region_sha256:
        raise FrozenValidationError("trusted fixed-region hash changed", reason_code=ReasonCode.PROTECTED_PREFIX)
    return ValidatedFrozenSource(
        task=trusted.task,
        proof_bytes=raw,
        source_bytes=source,
        prefix=trusted.prefix,
        suffix=trusted.suffix,
        marker_offset=trusted.marker_offset,
        template_sha256=trusted.template_sha256,
        fixed_region_sha256=trusted.fixed_region_sha256,
        proof_sha256=proof_hash,
        source_sha256=corpus.hash_source(source),
        task_hash=trusted.task.task_hash,
        metavariables_deferred=any(left.text == "?" and right.text == "_" and left.end == right.start for left, right in zip(result.tokens, result.tokens[1:])),
    )


validate_frozen_hole = validate_frozen
validate_proof = validate_frozen
freeze_proof = validate_frozen


_DECLARATION_TOKENS = frozenset({
    "theorem", "lemma", "def", "abbrev", "example", "axiom", "constant", "opaque",
    "inductive", "structure", "class", "instance", "macro", "elab", "syntax",
})

COMMAND_TOKENS = frozenset({
    "import", "namespace", "section", "end", "open", "set_option", "#",
    "include", "omit", "attribute", "local", "scoped", "initialize", "export",
})

FROZEN_COMMAND_TOKENS = frozenset({
    "axiom", "opaque", "constant", "theorem", "lemma", "def", "abbrev", "example",
    "inductive", "structure", "class", "instance", "macro", "elab", "syntax",
    "import", "namespace", "section", "end", "open", "set_option", "include", "omit",
    "attribute", "local", "scoped", "initialize", "export", "foreign",
})

FROZEN_FORBIDDEN_TOKENS = FROZEN_COMMAND_TOKENS | frozenset({
    "sorry", "admit", "sorryAx", "admitAx", "unsafe", "partial", "implemented_by", "extern",
    "run_tac", "native_decide", "exact?"})

WHOLE_BLOCKED_TOKENS = frozenset({
    "unsafe", "partial", "implemented_by", "extern", "run_tac", "elab", "macro", "syntax",
    "initialize", "foreign", "native_decide", "set_option", "include", "omit", "attribute",
    "local", "scoped", "open", "section", "#", "IO", "System", "File", "FS", "Socket",
    "Process", "Command", "Environment", "Subprocess", "unsafeCast", "ofReduceBool",
})


def _diff_categories(
    trusted: TrustedTemplate,
    source: bytes,
    result: LexResult | None = None,
) -> ProtectedSourceDiff:
    marker_bytes = MARKER.encode("ascii")
    has_marker = marker_bytes in source
    prefix_ok = source.startswith(trusted.prefix)
    suffix_ok = source.endswith(trusted.suffix)
    categories: list[str] = []
    reasons: list[str] = []
    if not prefix_ok:
        categories.append(ReasonCode.PROTECTED_PREFIX)
        reasons.append(ReasonCode.PROTECTED_PREFIX)
    if not suffix_ok:
        categories.append(ReasonCode.PROTECTED_SUFFIX)
        reasons.append(ReasonCode.PROTECTED_SUFFIX)
    expected_imports = ("\n".join(corpus.APPROVED_IMPORTS) + "\n").encode("utf-8")
    if not source.startswith(expected_imports):
        categories.append("imports_changed")
        reasons.append(ReasonCode.IMPORT_MUTATION)
    try:
        text = source.decode("utf-8", "strict")
    except UnicodeDecodeError:
        text = ""
    if text.count(f"namespace {corpus.NAMESPACE}") != 1 or text.count(f"end {corpus.NAMESPACE}") != 1:
        categories.append("namespace_changed")
        reasons.append(ReasonCode.NAMESPACE_MUTATION)
    if result is None:
        try:
            result = scan(source)
        except LexError:
            result = None
    if result is not None:
        tokens = result.tokens
        if any(token.text == "axiom" for token in tokens):
            categories.append("axiom_added")
            reasons.append(ReasonCode.AXIOM_ADDED)
        trusted_decl_count = sum(1 for token in scan(trusted.template_bytes).tokens if token.text in _DECLARATION_TOKENS)
        candidate_decl_count = sum(1 for token in tokens if token.text in _DECLARATION_TOKENS)
        if candidate_decl_count > trusted_decl_count:
            categories.append("extra_declaration")
            reasons.append(ReasonCode.EXTRA_DECLARATION)
        if not prefix_ok and any(token.text == "theorem" for token in tokens):
            categories.append("theorem_changed")
            reasons.append(ReasonCode.THEOREM_MUTATION)
            # A changed theorem declaration covers a changed binder or
            # proposition.  The finer label is useful for review even when a
            # byte-level diff cannot establish which part of the declaration
            # was intended by a model.
            categories.append("binder_changed")
            reasons.append(ReasonCode.BINDER_MUTATION)
    if has_marker:
        categories.append(ReasonCode.MARKER_RESIDUE)
        reasons.append(ReasonCode.MARKER_RESIDUE)
    # Preserve first occurrence while allowing a caller to use categories as
    # a stable set for regression assertions.
    categories = list(dict.fromkeys(categories))
    reasons = list(dict.fromkeys(reasons))
    return ProtectedSourceDiff(
        contract_accepted=prefix_ok and suffix_ok and not has_marker and not categories,
        prefix_unchanged=prefix_ok,
        suffix_unchanged=suffix_ok,
        changed_regions=tuple(categories),
        reason_codes=tuple(reasons),
        # The proof span is intentionally mutable and may change length.  A
        # differing byte at the marker boundary is therefore expected when
        # the protected prefix and suffix are both intact.
        first_changed_offset=(
            None
            if prefix_ok and suffix_ok
            else _first_difference(trusted.prefix + trusted.suffix, source)
        ),
        candidate_has_marker=has_marker,
    )


def protected_diff(
    task: TaskSpec | str | Mapping[str, Any],
    source: str | bytes | bytearray | memoryview,
    *,
    root: str | Path | None = None,
) -> ProtectedSourceDiff:
    trusted = TrustedTemplate.load(task, root=root)
    raw, _ = _coerce_submission(source, kind="source", reject_marker=False, reject_fence=False)
    return _diff_categories(trusted, raw)


classify_protected_diff = protected_diff
diff_protected_source = protected_diff


def _validate_whole_policy(raw: bytes, trusted: TrustedTemplate) -> LexResult:
    if not raw:
        raise WholeValidationError("whole-file source is empty", reason_code=ReasonCode.EMPTY_SOURCE)
    if not raw.endswith(b"\n"):
        raise WholeValidationError("whole-file source must end with LF", reason_code=ReasonCode.TRAILING_DATA)
    result = _active_tokens(raw)
    if not result.tokens:
        raise WholeValidationError("whole-file source has no active tokens", reason_code=ReasonCode.EMPTY_SOURCE)
    expected_imports = ("\n".join(corpus.APPROVED_IMPORTS) + "\n").encode("utf-8")
    if not raw.startswith(expected_imports):
        raise WholeValidationError("imports must equal the approved import block", reason_code=ReasonCode.IMPORT_MUTATION)
    for token in result.tokens:
        if token.text == "import" and token.start >= len(expected_imports):
            raise WholeValidationError("new or moved import is not allowed", reason_code=ReasonCode.IMPORT_MUTATION, offset=token.start, token=token.text)
        if token.text in WHOLE_BLOCKED_TOKENS:
            code = ReasonCode.FORBIDDEN_COMMAND if token.text in COMMAND_TOKENS or token.text == "#" else ReasonCode.HOST_EFFECT
            raise WholeValidationError("blocked whole-file source token", reason_code=code, offset=token.start, token=token.text)
    if result.comments and result.comments[-1].start >= result.tokens[-1].end:
        raise WholeValidationError("trailing comment is not accepted", reason_code=ReasonCode.TRAILING_COMMENT, offset=result.comments[-1].start)
    names = [token.text for token in result.tokens]
    if "namespace" not in names or "end" not in names:
        raise WholeValidationError("complete source must contain a namespace block", reason_code=ReasonCode.SOURCE_SHAPE)
    # A complete file has no active token after the namespace name belonging
    # to its final ``end``.  This catches prose and a second term early while
    # retaining arbitrary safe declarations before the theorem.
    end_positions = [i for i, token in enumerate(result.tokens) if token.text == "end"]
    if end_positions:
        last_end = end_positions[-1]
        tail = result.tokens[last_end + 1 :]
        # Named namespace endings contain one identifier or a dotted name.
        # A second unrelated identifier is prose or a second term, even when
        # it happens to fit inside the small tail bound.
        if len(tail) not in {1, 3} or (len(tail) == 3 and tail[1].text != "."):
            offset = tail[0].start if tail else result.tokens[last_end].end
            raise WholeValidationError("trailing source after namespace end", reason_code=ReasonCode.TRAILING_DATA, offset=offset)
    return result


@dataclass(frozen=True, slots=True)
class ValidatedWholeSource:
    """A complete source accepted by the safe whole-file policy."""

    task: TaskSpec
    source_bytes: bytes
    source_sha256: str
    task_hash: str
    template_sha256: str
    fixed_region_sha256: str
    protected: ProtectedSourceDiff
    prefix: bytes
    suffix: bytes
    reason_codes: tuple[str, ...] = ()
    mode: str = "whole_file"
    checker_accept: bool = True

    @property
    def source(self) -> bytes:
        return self.source_bytes

    @property
    def raw_source(self) -> bytes:
        return self.source_bytes

    @property
    def candidate_source(self) -> bytes:
        return self.source_bytes

    @property
    def contract_accepted(self) -> bool:
        return self.protected.contract_accepted

    @property
    def accepted(self) -> bool:
        return self.checker_accept

    @property
    def protected_diff(self) -> ProtectedSourceDiff:
        return self.protected

    @property
    def proof_bytes(self) -> bytes | None:
        trusted = self.protected
        # The proof region can be recovered exactly when both immutable spans
        # match.  An intentionally mutated whole file has no trusted proof
        # extraction and is left for Lean/behavioural classification.
        if not trusted.contract_accepted:
            return None
        return self.source_bytes[len(self.prefix) : len(self.source_bytes) - len(self.suffix)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task.id,
            "task_hash": self.task_hash,
            "source_sha256": self.source_sha256,
            "template_sha256": self.template_sha256,
            "fixed_region_sha256": self.fixed_region_sha256,
            "checker_accept": self.checker_accept,
            "contract_accepted": self.contract_accepted,
            "reason_codes": list(self.reason_codes),
            "protected_diff": self.protected.to_dict(),
        }


def validate_whole(
    task: TaskSpec | str | Mapping[str, Any],
    source: str | bytes | bytearray | memoryview,
    *,
    root: str | Path | None = None,
) -> ValidatedWholeSource:
    """Validate a bounded, safe complete Lean source without invoking Lean."""

    try:
        trusted = TrustedTemplate.load(task, root=root)
        raw, source_hash = _coerce_submission(source, kind="source")
        result = _validate_whole_policy(raw, trusted)
    except ValidationError as exc:
        raise _recast(exc, WholeValidationError) from exc
    except LockError as exc:
        raise WholeValidationError(
            str(exc),
            reason_code=exc.reason_code,
            reason_codes=exc.reason_codes,
            offset=exc.offset,
            token=exc.token,
            raw_sha256=exc.raw_sha256,
        ) from exc
    diff = _diff_categories(trusted, raw, result)
    reasons = tuple(dict.fromkeys(diff.reason_codes))
    return ValidatedWholeSource(
        task=trusted.task,
        source_bytes=raw,
        source_sha256=source_hash,
        task_hash=trusted.task.task_hash,
        template_sha256=trusted.template_sha256,
        fixed_region_sha256=trusted.fixed_region_sha256,
        protected=diff,
        prefix=trusted.prefix,
        suffix=trusted.suffix,
        reason_codes=reasons,
    )


validate_whole_file = validate_whole
validate_source = validate_whole


@dataclass(frozen=True, slots=True)
class ValidatedNoProof:
    """Strict abstention response.  It never reaches Lean."""

    task: TaskSpec | None
    mode: str
    disclosure: str = ""
    contract_accepted: bool = True
    checker_accept: bool = False
    reason_codes: tuple[str, ...] = ("no_proof",)

    @property
    def accepted(self) -> bool:
        return False


def validate_response(
    task: TaskSpec | str | Mapping[str, Any],
    response: ResponseEnvelope | str | bytes | bytearray | memoryview,
    mode: str,
    *,
    root: str | Path | None = None,
    allow_solved_metavariables: bool = False,
    final_metavariables: int | None = None,
) -> ValidatedFrozenSource | ValidatedWholeSource | ValidatedNoProof:
    """Parse one envelope and route it to the mode-specific validator."""

    if isinstance(response, ResponseEnvelope):
        envelope = response
    else:
        envelope = parse_response_for_mode(response, mode)
    if mode not in {"frozen_hole", "whole_file"}:
        raise ValueError("mode must be whole_file or frozen_hole")
    if envelope.kind == "no_proof":
        resolved = _resolve_task(task, root=root)
        return ValidatedNoProof(resolved, mode, envelope.disclosure)
    if mode == "frozen_hole":
        if envelope.kind != "proof":
            raise FrozenValidationError("response kind does not match frozen mode", reason_code=ReasonCode.KIND_MISMATCH)
        return validate_frozen(
            task,
            envelope.text,
            root=root,
            allow_solved_metavariables=allow_solved_metavariables,
            final_metavariables=final_metavariables,
        )
    if envelope.kind != "file":
        raise WholeValidationError("response kind does not match whole-file mode", reason_code=ReasonCode.KIND_MISMATCH)
    return validate_whole(task, envelope.text, root=root)


validate_submission = validate_response
check_response = validate_response


def materialize_frozen(
    task: TaskSpec | str | Mapping[str, Any],
    proof: str | bytes | bytearray | memoryview,
    *,
    root: str | Path | None = None,
) -> bytes:
    return validate_frozen(task, proof, root=root).source_bytes


__all__ = [
    "FROZEN_COMMAND_TOKENS",
    "FROZEN_FORBIDDEN_TOKENS",
    "LockError",
    "LockViolation",
    "FrozenLockError",
    "FrozenValidationError",
    "MAX_SUBMISSION_BYTES",
    "MARKER",
    "ProtectedSourceDiff",
    "ReasonCode",
    "TrustedTemplate",
    "ValidatedFrozenSource",
    "ValidatedNoProof",
    "ValidatedWholeSource",
    "ValidationError",
    "WholeLockError",
    "WholeValidationError",
    "WHOLE_BLOCKED_TOKENS",
    "check_response",
    "classify_protected_diff",
    "diff_protected_source",
    "freeze_proof",
    "load_trusted_template",
    "materialize_frozen",
    "parse_response_for_mode",
    "protected_diff",
    "validate_frozen",
    "validate_frozen_hole",
    "validate_proof",
    "validate_response",
    "validate_source",
    "validate_submission",
    "validate_whole",
    "validate_whole_file",
]
