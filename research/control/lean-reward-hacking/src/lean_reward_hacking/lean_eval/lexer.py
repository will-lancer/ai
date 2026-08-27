"""Small, conservative lexer for the proof-interface boundary.

This is deliberately not a Lean parser.  It has one job before Lean is
started: distinguish code from comments and quoted text, report byte offsets,
and make command-token policy checks deterministic.  The scanner understands
nested block comments and the quoted forms which are relevant to the lock.
Any syntax it does not understand remains an ordinary token and is left for
Lean to diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import unicodedata
from typing import Iterable, Iterator, Sequence


MAX_LEX_BYTES = 64 * 1024


class LexState(str, Enum):
    NORMAL = "NORMAL"
    CODE = "NORMAL"
    LINE_COMMENT = "LINE_COMMENT"
    BLOCK_COMMENT = "BLOCK_COMMENT"
    STRING = "STRING"
    RAW_STRING = "RAW_STRING"
    INTERPOLATION_TEXT = "INTERPOLATION_TEXT"
    INTERPOLATION_CODE = "INTERPOLATION_CODE"
    CHAR = "CHAR"
    QUOTED_IDENTIFIER = "QUOTED_IDENTIFIER"
    NAME_LITERAL = "NAME_LITERAL"


class LexError(ValueError):
    """A source could not be tokenized to a normal end state."""

    def __init__(self, message: str, *, code: str = "lex_error", offset: int | None = None) -> None:
        self.code = code
        self.reason_code = code
        self.offset = offset
        suffix = "" if offset is None else f" at byte {offset}"
        super().__init__(f"{message}{suffix}")


@dataclass(frozen=True, slots=True)
class LexToken:
    """One active token, with UTF-8 byte offsets into the original source."""

    text: str
    start: int
    end: int
    kind: str = "symbol"

    @property
    def value(self) -> str:
        return self.text

    @property
    def offset(self) -> int:
        return self.start


Token = LexToken


@dataclass(frozen=True, slots=True)
class LexSpan:
    """An inactive comment or quoted span in the original source."""

    start: int
    end: int
    kind: str


@dataclass(frozen=True, slots=True)
class LexResult:
    """Tokens and inactive spans from one source."""

    source: bytes
    tokens: tuple[LexToken, ...]
    comments: tuple[LexSpan, ...]
    state: LexState = LexState.NORMAL
    delimiters: tuple[str, ...] = ()

    @property
    def end_state(self) -> LexState:
        return self.state

    @property
    def normal_eof(self) -> bool:
        return self.state is LexState.NORMAL

    def active_marker_offsets(self, marker: str = "__RH_PROOF_HOLE__") -> tuple[int, ...]:
        return tuple(token.start for token in self.tokens if token.text == marker)


# Longest first.  Keeping operators as single tokens is also valid for the
# policy scanner, but these spellings make diagnostics and test fixtures much
# easier to read.
_MULTI_SYMBOLS = tuple(
    sorted(
        {
            ":=",
            "=>",
            "->",
            "<-",
            "<->",
            "++",
            "::",
            "==",
            "!=",
            "<=",
            ">=",
            "&&",
            "||",
            "<;>",
            ";;",
            ":>",
            "⟨",
            "⟩",
            "▸",
        },
        key=len,
        reverse=True,
    )
)


def _coerce_source(source: str | bytes | bytearray | memoryview) -> tuple[bytes, str]:
    if isinstance(source, str):
        try:
            raw = source.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise LexError("source is not strict UTF-8", code="invalid_utf8") from exc
    elif isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
    else:
        raise TypeError("source must be text or bytes")
    if len(raw) > MAX_LEX_BYTES:
        raise LexError("source exceeds the lexer size limit", code="size_limit")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise LexError("UTF-8 BOM is not allowed", code="bom", offset=0)
    if b"\x00" in raw:
        raise LexError("NUL is not allowed", code="nul", offset=raw.index(b"\x00"))
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise LexError("source is not strict UTF-8", code="invalid_utf8", offset=exc.start) from exc
    return raw, text


def _is_identifier_start(char: str) -> bool:
    if char == "_" or ("A" <= char <= "Z") or ("a" <= char <= "z"):
        return True
    # Lean accepts Unicode identifiers.  ``isidentifier`` on a one-character
    # string catches letters and a conservative set of identifier symbols.
    return char.isidentifier()


def _is_identifier_continue(char: str) -> bool:
    if _is_identifier_start(char) or char.isdigit() or char == "'":
        return True
    category = unicodedata.category(char)
    return category in {"Mn", "Mc", "Pc"}


def _byte_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def _is_allowed_whitespace(char: str) -> bool:
    return char in {" ", "\t", "\n", "\v", "\f"}


def _looks_like_char(text: str, index: int) -> bool:
    """Return whether an apostrophe begins a character literal.

    Apostrophes are legal in Lean identifiers (``foo'``), so treating every
    apostrophe as a quote would produce false unterminated-literal errors.
    A closing apostrophe before the next line break is enough to recognise a
    literal; Lean performs the more detailed character validity check.
    """

    if index + 1 >= len(text):
        return False
    cursor = index + 1
    escaped = False
    while cursor < len(text):
        char = text[cursor]
        if char == "\n":
            return False
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "'":
            return True
        cursor += 1
    return False


def _consume_quoted(text: str, index: int, delimiter: str, state: LexState) -> int:
    cursor = index + 1
    escaped = False
    while cursor < len(text):
        char = text[cursor]
        if escaped:
            escaped = False
            cursor += 1
            continue
        if char == "\\":
            escaped = True
            cursor += 1
            continue
        if char == delimiter:
            return cursor + 1
        cursor += 1
    raise LexError(
        f"unterminated {state.value.lower()}",
        code=f"unterminated_{state.value.lower()}",
        offset=index,
    )


def _consume_block_comment(text: str, index: int) -> int:
    cursor = index + 2
    depth = 1
    while cursor < len(text):
        if text.startswith("/-", cursor):
            depth += 1
            cursor += 2
        elif text.startswith("-/", cursor):
            depth -= 1
            cursor += 2
            if depth == 0:
                return cursor
        else:
            cursor += 1
    raise LexError("unterminated block comment", code="unterminated_block_comment", offset=index)


def _consume_name_literal(text: str, index: int) -> int:
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == "`":
            return cursor + 1
        if text[cursor] == "\n":
            break
        cursor += 1
    raise LexError("unterminated name literal", code="unterminated_name_literal", offset=index)


def _consume_raw_string(text: str, index: int) -> int:
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == '"':
            return cursor + 1
        cursor += 1
    raise LexError("unterminated raw string", code="unterminated_raw_string", offset=index)


def scan(source: str | bytes | bytearray | memoryview) -> LexResult:
    """Tokenize active Lean text and finish in :class:`LexState.NORMAL`.

    Offsets are bytes, while token text is decoded Unicode.  Comments and
    quoted spans are recorded separately and never appear in ``tokens``.
    """

    raw, text = _coerce_source(source)
    offsets = _byte_offsets(text)
    tokens: list[LexToken] = []
    comments: list[LexSpan] = []
    index = 0
    length = len(text)
    # Each entry is the expected closing character and the byte offset at
    # which its opener appeared.  Interpolation braces are marked separately
    # so the scanner can return to string text after the expression closes.
    delimiters: list[tuple[str, int, str]] = []
    interpolation_text = False
    interpolation_return = False
    raw_string = False

    def add_token(start: int, end: int, kind: str = "symbol") -> None:
        tokens.append(LexToken(text[start:end], offsets[start], offsets[end], kind))

    def add_symbol(start: int, end: int) -> None:
        symbol = text[start:end]
        if symbol in {"(", "[", "{", "⟨"}:
            expected = {
                "(": ")",
                "[": "]",
                "{": "}",
                "⟨": "⟩",
            }[symbol]
            delimiters.append((expected, offsets[start], "ordinary"))
        elif symbol in {")",
            "]",
            "}",
            "⟩",
        }:
            if not delimiters or delimiters[-1][0] != symbol:
                raise LexError("unbalanced delimiter", code="unbalanced_delimiter", offset=offsets[start])
            _, _, kind = delimiters.pop()
            if symbol == "}" and kind == "interpolation":
                nonlocal_interpolation[0] = True
        add_token(start, end, "symbol")

    # A mutable cell lets ``add_symbol`` switch back from interpolation code
    # without making the helper a second scanner state machine.
    nonlocal_interpolation = [False]

    while index < length:
        char = text[index]
        if interpolation_text:
            if char == "\r":
                raise LexError("CR line endings are not allowed", code="cr", offset=offsets[index])
            if char == "\\":
                if index + 1 >= length:
                    raise LexError("unterminated string escape", code="unterminated_string", offset=offsets[index])
                index += 2
                continue
            if char == '"':
                interpolation_text = False
                raw_string = False
                index += 1
                continue
            if char == "{" and not raw_string:
                add_token(index, index + 1, "symbol")
                delimiters.append(("}", offsets[index], "interpolation"))
                interpolation_text = False
                interpolation_return = True
                index += 1
                continue
            index += 1
            continue
        if nonlocal_interpolation[0]:
            # The closing brace that ended an interpolation is consumed by
            # the ordinary branch below.  The flag is reset here so a normal
            # source brace cannot accidentally enter string text.
            nonlocal_interpolation[0] = False
            interpolation_text = True
            interpolation_return = False
            continue
        if char == "\r":
            raise LexError("CR line endings are not allowed", code="cr", offset=offsets[index])
        if ord(char) < 32 and not _is_allowed_whitespace(char):
            raise LexError("control character is not allowed", code="control_char", offset=offsets[index])
        if char.isspace() and not _is_allowed_whitespace(char):
            raise LexError("non-ASCII whitespace is not allowed", code="unicode_whitespace", offset=offsets[index])
        if _is_allowed_whitespace(char):
            index += 1
            continue

        start = index
        if text.startswith("--", index):
            index += 2
            while index < length and text[index] != "\n":
                index += 1
            comments.append(LexSpan(offsets[start], offsets[index], "line_comment"))
            continue
        if text.startswith("/-", index):
            index = _consume_block_comment(text, index)
            comments.append(LexSpan(offsets[start], offsets[index], "block_comment"))
            continue
        if char == "`":
            index = _consume_name_literal(text, index)
            add_token(start, index, "name_literal")
            continue
        if char == '"':
            # Lean's interpolated strings use a short prefix such as s!"...".
            # String text is inactive, while expressions between braces return
            # to the normal scanner and therefore remain policy-visible.
            interpolation_prefix = (
                index >= 2
                and text[index - 1] == "!"
                and text[index - 2] in {"s", "f", "m", "t"}
            )
            if interpolation_prefix:
                interpolation_text = True
                raw_string = False
                index += 1
                continue
            if index >= 1 and text[index - 1] == "r":
                index = _consume_raw_string(text, index)
            else:
                index = _consume_quoted(text, index, '"', LexState.STRING)
            add_token(start, index, "string")
            continue
        if char == "'" and _looks_like_char(text, index):
            index = _consume_quoted(text, index, "'", LexState.CHAR)
            add_token(start, index, "char")
            continue
        if text.startswith("«", index):
            closing = text.find("»", index + 1)
            if closing < 0:
                raise LexError("unterminated quoted identifier", code="unterminated_quoted_identifier", offset=offsets[index])
            index = closing + 1
            add_token(start, index, "quoted_identifier")
            continue
        if _is_identifier_start(char):
            index += 1
            while index < length and _is_identifier_continue(text[index]):
                index += 1
            add_token(start, index, "identifier")
            continue
        if char.isdigit():
            index += 1
            while index < length and (text[index].isalnum() or text[index] in "_'"):
                index += 1
            add_token(start, index, "number")
            continue

        matched = False
        for symbol in _MULTI_SYMBOLS:
            if text.startswith(symbol, index):
                index += len(symbol)
                add_symbol(start, index)
                matched = True
                break
        if matched:
            continue
        index += 1
        add_symbol(start, index)

    if interpolation_text:
        raise LexError("unterminated interpolated string", code="unterminated_string", offset=offsets[max(0, index - 1)])
    if delimiters:
        _, offset, _ = delimiters[-1]
        raise LexError("unclosed delimiter", code="unbalanced_delimiter", offset=offset)
    return LexResult(raw, tuple(tokens), tuple(comments), LexState.NORMAL, tuple(item[0] for item in delimiters))


def lex(source: str | bytes | bytearray | memoryview) -> tuple[LexToken, ...]:
    """Compatibility API returning only active tokens."""

    return scan(source).tokens


tokenize = lex
lex_source = scan
scan_tokens = scan


def active_marker_offsets(
    source: str | bytes | bytearray | memoryview,
    marker: str = "__RH_PROOF_HOLE__",
) -> tuple[int, ...]:
    return scan(source).active_marker_offsets(marker)


def forbidden_tokens(
    source_or_tokens: str | bytes | bytearray | memoryview | Iterable[LexToken],
    forbidden: Iterable[str],
) -> tuple[LexToken, ...]:
    tokens = (
        scan(source_or_tokens).tokens
        if isinstance(source_or_tokens, (str, bytes, bytearray, memoryview))
        else tuple(source_or_tokens)
    )
    blocked = set(forbidden)
    return tuple(token for token in tokens if token.text in blocked)


def token_texts(tokens: Sequence[LexToken]) -> tuple[str, ...]:
    return tuple(token.text for token in tokens)


__all__ = [
    "LexError",
    "LexResult",
    "LexSpan",
    "LexState",
    "LexToken",
    "MAX_LEX_BYTES",
    "Token",
    "active_marker_offsets",
    "forbidden_tokens",
    "lex",
    "lex_source",
    "scan",
    "scan_tokens",
    "token_texts",
    "tokenize",
]
