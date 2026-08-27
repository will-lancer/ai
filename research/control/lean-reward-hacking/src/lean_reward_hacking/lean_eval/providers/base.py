"""Shared provider protocol for the Lean proof-interface evaluation.

The provider boundary is deliberately small.  It speaks bytes to an injected
transport, keeps provider-specific JSON mapping in the adapter modules, and
does not import an SDK.  A live adapter creates its transport only after the
shared :class:`~lean_reward_hacking.lean_eval.compute.PaidDispatchGate` has
authorised the request.  Offline tests pass a fake transport directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest
from urllib.request import urlopen

from ..canonical import canonical_bytes, canonical_dumps, domain_hash, strict_loads
from ..compute import PaidDispatchGate


MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 8_192


class ProviderError(RuntimeError):
    """Base exception for provider boundary failures."""


class ProviderConfigError(ValueError, ProviderError):
    """Raised when an adapter or request has an invalid configuration."""


class TransportError(ProviderError):
    """Raised when a transport cannot produce a bounded HTTP response."""


class ProviderHTTPError(ProviderError):
    """Raised for a non-successful provider HTTP status."""

    def __init__(self, status_code: int, message: str = "provider request failed") -> None:
        self.status_code = status_code
        super().__init__(f"{message} (HTTP {status_code})")


class ResponseParseError(ProviderError):
    """Raised when a provider response is not the expected strict schema."""


class ProviderAuthorizationError(ProviderError):
    """Raised when a live provider request lacks an exact preflight fact."""


class ResponseStatus(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ProviderPreflight:
    """Bound account/model metadata required by a live adapter."""

    provider: str
    model_id: str
    endpoint: str
    available: bool = True

    def __post_init__(self) -> None:
        for name in ("provider", "model_id", "endpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ProviderConfigError(f"preflight.{name} must be non-empty text")
        if not isinstance(self.available, bool):
            raise ProviderConfigError("preflight.available must be boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProviderPreflight":
        if not isinstance(value, Mapping):
            raise ProviderConfigError("preflight metadata must be an object")
        try:
            return cls(
                provider=value["provider"],
                model_id=value.get("model_id", value.get("model")),
                endpoint=value["endpoint"],
                available=value.get("available", True),
            )
        except KeyError as exc:
            raise ProviderConfigError(f"preflight field is missing: {exc.args[0]}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "available": self.available,
        }


@dataclass(frozen=True, slots=True)
class HTTPRequest:
    """One exact outbound HTTP request."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        method = self.method.upper() if isinstance(self.method, str) else ""
        if method not in {"POST", "GET"}:
            raise ProviderConfigError("HTTP method must be POST or GET")
        if not isinstance(self.url, str) or not self.url.startswith(("https://", "http://")):
            raise ProviderConfigError("HTTP URL must use http or https")
        if not isinstance(self.headers, Mapping):
            raise ProviderConfigError("HTTP headers must be a mapping")
        normalised: dict[str, str] = {}
        for key, value in self.headers.items():
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise ProviderConfigError("HTTP headers must contain text keys and values")
            if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
                raise ProviderConfigError("HTTP headers cannot contain line breaks")
            normalised[key] = value
        try:
            body = bytes(self.body)
        except (TypeError, ValueError) as exc:
            raise ProviderConfigError("HTTP body must be bytes") from exc
        if len(body) > MAX_REQUEST_BYTES:
            raise ProviderConfigError("HTTP request exceeds the size limit")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "headers", MappingProxyType(normalised))
        object.__setattr__(self, "body", body)

    @property
    def json(self) -> Any:
        try:
            return strict_loads(self.body, max_bytes=MAX_REQUEST_BYTES)
        except ValueError as exc:
            raise ProviderConfigError(f"request body is not strict JSON: {exc}") from exc


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    """Bounded response returned by a transport."""

    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""

    def __post_init__(self) -> None:
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int):
            raise TransportError("HTTP status must be an integer")
        if self.status_code < 100 or self.status_code > 599:
            raise TransportError("HTTP status is outside the valid range")
        if not isinstance(self.headers, Mapping):
            raise TransportError("HTTP response headers must be a mapping")
        normalised: dict[str, str] = {}
        for key, value in self.headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TransportError("HTTP response headers must be text")
            normalised[key] = value
        try:
            body = bytes(self.body)
        except (TypeError, ValueError) as exc:
            raise TransportError("HTTP response body must be bytes") from exc
        if len(body) > MAX_RESPONSE_BYTES:
            raise TransportError("HTTP response exceeds the size limit")
        object.__setattr__(self, "headers", MappingProxyType(normalised))
        object.__setattr__(self, "body", body)

    @property
    def status(self) -> int:
        return self.status_code


@runtime_checkable
class Transport(Protocol):
    """Minimal injectable transport required by an adapter."""

    def send(self, request: HTTPRequest) -> HTTPResponse:
        ...


HTTPTransport = Transport
TransportRequest = HTTPRequest
TransportResponse = HTTPResponse


class UrllibTransport:
    """Bounded production transport with no retries or implicit credentials."""

    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ProviderConfigError("transport timeout must be numeric")
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ProviderConfigError("transport timeout must be between 0 and 300 seconds")
        self.timeout_seconds = float(timeout_seconds)

    def send(self, request: HTTPRequest) -> HTTPResponse:
        if not isinstance(request, HTTPRequest):
            raise TransportError("transport expects an HTTPRequest")
        url_request = URLRequest(
            request.url,
            data=request.body if request.method == "POST" else None,
            headers=dict(request.headers),
            method=request.method,
        )
        try:
            with urlopen(url_request, timeout=self.timeout_seconds) as response:  # noqa: S310
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise TransportError("HTTP response exceeds the size limit")
                headers = {str(key): str(value) for key, value in response.headers.items()}
                return HTTPResponse(int(response.status), headers, body)
        except HTTPError as exc:
            try:
                body = exc.read(MAX_RESPONSE_BYTES + 1)
            except OSError:
                body = b""
            if len(body) > MAX_RESPONSE_BYTES:
                body = body[:MAX_RESPONSE_BYTES]
            headers = {str(key): str(value) for key, value in exc.headers.items()}
            return HTTPResponse(int(exc.code), headers, body)
        except (URLError, TimeoutError, OSError) as exc:
            raise TransportError("provider transport failed") from exc

    request = send


@dataclass(frozen=True, slots=True)
class Usage:
    """Strict token usage reported by a provider."""

    input_tokens: int
    output_tokens: int
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ResponseParseError(f"usage.{name} must be a non-negative integer")
        total = self.total_tokens
        if total is None:
            total = self.input_tokens + self.output_tokens
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ResponseParseError("usage.total_tokens must be a non-negative integer")
        if total != self.input_tokens + self.output_tokens:
            raise ResponseParseError("usage.total_tokens does not equal input plus output")
        object.__setattr__(self, "total_tokens", total)

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True, init=False)
class GenerationRequest:
    """Provider-neutral task request with deterministic retry context."""

    prompt: str
    system_prompt: str
    mode_instructions: str
    checker_feedback: tuple[str, ...]
    max_output_tokens: int
    attempt: int
    previous_response_id: str | None
    previous_assistant_content: tuple[Mapping[str, Any], ...]
    previous_model_parts: tuple[Mapping[str, Any], ...]
    previous_user_text: str | None
    metadata: Mapping[str, Any]
    estimated_microdollars: int
    temperature: float | None

    def __init__(
        self,
        prompt: str | None = None,
        *,
        task_prompt: str | None = None,
        user_prompt: str | None = None,
        system_prompt: str = "",
        mode_instructions: str = "",
        checker_feedback: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]] = (),
        feedback: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]] | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        attempt: int = 1,
        previous_response_id: str | None = None,
        previous_assistant_content: Sequence[Mapping[str, Any]] | None = None,
        assistant_blocks: Sequence[Mapping[str, Any]] | None = None,
        previous_model_parts: Sequence[Mapping[str, Any]] | None = None,
        model_parts: Sequence[Mapping[str, Any]] | None = None,
        previous_user_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        estimated_microdollars: int = 0,
        temperature: float | None = None,
    ) -> None:
        supplied = [value for value in (prompt, task_prompt, user_prompt) if value is not None]
        if not supplied:
            raise ProviderConfigError("a task prompt is required")
        if any(not isinstance(value, str) for value in supplied):
            raise ProviderConfigError("task prompts must be strings")
        if len(set(supplied)) != 1:
            raise ProviderConfigError("prompt aliases disagree")
        text = supplied[0]
        if not isinstance(system_prompt, str) or not isinstance(mode_instructions, str):
            raise ProviderConfigError("system and mode instructions must be strings")
        if any("\x00" in value for value in (text, system_prompt, mode_instructions)):
            raise ProviderConfigError("prompt text cannot contain NUL")
        if any(
            len(value.encode("utf-8", "strict")) > MAX_REQUEST_BYTES
            for value in (text, system_prompt, mode_instructions)
        ):
            raise ProviderConfigError("prompt text exceeds the size limit")
        if feedback is not None:
            if checker_feedback not in ((), "", None):
                raise ProviderConfigError("use checker_feedback or feedback, not both")
            checker_feedback = feedback
        normalized_feedback = _normalise_feedback(checker_feedback)
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
            raise ProviderConfigError("max_output_tokens must be an integer")
        if max_output_tokens < 1 or max_output_tokens > 1_000_000:
            raise ProviderConfigError("max_output_tokens is outside the supported range")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ProviderConfigError("attempt must be a one-based integer")
        if previous_response_id is not None:
            if not isinstance(previous_response_id, str) or not previous_response_id:
                raise ProviderConfigError("previous_response_id must be non-empty text")
        if previous_assistant_content is not None and assistant_blocks is not None:
            if tuple(previous_assistant_content) != tuple(assistant_blocks):
                raise ProviderConfigError("assistant content aliases disagree")
        if previous_assistant_content is None:
            previous_assistant_content = assistant_blocks
        if previous_model_parts is not None and model_parts is not None:
            if tuple(previous_model_parts) != tuple(model_parts):
                raise ProviderConfigError("model parts aliases disagree")
        if previous_model_parts is None:
            previous_model_parts = model_parts
        assistant_content = _normalise_blocks(previous_assistant_content, "previous_assistant_content")
        model_content = _normalise_blocks(previous_model_parts, "previous_model_parts")
        if previous_user_text is not None:
            if not isinstance(previous_user_text, str) or "\x00" in previous_user_text:
                raise ProviderConfigError("previous_user_text must be safe text")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ProviderConfigError("request metadata must be a mapping")
        if (
            isinstance(estimated_microdollars, bool)
            or not isinstance(estimated_microdollars, int)
            or estimated_microdollars < 0
        ):
            raise ProviderConfigError("estimated_microdollars must be non-negative")
        if temperature is not None:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
                raise ProviderConfigError("temperature must be numeric")
            if temperature < 0 or temperature > 2:
                raise ProviderConfigError("temperature is outside the supported range")
        object.__setattr__(self, "prompt", text)
        object.__setattr__(self, "system_prompt", system_prompt)
        object.__setattr__(self, "mode_instructions", mode_instructions)
        object.__setattr__(self, "checker_feedback", normalized_feedback)
        object.__setattr__(self, "max_output_tokens", max_output_tokens)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "previous_response_id", previous_response_id)
        object.__setattr__(self, "previous_assistant_content", assistant_content)
        object.__setattr__(self, "previous_model_parts", model_content)
        object.__setattr__(self, "previous_user_text", previous_user_text)
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata)))
        object.__setattr__(self, "estimated_microdollars", estimated_microdollars)
        object.__setattr__(self, "temperature", None if temperature is None else float(temperature))

    @property
    def task_prompt(self) -> str:
        return self.prompt

    @property
    def user_prompt(self) -> str:
        return self.prompt

    @property
    def feedback(self) -> tuple[str, ...]:
        return self.checker_feedback

    def rendered_user_text(self) -> str:
        sections = [self.prompt]
        if self.mode_instructions:
            sections.append("Mode instructions:\n" + self.mode_instructions)
        if self.checker_feedback:
            sections.append("Checker feedback:\n" + "\n\n".join(self.checker_feedback))
        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "mode_instructions": self.mode_instructions,
            "checker_feedback": list(self.checker_feedback),
            "max_output_tokens": self.max_output_tokens,
            "attempt": self.attempt,
            "previous_response_id": self.previous_response_id,
            "previous_assistant_content": [dict(item) for item in self.previous_assistant_content],
            "previous_model_parts": [dict(item) for item in self.previous_model_parts],
            "previous_user_text": self.previous_user_text,
            "metadata": dict(self.metadata),
            "estimated_microdollars": self.estimated_microdollars,
            "temperature": self.temperature,
        }


ProviderRequest = GenerationRequest


def _normalise_feedback(
    value: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]] | None,
) -> tuple[str, ...]:
    if value is None or value == "" or value == ():
        return ()
    if isinstance(value, str) or isinstance(value, Mapping):
        values: Sequence[str | Mapping[str, Any]] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise ProviderConfigError("checker feedback must be text, JSON objects, or a sequence")
    result: list[str] = []
    for item in values:
        if isinstance(item, str):
            if "\x00" in item:
                raise ProviderConfigError("checker feedback cannot contain NUL")
            result.append(item)
        elif isinstance(item, Mapping):
            try:
                encoded = canonical_dumps(item)
            except ValueError as exc:
                raise ProviderConfigError(f"checker feedback is not canonical JSON: {exc}") from exc
            result.append(encoded)
        else:
            raise ProviderConfigError("checker feedback entries must be text or mappings")
    return tuple(result)


def _normalise_blocks(
    value: Sequence[Mapping[str, Any]] | None,
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ProviderConfigError(f"{name} must be a sequence of objects")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ProviderConfigError(f"{name}[{index}] must be an object")
        try:
            checked = strict_loads(canonical_bytes(item))
        except ValueError as exc:
            raise ProviderConfigError(f"{name}[{index}] is not canonical JSON: {exc}") from exc
        if not isinstance(checked, Mapping):
            raise ProviderConfigError(f"{name}[{index}] must remain an object")
        result.append(dict(checked))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    """Capabilities exposed to evaluated models."""

    tools: tuple[str, ...] = ()
    shell: bool = False
    network: bool = False
    filesystem: bool = False
    model_weights: bool = False

    def __post_init__(self) -> None:
        if tuple(self.tools) != self.tools or any(not isinstance(item, str) for item in self.tools):
            raise ProviderConfigError("capability tools must be a tuple of strings")
        for name in ("shell", "network", "filesystem", "model_weights"):
            if not isinstance(getattr(self, name), bool):
                raise ProviderConfigError(f"capability {name} must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": list(self.tools),
            "shell": self.shell,
            "network": self.network,
            "filesystem": self.filesystem,
            "model_weights": self.model_weights,
        }


EMPTY_CAPABILITY_MANIFEST = CapabilityManifest()


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Normalized provider result retained by the run record layer."""

    provider: str
    requested_model_id: str
    returned_model_id: str
    provider_response_id: str
    text: str
    usage: Usage
    response_status: str = ResponseStatus.COMPLETED.value
    incomplete_reason: str = ""
    tool_calls: tuple[str, ...] = ()
    assistant_content: tuple[Mapping[str, Any], ...] = ()
    model_parts: tuple[Mapping[str, Any], ...] = ()
    status_code: int = 200
    raw_body: bytes = b""
    request_hash: str = ""
    response_hash: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        for name in ("provider", "requested_model_id", "returned_model_id", "provider_response_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ResponseParseError(f"{name} must be non-empty text")
        if not isinstance(self.text, str):
            raise ResponseParseError("response text must be text")
        if self.response_status not in {item.value for item in ResponseStatus}:
            raise ResponseParseError("unknown response status")
        if not isinstance(self.incomplete_reason, str) or not isinstance(self.error, str):
            raise ResponseParseError("response diagnostics must be text")
        if not isinstance(self.tool_calls, tuple) or any(not isinstance(item, str) for item in self.tool_calls):
            raise ResponseParseError("tool_calls must be a tuple of names")
        if not isinstance(self.assistant_content, tuple) or any(
            not isinstance(item, Mapping) for item in self.assistant_content
        ):
            raise ResponseParseError("assistant_content must be a tuple of objects")
        if not isinstance(self.model_parts, tuple) or any(
            not isinstance(item, Mapping) for item in self.model_parts
        ):
            raise ResponseParseError("model_parts must be a tuple of objects")
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int):
            raise ResponseParseError("response status code must be an integer")
        object.__setattr__(self, "raw_body", bytes(self.raw_body))

    @property
    def response_id(self) -> str:
        return self.provider_response_id

    @property
    def model_id(self) -> str:
        return self.returned_model_id

    @property
    def returned_model(self) -> str:
        return self.returned_model_id

    @property
    def completed(self) -> bool:
        return self.response_status == ResponseStatus.COMPLETED.value

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "provider": self.provider,
            "requested_model_id": self.requested_model_id,
            "returned_model_id": self.returned_model_id,
            "provider_response_id": self.provider_response_id,
            "text": self.text,
            "usage": self.usage.to_dict(),
            "response_status": self.response_status,
            "incomplete_reason": self.incomplete_reason,
            "tool_calls": list(self.tool_calls),
            "assistant_content": [dict(item) for item in self.assistant_content],
            "model_parts": [dict(item) for item in self.model_parts],
            "status_code": self.status_code,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "error": self.error,
        }
        if include_raw:
            result["raw_body"] = self.raw_body.decode("utf-8", "replace")
        return result


ProviderResult = ProviderResponse


class ProviderAdapter(Protocol):
    """Structural interface implemented by all adapters."""

    provider: str
    model_id: str

    @property
    def capabilities(self) -> CapabilityManifest:
        ...

    def generate(self, request: GenerationRequest | Mapping[str, Any]) -> ProviderResponse:
        ...


class BaseProviderAdapter:
    """Common transport, authorization, and response bookkeeping."""

    provider = ""
    model_id = ""
    endpoint = ""
    request_headers: Mapping[str, str] = MappingProxyType({})

    def __init__(
        self,
        *,
        model_id: str,
        transport: Transport | Any | None = None,
        transport_factory: Callable[[], Transport] | None = None,
        gate: PaidDispatchGate | None = None,
        config: Mapping[str, Any] | str | bytes | os.PathLike[str] | None = None,
        live: bool = False,
        cli_live: bool = False,
        approval: Any = None,
        credentials: Mapping[str, Any] | Sequence[str] | bool | None = None,
        preflight_ok: bool = False,
        preflight: ProviderPreflight | Mapping[str, Any] | None = None,
        account_preflight: ProviderPreflight | Mapping[str, Any] | None = None,
        api_key: str | None = None,
        transport_timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ProviderConfigError("model_id must be non-empty text")
        self._validate_model_id(model_id)
        if transport is not None and transport_factory is not None:
            raise ProviderConfigError("pass transport or transport_factory, not both")
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise ProviderConfigError("api_key must be non-empty text when supplied")
        if preflight is not None and account_preflight is not None:
            raise ProviderConfigError("pass preflight or account_preflight, not both")
        if preflight is None:
            preflight = account_preflight
        if isinstance(preflight, Mapping):
            preflight = ProviderPreflight.from_mapping(preflight)
        elif preflight is not None and not isinstance(preflight, ProviderPreflight):
            raise ProviderConfigError("preflight must be ProviderPreflight or mapping")
        self.model_id = model_id
        self._transport = transport
        self._transport_factory = transport_factory
        self._api_key = api_key
        self._credentials = credentials
        self._preflight = preflight
        self._live = live is True
        self._gate = gate
        if self._gate is None and self._live:
            gate_credentials: Mapping[str, Any] | Sequence[str] | bool | None = credentials
            if api_key:
                gate_credentials = {self.credential_name: api_key}
            self._gate = PaidDispatchGate(
                config,
                live=True,
                cli_live=cli_live,
                approval=approval,
                credentials=gate_credentials,
                preflight_ok=preflight_ok,
            )
        if self._live and self._transport_factory is None and self._transport is None:
            self._transport_factory = lambda: UrllibTransport(timeout_seconds=transport_timeout_seconds)
        if self._live and self._gate is None:
            raise ProviderConfigError("live adapters require a paid dispatch gate")

    @property
    def capabilities(self) -> CapabilityManifest:
        return EMPTY_CAPABILITY_MANIFEST

    @property
    def capability_manifest(self) -> CapabilityManifest:
        return self.capabilities

    @property
    def requested_model_id(self) -> str:
        return self.model_id

    def _validate_model_id(self, model_id: str) -> None:
        raise NotImplementedError

    @property
    def credential_name(self) -> str:
        return "API_KEY"

    def _credential(self) -> str | None:
        if self._api_key:
            return self._api_key
        value = self._credentials
        if isinstance(value, Mapping):
            for key in (self.credential_name, "api_key", "API_KEY"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for candidate in value:
                if isinstance(candidate, str) and candidate:
                    return candidate
        return None

    def _build_payload(self, request: GenerationRequest) -> Mapping[str, Any]:
        raise NotImplementedError

    def _parse_payload(self, body: bytes, request: GenerationRequest) -> ProviderResponse:
        raise NotImplementedError

    def build_http_request(self, request: GenerationRequest | Mapping[str, Any]) -> HTTPRequest:
        normalized = coerce_request(request)
        payload = self._build_payload(normalized)
        try:
            body = canonical_bytes(payload)
        except ValueError as exc:
            raise ProviderConfigError(f"provider request is not canonical JSON: {exc}") from exc
        headers = dict(self.request_headers)
        key = self._credential()
        if key:
            headers.update(self._credential_headers(key))
        headers["Content-Type"] = "application/json"
        headers.setdefault("Accept", "application/json")
        return HTTPRequest("POST", self.endpoint_for(normalized), headers, body)

    def endpoint_for(self, request: GenerationRequest) -> str:
        return self.endpoint

    def _credential_headers(self, key: str) -> Mapping[str, str]:
        return {"Authorization": "Bearer " + key}

    def generate(self, request: GenerationRequest | Mapping[str, Any]) -> ProviderResponse:
        normalized = coerce_request(request)
        if self._live:
            self._require_live_preflight()
            if self._credential() is None:
                raise ProviderAuthorizationError("live provider credentials are missing")
        http_request = self.build_http_request(normalized)
        request_hash = domain_hash("lean-eval/request", http_request.body)

        def callback_factory() -> Callable[[], HTTPResponse]:
            transport = self._make_transport()

            def callback() -> HTTPResponse:
                return _invoke_transport(transport, http_request)

            return callback

        if self._gate is not None:
            raw = self._gate.dispatch_factory(
                self.provider,
                callback_factory,
                model_id=self.model_id,
                estimated_microdollars=normalized.estimated_microdollars,
            )
        else:
            raw = callback_factory()()
        response = coerce_response(raw)
        if not (200 <= response.status_code < 300):
            raise ProviderHTTPError(response.status_code, _safe_provider_error(response.body))
        parsed = self._parse_payload(response.body, normalized)
        response_hash = domain_hash("lean-eval/response", response.body)
        return ProviderResponse(
            provider=parsed.provider,
            requested_model_id=parsed.requested_model_id,
            returned_model_id=parsed.returned_model_id,
            provider_response_id=parsed.provider_response_id,
            text=parsed.text,
            usage=parsed.usage,
            response_status=parsed.response_status,
            incomplete_reason=parsed.incomplete_reason,
            tool_calls=parsed.tool_calls,
            assistant_content=parsed.assistant_content,
            model_parts=parsed.model_parts,
            status_code=response.status_code,
            raw_body=response.body,
            request_hash=request_hash,
            response_hash=response_hash,
            error=parsed.error,
        )

    request = generate
    complete = generate

    def _make_transport(self) -> Any:
        if self._transport is not None:
            return self._transport
        if self._transport_factory is None:
            raise TransportError("no transport was supplied")
        return self._transport_factory()

    def _require_live_preflight(self) -> None:
        preflight = self._preflight
        if preflight is None:
            raise ProviderAuthorizationError("live provider account/model preflight is required")
        if not preflight.available:
            raise ProviderAuthorizationError("provider/model preflight reported unavailable")
        if preflight.provider != self.provider:
            raise ProviderAuthorizationError("provider preflight identity does not match adapter")
        if preflight.model_id != self.model_id:
            raise ProviderAuthorizationError("model preflight identity does not match adapter")
        if preflight.endpoint != self.endpoint:
            raise ProviderAuthorizationError("provider endpoint is outside the pinned allowlist")


def coerce_request(value: GenerationRequest | Mapping[str, Any]) -> GenerationRequest:
    if isinstance(value, GenerationRequest):
        return value
    if not isinstance(value, Mapping):
        raise ProviderConfigError("provider request must be GenerationRequest or mapping")
    try:
        return GenerationRequest(**dict(value))
    except TypeError as exc:
        raise ProviderConfigError(f"invalid provider request fields: {exc}") from exc


def coerce_response(value: Any) -> HTTPResponse:
    if isinstance(value, HTTPResponse):
        return value
    if isinstance(value, Mapping):
        status = value.get("status_code", value.get("status"))
        body = value.get("body", b"")
        headers = value.get("headers", {})
        return HTTPResponse(status, headers, body)
    raise TransportError("transport returned an unsupported response object")


def _invoke_transport(transport: Any, request: HTTPRequest) -> HTTPResponse:
    if transport is None:
        raise TransportError("transport factory returned None")
    method = getattr(transport, "send", None)
    if callable(method):
        return coerce_response(method(request))
    method = getattr(transport, "request", None)
    if callable(method):
        return coerce_response(method(request))
    if callable(transport):
        return coerce_response(transport(request))
    raise TransportError("transport must implement send or request")


def parse_json_object(body: bytes, *, provider: str) -> dict[str, Any]:
    try:
        value = strict_loads(body, max_bytes=MAX_RESPONSE_BYTES)
    except ValueError as exc:
        raise ResponseParseError(f"{provider} response is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ResponseParseError(f"{provider} response must be a JSON object")
    return value


def require_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ResponseParseError(f"{field_name} must be {'a' if allow_empty else 'non-empty'} string")
    if "\x00" in value:
        raise ResponseParseError(f"{field_name} contains NUL")
    return value


def require_object(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResponseParseError(f"{field_name} must be an object")
    return value


def require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponseParseError(f"{field_name} must be a non-negative integer")
    return value


def parse_usage(
    value: Any,
    *,
    provider: str,
    input_field: str = "input_tokens",
    output_field: str = "output_tokens",
    total_field: str | None = "total_tokens",
) -> Usage:
    usage = require_object(value, f"{provider}.usage")
    input_tokens = require_nonnegative_int(usage.get(input_field), f"usage.{input_field}")
    output_tokens = require_nonnegative_int(usage.get(output_field), f"usage.{output_field}")
    total: int | None = None
    if total_field is not None and total_field in usage:
        total = require_nonnegative_int(usage[total_field], f"usage.{total_field}")
    return Usage(input_tokens, output_tokens, total)


def extract_text_fragments(value: Any, *, provider: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ResponseParseError(f"{provider} content must be an array")
    fragments: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ResponseParseError(f"{provider} content[{index}] must be an object")
        text = item.get("text")
        if isinstance(text, str):
            fragments.append(text)
    if not fragments:
        raise ResponseParseError(f"{provider} response contains no visible text")
    return tuple(fragments)


def _safe_provider_error(body: bytes) -> str:
    """Return a short error marker without echoing credentials or raw text."""

    try:
        value = strict_loads(body, max_bytes=MAX_RESPONSE_BYTES)
    except ValueError:
        return "provider returned an error response"
    if isinstance(value, Mapping):
        error = value.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            if isinstance(code, str) and code:
                return "provider returned error code " + code[:80]
        message = value.get("message")
        if isinstance(message, str) and message:
            return "provider returned an error message"
    return "provider returned an error response"


__all__ = [
    "BaseProviderAdapter",
    "CapabilityManifest",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "EMPTY_CAPABILITY_MANIFEST",
    "GenerationRequest",
    "HTTPResponse",
    "HTTPRequest",
    "HTTPTransport",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "ProviderAdapter",
    "ProviderAuthorizationError",
    "ProviderConfigError",
    "ProviderError",
    "ProviderHTTPError",
    "ProviderPreflight",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderResult",
    "ResponseParseError",
    "ResponseStatus",
    "Transport",
    "TransportError",
    "TransportRequest",
    "TransportResponse",
    "Usage",
    "UrllibTransport",
    "coerce_request",
    "coerce_response",
    "extract_text_fragments",
    "parse_json_object",
    "parse_usage",
    "require_nonnegative_int",
    "require_object",
    "require_text",
]
