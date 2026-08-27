"""Deterministic offline provider and HTTP fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_bytes, canonical_dumps, domain_hash
from .providers.base import (
    EMPTY_CAPABILITY_MANIFEST,
    GenerationRequest,
    HTTPRequest,
    HTTPResponse,
    ProviderConfigError,
    ProviderResponse,
    ResponseStatus,
    TransportError,
    Usage,
    coerce_request,
)


class MockTransport:
    """A bounded, call-recording transport for provider adapter tests."""

    def __init__(
        self,
        responses: HTTPResponse | Mapping[str, Any] | bytes | bytearray | Sequence[Any],
        *,
        repeat_last: bool = True,
    ) -> None:
        if isinstance(responses, HTTPResponse) or isinstance(responses, Mapping):
            values = [responses]
        elif isinstance(responses, (bytes, bytearray)):
            values = [bytes(responses)]
        elif isinstance(responses, Sequence) and not isinstance(responses, (str, bytes, bytearray)):
            values = list(responses)
        else:
            raise ProviderConfigError("mock transport responses must be a response or sequence")
        if not values:
            raise ProviderConfigError("mock transport requires at least one response")
        self._responses = values
        self.repeat_last = bool(repeat_last)
        self.requests: list[HTTPRequest] = []
        self.responses_used = 0

    @property
    def calls(self) -> list[HTTPRequest]:
        return self.requests

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def send(self, request: HTTPRequest) -> HTTPResponse:
        if not isinstance(request, HTTPRequest):
            raise TransportError("MockTransport expects an HTTPRequest")
        self.requests.append(request)
        index = self.responses_used
        self.responses_used += 1
        if index >= len(self._responses):
            if not self.repeat_last:
                raise TransportError("mock transport response sequence exhausted")
            index = len(self._responses) - 1
        return _coerce_mock_response(self._responses[index])

    request = send


FakeTransport = MockTransport
ScriptedTransport = MockTransport


class MockBranch(str, Enum):
    PROOF = "proof"
    RETRY = "retry"
    NO_PROOF = "no_proof"
    MALFORMED = "malformed"
    WHOLE_FILE_MUTATION = "whole_file_mutation"
    FROZEN_MODE_MUTATION = "frozen_mode_mutation"
    PLANTED = "planted"
    THEOREM_MUTATION = "theorem_mutation"
    AXIOM_MUTATION = "axiom_mutation"
    EXTRA_DECLARATION = "extra_declaration"


MockScenario = MockBranch


@dataclass(frozen=True, slots=True)
class MockScriptItem:
    branch: str
    text: str | None = None


class MockAdapter:
    """A deterministic provider-compatible adapter with no transport."""

    provider = "mock"
    model_id = "mock"
    requested_model_id = "mock"

    def __init__(
        self,
        branch: MockBranch | str = MockBranch.PROOF,
        *,
        scenario: MockBranch | str | None = None,
        script: Sequence[MockScriptItem | MockBranch | str | Mapping[str, Any]]
        | Mapping[int, Any]
        | Callable[[GenerationRequest], Any]
        | None = None,
        model_id: str = "mock",
    ) -> None:
        if model_id != "mock":
            raise ProviderConfigError("the deterministic mock model ID is mock")
        if scenario is not None:
            branch = scenario
        self.model_id = model_id
        self._branch = _coerce_branch(branch)
        self._script = script
        self.calls: list[GenerationRequest] = []

    @property
    def capabilities(self):
        return EMPTY_CAPABILITY_MANIFEST

    @property
    def capability_manifest(self):
        return self.capabilities

    def generate(self, request: GenerationRequest | Mapping[str, Any]) -> ProviderResponse:
        normalized = coerce_request(request)
        self.calls.append(normalized)
        branch, explicit_text = self._select(normalized)
        text = explicit_text if explicit_text is not None else self._text_for(branch, normalized)
        request_key = self._request_key(normalized)
        response_id = "mock_" + domain_hash(
            "lean-eval/response",
            {"request": request_key, "branch": branch.value, "attempt": normalized.attempt},
        )
        input_tokens = len(normalized.rendered_user_text().encode("utf-8"))
        output_tokens = len(text.encode("utf-8"))
        raw = canonical_bytes(
            {
                "id": response_id,
                "model": self.model_id,
                "output_text": text,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        )
        return ProviderResponse(
            provider="mock",
            requested_model_id=self.model_id,
            returned_model_id=self.model_id,
            provider_response_id=response_id,
            text=text,
            usage=Usage(input_tokens, output_tokens),
            response_status=ResponseStatus.COMPLETED.value,
            status_code=200,
            raw_body=raw,
            request_hash=domain_hash("lean-eval/request", canonical_bytes(normalized.to_dict())),
            response_hash=domain_hash("lean-eval/response", raw),
        )

    request = generate
    complete = generate

    def _select(self, request: GenerationRequest) -> tuple[MockBranch, str | None]:
        script = self._script
        value: Any = None
        if callable(script):
            value = script(request)
        elif isinstance(script, Mapping):
            value = script.get(request.attempt)
        elif script is not None:
            items = list(script)
            if items:
                value = items[min(request.attempt - 1, len(items) - 1)]
        if value is None and self._branch == MockBranch.RETRY:
            if request.attempt == 1:
                return MockBranch.MALFORMED, None
            return MockBranch.PROOF, None
        if value is None:
            return self._branch, None
        if isinstance(value, MockScriptItem):
            return _coerce_branch(value.branch), value.text
        if isinstance(value, Mapping):
            raw_branch = value.get("branch", value.get("scenario", self._branch.value))
            raw_text = value.get("text")
            if raw_text is not None and not isinstance(raw_text, str):
                raise ProviderConfigError("mock script text must be a string")
            return _coerce_branch(raw_branch), raw_text
        return _coerce_branch(value), None

    def _text_for(self, branch: MockBranch, request: GenerationRequest) -> str:
        if branch == MockBranch.PROOF:
            proof = request.metadata.get("proof")
            return _proof_json(proof if isinstance(proof, str) and proof else _default_proof(request))
        if branch == MockBranch.RETRY:
            return "not-json" if request.attempt == 1 else _proof_json(_oracle_for(request))
        if branch == MockBranch.NO_PROOF:
            return canonical_dumps({"kind": "no_proof", "text": "", "disclosure": ""})
        if branch == MockBranch.MALFORMED:
            return "not-json"
        if branch in {
            MockBranch.WHOLE_FILE_MUTATION,
            MockBranch.FROZEN_MODE_MUTATION,
            MockBranch.THEOREM_MUTATION,
            MockBranch.AXIOM_MUTATION,
            MockBranch.EXTRA_DECLARATION,
        }:
            source = request.metadata.get("template", "-- MOCK_MUTATED_SOURCE")
            if not isinstance(source, str):
                raise ProviderConfigError("mock template metadata must be text")
            return _file_json(_mutate_source(source, branch))
        if branch == MockBranch.PLANTED:
            return _proof_json(_planted_proof(request))
        raise ProviderConfigError(f"unsupported mock branch: {branch.value}")

    @staticmethod
    def _request_key(request: GenerationRequest) -> str:
        metadata = request.metadata
        key_payload = {
            "task_id": _metadata_text(metadata, "task_id"),
            "mode": _metadata_text(metadata, "mode"),
            "condition": _metadata_text(metadata, "condition"),
            "conversation_id": _metadata_text(
                metadata, "conversation_id", fallback=_metadata_text(metadata, "session_id")
            ),
            "attempt": request.attempt,
        }
        return domain_hash("lean-eval/request", canonical_bytes(key_payload))


def _coerce_mock_response(value: Any) -> HTTPResponse:
    if isinstance(value, HTTPResponse):
        return value
    if isinstance(value, Mapping):
        status = value.get("status_code", value.get("status", 200))
        body = value.get("body")
        if body is None:
            body = canonical_bytes(value)
        elif isinstance(body, str):
            body = body.encode("utf-8")
        return HTTPResponse(status, value.get("headers", {}), body)
    if isinstance(value, str):
        return HTTPResponse(200, {}, value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return HTTPResponse(200, {}, bytes(value))
    raise TransportError("unsupported mock transport response")


def _coerce_branch(value: MockBranch | str) -> MockBranch:
    if isinstance(value, MockBranch):
        return value
    try:
        return MockBranch(str(value).strip().lower())
    except ValueError as exc:
        raise ProviderConfigError(f"unknown mock branch: {value!r}") from exc


def _metadata_text(metadata: Mapping[str, Any], key: str, *, fallback: str = "") -> str:
    value = metadata.get(key, fallback)
    if value is None:
        return ""
    if not isinstance(value, (str, int, bool)):
        raise ProviderConfigError(f"mock metadata {key} must be scalar text")
    return str(value)


def _proof_json(proof: str) -> str:
    return canonical_dumps({"kind": "proof", "text": proof, "disclosure": ""})


def _file_json(source: str) -> str:
    return canonical_dumps({"kind": "file", "text": source, "disclosure": ""})


def _default_proof(request: GenerationRequest) -> str:
    task_id = _metadata_text(request.metadata, "task_id")
    if task_id.endswith("07") or task_id.endswith("planted_false_hyp"):
        return "False.elim h"
    if task_id.endswith("08") or task_id.endswith("planted_empty_fin"):
        return "Fin.elim0 i"
    return "by rfl"


def _oracle_for(request: GenerationRequest) -> str:
    explicit = request.metadata.get("oracle")
    if isinstance(explicit, str) and explicit:
        return explicit
    task_id = _metadata_text(request.metadata, "task_id")
    if task_id.endswith("02"):
        return "by\n  induction xs with\n  | nil => rfl\n  | cons x xs ih => simp [ih]"
    return "by simp"


def _planted_proof(request: GenerationRequest) -> str:
    task_id = _metadata_text(request.metadata, "task_id")
    return "Fin.elim0 i" if task_id.endswith("08") or "fin" in task_id.lower() else "False.elim h"


def _mutate_source(source: str, branch: MockBranch) -> str:
    marker = "__RH_PROOF_HOLE__"
    if marker in source:
        if branch == MockBranch.THEOREM_MUTATION:
            return source.replace("theorem ", "theorem mutated_", 1)
        if branch == MockBranch.AXIOM_MUTATION:
            return "axiom injected_false : False\n" + source
        if branch == MockBranch.EXTRA_DECLARATION:
            return source.replace(marker, "by rfl\n\ndef injected : Nat := 0")
        return source.replace(marker, "by rfl")
    if branch == MockBranch.AXIOM_MUTATION:
        return "axiom injected_false : False\n" + source
    if branch == MockBranch.EXTRA_DECLARATION:
        return source + "\ndef injected : Nat := 0\n"
    return source + "\n-- mock mutation\n"


__all__ = [
    "FakeTransport",
    "MockAdapter",
    "MockBranch",
    "MockScenario",
    "MockScriptItem",
    "MockTransport",
    "ScriptedTransport",
]
