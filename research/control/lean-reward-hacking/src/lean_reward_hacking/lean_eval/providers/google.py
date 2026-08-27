"""Google Gemini Generate Content adapter for the Lean-eval boundary."""

from __future__ import annotations

from typing import Any, Mapping

from .base import (
    BaseProviderAdapter,
    GenerationRequest,
    ProviderResponse,
    ResponseParseError,
    ResponseStatus,
    parse_json_object,
    require_object,
    require_nonnegative_int,
    require_text,
    Usage,
)


PROVIDER = "google"
GOOGLE_MODEL = "gemini-3.7-flash"
GOOGLE_GENERATE_CONTENT_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/" + GOOGLE_MODEL + ":generateContent"
)
GOOGLE_THINKING_LEVEL = "MEDIUM"


class GoogleAdapter(BaseProviderAdapter):
    """Transport-injected Gemini Generate Content client."""

    provider = PROVIDER
    endpoint = GOOGLE_GENERATE_CONTENT_ENDPOINT
    credential_name = "GOOGLE_API_KEY"

    def __init__(self, model_id: str = GOOGLE_MODEL, **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)

    def _validate_model_id(self, model_id: str) -> None:
        if model_id != GOOGLE_MODEL:
            raise ValueError("Google adapter requires gemini-3.7-flash")

    def _credential_headers(self, key: str) -> Mapping[str, str]:
        return {"x-goog-api-key": key}

    def endpoint_for(self, request: GenerationRequest) -> str:
        return GOOGLE_GENERATE_CONTENT_ENDPOINT

    def _build_payload(self, request: GenerationRequest) -> Mapping[str, Any]:
        contents: list[Mapping[str, Any]] = []
        if request.previous_model_parts:
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": request.previous_user_text or request.rendered_user_text()}],
                }
            )
            contents.append(
                {
                    "role": "model",
                    "parts": [dict(part) for part in request.previous_model_parts],
                }
            )
        contents.append({"role": "user", "parts": [{"text": request.rendered_user_text()}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_output_tokens,
                "thinkingConfig": {
                    "thinkingLevel": GOOGLE_THINKING_LEVEL,
                    "includeThoughts": False,
                },
            },
        }
        if request.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        return payload

    @staticmethod
    def parse_response(
        body: bytes,
        request: GenerationRequest,
        *,
        requested_model_id: str,
    ) -> ProviderResponse:
        value = parse_json_object(body, provider=PROVIDER)
        response_id = require_text(value.get("responseId"), "google.responseId")
        returned_model = require_text(value.get("modelVersion"), "google.modelVersion")
        usage = _parse_google_usage(value.get("usageMetadata"))
        candidates = value.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ResponseParseError("google.candidates must be a non-empty array")
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise ResponseParseError("google.candidates[0] must be an object")
        content_object = require_object(candidate.get("content"), "google.candidates[0].content")
        parts = content_object.get("parts")
        if not isinstance(parts, list):
            raise ResponseParseError("google candidate parts must be an array")
        visible: list[str] = []
        tool_calls: list[str] = []
        model_parts: list[Mapping[str, Any]] = []
        for index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                raise ResponseParseError(f"google candidate parts[{index}] must be an object")
            part_copy = dict(part)
            model_parts.append(part_copy)
            thought = part.get("thought", False)
            if not isinstance(thought, bool):
                raise ResponseParseError(f"google candidate parts[{index}].thought must be boolean")
            if thought:
                continue
            text = part.get("text")
            if text is not None:
                if not isinstance(text, str):
                    raise ResponseParseError(f"google candidate parts[{index}].text must be text")
                visible.append(text)
            for key in ("functionCall", "executableCode", "codeExecutionResult", "function_call"):
                call = part.get(key)
                if call is None:
                    continue
                if not isinstance(call, Mapping):
                    raise ResponseParseError(f"google candidate parts[{index}].{key} must be an object")
                name = call.get("name")
                tool_calls.append(
                    require_text(name, f"google candidate parts[{index}].{key}.name")
                    if name is not None
                    else key
                )
        finish_reason = candidate.get("finishReason", "STOP")
        if not isinstance(finish_reason, str):
            raise ResponseParseError("google finishReason must be a string")
        if finish_reason == "MAX_TOKENS":
            response_status = ResponseStatus.INCOMPLETE.value
        elif finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
            response_status = ResponseStatus.ERROR.value
        elif finish_reason in {"STOP", "OTHER", "FINISH_REASON_UNSPECIFIED"}:
            response_status = ResponseStatus.COMPLETED.value
        else:
            raise ResponseParseError(f"unknown Google finishReason: {finish_reason}")
        text = "".join(visible)
        if response_status == ResponseStatus.COMPLETED.value and not text and not tool_calls:
            raise ResponseParseError("completed Google response contains no visible text")
        return ProviderResponse(
            provider=PROVIDER,
            requested_model_id=requested_model_id,
            returned_model_id=returned_model,
            provider_response_id=response_id,
            text=text,
            usage=usage,
            response_status=response_status,
            incomplete_reason=finish_reason if response_status == ResponseStatus.INCOMPLETE.value else "",
            tool_calls=tuple(tool_calls),
            model_parts=tuple(model_parts),
            assistant_content=tuple(model_parts),
            error=finish_reason if response_status == ResponseStatus.ERROR.value else "",
        )

    def _parse_payload(self, body: bytes, request: GenerationRequest) -> ProviderResponse:
        return self.parse_response(body, request, requested_model_id=self.model_id)


def _parse_google_usage(value: Any) -> Usage:
    usage = require_object(value, "google.usageMetadata")
    input_tokens = require_nonnegative_int(usage.get("promptTokenCount"), "usage.promptTokenCount")
    output_tokens = require_nonnegative_int(usage.get("candidatesTokenCount"), "usage.candidatesTokenCount")
    total = usage.get("totalTokenCount")
    if total is not None:
        total = require_nonnegative_int(total, "usage.totalTokenCount")
    return Usage(input_tokens, output_tokens, total)


GoogleProvider = GoogleAdapter
GeminiAdapter = GoogleAdapter


__all__ = [
    "GOOGLE_GENERATE_CONTENT_ENDPOINT",
    "GOOGLE_MODEL",
    "GOOGLE_THINKING_LEVEL",
    "GeminiAdapter",
    "GoogleAdapter",
    "GoogleProvider",
]
