"""Anthropic Messages API adapter for the Lean-eval provider boundary."""

from __future__ import annotations

from typing import Any, Mapping

from .base import (
    BaseProviderAdapter,
    GenerationRequest,
    ProviderResponse,
    ResponseParseError,
    ResponseStatus,
    parse_json_object,
    parse_usage,
    require_text,
)


PROVIDER = "anthropic"
ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-opus-5"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_EFFORT = "medium"


class AnthropicAdapter(BaseProviderAdapter):
    """Transport-injected Claude Messages client."""

    provider = PROVIDER
    endpoint = ANTHROPIC_MESSAGES_ENDPOINT
    credential_name = "ANTHROPIC_API_KEY"

    def __init__(self, model_id: str = ANTHROPIC_MODEL, **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)

    def _validate_model_id(self, model_id: str) -> None:
        if model_id != ANTHROPIC_MODEL:
            raise ValueError("Anthropic adapter requires claude-opus-5")

    def _credential_headers(self, key: str) -> Mapping[str, str]:
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}

    def _build_payload(self, request: GenerationRequest) -> Mapping[str, Any]:
        current_message = {
            "role": "user",
            "content": [{"type": "text", "text": request.rendered_user_text()}],
        }
        messages: list[Mapping[str, Any]] = []
        if request.previous_assistant_content:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": request.previous_user_text or request.rendered_user_text(),
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [dict(block) for block in request.previous_assistant_content],
                }
            )
        messages.append(current_message)
        return {
            "model": self.model_id,
            "max_tokens": request.max_output_tokens,
            "system": request.system_prompt,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": ANTHROPIC_EFFORT},
            "tools": [],
            "stream": False,
        }

    @staticmethod
    def parse_response(
        body: bytes,
        request: GenerationRequest,
        *,
        requested_model_id: str,
    ) -> ProviderResponse:
        value = parse_json_object(body, provider=PROVIDER)
        response_id = require_text(value.get("id"), "anthropic.id")
        returned_model = require_text(value.get("model"), "anthropic.model")
        error = _error_marker(value.get("error"))
        if "usage" not in value and error:
            from .base import Usage

            usage = Usage(0, 0)
        else:
            usage = parse_usage(value.get("usage"), provider=PROVIDER)
        content = value.get("content")
        if not isinstance(content, list):
            raise ResponseParseError("anthropic.content must be an array")
        visible: list[str] = []
        tool_calls: list[str] = []
        blocks: list[Mapping[str, Any]] = []
        for index, block in enumerate(content):
            if not isinstance(block, Mapping):
                raise ResponseParseError(f"anthropic.content[{index}] must be an object")
            block_copy = dict(block)
            blocks.append(block_copy)
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ResponseParseError(f"anthropic.content[{index}].text must be text")
                visible.append(text)
            elif block_type in {"tool_use", "server_tool_use", "computer_call"}:
                name = block.get("name")
                tool_calls.append(
                    require_text(name, f"anthropic.content[{index}].name")
                    if name is not None
                    else str(block_type)
                )
            elif block_type in {"thinking", "redacted_thinking"}:
                continue
            elif block_type is not None and not isinstance(block_type, str):
                raise ResponseParseError(f"anthropic.content[{index}].type must be text")
        stop_reason = value.get("stop_reason", "end_turn")
        if not isinstance(stop_reason, str):
            raise ResponseParseError("anthropic.stop_reason must be a string")
        if stop_reason in {"max_tokens", "pause_turn"}:
            response_status = ResponseStatus.INCOMPLETE.value
        elif stop_reason == "refusal":
            response_status = ResponseStatus.ERROR.value
        elif stop_reason in {"end_turn", "stop_sequence", "tool_use"}:
            response_status = ResponseStatus.COMPLETED.value
        else:
            raise ResponseParseError(f"unknown Anthropic stop_reason: {stop_reason}")
        if error:
            response_status = ResponseStatus.ERROR.value
        text = "".join(visible)
        if response_status == ResponseStatus.COMPLETED.value and not text and not tool_calls and not error:
            raise ResponseParseError("completed Anthropic response contains no visible text")
        return ProviderResponse(
            provider=PROVIDER,
            requested_model_id=requested_model_id,
            returned_model_id=returned_model,
            provider_response_id=response_id,
            text=text,
            usage=usage,
            response_status=response_status,
            incomplete_reason=stop_reason if response_status == ResponseStatus.INCOMPLETE.value else "",
            tool_calls=tuple(tool_calls),
            assistant_content=tuple(blocks),
            error=error or ("refusal" if stop_reason == "refusal" else ""),
        )

    def _parse_payload(self, body: bytes, request: GenerationRequest) -> ProviderResponse:
        return self.parse_response(body, request, requested_model_id=self.model_id)


def _error_marker(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        error_type = value.get("type")
        if isinstance(error_type, str) and error_type:
            return error_type[:128]
        return "provider_error"
    if isinstance(value, str) and value:
        return value[:128]
    raise ResponseParseError("anthropic.error must be an object or string")


AnthropicProvider = AnthropicAdapter


__all__ = [
    "ANTHROPIC_EFFORT",
    "ANTHROPIC_MESSAGES_ENDPOINT",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_VERSION",
    "AnthropicAdapter",
    "AnthropicProvider",
]
