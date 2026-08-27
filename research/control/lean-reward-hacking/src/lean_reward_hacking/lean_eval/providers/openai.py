"""OpenAI Responses API adapter used by the Lean-eval handoff."""

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
    require_object,
    require_text,
)


PROVIDER = "openai"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_MODELS = frozenset({"gpt-5.6-sol", "gpt-5.6-luna"})
OPENAI_REASONING_EFFORT = "medium"


class OpenAIAdapter(BaseProviderAdapter):
    """Transport-injected OpenAI Responses client."""

    provider = PROVIDER
    endpoint = OPENAI_RESPONSES_ENDPOINT
    credential_name = "OPENAI_API_KEY"

    def __init__(self, model_id: str = "gpt-5.6-sol", **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)

    def _validate_model_id(self, model_id: str) -> None:
        if model_id not in OPENAI_MODELS:
            raise ValueError("OpenAI adapter requires gpt-5.6-sol or gpt-5.6-luna")

    def _credential_headers(self, key: str) -> Mapping[str, str]:
        return {"Authorization": "Bearer " + key}

    def _build_payload(self, request: GenerationRequest) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "instructions": request.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.rendered_user_text()}],
                },
            ],
            "reasoning": {"effort": OPENAI_REASONING_EFFORT},
            "max_output_tokens": request.max_output_tokens,
            "tools": [],
            "tool_choice": "none",
            "store": True,
        }
        if request.previous_response_id is not None:
            payload["previous_response_id"] = request.previous_response_id
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    @staticmethod
    def parse_response(
        body: bytes,
        request: GenerationRequest,
        *,
        requested_model_id: str,
    ) -> ProviderResponse:
        value = parse_json_object(body, provider=PROVIDER)
        response_id = require_text(value.get("id"), "openai.id")
        returned_model = require_text(value.get("model"), "openai.model")
        reported_previous = value.get("previous_response_id")
        if reported_previous is not None:
            reported_previous = require_text(reported_previous, "openai.previous_response_id")
            if reported_previous != request.previous_response_id:
                raise ResponseParseError("OpenAI response belongs to a different response chain")

        raw_status = value.get("status", ResponseStatus.COMPLETED.value)
        if not isinstance(raw_status, str):
            raise ResponseParseError("openai.status must be a string")
        if raw_status == "completed":
            response_status = ResponseStatus.COMPLETED.value
        elif raw_status == "incomplete":
            response_status = ResponseStatus.INCOMPLETE.value
        elif raw_status in {"failed", "cancelled", "error"}:
            response_status = ResponseStatus.ERROR.value
        else:
            raise ResponseParseError(f"unknown OpenAI response status: {raw_status}")

        usage_value = value.get("usage")
        if usage_value is None and response_status == ResponseStatus.ERROR.value:
            usage = _zero_usage()
        else:
            usage = parse_usage(usage_value, provider=PROVIDER)

        fragments: list[str] = []
        tool_calls: list[str] = []
        assistant_content: list[Mapping[str, Any]] = []
        output = value.get("output")
        if output is not None:
            if not isinstance(output, list):
                raise ResponseParseError("openai.output must be an array")
            for index, item in enumerate(output):
                if not isinstance(item, Mapping):
                    raise ResponseParseError(f"openai.output[{index}] must be an object")
                item_type = item.get("type")
                if item_type in {"function_call", "custom_tool_call", "computer_call", "tool_call"}:
                    name = item.get("name")
                    tool_calls.append(
                        require_text(name, f"openai.output[{index}].name")
                        if name is not None
                        else str(item_type)
                    )
                content = item.get("content")
                if content is not None:
                    if not isinstance(content, list):
                        raise ResponseParseError(f"openai.output[{index}].content must be an array")
                    for block_index, block in enumerate(content):
                        if not isinstance(block, Mapping):
                            raise ResponseParseError(
                                f"openai.output[{index}].content[{block_index}] must be an object"
                            )
                        assistant_content.append(dict(block))
                        block_type = block.get("type")
                        if block_type in {"output_text", "text"}:
                            text = block.get("text")
                            if not isinstance(text, str):
                                raise ResponseParseError(
                                    f"openai.output[{index}].content[{block_index}].text must be text"
                                )
                            fragments.append(text)
                        elif block_type in {"function_call", "custom_tool_call"}:
                            name = block.get("name")
                            tool_calls.append(
                                require_text(name, "openai tool call name")
                                if name is not None
                                else str(block_type)
                            )
        output_text = value.get("output_text")
        if output_text is not None:
            if not isinstance(output_text, str):
                raise ResponseParseError("openai.output_text must be text")
            if not fragments:
                fragments.append(output_text)

        error = _error_marker(value.get("error"))
        incomplete_reason = ""
        incomplete_details = value.get("incomplete_details")
        if incomplete_details is not None:
            details = require_object(incomplete_details, "openai.incomplete_details")
            reason = details.get("reason")
            if reason is not None:
                incomplete_reason = require_text(reason, "openai.incomplete_details.reason")
        text = "".join(fragments)
        if response_status == ResponseStatus.COMPLETED.value and not text and not tool_calls and not error:
            raise ResponseParseError("completed OpenAI response contains no visible text")
        return ProviderResponse(
            provider=PROVIDER,
            requested_model_id=requested_model_id,
            returned_model_id=returned_model,
            provider_response_id=response_id,
            text=text,
            usage=usage,
            response_status=response_status,
            incomplete_reason=incomplete_reason,
            tool_calls=tuple(tool_calls),
            assistant_content=tuple(assistant_content),
            error=error,
        )

    def _parse_payload(self, body: bytes, request: GenerationRequest) -> ProviderResponse:
        return self.parse_response(body, request, requested_model_id=self.model_id)


def _zero_usage():
    from .base import Usage

    return Usage(0, 0)


def _error_marker(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        code = value.get("code")
        error_type = value.get("type")
        if isinstance(code, str) and code:
            return code[:128]
        if isinstance(error_type, str) and error_type:
            return error_type[:128]
        return "provider_error"
    if isinstance(value, str) and value:
        return value[:128]
    raise ResponseParseError("openai.error must be an object or string")


OpenAIProvider = OpenAIAdapter


__all__ = [
    "OPENAI_MODELS",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_RESPONSES_ENDPOINT",
    "OpenAIAdapter",
    "OpenAIProvider",
]
