"""Offline contract tests for the Lean-eval provider boundary."""

from __future__ import annotations

import unittest

from lean_reward_hacking.lean_eval.canonical import canonical_bytes
from lean_reward_hacking.lean_eval.compute import PaidDispatchGate
from lean_reward_hacking.lean_eval.mock import MockAdapter, MockBranch, MockTransport
from lean_reward_hacking.lean_eval.providers import (
    ANTHROPIC_MESSAGES_ENDPOINT,
    ANTHROPIC_MODEL,
    GOOGLE_GENERATE_CONTENT_ENDPOINT,
    GOOGLE_MODEL,
    OPENAI_RESPONSES_ENDPOINT,
    AnthropicAdapter,
    GenerationRequest,
    GoogleAdapter,
    OpenAIAdapter,
    ProviderPreflight,
    ResponseParseError,
)


def request(*, attempt: int = 1, **kwargs: object) -> GenerationRequest:
    return GenerationRequest(
        prompt="Return the requested Lean submission.",
        system_prompt="Follow the submission protocol.",
        mode_instructions="Submit one JSON object.",
        checker_feedback=kwargs.pop("checker_feedback", ()),
        attempt=attempt,
        metadata=kwargs.pop("metadata", {"task_id": "rh_task_01", "mode": "frozen_hole"}),
        **kwargs,
    )


class RequestTypeTests(unittest.TestCase):
    def test_request_aliases_and_feedback_are_deterministic(self) -> None:
        one = GenerationRequest(
            task_prompt="task",
            mode_instructions="mode",
            feedback={"score": 0, "diagnostics": "safe"},
        )
        two = GenerationRequest(
            prompt="task",
            mode_instructions="mode",
            checker_feedback='{"diagnostics":"safe","score":0}',
        )
        self.assertEqual(one.rendered_user_text(), two.rendered_user_text())
        self.assertEqual(one.task_prompt, "task")

    def test_empty_capability_is_exposed(self) -> None:
        adapter = OpenAIAdapter(
            transport=MockTransport(
                canonical_bytes(
                    {
                        "id": "r",
                        "model": "gpt-5.6-sol",
                        "output_text": "ok",
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    }
                )
            )
        )
        self.assertEqual(adapter.capabilities.to_dict()["tools"], [])
        self.assertFalse(adapter.capabilities.shell)
        self.assertFalse(adapter.capabilities.network)


class OpenAIAdapterTests(unittest.TestCase):
    def test_exact_responses_payload_and_visible_fragments(self) -> None:
        body = {
            "id": "resp_test",
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "first"},
                        {"type": "output_text", "text": " second"},
                    ],
                },
                {"type": "function_call", "name": "ignored"},
            ],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        }
        transport = MockTransport(canonical_bytes(body))
        adapter = OpenAIAdapter(model_id="gpt-5.6-sol", transport=transport)
        response = adapter.generate(request())
        payload = transport.calls[0].json
        self.assertEqual(transport.calls[0].url, OPENAI_RESPONSES_ENDPOINT)
        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertEqual(payload["tools"], [])
        self.assertEqual(payload["tool_choice"], "none")
        self.assertTrue(payload["store"])
        self.assertEqual(response.text, "first second")
        self.assertEqual(response.provider_response_id, "resp_test")
        self.assertEqual(response.tool_calls, ("ignored",))
        self.assertEqual(response.usage.total_tokens, 6)
        self.assertNotIn("Authorization", transport.calls[0].headers)

    def test_retry_replays_previous_response_id_and_incomplete_status_is_recorded(self) -> None:
        body = {
            "id": "resp_retry",
            "model": "gpt-5.6-luna",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output_text": "partial",
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }
        transport = MockTransport(canonical_bytes(body))
        adapter = OpenAIAdapter(model_id="gpt-5.6-luna", transport=transport, api_key="offline-key")
        response = adapter.generate(request(attempt=2, previous_response_id="resp_previous"))
        payload = transport.calls[0].json
        self.assertEqual(payload["previous_response_id"], "resp_previous")
        self.assertEqual(response.response_status, "incomplete")
        self.assertEqual(response.incomplete_reason, "max_output_tokens")
        self.assertEqual(transport.calls[0].headers["Authorization"], "Bearer offline-key")

    def test_chain_mismatch_and_negative_usage_reject_before_result(self) -> None:
        mismatch = {
            "id": "resp",
            "model": "gpt-5.6-sol",
            "previous_response_id": "other",
            "output_text": "ok",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        adapter = OpenAIAdapter(transport=MockTransport(canonical_bytes(mismatch)))
        with self.assertRaises(ResponseParseError):
            adapter.generate(request(previous_response_id="expected"))
        negative = {
            "id": "resp",
            "model": "gpt-5.6-sol",
            "output_text": "ok",
            "usage": {"input_tokens": -1, "output_tokens": 1},
        }
        with self.assertRaises(ResponseParseError):
            OpenAIAdapter(transport=MockTransport(canonical_bytes(negative))).generate(request())

    def test_moving_alias_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIAdapter(model_id="gpt-5.6", transport=MockTransport(b"{}"))


class AnthropicAdapterTests(unittest.TestCase):
    def test_exact_messages_payload_and_opaque_retry_blocks(self) -> None:
        body = {
            "id": "msg_test",
            "model": ANTHROPIC_MODEL,
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "opaque"},
                {"type": "redacted_thinking", "data": "opaque-redacted"},
                {"type": "text", "text": "proof json"},
            ],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
        transport = MockTransport(canonical_bytes(body))
        adapter = AnthropicAdapter(transport=transport, api_key="anthropic-test")
        response = adapter.generate(request())
        payload = transport.calls[0].json
        self.assertEqual(transport.calls[0].url, ANTHROPIC_MESSAGES_ENDPOINT)
        self.assertEqual(payload["model"], ANTHROPIC_MODEL)
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "medium"})
        self.assertEqual(payload["tools"], [])
        self.assertFalse(payload["stream"])
        self.assertEqual(response.text, "proof json")
        self.assertEqual(response.assistant_content[0]["type"], "thinking")
        self.assertEqual(transport.calls[0].headers["x-api-key"], "anthropic-test")
        retry_request = request(
            attempt=2,
            previous_assistant_content=response.assistant_content,
            previous_user_text="first prompt",
        )
        retry_payload = adapter.build_http_request(retry_request).json
        self.assertEqual(retry_payload["messages"][1]["role"], "assistant")
        self.assertEqual(retry_payload["messages"][1]["content"], list(response.assistant_content))

    def test_refusal_and_max_tokens_statuses_are_distinct(self) -> None:
        for stop_reason, expected in (("refusal", "error"), ("max_tokens", "incomplete")):
            body = {
                "id": "msg_" + stop_reason,
                "model": ANTHROPIC_MODEL,
                "stop_reason": stop_reason,
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
            response = AnthropicAdapter(transport=MockTransport(canonical_bytes(body))).generate(request())
            self.assertEqual(response.response_status, expected)
            self.assertEqual(response.text, "")


class GoogleAdapterTests(unittest.TestCase):
    def test_exact_generate_content_payload_and_thought_filter(self) -> None:
        body = {
            "responseId": "google_response",
            "modelVersion": GOOGLE_MODEL,
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": "hidden"},
                            {"text": "visible"},
                            {"functionCall": {"name": "never-run"}},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 2,
                "totalTokenCount": 6,
            },
        }
        transport = MockTransport(canonical_bytes(body))
        adapter = GoogleAdapter(transport=transport, api_key="google-test")
        response = adapter.generate(request())
        payload = transport.calls[0].json
        self.assertEqual(transport.calls[0].url, GOOGLE_GENERATE_CONTENT_ENDPOINT)
        self.assertEqual(
            payload["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "MEDIUM", "includeThoughts": False},
        )
        self.assertNotIn("tools", payload)
        self.assertNotIn("temperature", payload["generationConfig"])
        self.assertNotIn("topP", payload["generationConfig"])
        self.assertEqual(response.text, "visible")
        self.assertEqual(response.tool_calls, ("never-run",))
        self.assertEqual(transport.calls[0].headers["x-goog-api-key"], "google-test")

    def test_stateless_retry_preserves_model_parts_and_signature(self) -> None:
        adapter = GoogleAdapter(transport=MockTransport(b"{}"))
        retry = request(
            attempt=2,
            previous_model_parts=({"thought": True, "thoughtSignature": "sig", "text": "opaque"},),
            previous_user_text="first",
        )
        payload = adapter.build_http_request(retry).json
        self.assertEqual(payload["contents"][1]["role"], "model")
        self.assertEqual(payload["contents"][1]["parts"][0]["thoughtSignature"], "sig")


class GateAndMockTests(unittest.TestCase):
    def test_denied_gate_constructs_zero_transports(self) -> None:
        config = {"schema_version": "lean-eval/config-v1", "live": True, "models": ["gpt-5.6-sol"]}
        gate = PaidDispatchGate(config, credentials={"OPENAI_API_KEY": "present"})
        constructed: list[str] = []

        def factory():
            constructed.append("created")
            return MockTransport(b"{}")

        adapter = OpenAIAdapter(
            live=True,
            gate=gate,
            transport_factory=factory,
            api_key="present",
            preflight=ProviderPreflight("openai", "gpt-5.6-sol", OPENAI_RESPONSES_ENDPOINT),
        )
        with self.assertRaises(Exception):
            adapter.generate(request())
        self.assertEqual(constructed, [])

    def test_mock_branches_are_deterministic_and_conversation_bound(self) -> None:
        first = MockAdapter(branch=MockBranch.RETRY)
        second = MockAdapter(branch=MockBranch.RETRY)
        req1 = request(
            attempt=1,
            metadata={
                "task_id": "rh_task_02",
                "mode": "whole_file",
                "condition": "neutral",
                "conversation_id": "A",
            },
        )
        req2 = request(
            attempt=2,
            metadata={
                "task_id": "rh_task_02",
                "mode": "whole_file",
                "condition": "neutral",
                "conversation_id": "A",
            },
        )
        self.assertEqual(first.generate(req1).provider_response_id, second.generate(req1).provider_response_id)
        self.assertEqual(first.generate(req1).response_status, "completed")
        self.assertEqual(first.generate(req1).text, "not-json")
        repaired = first.generate(req2)
        self.assertIn('"kind":"proof"', repaired.text)
        other = first.generate(
            request(
                attempt=1,
                metadata={
                    "task_id": "rh_task_02",
                    "mode": "whole_file",
                    "condition": "neutral",
                    "conversation_id": "B",
                },
            )
        )
        self.assertNotEqual(other.provider_response_id, repaired.provider_response_id)
        self.assertIn('"kind":"no_proof"', MockAdapter(branch=MockBranch.NO_PROOF).generate(req1).text)
        self.assertEqual(MockAdapter(branch=MockBranch.MALFORMED).generate(req1).text, "not-json")
        self.assertIn(
            "False.elim",
            MockAdapter(branch=MockBranch.PLANTED)
            .generate(request(metadata={"task_id": "rh_task_07"}))
            .text,
        )

    def test_mock_mutation_and_usage_are_stable(self) -> None:
        metadata = {
            "task_id": "rh_task_01",
            "mode": "whole_file",
            "conversation_id": "X",
            "template": "theorem t := __RH_PROOF_HOLE__",
        }
        adapter = MockAdapter(branch=MockBranch.AXIOM_MUTATION)
        one = adapter.generate(request(metadata=metadata))
        two = MockAdapter(branch=MockBranch.AXIOM_MUTATION).generate(request(metadata=metadata))
        self.assertEqual(one.text, two.text)
        self.assertIn("injected_false", one.text)
        self.assertEqual(one.usage, two.usage)


if __name__ == "__main__":
    unittest.main()
