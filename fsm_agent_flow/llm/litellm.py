"""LiteLLM adapter (multi-provider, OpenAI-compatible format)."""

from __future__ import annotations

import json
from typing import Any

from ..tools import ToolSpec
from .adapter import LLMResponse, Message, ToolCall


class LiteLLMAdapter:
    """Adapter using LiteLLM for multi-provider support.
    
    Args:
        model: Model string with provider prefix (e.g., "gemini/gemini-pro-latest")
        track_cost: If True, accumulates cost in .turn_cost attribute (resets on each chat call)
        **kwargs: Additional kwargs passed to litellm.completion()
    """

    def __init__(self, model: str = "gpt-4o", *, track_cost: bool = False, **kwargs):
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError("Install litellm: pip install 'fsm-agent-flow[litellm]'")
        self.model = model
        self._extra_kwargs = kwargs
        self._track_cost = track_cost
        self.turn_cost: float = 0.0

    def reset_turn(self) -> None:
        """Reset per-turn cost accumulator."""
        self.turn_cost = 0.0

    def format_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in tools]

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import litellm

        msgs = [_convert_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            # Enable thinking mode for Gemini to get thought_signature in tool calls
            # LiteLLM auto-extracts thought_signature into provider_specific_fields
            # and auto-injects it back when replaying assistant messages with tool_calls
            "reasoning_effort": "low",
            **self._extra_kwargs,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools

        response = litellm.completion(**kwargs)
        choice = response.choices[0]

        # Track cost using LiteLLM's built-in cost calculation (accumulate per turn)
        if self._track_cost:
            self.turn_cost += litellm.completion_cost(response)

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                # Extract provider_specific_fields (contains thought_signature for Gemini)
                psf = getattr(tc, 'provider_specific_fields', {}) or {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        provider_specific_fields=psf,
                    )
                )

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
        )


def _convert_message(m: Message) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        msg["content"] = m.content
    if m.tool_calls:
        tool_calls_converted = []
        for tc in m.tool_calls:
            tc_dict = {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            # Preserve provider_specific_fields (contains thought_signature for Gemini)
            # LiteLLM stores thought_signature here and auto-injects it when sending to Gemini
            if hasattr(tc, 'provider_specific_fields') and tc.provider_specific_fields:
                tc_dict["provider_specific_fields"] = tc.provider_specific_fields
            tool_calls_converted.append(tc_dict)
        msg["tool_calls"] = tool_calls_converted
    if m.tool_call_id is not None:
        msg["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        msg["name"] = m.name
    return msg
