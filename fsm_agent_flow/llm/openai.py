"""OpenAI-compatible LLM adapter."""

from __future__ import annotations

import json
from typing import Any

from ..tools import ToolSpec
from .adapter import LLMResponse, Message, ToolCall


class OpenAIAdapter:
    """Adapter for the OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, **kwargs):
        try:
            import openai
        except ImportError:
            raise ImportError("Install openai: pip install 'fsm-agent-flow[openai]'")
        self.model = model
        self._client = openai.OpenAI(api_key=api_key, **kwargs)

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
        msgs = [self._convert_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
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

    @staticmethod
    def _convert_message(m: Message) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            msg["content"] = m.content
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            msg["name"] = m.name
        return msg
